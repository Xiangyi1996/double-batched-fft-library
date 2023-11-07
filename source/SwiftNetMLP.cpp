#define TM 8
#define TK 16
#define TN 16
// #define TM 8
// #define TK 16
// #define TN 8
#define SKEW 0
#define SG_SIZE 16
// #define SG_SIZE 8

// #define SMALL_BATCH_SIZES

#include "SwiftNetMLP.h"
#include "common.h"
#include "common_device.h"
#include "mkl.h"
#include "mkl_omp_offload.h"
#include "oneapi/mkl.hpp"
#include "trainer.h"
#include <CL/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
using bf16 = sycl::ext::oneapi::bfloat16;
using namespace tinydpcppnn::builtin;

// fused operation which does forward_pass+error computation + backward pass
template <int WIDTH, Activation activation, Activation output_activation>
std::vector<sycl::event> mlp_swift_fused(queue &q, bf16 const *const __restrict__ weights_ptr,
                                         bf16 const *const __restrict__ weightsT_ptr,
                                         bf16 const *const __restrict__ inputs_ptr,  // input to forward pass
                                         bf16 const *const __restrict__ targets_ptr, // targets for error computation
                                         bf16 *const __restrict__ output_ptr, // gradients output after backward pass
                                         bf16 *const __restrict__ intermediate_output_forward,
                                         bf16 *const __restrict__ intermediate_output_backward,
                                         const int n_hidden_layers, const int M, const std::vector<sycl::event> &deps) {
    // reuse of B, this is in subgroups, ONLY works for 64 nor
    // note that large grf mode requires this to be set to 32
    const int SGS_IN_WG = 64;
    // dimensions are M = batch_size, N = WIDTH = K = 64;
    static_assert(TK == SG_SIZE);
    static_assert(TN == TK);
    if constexpr (WIDTH != 64) throw std::invalid_argument("Current implementation only works for a WIDTH of 64");
    assert(M % TM == 0); // make sure there is no remainder and no out of bounds accesses
    // Note that TN = TK = SG_SIZE

    // One Block Row has TM rows an N columns.
    auto e = q.submit([&](handler &cgh) {
        cgh.depends_on(deps);
        local_accessor<bf16, 1> B(range<1>(WIDTH * WIDTH),
                                  cgh); // weights matrix. 64*64*2 byte = 8 kb. Thus, can have up to 16 WGs per Xe Core.
        local_accessor<bf16, 1> Atmp(range<1>(TM * WIDTH * SGS_IN_WG),
                                     cgh); // buffer for loading joint matrices. 8*64*64*2byte = 64kb. TODO: check if
                                           // this is too much. If so, split in half
        // number of SGS is given by batch_size / (TM), since batch_size is the number of rows in the output

        cgh.parallel_for(
            nd_range<1>(std::max(M / TM * SG_SIZE, SGS_IN_WG * SG_SIZE),
                        SGS_IN_WG * SG_SIZE), // assuming here that the number of block rows is divisable by SGS_IN_WG
            [=](nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                const int wg_id = item.get_group().get_group_id();
                auto sg = item.get_sub_group();
                const uint16_t loc_id = sg.get_local_id()[0]; // is in 0-15
                const uint16_t sg_id = sg.get_group_id()[0];  // is in 0-63
#ifdef SMALL_BATCH_SIZES
                const bool sg_has_no_blockrow = (wg_id * TM * SGS_IN_WG + sg_id * TM) >= M;
#endif

                /// Start with forward pass

                const uint16_t sg_offset_B =
                    sg_id * WIDTH *
                    (WIDTH / SGS_IN_WG); // we assume SGS_IN_WG divides K //is in the rang 0-64*64=0-4096
                const uint16_t sg_offset_A = sg_id * WIDTH * TM; // offset in WG is in the range 0-64*64*8=0-32K
                const int total_offset_A = wg_id * SGS_IN_WG * WIDTH * TM + sg_offset_A; // offset in current block
                __builtin_IB_lsc_prefetch_global_uint4(
                    (const __attribute__((opencl_global)) uint32_t *)(inputs_ptr + total_offset_A) + loc_id, 0,
                    LSC_LDCC_L1C_L3UC);
                __builtin_IB_lsc_prefetch_global_uint4(
                    (const __attribute__((opencl_global)) uint32_t *)(inputs_ptr + total_offset_A) + loc_id +
                        4 * SG_SIZE,
                    0, LSC_LDCC_L1C_L3UC);
                __builtin_IB_lsc_prefetch_global_uint4(
                    (const __attribute__((opencl_global)) uint32_t *)(inputs_ptr + total_offset_A) + loc_id +
                        8 * SG_SIZE,
                    0, LSC_LDCC_L1C_L3UC);
                __builtin_IB_lsc_prefetch_global_uint4(
                    (const __attribute__((opencl_global)) uint32_t *)(inputs_ptr + total_offset_A) + loc_id +
                        16 * SG_SIZE,
                    0, LSC_LDCC_L1C_L3UC);

                // Load B into slm
                /// ATTENTION: this version only works for K = SGS_IN_WG and NBLOCKCOLS_PER_SG = 4
                ((int32_t *)(&B[sg_offset_B]))[loc_id] = ((int32_t *)(weights_ptr + sg_offset_B))[loc_id];
                ((int32_t *)(&B[sg_offset_B]))[loc_id + SG_SIZE] =
                    ((int32_t *)(weights_ptr + sg_offset_B))[loc_id + SG_SIZE];

// load input
/// Alternative of loading A through SLM to avoid inefficient access to HBM
#ifdef SMALL_BATCH_SIZES
                if (!sg_has_no_blockrow)
#endif
                {
                    sycl::vec<int32_t, TM> tmp16avalues0 =
                        sg.load<8>(global_ptr<int32_t>((int32_t *)(inputs_ptr + total_offset_A)));
                    sycl::vec<int32_t, TM> tmp16avalues1 =
                        sg.load<8>(global_ptr<int32_t>((int32_t *)(inputs_ptr + total_offset_A + 4 * WIDTH)));

                    sg.store<8>(local_ptr<int32_t>((int32_t *)&Atmp[sg_offset_A]), tmp16avalues0);
                    sg.store<8>(local_ptr<int32_t>((int32_t *)(&Atmp[sg_offset_A + 4 * WIDTH])), tmp16avalues1);
                    // we do not need SLM barrier since each SG writes and reads only its own data.
                    // load A matrix from slm to joint_matrices. This is done to avoid inefficient bf16 HBM access

                    // ATTENTION: current inputs are positive and this is not really tested
                    if constexpr (activation == Activation::ReLU) {
                        // update tmp32avalues with bit shenanigans to avoid one pass through slm
                        // Basically just checks the sign bits of the two bf16 values stored in one int32_t
                        int32_t bitmask = 0b11111111111111110000000000000000;
                        for (uint8_t iter = 0; iter < TM; iter++) {
                            if ((tmp16avalues0[iter] >> 31) & 1) tmp16avalues0[iter] &= ~bitmask;
                            if ((tmp16avalues0[iter] >> 15) & 1) tmp16avalues0[iter] &= bitmask;

                            if ((tmp16avalues1[iter] >> 31) & 1) tmp16avalues1[iter] &= ~bitmask;
                            if ((tmp16avalues1[iter] >> 15) & 1) tmp16avalues1[iter] &= bitmask;
                        }
                    }
                    // else {} //nothing to be done, implement other activations here

                    sg.store<8>(global_ptr<int32_t>((int32_t *)(intermediate_output_forward + total_offset_A)),
                                tmp16avalues0);
                    sg.store<8>(
                        global_ptr<int32_t>((int32_t *)(intermediate_output_forward + total_offset_A + 4 * WIDTH)),
                        tmp16avalues1);
                }

                joint_matrix<sub_group, bf16, use::a, TM, TK, layout::row_major> A_block0, A_block1, A_block2, A_block3;
#ifdef SMALL_BATCH_SIZES
                if (!sg_has_no_blockrow)
#endif
                {
                    joint_matrix_load(sg, A_block0, local_ptr<bf16>(&Atmp[sg_offset_A + 0 * SG_SIZE]), WIDTH);
                    joint_matrix_load(sg, A_block1, local_ptr<bf16>(&Atmp[sg_offset_A + 1 * SG_SIZE]), WIDTH);
                    joint_matrix_load(sg, A_block2, local_ptr<bf16>(&Atmp[sg_offset_A + 2 * SG_SIZE]), WIDTH);
                    joint_matrix_load(sg, A_block3, local_ptr<bf16>(&Atmp[sg_offset_A + 3 * SG_SIZE]), WIDTH);
                }

                // We have n_hidden_layers. Thus n_hidden_layers - 1 gemms between
                // the layers (layer 0 -> GEMM -> layer1 -> GEMM -> layer2 -> etc.)
                // Since we also do the GEMM from input to hidden layer 0,
                // we perform n_hidden_layers GEMMS.
                joint_matrix<sub_group, bf16, use::b, TK, TN, sycl::ext::intel::experimental::matrix::layout::packed>
                    B_block;
                joint_matrix<sub_group, float, use::accumulator, TM, TN> C_block0, C_block1, C_block2, C_block3;

                for (uint8_t layer = 0; layer < n_hidden_layers; layer++) {

// reset result matrix
#ifdef SMALL_BATCH_SIZES
                    if (!sg_has_no_blockrow)
#endif
                    {
                        joint_matrix_fill(sg, C_block0, 0.0f);
                        joint_matrix_fill(sg, C_block1, 0.0f);
                        joint_matrix_fill(sg, C_block2, 0.0f);
                        joint_matrix_fill(sg, C_block3, 0.0f);
                    }

                    item.barrier(sycl::access::fence_space::local_space); // wait for B to be loaded

// block axpy scheme
#ifdef SMALL_BATCH_SIZES
                    if (!sg_has_no_blockrow)
#endif
                    {
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0 * WIDTH * TK + 0 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block0 = joint_matrix_mad(sg, A_block0, B_block, C_block0);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0 * WIDTH * TK + 1 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block1 = joint_matrix_mad(sg, A_block0, B_block, C_block1);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0 * WIDTH * TK + 2 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block2 = joint_matrix_mad(sg, A_block0, B_block, C_block2);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0 * WIDTH * TK + 3 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block3 = joint_matrix_mad(sg, A_block0, B_block, C_block3);

                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[1 * WIDTH * TK + 0 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block0 = joint_matrix_mad(sg, A_block1, B_block, C_block0);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[1 * WIDTH * TK + 1 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block1 = joint_matrix_mad(sg, A_block1, B_block, C_block1);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[1 * WIDTH * TK + 3 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block2 = joint_matrix_mad(sg, A_block1, B_block, C_block2);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[1 * WIDTH * TK + 0 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block3 = joint_matrix_mad(sg, A_block1, B_block, C_block3);

                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[2 * WIDTH * TK + 0 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block0 = joint_matrix_mad(sg, A_block2, B_block, C_block0);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[2 * WIDTH * TK + 1 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block1 = joint_matrix_mad(sg, A_block2, B_block, C_block1);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[2 * WIDTH * TK + 2 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block2 = joint_matrix_mad(sg, A_block2, B_block, C_block2);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[2 * WIDTH * TK + 3 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block3 = joint_matrix_mad(sg, A_block2, B_block, C_block3);

                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[3 * WIDTH * TK + 0 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block0 = joint_matrix_mad(sg, A_block3, B_block, C_block0);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[3 * WIDTH * TK + 1 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block1 = joint_matrix_mad(sg, A_block3, B_block, C_block1);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[3 * WIDTH * TK + 2 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block2 = joint_matrix_mad(sg, A_block3, B_block, C_block2);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[3 * WIDTH * TK + 3 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block3 = joint_matrix_mad(sg, A_block3, B_block, C_block3);
                    }

                    // Load next B into slm
                    /// ATTENTION: this version only works for K = SGS_IN_WG and NBLOCKCOLS_PER_SG = 4
                    item.barrier(sycl::access::fence_space::local_space);
                    sg.store<2>(local_ptr<int32_t>((int32_t *)(&B[sg_offset_B])),
                                sg.load<2>(global_ptr<int32_t>(
                                    (int32_t *)(weights_ptr + (layer + 1) * WIDTH * WIDTH + sg_offset_B))));

// This can be done in the future with joint_matrix_copy and a joint_maitrx_apply.
#ifdef SMALL_BATCH_SIZES
                    if (!sg_has_no_blockrow)
#endif
                    {
                        auto Ci_data0 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block0);
                        auto Ai_data0 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block0);
                        auto Ci_data1 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block1);
                        auto Ai_data1 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block1);
                        auto Ci_data2 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block2);
                        auto Ai_data2 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block2);
                        auto Ci_data3 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block3);
                        auto Ai_data3 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block3);
                        // if constexpr (activation == Activation::ReLU)
                        //{
                        for (int rowiter = 0; rowiter < Ci_data0.length(); rowiter++) // should be TM in length
                        {
                            Ai_data0[rowiter] =
                                fmax((bf16)0, (bf16)Ci_data0[rowiter]); // tmpCi < (bf16)0 ? (bf16)0 : tmpCi;
                            Ai_data1[rowiter] = fmax((bf16)0, (bf16)Ci_data1[rowiter]);
                            Ai_data2[rowiter] = fmax((bf16)0, (bf16)Ci_data2[rowiter]);
                            Ai_data3[rowiter] = fmax((bf16)0, (bf16)Ci_data3[rowiter]);
                        }
                        //}
                        // else if constexpr (activation == Activation::None)
                        // {
                        //     for (int rowiter = 0; rowiter < Ci_data0.length(); rowiter++) //should be TM in length
                        //     {
                        //         Ai_data0[rowiter] = (bf16)Ci_data0[rowiter];
                        //         Ai_data1[rowiter] = (bf16)Ci_data1[rowiter];
                        //         Ai_data2[rowiter] = (bf16)Ci_data2[rowiter];
                        //         Ai_data3[rowiter] = (bf16)Ci_data3[rowiter];
                        //     }
                        // }
                        // //else none

                        sycl::ext::intel::experimental::matrix::joint_matrix_store(
                            sg, A_block0, local_ptr<bf16>(&Atmp[sg_offset_A + 0 * SG_SIZE]), WIDTH);
                        sycl::ext::intel::experimental::matrix::joint_matrix_store(
                            sg, A_block1, local_ptr<bf16>(&Atmp[sg_offset_A + 1 * SG_SIZE]), WIDTH);
                        sycl::ext::intel::experimental::matrix::joint_matrix_store(
                            sg, A_block2, local_ptr<bf16>(&Atmp[sg_offset_A + 2 * SG_SIZE]), WIDTH);
                        sycl::ext::intel::experimental::matrix::joint_matrix_store(
                            sg, A_block3, local_ptr<bf16>(&Atmp[sg_offset_A + 3 * SG_SIZE]), WIDTH);
                        /// Alternative of loading A through SLM to avoid inefficient access to HBM
                        // we do not need SLM barrier since each SG writes and reads only its own data.

                        // for (uint8_t iter = 0; iter < TM; iter++) {
                        //     *((int32_t*)(intermediate_output_forward+loc_offset_A+iter*WIDTH)+loc_id) =
                        //     *(((int32_t*)&Atmp[sg_offset_A+iter*WIDTH])+loc_id);
                        //     *((int32_t*)(intermediate_output_forward+loc_offset_A+iter*WIDTH)+loc_id + SG_SIZE) =
                        //     *(((int32_t*)&Atmp[sg_offset_A+iter*WIDTH])+loc_id+SG_SIZE);
                        // }
                        const int loc_offset_A = (layer + 1) * M * WIDTH + total_offset_A;
                        sg.store<8>(global_ptr<int32_t>((int32_t *)(intermediate_output_forward + loc_offset_A)),
                                    sg.load<8>(local_ptr<int32_t>((int32_t *)(&Atmp[sg_offset_A]))));
                        sg.store<8>(
                            global_ptr<int32_t>((int32_t *)(intermediate_output_forward + loc_offset_A + 4 * WIDTH)),
                            sg.load<8>(local_ptr<int32_t>((int32_t *)(&Atmp[sg_offset_A + 4 * WIDTH]))));
                    }
                }

// generate output, i.e. last GEMM, differs since it uses output_activation
// reset result matrix
#ifdef SMALL_BATCH_SIZES
                if (!sg_has_no_blockrow)
#endif
                {
                    joint_matrix_fill(sg, C_block0, 0.0f);
                    joint_matrix_fill(sg, C_block1, 0.0f);
                    joint_matrix_fill(sg, C_block2, 0.0f);
                    joint_matrix_fill(sg, C_block3, 0.0f);
                }

                item.barrier(sycl::access::fence_space::local_space); // wait for B to be loaded

// block axpy scheme
#ifdef SMALL_BATCH_SIZES
                if (!sg_has_no_blockrow)
#endif
                {
                    joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0 * WIDTH * TK + 0 * 2 * TN]),
                                      2 * WIDTH); // 2*TN since weights are in VNNI format
                    C_block0 = joint_matrix_mad(sg, A_block0, B_block, C_block0);
                    joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0 * WIDTH * TK + 1 * 2 * TN]),
                                      2 * WIDTH); // 2*TN since weights are in VNNI format
                    C_block1 = joint_matrix_mad(sg, A_block0, B_block, C_block1);
                    joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0 * WIDTH * TK + 2 * 2 * TN]),
                                      2 * WIDTH); // 2*TN since weights are in VNNI format
                    C_block2 = joint_matrix_mad(sg, A_block0, B_block, C_block2);
                    joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0 * WIDTH * TK + 3 * 2 * TN]),
                                      2 * WIDTH); // 2*TN since weights are in VNNI format
                    C_block3 = joint_matrix_mad(sg, A_block0, B_block, C_block3);

                    joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[1 * WIDTH * TK + 0 * 2 * TN]),
                                      2 * WIDTH); // 2*TN since weights are in VNNI format
                    C_block0 = joint_matrix_mad(sg, A_block1, B_block, C_block0);
                    joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[1 * WIDTH * TK + 1 * 2 * TN]),
                                      2 * WIDTH); // 2*TN since weights are in VNNI format
                    C_block1 = joint_matrix_mad(sg, A_block1, B_block, C_block1);
                    joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[1 * WIDTH * TK + 2 * 2 * TN]),
                                      2 * WIDTH); // 2*TN since weights are in VNNI format
                    C_block2 = joint_matrix_mad(sg, A_block1, B_block, C_block2);
                    joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[1 * WIDTH * TK + 3 * 2 * TN]),
                                      2 * WIDTH); // 2*TN since weights are in VNNI format
                    C_block3 = joint_matrix_mad(sg, A_block1, B_block, C_block3);

                    joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[2 * WIDTH * TK + 0 * 2 * TN]),
                                      2 * WIDTH); // 2*TN since weights are in VNNI format
                    C_block0 = joint_matrix_mad(sg, A_block2, B_block, C_block0);
                    joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[2 * WIDTH * TK + 1 * 2 * TN]),
                                      2 * WIDTH); // 2*TN since weights are in VNNI format
                    C_block1 = joint_matrix_mad(sg, A_block2, B_block, C_block1);
                    joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[2 * WIDTH * TK + 2 * 2 * TN]),
                                      2 * WIDTH); // 2*TN since weights are in VNNI format
                    C_block2 = joint_matrix_mad(sg, A_block2, B_block, C_block2);
                    joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[2 * WIDTH * TK + 3 * 2 * TN]),
                                      2 * WIDTH); // 2*TN since weights are in VNNI format
                    C_block3 = joint_matrix_mad(sg, A_block2, B_block, C_block3);

                    joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[3 * WIDTH * TK + 0 * 2 * TN]),
                                      2 * WIDTH); // 2*TN since weights are in VNNI format
                    C_block0 = joint_matrix_mad(sg, A_block3, B_block, C_block0);
                    joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[3 * WIDTH * TK + 1 * 2 * TN]),
                                      2 * WIDTH); // 2*TN since weights are in VNNI format
                    C_block1 = joint_matrix_mad(sg, A_block3, B_block, C_block1);
                    joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[3 * WIDTH * TK + 2 * 2 * TN]),
                                      2 * WIDTH); // 2*TN since weights are in VNNI format
                    C_block2 = joint_matrix_mad(sg, A_block3, B_block, C_block2);
                    joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[3 * WIDTH * TK + 3 * 2 * TN]),
                                      2 * WIDTH); // 2*TN since weights are in VNNI format
                    C_block3 = joint_matrix_mad(sg, A_block3, B_block, C_block3);

                    /// TODO: Output activation in what follows. Here == None
                    // This can be done in the future with joint_matrix_copy and a joint_maitrx_apply.
                    // This can be done in the future with joint_matrix_copy and a joint_maitrx_apply.
                    auto Ci_data0 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block0);
                    auto Ai_data0 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block0);
                    auto Ci_data1 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block1);
                    auto Ai_data1 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block1);
                    auto Ci_data2 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block2);
                    auto Ai_data2 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block2);
                    auto Ci_data3 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block3);
                    auto Ai_data3 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block3);
                    // if constexpr (/*output_activation == Activation::ReLU*/false) //output activation is always none
                    // for this test. We do not have a template parameter for this
                    // {
                    //     for (int rowiter = 0; rowiter < Ci_data0.length(); rowiter++) //should be TM in length
                    //     {
                    //         Ai_data0[rowiter] = fmax((bf16)0, (bf16)Ci_data0[rowiter]);//tmpCi < (bf16)0 ? (bf16)0 :
                    //         tmpCi; Ai_data1[rowiter] = fmax((bf16)0, (bf16)Ci_data1[rowiter]); Ai_data2[rowiter] =
                    //         fmax((bf16)0, (bf16)Ci_data2[rowiter]); Ai_data3[rowiter] = fmax((bf16)0,
                    //         (bf16)Ci_data3[rowiter]);
                    //     }
                    // }
                    // else if constexpr (/*output_activation == Activation::None*/true) //for this tests, alway
                    // activation true
                    // {
                    for (uint8_t rowiter = 0; rowiter < Ci_data0.length(); rowiter++) // should be TM in length
                    {
                        Ai_data0[rowiter] = (bf16)Ci_data0[rowiter];
                        Ai_data1[rowiter] = (bf16)Ci_data1[rowiter];
                        Ai_data2[rowiter] = (bf16)Ci_data2[rowiter];
                        Ai_data3[rowiter] = (bf16)Ci_data3[rowiter];
                    }
                    //}

                    const int loc_offset_A = (n_hidden_layers + 1) * M * WIDTH + total_offset_A;
                    // load A matrix from slm to joint_matrices. This is done to avoid inefficient bf16 HBM access
                    sycl::ext::intel::experimental::matrix::joint_matrix_store(
                        sg, A_block0, local_ptr<bf16>(&Atmp[sg_offset_A + 0 * SG_SIZE]), WIDTH);
                    sycl::ext::intel::experimental::matrix::joint_matrix_store(
                        sg, A_block1, local_ptr<bf16>(&Atmp[sg_offset_A + 1 * SG_SIZE]), WIDTH);
                    sycl::ext::intel::experimental::matrix::joint_matrix_store(
                        sg, A_block2, local_ptr<bf16>(&Atmp[sg_offset_A + 2 * SG_SIZE]), WIDTH);
                    sycl::ext::intel::experimental::matrix::joint_matrix_store(
                        sg, A_block3, local_ptr<bf16>(&Atmp[sg_offset_A + 3 * SG_SIZE]), WIDTH);
                    /// Alternative of loading A through SLM to avoid inefficient access to HBM
                    // we do not need SLM barrier since each SG writes and reads only its own data.
                    //  for (uint8_t iter = 0; iter < TM; iter++) {
                    //      *((int32_t*)(intermediate_output_forward+loc_offset_A+iter*WIDTH)+loc_id) =
                    //      *(((int32_t*)&Atmp[sg_offset_A+iter*WIDTH])+loc_id);
                    //      *((int32_t*)(intermediate_output_forward+loc_offset_A+iter*WIDTH)+loc_id + SG_SIZE) =
                    //      *(((int32_t*)&Atmp[sg_offset_A+iter*WIDTH])+loc_id+SG_SIZE);
                    //  }
                    sg.store<8>(global_ptr<int32_t>((int32_t *)(intermediate_output_forward + loc_offset_A)),
                                sg.load<8>(local_ptr<int32_t>((int32_t *)(&Atmp[sg_offset_A]))));
                    sg.store<8>(
                        global_ptr<int32_t>((int32_t *)(intermediate_output_forward + loc_offset_A + 4 * WIDTH)),
                        sg.load<8>(local_ptr<int32_t>((int32_t *)(&Atmp[sg_offset_A + 4 * WIDTH]))));
                }

            /// Compute L2 loss and gradients as input for backward pass
#ifdef SMALL_BATCH_SIZES
                if (!sg_has_no_blockrow)
#endif
                {
                    const float inv_N_total_elements = 2.0f / (M * WIDTH);
                    for (int elemiter = 0; elemiter < TM * WIDTH / 2; elemiter += SG_SIZE) { // row
                        const int32_t tmp_target =
                            sg.load((int32_t *)(targets_ptr + total_offset_A) + elemiter); // hbm access. May be slow
                        int32_t tmp_source = sg.load((int32_t *)&Atmp[sg_offset_A + 2 * elemiter]);
                        ((bf16 *)&tmp_source)[0] -= ((bf16 *)&tmp_target)[0];
                        ((bf16 *)&tmp_source)[1] -= ((bf16 *)&tmp_target)[1];
                        ((bf16 *)&tmp_source)[0] *= inv_N_total_elements;
                        ((bf16 *)&tmp_source)[1] *= inv_N_total_elements;
                        sg.store(((int32_t *)&Atmp[sg_offset_A + 2 * elemiter]), tmp_source);
                    }
                    // //for testing purposed, we just take the target as input
                    // for (int elemiter = 0; elemiter < TM*K/2; elemiter+=SG_SIZE) { //row
                    //     int32_t tmp_target = ((int32_t*)(targets_ptr+total_offset_A))[elemiter + loc_id]; //hbm
                    //     access. May be slow Atmp[sg_offset_A + 2*elemiter + 2*loc_id] = ((bf16*)&tmp_target)[0];
                    //     Atmp[sg_offset_A + 2*elemiter + 2*loc_id+1] = ((bf16*)&tmp_target)[1];
                    // }
                }
                // A tmp now holds the grads. We can start the backward pass.

                /// ATTENTION: this version only works for K = SGS_IN_WG and NBLOCKCOLS_PER_SG = 4
                item.barrier(sycl::access::fence_space::local_space);
                // ((int32_t *)(&B[sg_offset_B]))[loc_id] = ((int32_t*)(weightsT_ptr+n_hidden_layers*WIDTH*WIDTH +
                // sg_offset_B))[loc_id];
                // ((int32_t *)(&B[sg_offset_B]))[loc_id+SG_SIZE] = ((int32_t*)(weightsT_ptr+n_hidden_layers*WIDTH*WIDTH
                // + sg_offset_B))[loc_id+SG_SIZE];
                sg.store<2>(local_ptr<int32_t>((int32_t *)(&B[sg_offset_B])),
                            sg.load<2>(global_ptr<int32_t>(
                                (int32_t *)(weightsT_ptr + n_hidden_layers * WIDTH * WIDTH + sg_offset_B))));

#ifdef SMALL_BATCH_SIZES
                if (!sg_has_no_blockrow)
#endif
                {
                    // we do not need SLM barrier since each SG writes and reads only its own data.
                    // load A matrix from slm to joint_matrices. This is done to avoid inefficient bf16 HBM access
                    joint_matrix_load(sg, A_block0, local_ptr<bf16>(&Atmp[sg_offset_A + 0 * SG_SIZE]), WIDTH);
                    joint_matrix_load(sg, A_block1, local_ptr<bf16>(&Atmp[sg_offset_A + 1 * SG_SIZE]), WIDTH);
                    joint_matrix_load(sg, A_block2, local_ptr<bf16>(&Atmp[sg_offset_A + 2 * SG_SIZE]), WIDTH);
                    joint_matrix_load(sg, A_block3, local_ptr<bf16>(&Atmp[sg_offset_A + 3 * SG_SIZE]), WIDTH);

                    // activate the A_blocks with output activation based on forward
                    sycl::vec<int32_t, TM> tmp16avalues0, tmp16avalues1;
                    // if constexpr (output_activation == Activation::ReLU) //is this necessary or garbage since forward
                    // is ReLU activated as well (i.e. >= 0) and never triggers < 0 case.
                    // {
                    //     //reuse Atmp for loading of the data through SLM.
                    //     for (int iter = 0; iter < TM; iter++) {
                    //         *(((int32_t*)&Atmp[sg_offset_A+iter*WIDTH])+loc_id) =
                    //             *((int32_t*)(intermediate_output_forward + (n_hidden_layers+1)*M*WIDTH +
                    //             total_offset_A+iter*WIDTH)+loc_id);
                    //         *(((int32_t*)&Atmp[sg_offset_A+iter*WIDTH])+loc_id + SG_SIZE) =
                    //             *((int32_t*)(intermediate_output_forward + (n_hidden_layers+1)*M*WIDTH +
                    //             total_offset_A+iter*WIDTH)+loc_id+SG_SIZE);
                    //     }

                    //     auto Ai_data0 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block0);
                    //     auto Ai_data1 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block1);
                    //     auto Ai_data2 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block2);
                    //     auto Ai_data3 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block3);
                    //     for (int rowiter = 0; rowiter < Ai_data0.length(); rowiter++)
                    //     {
                    //         Ai_data0[rowiter] = fmax((bf16)0, Atmp[sg_offset_A + rowiter*WIDTH + 0*TK + loc_id]);
                    //         Ai_data1[rowiter] = fmax((bf16)0, Atmp[sg_offset_A + rowiter*WIDTH + 1*TK + loc_id]);
                    //         Ai_data2[rowiter] = fmax((bf16)0, Atmp[sg_offset_A + rowiter*WIDTH + 2*TK + loc_id]);
                    //         Ai_data3[rowiter] = fmax((bf16)0, Atmp[sg_offset_A + rowiter*WIDTH + 3*TK + loc_id]);
                    //     }

                    //     //when done, store activated A matrices back to slm
                    //     sycl::ext::intel::experimental::matrix::joint_matrix_store(sg, A_block0,
                    //     Atmp.get_pointer()+sg_offset_A + 0*SG_SIZE, WIDTH);
                    //     sycl::ext::intel::experimental::matrix::joint_matrix_store(sg, A_block1,
                    //     Atmp.get_pointer()+sg_offset_A + 1*SG_SIZE, WIDTH);
                    //     sycl::ext::intel::experimental::matrix::joint_matrix_store(sg, A_block2,
                    //     Atmp.get_pointer()+sg_offset_A + 2*SG_SIZE, WIDTH);
                    //     sycl::ext::intel::experimental::matrix::joint_matrix_store(sg, A_block3,
                    //     Atmp.get_pointer()+sg_offset_A + 3*SG_SIZE, WIDTH);

                    //     //and load again in registers
                    //     for (int iter = 0; iter < TM; iter++) {
                    //         tmp16avalues0[iter] = *(((int32_t*)&Atmp[sg_offset_A+iter*WIDTH])+loc_id);
                    //         tmp16avalues1[iter] = *(((int32_t*)&Atmp[sg_offset_A+iter*WIDTH])+loc_id + SG_SIZE);
                    //     }
                    // }
                    // else if constexpr (output_activation == Activation::None) {
                    // for (uint8_t iter = 0; iter < TM; iter++) {
                    //     tmp16avalues0[iter] = *(((int32_t*)&Atmp[sg_offset_A+iter*WIDTH])+loc_id);
                    //     tmp16avalues1[iter] = *(((int32_t*)&Atmp[sg_offset_A+iter*WIDTH])+loc_id + SG_SIZE);
                    // }
                    tmp16avalues0 = sg.load<8>(local_ptr<int32_t>((int32_t *)(&Atmp[sg_offset_A])));
                    tmp16avalues1 = sg.load<8>(local_ptr<int32_t>((int32_t *)(&Atmp[sg_offset_A + 4 * WIDTH])));
                    //} else //nothing to be done

                    // Store this activated input at the end of interm_output for reuse in the update (GEMMS below)
                    const int loc_offset_A = n_hidden_layers * M * WIDTH + total_offset_A;

                    /// store the activated a values of the input to intermediate output
                    for (uint8_t iter = 0; iter < TM; iter++) {
                        *((int32_t *)(intermediate_output_backward + loc_offset_A + iter * WIDTH) + loc_id) =
                            tmp16avalues0[iter];
                        *((int32_t *)(intermediate_output_backward + loc_offset_A + iter * WIDTH) + loc_id + SG_SIZE) =
                            tmp16avalues1[iter];
                    }
                }

                // We have n_hidden_layers. Thus n_hidden_layers - 1 gemms between
                // the layers (layer 0 -> GEMM -> layer1 -> GEMM -> layer2 -> etc.)
                // Since we also do the GEMM from input to hidden layer 0,
                // we perform n_hidden_layers GEMMS.
                for (uint8_t layer = n_hidden_layers; layer > 0; layer--) // we are also doing output->last hidden layer
                {

// reset result matrix
#ifdef SMALL_BATCH_SIZES
                    if (!sg_has_no_blockrow)
#endif
                    {
                        joint_matrix_fill(sg, C_block0, 0.0f);
                        joint_matrix_fill(sg, C_block1, 0.0f);
                        joint_matrix_fill(sg, C_block2, 0.0f);
                        joint_matrix_fill(sg, C_block3, 0.0f);
                    }

                    item.barrier(sycl::access::fence_space::local_space); // wait for B to be done storing

// block axpy scheme
#ifdef SMALL_BATCH_SIZES
                    if (!sg_has_no_blockrow)
#endif
                    {
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0 * WIDTH * TK + 0 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block0 = joint_matrix_mad(sg, A_block0, B_block, C_block0);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0 * WIDTH * TK + 1 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block1 = joint_matrix_mad(sg, A_block0, B_block, C_block1);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0 * WIDTH * TK + 2 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block2 = joint_matrix_mad(sg, A_block0, B_block, C_block2);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0 * WIDTH * TK + 3 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block3 = joint_matrix_mad(sg, A_block0, B_block, C_block3);

                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[1 * WIDTH * TK + 0 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block0 = joint_matrix_mad(sg, A_block1, B_block, C_block0);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[1 * WIDTH * TK + 1 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block1 = joint_matrix_mad(sg, A_block1, B_block, C_block1);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[1 * WIDTH * TK + 2 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block2 = joint_matrix_mad(sg, A_block1, B_block, C_block2);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[1 * WIDTH * TK + 3 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block3 = joint_matrix_mad(sg, A_block1, B_block, C_block3);

                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[2 * WIDTH * TK + 0 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block0 = joint_matrix_mad(sg, A_block2, B_block, C_block0);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[2 * WIDTH * TK + 1 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block1 = joint_matrix_mad(sg, A_block2, B_block, C_block1);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[2 * WIDTH * TK + 2 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block2 = joint_matrix_mad(sg, A_block2, B_block, C_block2);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[2 * WIDTH * TK + 3 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block3 = joint_matrix_mad(sg, A_block2, B_block, C_block3);

                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[3 * WIDTH * TK + 0 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block0 = joint_matrix_mad(sg, A_block3, B_block, C_block0);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[3 * WIDTH * TK + 1 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block1 = joint_matrix_mad(sg, A_block3, B_block, C_block1);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[3 * WIDTH * TK + 2 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block2 = joint_matrix_mad(sg, A_block3, B_block, C_block2);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[3 * WIDTH * TK + 3 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block3 = joint_matrix_mad(sg, A_block3, B_block, C_block3);
                    }

                    // load B for next iteration into SLM
                    if (layer > 1) {
                        item.barrier(sycl::access::fence_space::local_space);
                        // ((int32_t *)(&B[sg_offset_B]))[loc_id] = ((int32_t*)(weightsT_ptr+(layer-1)*WIDTH*WIDTH +
                        // sg_offset_B))[loc_id];
                        // ((int32_t *)(&B[sg_offset_B]))[loc_id+SG_SIZE] =
                        // ((int32_t*)(weightsT_ptr+(layer-1)*WIDTH*WIDTH + sg_offset_B))[loc_id+SG_SIZE];
                        sg.store<2>(local_ptr<int32_t>((int32_t *)(&B[sg_offset_B])),
                                    sg.load<2>(global_ptr<int32_t>(
                                        (int32_t *)(weightsT_ptr + (layer - 1) * WIDTH * WIDTH + sg_offset_B))));
                    }

// This can be done in the future with joint_matrix_copy and a joint_maitrx_apply.
#ifdef SMALL_BATCH_SIZES
                    if (!sg_has_no_blockrow)
#endif
                    {
                        auto Ci_data0 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block0);
                        auto Ai_data0 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block0);
                        auto Ci_data1 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block1);
                        auto Ai_data1 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block1);
                        auto Ci_data2 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block2);
                        auto Ai_data2 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block2);
                        auto Ci_data3 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block3);
                        auto Ai_data3 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block3);
                        /// ATTENTION: forward is already activated in the forward pass.
                        // In general, we are always using same input and output activation for
                        // forward pass and backward pass (Is this true?). Do not need to activate again.
                        //  if constexpr (false) //forward is ReLU activated, thus always >= 0. The condition is never
                        //  activated
                        //  {
                        //      // sycl::vec<int32_t,16> tmp32avalues =  sg.load<16>(multi_ptr<const int32_t,
                        //      access::address_space::global_space>(reinterpret_cast<const int32_t*>(
                        //      //     forward+(n_hidden_matrices+2)*M*K + total_offset_A)));
                        //      // sg.store<16>(multi_ptr<int32_t,
                        //      access::address_space::local_space>(reinterpret_cast<int32_t*>(Atmp.get_pointer().get()
                        //      + sg_offset_A)),
                        //      //     tmp32avalues);

                        //     ///HERE, implement all the activations

                        // }
                        // else if constexpr (activation == Activation::ReLU || activation == Activation::None)
                        // //nothing to be done since forw is ReLU activated an thus >=0
                        // {
                        for (uint8_t rowiter = 0; rowiter < Ci_data0.length(); rowiter++) // should be TM in length
                        {
                            Ai_data0[rowiter] = (bf16)Ci_data0[rowiter];
                            Ai_data1[rowiter] = (bf16)Ci_data1[rowiter];
                            Ai_data2[rowiter] = (bf16)Ci_data2[rowiter];
                            Ai_data3[rowiter] = (bf16)Ci_data3[rowiter];
                        }
                        //}
                        // else nothing to be done

                        // store A
                        sycl::ext::intel::experimental::matrix::joint_matrix_store(
                            sg, A_block0, local_ptr<bf16>(&Atmp[sg_offset_A + 0 * SG_SIZE]), WIDTH);
                        sycl::ext::intel::experimental::matrix::joint_matrix_store(
                            sg, A_block1, local_ptr<bf16>(&Atmp[sg_offset_A + 1 * SG_SIZE]), WIDTH);
                        sycl::ext::intel::experimental::matrix::joint_matrix_store(
                            sg, A_block2, local_ptr<bf16>(&Atmp[sg_offset_A + 2 * SG_SIZE]), WIDTH);
                        sycl::ext::intel::experimental::matrix::joint_matrix_store(
                            sg, A_block3, local_ptr<bf16>(&Atmp[sg_offset_A + 3 * SG_SIZE]), WIDTH);
                        /// Alternative of loading A through SLM to avoid inefficient access to HBM
                        // we do not need SLM barrier since each SG writes and reads only its own data.
                        const int loc_offset_A = (layer - 1) * M * WIDTH + total_offset_A;
                        for (uint8_t iter = 0; iter < TM; iter++) {
                            *((int32_t *)(intermediate_output_backward + loc_offset_A + iter * WIDTH) + loc_id) =
                                *(((int32_t *)&Atmp[sg_offset_A + iter * WIDTH]) + loc_id);

                            *((int32_t *)(intermediate_output_backward + loc_offset_A + iter * WIDTH) + loc_id +
                              SG_SIZE) = *(((int32_t *)&Atmp[sg_offset_A + iter * WIDTH]) + loc_id + SG_SIZE);
                        }
                        // sg.store<8>(global_ptr<int32_t>((int32_t*)(intermediate_output_backward+loc_offset_A)),
                        // sg.load<8>(local_ptr<int32_t>((int32_t*)(&Atmp[sg_offset_A]))));
                        // sg.store<8>(global_ptr<int32_t>((int32_t*)(intermediate_output_backward+loc_offset_A+4*WIDTH)),
                        // sg.load<8>(local_ptr<int32_t>((int32_t*)(&Atmp[sg_offset_A+4*WIDTH]))));
                    }
                }
            });
    });

    /// TODO: merge this with the above systolic code
    /// TODO: check offsets.
    // NOTE: MKL gemm_batch is slower.
    std::vector<sycl::event> events(n_hidden_layers + 1);
    for (int iter = 0; iter < n_hidden_layers + 1; iter++) {
        // Perform GEMM operation
        events[iter] = oneapi::mkl::blas::row_major::gemm(
            q, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, WIDTH, WIDTH, M, 1.0f,
            reinterpret_cast<const oneapi::mkl::bfloat16 *>(intermediate_output_forward) + iter * M * WIDTH, WIDTH,
            reinterpret_cast<oneapi::mkl::bfloat16 *>(intermediate_output_backward) + iter * M * WIDTH, WIDTH, 1.0f,
            reinterpret_cast<oneapi::mkl::bfloat16 *>(output_ptr) + iter * WIDTH * WIDTH, WIDTH, {e});
    }

    return events;
}

/**
 * Performs forward pass for the Swift MLP model.
 *
 * @param q                  SYCL queue for command submission.
 * @param output_activation The type of activation to be applied for output
 * layer.
 * @param weights            Device memory containing weights for the model.
 * @param inputs             Device memory containing input data.
 * @param intermediate_output Pointer to intermediate output memory.
 * @param act_mem            Pointer to activation memory.
 * @param act_mem_temp       Pointer to temporary activation memory.
 * @param output             Device memory for storing the output.
 * @param n_hidden_layers    Number of hidden layers.
 * @param input_width        Width of the input data.
 * @param output_width       Width of the output data.
 * @param batch_size         Batch size of the data.
 * @tparam WIDTH             Width of the layers.
 * @tparam activation        Type of activation for hidden layers.
 */
// Version in which each subgroup computes full block rows in the output
// Writes full B into slm.
// Option to have a variable number of block rows per sg.
// initial version tries to keep A in registers and use element-wise casting
// between the joint matrices.
template <int WIDTH, Activation activation, bool INFERENCE>
std::vector<sycl::event> mlp_swift_forward(queue &q, bf16 const *const __restrict__ weights_ptr,
                                           bf16 const *const __restrict__ inputs_ptr,
                                           bf16 *const __restrict__ intermediate_output, const int n_hidden_layers,
                                           const int batch_size, const std::vector<sycl::event> &deps) {
    // Indicates how many joint_matrix rows (i.e. time TM actual rows) are done by one
    // sub-group. ONLY works for = 1 right now.
    // reuse of B, this is in subgroups, ONLY works for 64 nor
    // note that large grf mode requires this to be set to 32
    constexpr int SGS_IN_WG = 64;
    // dimensions are M = batch_size, N = WIDTH = K = 64;
    const int M = batch_size;
    constexpr int N = WIDTH;
    constexpr int K = WIDTH;
    static_assert(TK == SG_SIZE);
    static_assert(TN == TK);
    if constexpr (WIDTH != 64) throw std::invalid_argument("Current implementation only works for a WIDTH of 64");
    assert(M % TM == 0); // make sure there is no remainder and no out of bounds accesses
    // Note that TN = TK = SG_SIZE
    constexpr int NBLOCKCOLS_PER_SG = N / SG_SIZE;

    // One Block Row has TM rows an N columns.
    auto e = q.submit([&](handler &cgh) {
        cgh.depends_on(deps);
        local_accessor<bf16, 1> B(range<1>(K * N),
                                  cgh); // weights matrix. 64*64*2 byte = 8 kb. Thus, can have up to 16 WGs per Xe Core.
        local_accessor<bf16, 1> Atmp(range<1>(TM * K * SGS_IN_WG),
                                     cgh); // buffer for loading joint matrices. 8*64*64*2byte = 64kb. TODO: check if
                                           // this is too much. If so, split in half
        // number of SGS is given by batch_size / TM, since batch_size is the number of rows in the output

        cgh.parallel_for(
            nd_range<1>(std::max(M / TM * SG_SIZE, SGS_IN_WG * SG_SIZE),
                        SGS_IN_WG * SG_SIZE), // assuming here that the number of block rows is divisable by SGS_IN_WG
            [=](nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                // we assume we start with the input.
                // we assume input_width = WIDTH = output_width
                // Initialize: Load input into A
                // Iterate
                // 1: Load B= weights in SLM
                // 2: Do multiplication of a stripe of A with all of B to get a stripe of C (stripe == all the block
                // rows of one SG) 3: Apply the activation function on C while writing back to A 3a: write to
                // intermediate_output, if necessary.

                // After loop, check if we are in the INFERENCE case. In that case, write the last result to
                // intermediate output.
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////
                const int wg_id = item.get_group().get_group_id();
                auto sg = item.get_sub_group();
                const int loc_id = sg.get_local_id()[0];
                const int sg_id = sg.get_group_id()[0];
#ifdef SMALL_BATCH_SIZES
                const bool sg_has_no_blockrow = (wg_id * TM * SGS_IN_WG + sg_id * TM) >= M;
#endif

                bf16 const *weights_ptr_loc = weights_ptr;

                const int sg_offset_B = sg_id * N * K / SGS_IN_WG; // we assume this is divisible
                const int sg_offset_A = sg_id * K * TM;
                const int total_offset_A = wg_id * SGS_IN_WG * K * TM + sg_offset_A;

                // Load B into slm
                /// ATTENTION: this version only works for K = SGS_IN_WG and NBLOCKCOLS_PER_SG = 4
                ((int32_t *)(&B[sg_offset_B]))[loc_id] = ((int32_t *)(weights_ptr_loc + sg_offset_B))[loc_id];
                ((int32_t *)(&B[sg_offset_B]))[loc_id + SG_SIZE] =
                    ((int32_t *)(weights_ptr_loc + sg_offset_B))[loc_id + SG_SIZE];
                weights_ptr_loc += K * N;

// load input in slm
/// Alternative of loading A through SLM to avoid inefficient access to HBM
#ifdef SMALL_BATCH_SIZES
                if (!sg_has_no_blockrow)
#endif
                {
                    sycl::vec<int32_t, TM> tmp16avalues0, tmp16avalues1;
                    for (int iter = 0; iter < TM; iter++) {
                        tmp16avalues0[iter] = *((int32_t *)(inputs_ptr + total_offset_A) + loc_id + iter * K / 2);
                        tmp16avalues1[iter] =
                            *((int32_t *)(inputs_ptr + total_offset_A) + loc_id + iter * K / 2 + SG_SIZE);
                    }

                    for (int iter = 0; iter < TM; iter++) {
                        *(((int32_t *)&Atmp[sg_offset_A]) + loc_id + iter * K / 2) = tmp16avalues0[iter];
                        *(((int32_t *)&Atmp[sg_offset_A]) + loc_id + iter * K / 2 + SG_SIZE) = tmp16avalues1[iter];
                    }
                    // we do not need SLM barrier since each SG writes and reads only its own data.
                    // load A matrix from slm to joint_matrices. This is done to avoid inefficient bf16 HBM access

                    // ATTENTION: current inputs are positive and this is not really tested
                    if constexpr (!INFERENCE) // write input into intermediate_output, required for the backward_pass in
                                              // training
                    {
                        if constexpr (activation == Activation::ReLU) {
                            // update tmp32avalues with bit shenanigans to avoid one pass through slm
                            int32_t bitmask = 0b11111111111111110000000000000000;
                            for (int iter = 0; iter < TM; iter++) {
                                if ((tmp16avalues0[iter] >> 31) & 1) tmp16avalues0[iter] &= ~bitmask;
                                if ((tmp16avalues0[iter] >> 15) & 1) tmp16avalues0[iter] &= bitmask;

                                if ((tmp16avalues1[iter] >> 31) & 1) tmp16avalues1[iter] &= ~bitmask;
                                if ((tmp16avalues1[iter] >> 15) & 1) tmp16avalues1[iter] &= bitmask;
                            }
                        }
                        // else {} //nothing to be done

                        for (int iter = 0; iter < TM; iter++) {
                            *((int32_t *)(intermediate_output + total_offset_A) + loc_id + iter * K / 2) =
                                tmp16avalues0[iter];
                            *((int32_t *)(intermediate_output + total_offset_A) + loc_id + iter * K / 2 + SG_SIZE) =
                                tmp16avalues1[iter];
                        }
                    }
                }

                joint_matrix<sub_group, bf16, use::a, TM, TK, layout::row_major> A_block0, A_block1, A_block2, A_block3;
#ifdef SMALL_BATCH_SIZES
                if (!sg_has_no_blockrow)
#endif
                {
                    joint_matrix_load(sg, A_block0, local_ptr<bf16>(&Atmp[sg_offset_A + 0 * SG_SIZE]), K);
                    joint_matrix_load(sg, A_block1, local_ptr<bf16>(&Atmp[sg_offset_A + 1 * SG_SIZE]), K);
                    joint_matrix_load(sg, A_block2, local_ptr<bf16>(&Atmp[sg_offset_A + 2 * SG_SIZE]), K);
                    joint_matrix_load(sg, A_block3, local_ptr<bf16>(&Atmp[sg_offset_A + 3 * SG_SIZE]), K);
                }

                // We have n_hidden_layers. Thus n_hidden_layers - 1 gemms between
                // the layers (layer 0 -> GEMM -> layer1 -> GEMM -> layer2 -> etc.)
                // Since we also do the GEMM from input to hidden layer 0,
                // we perform n_hidden_layers GEMMS.
                joint_matrix<sub_group, bf16, use::b, TK, TN, sycl::ext::intel::experimental::matrix::layout::packed>
                    B_block;
                joint_matrix<sub_group, float, use::accumulator, TM, TN> C_block0, C_block1, C_block2, C_block3;

                for (int layer = 0; layer < n_hidden_layers; layer++) {
// reset result matrix
#ifdef SMALL_BATCH_SIZES
                    if (!sg_has_no_blockrow)
#endif
                    {
                        joint_matrix_fill(sg, C_block0, 0.0f);
                        joint_matrix_fill(sg, C_block1, 0.0f);
                        joint_matrix_fill(sg, C_block2, 0.0f);
                        joint_matrix_fill(sg, C_block3, 0.0f);
                    }

                    item.barrier(sycl::access::fence_space::local_space);

// block axpy scheme
#ifdef SMALL_BATCH_SIZES
                    if (!sg_has_no_blockrow)
#endif
                    {
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0 * WIDTH * TK + 0 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block0 = joint_matrix_mad(sg, A_block0, B_block, C_block0);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0 * WIDTH * TK + 1 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block1 = joint_matrix_mad(sg, A_block0, B_block, C_block1);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0 * WIDTH * TK + 2 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block2 = joint_matrix_mad(sg, A_block0, B_block, C_block2);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0 * WIDTH * TK + 3 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block3 = joint_matrix_mad(sg, A_block0, B_block, C_block3);

                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[1 * WIDTH * TK + 0 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block0 = joint_matrix_mad(sg, A_block1, B_block, C_block0);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[1 * WIDTH * TK + 1 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block1 = joint_matrix_mad(sg, A_block1, B_block, C_block1);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[1 * WIDTH * TK + 2 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block2 = joint_matrix_mad(sg, A_block1, B_block, C_block2);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[1 * WIDTH * TK + 3 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block3 = joint_matrix_mad(sg, A_block1, B_block, C_block3);

                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[2 * WIDTH * TK + 0 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block0 = joint_matrix_mad(sg, A_block2, B_block, C_block0);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[2 * WIDTH * TK + 1 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block1 = joint_matrix_mad(sg, A_block2, B_block, C_block1);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[2 * WIDTH * TK + 2 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block2 = joint_matrix_mad(sg, A_block2, B_block, C_block2);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[2 * WIDTH * TK + 3 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block3 = joint_matrix_mad(sg, A_block2, B_block, C_block3);

                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[3 * WIDTH * TK + 0 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block0 = joint_matrix_mad(sg, A_block3, B_block, C_block0);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[3 * WIDTH * TK + 1 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block1 = joint_matrix_mad(sg, A_block3, B_block, C_block1);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[3 * WIDTH * TK + 2 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block2 = joint_matrix_mad(sg, A_block3, B_block, C_block2);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[3 * WIDTH * TK + 3 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block3 = joint_matrix_mad(sg, A_block3, B_block, C_block3);
                    }

                    // Load next B into slm

                    /// ATTENTION: this version only works for K = SGS_IN_WG and NBLOCKCOLS_PER_SG = 4
                    item.barrier(sycl::access::fence_space::local_space); // make sure all the reads are done before we
                                                                          // write into slm again
                    ((int32_t *)(&B[sg_offset_B]))[loc_id] = ((int32_t *)(weights_ptr_loc + sg_offset_B))[loc_id];
                    ((int32_t *)(&B[sg_offset_B]))[loc_id + SG_SIZE] =
                        ((int32_t *)(weights_ptr_loc + sg_offset_B))[loc_id + SG_SIZE];
                    weights_ptr_loc += K * N;

// This can be done in the future with joint_matrix_copy and a joint_maitrx_apply.
#ifdef SMALL_BATCH_SIZES
                    if (!sg_has_no_blockrow)
#endif
                    {
                        auto Ci_data0 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block0);
                        auto Ai_data0 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block0);
                        auto Ci_data1 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block1);
                        auto Ai_data1 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block1);
                        auto Ci_data2 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block2);
                        auto Ai_data2 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block2);
                        auto Ci_data3 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block3);
                        auto Ai_data3 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block3);
                        if constexpr (activation == Activation::ReLU) {
                            for (int rowiter = 0; rowiter < Ci_data0.length(); rowiter++) // should be TM in length
                            {
                                Ai_data0[rowiter] = fmax((bf16)0, (bf16)Ci_data0[rowiter]);
                                Ai_data1[rowiter] = fmax((bf16)0, (bf16)Ci_data1[rowiter]);
                                Ai_data2[rowiter] = fmax((bf16)0, (bf16)Ci_data2[rowiter]);
                                Ai_data3[rowiter] = fmax((bf16)0, (bf16)Ci_data3[rowiter]);
                            }
                        } else if constexpr (activation == Activation::None) {
                            for (int rowiter = 0; rowiter < Ci_data0.length(); rowiter++) // should be TM in length
                            {
                                Ai_data0[rowiter] = (bf16)Ci_data0[rowiter];
                                Ai_data1[rowiter] = (bf16)Ci_data1[rowiter];
                                Ai_data2[rowiter] = (bf16)Ci_data2[rowiter];
                                Ai_data3[rowiter] = (bf16)Ci_data3[rowiter];
                            }
                        }
                        // else none

                        /// Store intermediate result in case of forward_pass
                        // Don't ask me what is going on with these indices and why inference write
                        // to a different position for the output than forward_pass.
                        if constexpr (!INFERENCE) {
                            const int loc_offset_A = (layer + 1) * M * K + total_offset_A;
                            sycl::ext::intel::experimental::matrix::joint_matrix_store(
                                sg, A_block0, local_ptr<bf16>(&Atmp[sg_offset_A + 0 * SG_SIZE]), K);
                            sycl::ext::intel::experimental::matrix::joint_matrix_store(
                                sg, A_block1, local_ptr<bf16>(&Atmp[sg_offset_A + 1 * SG_SIZE]), K);
                            sycl::ext::intel::experimental::matrix::joint_matrix_store(
                                sg, A_block2, local_ptr<bf16>(&Atmp[sg_offset_A + 2 * SG_SIZE]), K);
                            sycl::ext::intel::experimental::matrix::joint_matrix_store(
                                sg, A_block3, local_ptr<bf16>(&Atmp[sg_offset_A + 3 * SG_SIZE]), K);
                            /// Alternative of loading A through SLM to avoid inefficient access to HBM
                            // we do not need SLM barrier since each SG writes and reads only its own data.
                            for (int iter = 0; iter < TM; iter++) {
                                *((int32_t *)(intermediate_output + loc_offset_A + iter * K) + loc_id) =
                                    *(((int32_t *)&Atmp[sg_offset_A + iter * K]) + loc_id);
                                *((int32_t *)(intermediate_output + loc_offset_A + iter * K) + loc_id + SG_SIZE) =
                                    *(((int32_t *)&Atmp[sg_offset_A + iter * K]) + loc_id + SG_SIZE);
                            }
                        }
                    }
                }

// generate output, i.e. last GEMM

// reset result matrix
#ifdef SMALL_BATCH_SIZES
                if (!sg_has_no_blockrow)
#endif
                {
                    joint_matrix_fill(sg, C_block0, 0.0f);
                    joint_matrix_fill(sg, C_block1, 0.0f);
                    joint_matrix_fill(sg, C_block2, 0.0f);
                    joint_matrix_fill(sg, C_block3, 0.0f);
                }

                item.barrier(sycl::access::fence_space::local_space); // wait for B to be loaded

// block axpy scheme
#ifdef SMALL_BATCH_SIZES
                if (!sg_has_no_blockrow)
#endif
                {
                    joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0 * WIDTH * TK + 0 * 2 * TN]),
                                      2 * WIDTH); // 2*TN since weights are in VNNI format
                    C_block0 = joint_matrix_mad(sg, A_block0, B_block, C_block0);
                    joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0 * WIDTH * TK + 1 * 2 * TN]),
                                      2 * WIDTH); // 2*TN since weights are in VNNI format
                    C_block1 = joint_matrix_mad(sg, A_block0, B_block, C_block1);
                    joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0 * WIDTH * TK + 2 * 2 * TN]),
                                      2 * WIDTH); // 2*TN since weights are in VNNI format
                    C_block2 = joint_matrix_mad(sg, A_block0, B_block, C_block2);
                    joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0 * WIDTH * TK + 3 * 2 * TN]),
                                      2 * WIDTH); // 2*TN since weights are in VNNI format
                    C_block3 = joint_matrix_mad(sg, A_block0, B_block, C_block3);

                    joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[1 * WIDTH * TK + 0 * 2 * TN]),
                                      2 * WIDTH); // 2*TN since weights are in VNNI format
                    C_block0 = joint_matrix_mad(sg, A_block1, B_block, C_block0);
                    joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[1 * WIDTH * TK + 1 * 2 * TN]),
                                      2 * WIDTH); // 2*TN since weights are in VNNI format
                    C_block1 = joint_matrix_mad(sg, A_block1, B_block, C_block1);
                    joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[1 * WIDTH * TK + 2 * 2 * TN]),
                                      2 * WIDTH); // 2*TN since weights are in VNNI format
                    C_block2 = joint_matrix_mad(sg, A_block1, B_block, C_block2);
                    joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[1 * WIDTH * TK + 3 * 2 * TN]),
                                      2 * WIDTH); // 2*TN since weights are in VNNI format
                    C_block3 = joint_matrix_mad(sg, A_block1, B_block, C_block3);

                    joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[2 * WIDTH * TK + 0 * 2 * TN]),
                                      2 * WIDTH); // 2*TN since weights are in VNNI format
                    C_block0 = joint_matrix_mad(sg, A_block2, B_block, C_block0);
                    joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[2 * WIDTH * TK + 1 * 2 * TN]),
                                      2 * WIDTH); // 2*TN since weights are in VNNI format
                    C_block1 = joint_matrix_mad(sg, A_block2, B_block, C_block1);
                    joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[2 * WIDTH * TK + 2 * 2 * TN]),
                                      2 * WIDTH); // 2*TN since weights are in VNNI format
                    C_block2 = joint_matrix_mad(sg, A_block2, B_block, C_block2);
                    joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[2 * WIDTH * TK + 3 * 2 * TN]),
                                      2 * WIDTH); // 2*TN since weights are in VNNI format
                    C_block3 = joint_matrix_mad(sg, A_block2, B_block, C_block3);

                    joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[3 * WIDTH * TK + 0 * 2 * TN]),
                                      2 * WIDTH); // 2*TN since weights are in VNNI format
                    C_block0 = joint_matrix_mad(sg, A_block3, B_block, C_block0);
                    joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[3 * WIDTH * TK + 1 * 2 * TN]),
                                      2 * WIDTH); // 2*TN since weights are in VNNI format
                    C_block1 = joint_matrix_mad(sg, A_block3, B_block, C_block1);
                    joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[3 * WIDTH * TK + 2 * 2 * TN]),
                                      2 * WIDTH); // 2*TN since weights are in VNNI format
                    C_block2 = joint_matrix_mad(sg, A_block3, B_block, C_block2);
                    joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[3 * WIDTH * TK + 3 * 2 * TN]),
                                      2 * WIDTH); // 2*TN since weights are in VNNI format
                    C_block3 = joint_matrix_mad(sg, A_block3, B_block, C_block3);

                    /// TODO: Output activation in what follows. Here == None
                    // This can be done in the future with joint_matrix_copy and a joint_maitrx_apply.
                    // This can be done in the future with joint_matrix_copy and a joint_maitrx_apply.
                    auto Ci_data0 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block0);
                    auto Ai_data0 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block0);
                    auto Ci_data1 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block1);
                    auto Ai_data1 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block1);
                    auto Ci_data2 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block2);
                    auto Ai_data2 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block2);
                    auto Ci_data3 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block3);
                    auto Ai_data3 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block3);
                    if constexpr (/*output_activation == Activation::ReLU*/ false) // output activation is always none
                                                                                   // for this test. We do not have a
                                                                                   // template parameter for this
                    {
                        for (int rowiter = 0; rowiter < Ci_data0.length(); rowiter++) // should be TM in length
                        {
                            Ai_data0[rowiter] =
                                fmax((bf16)0, (bf16)Ci_data0[rowiter]); // tmpCi < (bf16)0 ? (bf16)0 : tmpCi;
                            Ai_data1[rowiter] = fmax((bf16)0, (bf16)Ci_data1[rowiter]);
                            Ai_data2[rowiter] = fmax((bf16)0, (bf16)Ci_data2[rowiter]);
                            Ai_data3[rowiter] = fmax((bf16)0, (bf16)Ci_data3[rowiter]);
                        }
                    } else if constexpr (/*output_activation == Activation::None*/ true) // for this tests, alway
                                                                                         // activation true
                    {
                        for (int rowiter = 0; rowiter < Ci_data0.length(); rowiter++) // should be TM in length
                        {
                            Ai_data0[rowiter] = (bf16)Ci_data0[rowiter];
                            Ai_data1[rowiter] = (bf16)Ci_data1[rowiter];
                            Ai_data2[rowiter] = (bf16)Ci_data2[rowiter];
                            Ai_data3[rowiter] = (bf16)Ci_data3[rowiter];
                        }
                    }

                    const int loc_offset_A = (n_hidden_layers + 1) * M * K + total_offset_A;
                    // load A matrix from slm to joint_matrices. This is done to avoid inefficient bf16 HBM access
                    sycl::ext::intel::experimental::matrix::joint_matrix_store(
                        sg, A_block0, local_ptr<bf16>(&Atmp[sg_offset_A + 0 * SG_SIZE]), K);
                    sycl::ext::intel::experimental::matrix::joint_matrix_store(
                        sg, A_block1, local_ptr<bf16>(&Atmp[sg_offset_A + 1 * SG_SIZE]), K);
                    sycl::ext::intel::experimental::matrix::joint_matrix_store(
                        sg, A_block2, local_ptr<bf16>(&Atmp[sg_offset_A + 2 * SG_SIZE]), K);
                    sycl::ext::intel::experimental::matrix::joint_matrix_store(
                        sg, A_block3, local_ptr<bf16>(&Atmp[sg_offset_A + 3 * SG_SIZE]), K);
                    /// Alternative of loading A through SLM to avoid inefficient access to HBM
                    // we do not need SLM barrier since each SG writes and reads only its own data.

                    for (int iter = 0; iter < TM; iter++) {
                        *((int32_t *)(intermediate_output + loc_offset_A + iter * K) + loc_id) =
                            *(((int32_t *)&Atmp[sg_offset_A + iter * K]) + loc_id);
                        *((int32_t *)(intermediate_output + loc_offset_A + iter * K) + loc_id + SG_SIZE) =
                            *(((int32_t *)&Atmp[sg_offset_A + iter * K]) + loc_id + SG_SIZE);
                    }
                }
            });
    });

    return {e};
}

/**
 * Backward pass for gradient calculation in the SwiftNet model.
 *
 * @param q                 SYCL queue for command submission.
 * @param weights_transposed Pointer to transposed and packed weights.
 * @param deltas            Pointer to delta values.
 * @param grads_matrices    Pointer to matrices for gradients.
 * @param out_inter         Pointer to intermediate outputs.
 * @param delta_temp        Pointer to temporary delta memory.
 * @param forward           Pointer to forward pass intermediate outputs.
 * @param n_hidden_matrices Number of hidden matrix multiplications.
 * @param batch_size        Batch size of the data.
 * @tparam WIDTH            Width of the matrices.
 * @tparam ACTIVATION       Type of activation for hidden layers.
 */
template <int WIDTH, Activation activation, Activation output_activation = Activation::None>
std::vector<sycl::event>
mlp_swiftnet_backward(queue &q, bf16 const *const __restrict__ weights_ptr, bf16 const *const __restrict__ inputs_ptr,
                      bf16 *const __restrict__ output_ptr, bf16 *const __restrict__ intermediate_output,
                      bf16 const *const __restrict__ forward, const int n_hidden_matrices, const int batch_size,
                      const std::vector<sycl::event> &deps) {
    // here, weights are already transposed and packed
    // in deltas, the last layer has already been calculated
    constexpr int NBLOCKROWS_PER_SG = 1; // ONLY works for = 1 right now.
    constexpr int SGS_IN_WG = 64;        // reuse of B, this is in subgroups
    // dimensions are M = batch_size, N = WIDTH = K = 64;
    /// ATTENTION: currently only works for batch sizes which are powers of 2
    // and which are larger than 512 (TM*SGS_IN_WG)
    const int M = batch_size;
    constexpr int N = WIDTH;
    constexpr int K = WIDTH;
    constexpr int NBLOCKCOLS_PER_SG = N / SG_SIZE; // Note that TN = TK = SG_SIZE

    static_assert(TK == SG_SIZE);
    static_assert(TN == TK);
    if constexpr (WIDTH != 64) throw std::invalid_argument("Current implementation only works for a WIDTH of 64");
    assert(M % TM == 0); // make sure there is no remainder and no out of bounds accesses

    // One Block Row has TM rows an N columns.
    auto e = q.submit([&](handler &cgh) {
        cgh.depends_on(deps);
        // weights matrix. 64*64*2 byte = 8 kb. Thus, can have up to 16 WGs per Xe Core.
        local_accessor<bf16, 1> B(range<1>(K * N), cgh);
        // buffer for loading joint matrices. 8*64*64*2byte = 64kb. TODO: check if this is too much. If so, split in
        // half
        local_accessor<bf16, 1> Atmp(range<1>(TM * K * SGS_IN_WG), cgh);

        // number of SGS is given by batch_size / (NBLOCKROWS_PER_SG * TM), since batch_size is the number of rows in
        // the output
        cgh.parallel_for(
            nd_range<1>(std::max(M / (NBLOCKROWS_PER_SG * TM) * SG_SIZE, SGS_IN_WG * SG_SIZE),
                        SGS_IN_WG * SG_SIZE), // assuming here that the number of block rows is divisable by SGS_IN_WG
            [=](nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                const int wg_id = item.get_group().get_group_id();
                auto sg = item.get_sub_group();
                const int loc_id = sg.get_local_id()[0];
                const int sg_id = sg.get_group_id()[0];
#ifdef SMALL_BATCH_SIZES
                const bool sg_has_no_blockrow =
                    (wg_id * TM * NBLOCKROWS_PER_SG * SGS_IN_WG + sg_id * TM * NBLOCKROWS_PER_SG) >= M;
#endif

                bf16 const *weights_ptr_loc = weights_ptr + (n_hidden_matrices + 1) * K * N;

                // Load B= transposed weights into slm
                constexpr int nrows_per_sg_B = K / SGS_IN_WG; // we assume this is divisible
                constexpr int nelems_per_sg_B = N * nrows_per_sg_B;
                const int sg_offset_B = sg_id * nelems_per_sg_B;

                /// ATTENTION: this version only works for K = SGS_IN_WG and NBLOCKCOLS_PER_SG = 4
                const sycl::vec<float, 2> tmp4bvalues = sg.load<2>(
                    multi_ptr<const float, access::address_space::global_space>(reinterpret_cast<const float *>(
                        weights_ptr_loc + sg_offset_B))); // load two block cols in one go to load 1 cache line
                sg.store<2>(
                    multi_ptr<float, access::address_space::local_space>(reinterpret_cast<float *>(&B[sg_offset_B])),
                    tmp4bvalues);
                weights_ptr_loc -= K * N; // decrease weights pointer by one layer

                constexpr int n_rows_per_sg_A = NBLOCKROWS_PER_SG * TM;
                constexpr int nelems_per_sg_A = K * n_rows_per_sg_A;
                const int sg_offset_A = sg_id * nelems_per_sg_A;
                const int wg_offset_A = wg_id * SGS_IN_WG * nelems_per_sg_A;
                const int total_offset_A = wg_offset_A + sg_offset_A;

                joint_matrix<sub_group, bf16, use::a, TM, TK, layout::row_major> A_block0, A_block1, A_block2, A_block3;
#ifdef SMALL_BATCH_SIZES
                if (!sg_has_no_blockrow)
#endif
                {
                    sycl::vec<int32_t, TM> tmp16avalues0, tmp16avalues1;
                    for (int iter = 0; iter < TM; iter++) {
                        tmp16avalues0[iter] = *((int32_t *)(inputs_ptr + total_offset_A) + loc_id + iter * K / 2);
                        tmp16avalues1[iter] =
                            *((int32_t *)(inputs_ptr + total_offset_A) + loc_id + iter * K / 2 + SG_SIZE);
                    }

                    for (int iter = 0; iter < TM; iter++) {
                        *(((int32_t *)&Atmp[sg_offset_A]) + loc_id + iter * K / 2) = tmp16avalues0[iter];
                        *(((int32_t *)&Atmp[sg_offset_A]) + loc_id + iter * K / 2 + SG_SIZE) = tmp16avalues1[iter];
                    }

                    // we do not need SLM barrier since each SG writes and reads only its own data.
                    // load A matrix from slm to joint_matrices. This is done to avoid inefficient bf16 HBM access
                    joint_matrix_load(sg, A_block0, local_ptr<bf16>(&Atmp[sg_offset_A + 0 * SG_SIZE]), K);
                    joint_matrix_load(sg, A_block1, local_ptr<bf16>(&Atmp[sg_offset_A + 1 * SG_SIZE]), K);
                    joint_matrix_load(sg, A_block2, local_ptr<bf16>(&Atmp[sg_offset_A + 2 * SG_SIZE]), K);
                    joint_matrix_load(sg, A_block3, local_ptr<bf16>(&Atmp[sg_offset_A + 3 * SG_SIZE]), K);

                    // activate the A_blocks with output activation based on forward
                    if constexpr (output_activation ==
                                  Activation::ReLU) // is this necessary or garbage since forward is ReLU activated as
                                                    // well (i.e. >= 0) and never triggers < 0 case.
                    {
                        // reuse Atmp for loading of the data through SLM.
                        for (int iter = 0; iter < TM; iter++) {
                            *(((int32_t *)&Atmp[sg_offset_A]) + loc_id + iter * K / 2) =
                                *((int32_t *)(forward + (n_hidden_matrices + 2) * M * K + total_offset_A) + loc_id +
                                  iter * K / 2);
                            *(((int32_t *)&Atmp[sg_offset_A]) + loc_id + iter * K / 2 + SG_SIZE) =
                                *((int32_t *)(forward + (n_hidden_matrices + 2) * M * K + total_offset_A) + loc_id +
                                  iter * K / 2 + SG_SIZE);
                        }

                        auto Ai_data0 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block0);
                        auto Ai_data1 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block1);
                        auto Ai_data2 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block2);
                        auto Ai_data3 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block3);
                        for (int rowiter = 0; rowiter < Ai_data0.length(); rowiter++) {
                            Ai_data0[rowiter] = fmax((bf16)0, Atmp[sg_offset_A + rowiter * K + 0 * TK + loc_id]);
                            Ai_data1[rowiter] = fmax((bf16)0, Atmp[sg_offset_A + rowiter * K + 1 * TK + loc_id]);
                            Ai_data2[rowiter] = fmax((bf16)0, Atmp[sg_offset_A + rowiter * K + 2 * TK + loc_id]);
                            Ai_data3[rowiter] = fmax((bf16)0, Atmp[sg_offset_A + rowiter * K + 3 * TK + loc_id]);
                        }

                        // when done, store activated A matrices back to slm
                        sycl::ext::intel::experimental::matrix::joint_matrix_store(
                            sg, A_block0, local_ptr<bf16>(&Atmp[sg_offset_A + 0 * SG_SIZE]), K);
                        sycl::ext::intel::experimental::matrix::joint_matrix_store(
                            sg, A_block1, local_ptr<bf16>(&Atmp[sg_offset_A + 1 * SG_SIZE]), K);
                        sycl::ext::intel::experimental::matrix::joint_matrix_store(
                            sg, A_block2, local_ptr<bf16>(&Atmp[sg_offset_A + 2 * SG_SIZE]), K);
                        sycl::ext::intel::experimental::matrix::joint_matrix_store(
                            sg, A_block3, local_ptr<bf16>(&Atmp[sg_offset_A + 3 * SG_SIZE]), K);

                        // and load again in registers
                        for (int iter = 0; iter < TM; iter++) {
                            tmp16avalues0[iter] = *(((int32_t *)&Atmp[sg_offset_A]) + loc_id + iter * K / 2);
                            tmp16avalues1[iter] = *(((int32_t *)&Atmp[sg_offset_A]) + loc_id + iter * K / 2 + SG_SIZE);
                        }
                    }
                    // else { } //nothing to be done

                    // Store this activated input at the end of interm_output for reuse in the update (GEMMS below)
                    const int loc_offset_A = (n_hidden_matrices + 1) * M * K + total_offset_A;

                    /// store the activated a values of the input to intermediate output
                    for (int iter = 0; iter < TM; iter++) {
                        *((int32_t *)(intermediate_output + loc_offset_A) + loc_id + iter * K / 2) =
                            tmp16avalues0[iter];
                        *((int32_t *)(intermediate_output + loc_offset_A) + loc_id + iter * K / 2 + SG_SIZE) =
                            tmp16avalues1[iter];
                    }
                }

                // Define matrices
                joint_matrix<sub_group, bf16, use::b, TK, TN, sycl::ext::intel::experimental::matrix::layout::packed>
                    B_block;
                joint_matrix<sub_group, float, use::accumulator, TM, TN> C_block0, C_block1, C_block2, C_block3;

                // We have n_hidden_layers. Thus n_hidden_layers - 1 gemms between
                // the layers (layer 0 -> GEMM -> layer1 -> GEMM -> layer2 -> etc.)
                // Since we also do the GEMM from input to hidden layer 0,
                // we perform n_hidden_layers GEMMS.
                for (int layer = n_hidden_matrices + 1; layer > 0;
                     layer--) // we are also doing output->last hidden layer
                {
                    const int loc_offset_A = (layer - 1) * M * K + total_offset_A;
// __builtin_IB_lsc_prefetch_global_uint4((const __attribute__((opencl_global))
//                        uint32_t *)(intermediate_output+loc_offset_A),
//                        0, LSC_LDCC_L1C_L3C);

// reset result matrix
#ifdef SMALL_BATCH_SIZES
                    if (!sg_has_no_blockrow)
#endif
                    {
                        joint_matrix_fill(sg, C_block0, 0.0f);
                        joint_matrix_fill(sg, C_block1, 0.0f);
                        joint_matrix_fill(sg, C_block2, 0.0f);
                        joint_matrix_fill(sg, C_block3, 0.0f);
                    }

                    item.barrier(sycl::access::fence_space::local_space); // wait for B to be done storing

// block axpy scheme
#ifdef SMALL_BATCH_SIZES
                    if (!sg_has_no_blockrow)
#endif
                    {
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0 * WIDTH * TK + 0 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block0 = joint_matrix_mad(sg, A_block0, B_block, C_block0);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0 * WIDTH * TK + 1 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block1 = joint_matrix_mad(sg, A_block0, B_block, C_block1);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0 * WIDTH * TK + 2 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block2 = joint_matrix_mad(sg, A_block0, B_block, C_block2);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0 * WIDTH * TK + 3 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block3 = joint_matrix_mad(sg, A_block0, B_block, C_block3);

                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[1 * WIDTH * TK + 0 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block0 = joint_matrix_mad(sg, A_block1, B_block, C_block0);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[1 * WIDTH * TK + 1 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block1 = joint_matrix_mad(sg, A_block1, B_block, C_block1);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[1 * WIDTH * TK + 2 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block2 = joint_matrix_mad(sg, A_block1, B_block, C_block2);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[1 * WIDTH * TK + 3 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block3 = joint_matrix_mad(sg, A_block1, B_block, C_block3);

                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[2 * WIDTH * TK + 0 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block0 = joint_matrix_mad(sg, A_block2, B_block, C_block0);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[2 * WIDTH * TK + 1 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block1 = joint_matrix_mad(sg, A_block2, B_block, C_block1);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[2 * WIDTH * TK + 2 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block2 = joint_matrix_mad(sg, A_block2, B_block, C_block2);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[2 * WIDTH * TK + 3 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block3 = joint_matrix_mad(sg, A_block2, B_block, C_block3);

                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[3 * WIDTH * TK + 0 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block0 = joint_matrix_mad(sg, A_block3, B_block, C_block0);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[3 * WIDTH * TK + 1 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block1 = joint_matrix_mad(sg, A_block3, B_block, C_block1);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[3 * WIDTH * TK + 2 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block2 = joint_matrix_mad(sg, A_block3, B_block, C_block2);
                        joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[3 * WIDTH * TK + 3 * 2 * TN]),
                                          2 * WIDTH); // 2*TN since weights are in VNNI format
                        C_block3 = joint_matrix_mad(sg, A_block3, B_block, C_block3);
                    }

                    // load B for next iteration into SLM
                    if (layer > 1) {
                        const sycl::vec<float, 2> tmp4bvalues = sg.load<2>(
                            multi_ptr<const float, access::address_space::global_space>(reinterpret_cast<const float *>(
                                weights_ptr_loc + sg_offset_B))); // load two block cols in one go to load 1 cache line
                        item.barrier(sycl::access::fence_space::local_space); // wait for B to done accessing before
                                                                              // storing new values in
                        sg.store<2>(multi_ptr<float, access::address_space::local_space>(
                                        reinterpret_cast<float *>(&B[sg_offset_B])),
                                    tmp4bvalues);
                        weights_ptr_loc -= K * N; // decrease weights pointer by one layer
                    }

// This can be done in the future with joint_matrix_copy and a joint_maitrx_apply.
#ifdef SMALL_BATCH_SIZES
                    if (!sg_has_no_blockrow)
#endif
                    {
                        auto Ci_data0 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block0);
                        auto Ai_data0 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block0);
                        auto Ci_data1 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block1);
                        auto Ai_data1 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block1);
                        auto Ci_data2 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block2);
                        auto Ai_data2 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block2);
                        auto Ci_data3 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block3);
                        auto Ai_data3 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block3);
                        /// ATTENTION: forward is already activated in the forward pass.
                        // In general, we are always using same input and output activation for
                        // forward pass and backward pass (Is this true?). Do not need to activate again.
                        if constexpr (false) // forward is ReLU activated, thus always >= 0. The condition is never
                                             // activated
                        {
                            // sycl::vec<int32_t,16> tmp32avalues =  sg.load<16>(multi_ptr<const int32_t,
                            // access::address_space::global_space>(reinterpret_cast<const int32_t*>(
                            //     forward+(n_hidden_matrices+2)*M*K + total_offset_A)));
                            // sg.store<16>(multi_ptr<int32_t,
                            // access::address_space::local_space>(reinterpret_cast<int32_t*>(Atmp.get_pointer().get() +
                            // sg_offset_A)),
                            //     tmp32avalues);

                            /// HERE, implement all the activations

                        } else if constexpr (activation == Activation::ReLU ||
                                             activation == Activation::None) // nothing to be done since forw is ReLU
                                                                             // activated an thus >=0
                        {
                            for (int rowiter = 0; rowiter < Ci_data0.length(); rowiter++) // should be TM in length
                            {
                                Ai_data0[rowiter] = (bf16)Ci_data0[rowiter];
                                Ai_data1[rowiter] = (bf16)Ci_data1[rowiter];
                                Ai_data2[rowiter] = (bf16)Ci_data2[rowiter];
                                Ai_data3[rowiter] = (bf16)Ci_data3[rowiter];
                            }
                        }
                        // esle nothing to be done

                        // store A
                        sycl::ext::intel::experimental::matrix::joint_matrix_store(
                            sg, A_block0, local_ptr<bf16>(&Atmp[sg_offset_A + 0 * SG_SIZE]), K);
                        sycl::ext::intel::experimental::matrix::joint_matrix_store(
                            sg, A_block1, local_ptr<bf16>(&Atmp[sg_offset_A + 1 * SG_SIZE]), K);
                        sycl::ext::intel::experimental::matrix::joint_matrix_store(
                            sg, A_block2, local_ptr<bf16>(&Atmp[sg_offset_A + 2 * SG_SIZE]), K);
                        sycl::ext::intel::experimental::matrix::joint_matrix_store(
                            sg, A_block3, local_ptr<bf16>(&Atmp[sg_offset_A + 3 * SG_SIZE]), K);
                        /// Alternative of loading A through SLM to avoid inefficient access to HBM
                        // we do not need SLM barrier since each SG writes and reads only its own data.
                        for (int iter = 0; iter < TM; iter++) {
                            *((int32_t *)(intermediate_output + loc_offset_A) + loc_id + iter * K / 2) =
                                *(((int32_t *)&Atmp[sg_offset_A]) + loc_id + iter * K / 2);

                            *((int32_t *)(intermediate_output + loc_offset_A) + loc_id + iter * K / 2 + SG_SIZE) =
                                *(((int32_t *)&Atmp[sg_offset_A]) + loc_id + iter * K / 2 + SG_SIZE);
                        }
                    }
                }
            });
    });

    /// TODO: merge this with the above systolic code
    /// TODO: check offsets.
    // NOTE: MKL gemm_batch is slower.
    std::vector<sycl::event> events(n_hidden_matrices + 2);
    for (int iter = 0; iter < n_hidden_matrices + 2; iter++) {
        // Perform GEMM operation
        events[iter] = oneapi::mkl::blas::row_major::gemm(
            q, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, N, K, M, 1.0f,
            reinterpret_cast<const oneapi::mkl::bfloat16 *>(forward) + iter * M * K, N,
            reinterpret_cast<oneapi::mkl::bfloat16 *>(intermediate_output) + iter * M * K, K, 1.0f,
            reinterpret_cast<oneapi::mkl::bfloat16 *>(output_ptr) + iter * K * N, K, {e});
    }

    return events;
}

/**
 * Constructor for the SwiftNetMLP class.
 *
 * @param q                  SYCL queue for command submission.
 * @param input_width        Width of the input data.
 * @param output_width       Width of the output data.
 * @param n_hidden_layers    Number of hidden layers.
 * @param activation         Activation function for hidden layers.
 * @param output_activation  Activation function for the output layer.
 * @param batch_size         Batch size of the data.
 * @tparam WIDTH             Width of the matrices.
 */
template <int WIDTH>
SwiftNetMLP<WIDTH>::SwiftNetMLP(queue q, int input_width, int output_width, int n_hidden_layers, Activation activation,
                                Activation output_activation)
    : m_inputs_width{input_width}, m_net_width{WIDTH}, m_output_width{output_width}, m_n_hidden_layers{n_hidden_layers},
      m_activation{activation}, m_output_activation{output_activation}, m_inputs_width_padded{WIDTH},
      m_output_width_padded{WIDTH} /*TODO: replace later with smallest, nearest common divisor*/ {
    // Store provided parameters
    m_q = q;
    m_n_hidden_matrices = m_n_hidden_layers - 1;

    check_parameters();

    // As the systolic matrix multiplication works in multiples of 8/16, we cannot have arbitrary input and output
    // width. To get the correct width as defined by input_width and output_width, we pad later with zeros

    // Allocate memory for various matrices
    m_weightsT_matrices.allocate2(m_net_width * m_inputs_width_padded +
                                      (m_net_width * m_net_width) * m_n_hidden_matrices +
                                      m_net_width * m_output_width_padded,
                                  m_q);

    m_weights_matrices.allocate2(m_net_width * m_inputs_width_padded +
                                     (m_net_width * m_net_width) * m_n_hidden_matrices +
                                     m_net_width * m_output_width_padded,
                                 m_q);

    m_weights_matrices_inferences.allocate2(m_net_width * m_inputs_width_padded +
                                                (m_net_width * m_net_width) * m_n_hidden_matrices +
                                                m_net_width * m_output_width_padded,
                                            m_q);

    m_grads_matrices.allocate2(m_net_width * m_inputs_width_padded + (m_net_width * m_net_width) * m_n_hidden_matrices +
                                   m_net_width * m_output_width_padded,
                               m_q);

    // Initialize constants and allocations

    // note that the memory on m_deltas (also called loss sometimes) is
    // "flexible". It doesn't allow m_output_width > WIDTH, as in the
    // last layer backward pass, the m_output_width is first written
}

template <int WIDTH> void SwiftNetMLP<WIDTH>::check_parameters() {
    if (m_inputs_width <= 0) {
        std::string errorMessage =
            "Input width of " + std::to_string(m_inputs_width) + " is not supported. Value must be larger than 0.";
        throw std::runtime_error(errorMessage);
    }

    if (m_output_width <= 0) {
        std::string errorMessage =
            "Output width of " + std::to_string(m_output_width) + " is not supported. Value must be larger than 0.";
        throw std::runtime_error(errorMessage);
    }

    if (m_inputs_width > WIDTH) {
        std::string errorMessage = "Input width of " + std::to_string(m_inputs_width) +
                                   " is not supported. Value must be <= WIDTH (" + std::to_string(WIDTH) + ").";
        throw std::runtime_error(errorMessage);
    }

    if (m_output_width > WIDTH) {
        std::string errorMessage = "Input width of " + std::to_string(m_output_width) +
                                   " is not supported. Value must be <= WIDTH (" + std::to_string(WIDTH) + ").";
        throw std::runtime_error(errorMessage);
    }

    if (m_n_hidden_layers <= 0) {
        std::string errorMessage = "N hidden layers is " + std::to_string(m_output_width) +
                                   " but must be >= 1, i.e., 1 hidden layer and 1 output layer.";
        throw std::runtime_error(errorMessage);
    }

    if (m_activation != Activation::ReLU) {
        throw std::runtime_error("m_activation must be ReLU for now.");
    }
    if (m_output_activation != Activation::None) {
        throw std::runtime_error("m_output_activation must be None for now.");
    }
}
template <int WIDTH> SwiftNetMLP<WIDTH>::~SwiftNetMLP() {}
/**
 * Get a pointer to the gradients matrices.
 *
 * @return A pointer to the gradients matrices.
 */
template <int WIDTH> DeviceMem<bf16> *SwiftNetMLP<WIDTH>::get_grads_matrices() { return &m_grads_matrices; }

/**
 * Get a pointer to the weights matrices.
 *
 * @return A pointer to the weights matrices.
 */
template <int WIDTH> DeviceMem<bf16> *SwiftNetMLP<WIDTH>::get_weights_matrices() { return &m_weights_matrices; }

/**
 * Get a pointer to the transposed weights matrices.
 *
 * @return A pointer to the transposed weights matrices.
 */
template <int WIDTH> DeviceMem<bf16> *SwiftNetMLP<WIDTH>::get_weightsT_matrices() { return &m_weightsT_matrices; }

/**
 * Initialize parameters for the neural network.
 * This function initializes the weights matrices with uniform random values.
 */
template <int WIDTH> void SwiftNetMLP<WIDTH>::initialize_params(int use_easy) {
    // Initialize weights matrices with uniform random values, you can choose a
    // different initialization ( see in DeviceMem.cpp )

    if (use_easy == 1) {
        m_weights_matrices.initialize_arange(m_q, m_inputs_width_padded, m_net_width, m_output_width_padded,
                                             m_n_hidden_matrices);
    } else if (use_easy == 2) {
        m_weights_matrices.initialize_constant(0.01, m_q);
    } else if (use_easy == 3) {
        m_weights_matrices.initialize_constant(-0.01, m_q);
    } else {
        m_weights_matrices.initialize_he_normal(m_inputs_width_padded, m_q);
    }

    zero_pad_weight_matrix();

    m_weights_matrices.make_transposed(m_weightsT_matrices, m_inputs_width_padded, m_net_width, m_output_width_padded,
                                       m_n_hidden_matrices, m_q);
};

template <int WIDTH> void SwiftNetMLP<WIDTH>::zero_pad_weight_matrix() {
    m_weights_matrices.zero_pad_input(m_inputs_width, m_inputs_width_padded, m_net_width, m_q);

    m_weights_matrices.zero_pad_output(m_output_width, m_inputs_width_padded, m_net_width, m_output_width_padded,
                                       m_n_hidden_matrices, m_q);
}

/**
 * Save the neural network parameters to a file.
 *
 * @param filename The name of the file to save the parameters to.
 */
template <int WIDTH> void SwiftNetMLP<WIDTH>::save_to_file(std::string filename) {
    // Open the file for writing
    std::ofstream file;
    file.open(filename);

    // Write parameters to the file
    file << m_inputs_width << "\n";
    file << m_net_width << "\n";
    file << m_output_width << "\n";
    file << m_n_hidden_layers << "\n";
    file << m_n_hidden_matrices << "\n";

    // Write each value of the weights matrices to the file
    for (int i = 0; i < m_weights_matrices.size(); i++) {
        file << m_weights_matrices.data()[i] << "\n";
    }

    // Close the file
    file.close();
    return;
}

/**
 * Load neural network parameters from a file.
 *
 * @param filename The name of the file to load parameters from.
 */
template <int WIDTH> void SwiftNetMLP<WIDTH>::load_from_file(std::string filename) {
    // Open the file for reading
    std::ifstream file;
    file.open(filename);
    std::string line;

    // Read parameters from the file
    file >> m_inputs_width;
    file >> m_net_width;
    file >> m_output_width;
    file >> m_n_hidden_layers;
    file >> m_n_hidden_matrices;

    // Read each value from the file and set it as a bf16 value in weights
    // matrices
    for (int i = 0; i < m_weights_matrices.size(); i++) {
        float x;
        file >> x;
        m_weights_matrices.data()[i] = bf16(x);
    }

    // Close the file
    file.close();

    // Make the weights matrices transposed using the transposed weights matrices
    m_weights_matrices.make_transposed(m_weightsT_matrices, m_inputs_width_padded, m_net_width, m_output_width,
                                       m_n_hidden_matrices, m_q);
    return;
}

/**
 * Free memory allocated on the device for various arrays.
 *
 * @param q The SYCL queue used for device operations.
 */
template <int WIDTH> void SwiftNetMLP<WIDTH>::free_mem(queue q) {
    m_grads_matrices.free_mem(q); // output after backward pass
    m_weights_matrices.free_mem(q);
    m_weightsT_matrices.free_mem(q);
}

/**
 * Perform a forward pass of the SwiftNetMLP model.
 *
 * @param input The input data on the device.
 * @param forward Pointer to the forward intermediate array.
 * The output is stored at the end of the array 'forward'
 */
template <int WIDTH>
std::vector<sycl::event> SwiftNetMLP<WIDTH>::forward_pass(const DeviceMem<bf16> &input, float *forward,
                                                          const size_t batch_size,
                                                          const std::vector<sycl::event> &deps) {
    // Static assertion and assertion checks
    static_assert(WIDTH % 64 == 0, "Width must be a multiple of 64.");

    if ((batch_size % 16) != 0) {
        throw std::invalid_argument("Batch size is not divisible by 16.");
    }

    if (batch_size < 512) {
        throw std::invalid_argument("Batch size must be >= 512, but is " + std::to_string(batch_size));
    }
    // this is necessary for backward pass
    // Get a pointer to the weights matrices data
    bf16 *const Forwardbf16 = reinterpret_cast<bf16 *>(forward);

    // if (m_inputs_width != WIDTH) // Should only be triggered on first layer
    //     throw std::invalid_argument("m_inputs_width has to be the same as WIDTH");
    // if (m_output_width != WIDTH) // Should only be triggered on first layer
    //     throw std::invalid_argument("m_output_width has to be the same as WIDTH");

    // Perform forward pass based on activation function
    switch (m_activation) {
    case Activation::None:
        return mlp_swift_forward<WIDTH, Activation::None, false>(m_q, m_weights_matrices.data(), input.data(),
                                                                 Forwardbf16, m_n_hidden_layers, batch_size, deps);
        break;
    // case Activation::Exponential:
    // return mlp_swift_forward<WIDTH, Activation::Exponential, false>(
    //     m_q, m_weights_matrices, input,
    //     Forwardbf16,
    //     m_n_hidden_layers,
    //     m_batch_size, deps);
    //     break;
    // case Activation::Sigmoid:
    // return mlp_swift_forward<WIDTH, Activation::Sigmoid, false>(
    //     m_q, m_weights_matrices, input,
    //     Forwardbf16,
    //     m_n_hidden_layers,
    //     m_batch_size, deps);
    //     break;
    case Activation::ReLU:
        return mlp_swift_forward<WIDTH, Activation::ReLU, false>(m_q, m_weights_matrices.data(), input.data(),
                                                                 Forwardbf16, m_n_hidden_layers, batch_size, deps);
        break;
    // case Activation::LeakyReLU:
    // return mlp_swift_forward<WIDTH, Activation::LeakyReLU, false>(
    //     m_q, m_weights_matrices, input,
    //     Forwardbf16,
    //     m_n_hidden_layers,
    //     m_batch_size, deps);
    //     break;
    // case Activation::Squareplus:
    // return mlp_swift_forward<WIDTH, Activation::Squareplus, false>(
    //     m_q, m_weights_matrices, input,
    //     Forwardbf16,
    //     m_n_hidden_layers,
    //     m_batch_size, deps);
    //     break;
    // case Activation::Softplus:
    // return mlp_swift_forward<WIDTH, Activation::Softplus, false>(
    //     m_q, m_weights_matrices, input,
    //     Forwardbf16,
    //     m_n_hidden_layers,
    //     m_batch_size, deps);
    //     break;
    // case Activation::Tanh:
    // return mlp_swift_forward<WIDTH, Activation::Tanh, false>(
    //     m_q, m_weights_matrices, input,
    //     Forwardbf16,
    //     m_n_hidden_layers,
    //     m_batch_size, deps);
    //     break;
    default:
        return {};
    }
}

/**
 * Perform a forward pass of the SwiftNetMLP model.
 *
 * @param input The input data on the device.
 * @param forward Pointer to the forward intermediate array. In inference this is not used for intermediate values.
 * The output is stored at the end of the array 'forward'
 */
template <int WIDTH>
std::vector<sycl::event> SwiftNetMLP<WIDTH>::inference(const DeviceMem<bf16> &input, float *const forward,
                                                       const size_t batch_size, const std::vector<sycl::event> &deps) {
    static_assert(WIDTH % 64 == 0, "Width must be multiple of 64.");
    assert(batch_size % 64 == 0);

    // if (m_inputs_width != WIDTH || m_output_width != WIDTH) {
    //     throw std::invalid_argument("inputs_width != WIDTH or output_width!= WIDTH is not supported");
    // }

    bf16 *const Forwardbf16 = reinterpret_cast<bf16 *>(forward);

    switch (m_activation) {
    case Activation::None:
        return mlp_swift_forward<WIDTH, Activation::None, true>(m_q, m_weights_matrices.data(), input.data(),
                                                                Forwardbf16, m_n_hidden_layers, batch_size, deps);
        break;
    // case Activation::Exponential:
    //     return mlp_swift_forward<WIDTH, Activation::Exponential, true>(
    //         m_q, m_weights_matrices, input, Forwardbf16,
    //         m_n_hidden_layers, m_batch_size, deps);
    // break;
    // case Activation::Sigmoid:
    //     return mlp_swift_forward<WIDTH, Activation::Sigmoid, true>(
    //         m_q, m_weights_matrices, input, Forwardbf16,
    //         m_n_hidden_layers, m_batch_size, deps);
    // break;
    case Activation::ReLU:
        return mlp_swift_forward<WIDTH, Activation::ReLU, true>(m_q, m_weights_matrices.data(), input.data(),
                                                                Forwardbf16, m_n_hidden_layers, batch_size, deps);
        break;
    // case Activation::LeakyReLU:
    //     return mlp_swift_forward<WIDTH, Activation::LeakyReLU, true>(
    //         m_q, m_weights_matrices, input, Forwardbf16,
    //         m_n_hidden_layers, m_batch_size, deps);
    // break;
    // case Activation::Squareplus:
    //     return mlp_swift_forward<WIDTH, Activation::Squareplus, true>(
    //         m_q, m_weights_matrices, input, Forwardbf16,
    //         m_n_hidden_layers, m_batch_size, deps);
    // break;
    // case Activation::Softplus:
    //     return mlp_swift_forward<WIDTH, Activation::Softplus, true>(
    //         m_q, m_weights_matrices, input, Forwardbf16,
    //         m_n_hidden_layers, m_batch_size, deps);
    // break;
    // case Activation::Tanh:
    //     return mlp_swift_forward<WIDTH, Activation::Tanh, true>(
    //         m_q, m_weights_matrices, input, Forwardbf16,
    //         m_n_hidden_layers, m_batch_size, deps);
    // break;
    default:
        throw std::runtime_error{"Unsupported activation."};
    }
}

/**
 * Perform the backward pass of the neural network.
 *
 * @param grads The gradients on the device. Input for the backward pass
 * @param out_inter Intermediate array for storing outputs. This is filled as part of the backward pass
 * @param forward Pointer to the forward intermediate array which was filled in the forw pass
 */
template <int WIDTH>
std::vector<sycl::event> SwiftNetMLP<WIDTH>::backward_pass(const DeviceMem<bf16> &grads, float *const out_inter,
                                                           float const *const forward, const size_t batch_size,
                                                           const std::vector<sycl::event> &deps) {
    // Compute activation backpropagation using parallel_for
    bf16 const *const Forwardbf16 = reinterpret_cast<const bf16 *>(forward);
    bf16 *const out_interbf16 = reinterpret_cast<bf16 *>(out_inter);

    // Choose appropriate mlp_swiftnet_backward based on activation
    // We are onyl doinh output_activation==none right now
    switch (m_activation) {
    case Activation::None:
        return mlp_swiftnet_backward<WIDTH, Activation::None>(m_q, m_weightsT_matrices.data(), grads.data(),
                                                              m_grads_matrices.data(), out_interbf16, Forwardbf16,
                                                              m_n_hidden_matrices, batch_size, deps);
        break;
    case Activation::ReLU:
        return mlp_swiftnet_backward<WIDTH, Activation::ReLU>(m_q, m_weightsT_matrices.data(), grads.data(),
                                                              m_grads_matrices.data(), out_interbf16, Forwardbf16,
                                                              m_n_hidden_matrices, batch_size, deps);
        break;
    // case Activation::LeakyReLU:
    // return mlp_swiftnet_backward<WIDTH, Activation::LeakyReLU>(
    //     m_q, m_weightsT_matrices, grads, m_grads_matrices,
    //     out_interbf16, Forwardbf16,
    //    m_n_hidden_matrices, m_batch_size,deps);
    //     break;
    // case Activation::Exponential:
    // return mlp_swiftnet_backward<WIDTH, Activation::Exponential>(
    //     m_q, m_weightsT_matrices, grads, m_grads_matrices,
    //     out_interbf16, Forwardbf16,
    //     m_n_hidden_matrices, m_batch_size,deps);
    //     break;
    // case Activation::Sigmoid:
    // return mlp_swiftnet_backward<WIDTH, Activation::Sigmoid>(
    //     m_q, m_weightsT_matrices, grads, m_grads_matrices,
    //     out_interbf16, Forwardbf16,
    //     m_n_hidden_matrices, m_batch_size,deps);
    //     break;
    // case Activation::Tanh:
    // return mlp_swiftnet_backward<WIDTH, Activation::Tanh>(
    //     m_q, m_weightsT_matrices, grads, m_grads_matrices,
    //     out_interbf16, Forwardbf16,
    //     m_n_hidden_matrices, m_batch_size,deps);
    //     break;
    default:
        return {};
    }
}

template <int WIDTH>
std::vector<sycl::event> SwiftNetMLP<WIDTH>::training(const DeviceMem<bf16> &input, const DeviceMem<bf16> &target,
                                                      float *const intermediate_output_forward,
                                                      float *const intermediate_output_backward,
                                                      const size_t batch_size, const std::vector<sycl::event> &deps) {
    // Compute activation backpropagation using parallel_for
    bf16 *const intermediate_output_forwardbf16 = reinterpret_cast<bf16 *>(intermediate_output_forward);
    bf16 *const intermediate_output_backwardbf16 = reinterpret_cast<bf16 *>(intermediate_output_backward);

    return mlp_swift_fused<WIDTH, Activation::ReLU, Activation::None>(
        m_q, m_weights_matrices.data(), m_weightsT_matrices.data(),
        input.data(),            // input to forward pass
        target.data(),           // targets for error computation
        m_grads_matrices.data(), // gradients output after backward pass
        intermediate_output_forwardbf16, intermediate_output_backwardbf16, m_n_hidden_layers, batch_size, deps);
}

template <int WIDTH> void SwiftNetMLP<WIDTH>::set_params(float *params) {
    auto p = m_weights_matrices.data();
    int s = m_weights_matrices.size();

    m_q.parallel_for<>(range<1>(s), [=](id<1> idx) { p[idx] = bf16(params[idx]); }).wait();
    m_weights_matrices.make_transposed(m_weightsT_matrices, m_inputs_width_padded, WIDTH, m_output_width,
                                       m_n_hidden_matrices, m_q);
}

template <int WIDTH> void SwiftNetMLP<WIDTH>::set_params(std::vector<bf16> &params) {
    m_weights_matrices.copy_from_host(params, m_q);
    m_weights_matrices.make_transposed(m_weightsT_matrices, m_inputs_width_padded, WIDTH, m_output_width,
                                       m_n_hidden_matrices, m_q);
}

template <int WIDTH> std::vector<bf16> SwiftNetMLP<WIDTH>::get_weights_matrices_as_vector() {
    std::vector<bf16> list_float(m_weights_matrices.size());
    m_weights_matrices.copy_to_host(list_float, m_q);
    return list_float;
}

template <int WIDTH> std::vector<bf16> SwiftNetMLP<WIDTH>::get_weightsT_matrices_as_vector() {
    std::vector<bf16> list_float(m_weightsT_matrices.size());
    m_weightsT_matrices.copy_to_host(list_float, m_q);
    return list_float;
}

template class SwiftNetMLP<64>;
template class SwiftNetMLP<128>;