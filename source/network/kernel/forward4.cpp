
#include "kernel.h"
#include "kernel_helper.h"

namespace tinydpcppnn {
namespace kernels {

// general forward which can do forward and inference, small batchsizes,
// large batchsizes, and all input and output widths and all types
// with 4 matrices per row in the output (i.e. WIDTH = 4*TN, TN depends on the device)
template <typename T, typename Tc, int INPUT_WIDTH, int OUTPUT_WIDTH, Activation activation,
          Activation output_activation, bool INFERENCE, size_t TN>
std::vector<sycl::event> mlp_swift_forward_4(sycl::queue &q, T const *const __restrict__ weights_ptr,
                                             T const *const __restrict__ inputs_ptr,
                                             T *const __restrict__ intermediate_output, const int n_hidden_layers,
                                             const int M, const std::vector<sycl::event> &deps) {
    // reuse of B, this is in subgroups, ONLY works for 64
    // note that large grf mode requires this to be set to 32, but then this code does not work anymore
    constexpr int SG_SIZE = TN;
    constexpr int SGS_IN_WG = q.get_device(info::device::max_work_group_size).get_info<>() / SG_SIZE; // maximum number
    constexpr int WIDTH = 4 * TN;
    constexpr size_t TM = 8;                                             // this may be adjusted in the future
    constexpr size_t TK = 8 * std::min<size_t>(8, 32 / (8 * sizeof(T))); // This depends on the datatype T
    assert(M % TM == 0); // make sure there is no remainder and no out of bounds accesses
    static_assert(INPUT_WIDTH == WIDTH);
    static_assert(OUTPUT_WIDTH == WIDTH);

    // One Block Row has TM rows an N columns.
    auto e = q.submit([&](handler &cgh) {
        cgh.depends_on(deps);
        local_accessor<T, 1> B(range<1>(WIDTH * WIDTH),
                               cgh); // weights matrix. 64*64*2 byte = 8 kb. Thus, can have up to 16 WGs per Xe Core.
        local_accessor<T, 1> Atmp(range<1>(TM * WIDTH * SGS_IN_WG),
                                  cgh); // buffer for loading joint matrices. 8*64*64*2byte = 64kb. TODO: check if
                                        // this is too much. If so, split in half
        // number of SGS is given by batch_size / TM, since batch_size is the number of rows in the output
        cgh.parallel_for(
            nd_range<1>(M / TM * SG_SIZE,
                        SGS_IN_WG * SG_SIZE), // assuming here that the number of block rows is divisable by SGS_IN_WG
            [=](nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                auto sg = item.get_sub_group();

                T const *weights_ptr_loc = weights_ptr;
                auto A_sg_start = local_ptr<T>(&Atmp[sg.get_group_id()[0] * WIDTH * TM]);
                auto B_ptr = local_ptr<T>(&B[0]);

                // offset in all the data
                const int wg_and_sg_offset_A =
                    item.get_group().get_group_id() * SGS_IN_WG * WIDTH * TM + sg.get_group_id()[0] * WIDTH * TM;
                int layer_offset_A = M * WIDTH + wg_and_sg_offset_A;

                helpers::moveMemory<WIDTH, WIDTH, T>(item, global_ptr<T>(weights_ptr_loc), B_ptr);
                weights_ptr_loc += WIDTH * WIDTH; // next weight matrix

                // load input in slm
                helpers::moveMemorySG<TM, WIDTH, T>(sg, global_ptr<T>(inputs_ptr + wg_and_sg_offset_A), A_sg_start);

                // if not inference activate and store in intermediate output
                if constexpr (!INFERENCE)
                    helpers::applyActivation(sg, A_sg_start, global_ptr<T>(intermediate_output + wg_and_sg_offset_A));

                joint_matrix<sub_group, Tc, use::accumulator, TM, TN> C_block0, C_block1, C_block2,
                    C_block3; // this is the only reason why this is not general
                for (int layer = 0; layer < n_hidden_layers; layer++) {
                    // reset result matrices
                    helpers::zeroMatrices(sg, C_block0, C_block1, C_block2, C_block3);

                    // ensure weight matrix is loaded
                    item.barrier(sycl::access::fence_space::local_space);

                    helpers::MAD(sg, A_sg_start, B_ptr, C_block0, C_block1, C_block2, C_block3);

                    item.barrier(sycl::access::fence_space::local_space);
                    // load next weight matrix
                    helpers::moveMemory<WIDTH, WIDTH, T>(item, global_ptr<T>(weights_ptr_loc), B_ptr);

                    // activate and save
                    applyActivation<activation>(sg, C_block0, A_sg_start);
                    applyActivation<activation>(sg, C_block1, A_sg_start + TN);
                    applyActivation<activation>(sg, C_block2, A_sg_start + 2 * TN);
                    applyActivation<activation>(sg, C_block3, A_sg_start + 3 * TN);

                    if constexpr (!INFERENCE) {
                        helpers::moveMemorySG<TM, WIDTH, T>(sg, A_sg_start,
                                                            global_ptr<T>(intermediate_output + layer_offset_A));
                    }
                    layer_offset_A += M * WIWDTH;
                }

                // generate output, i.e. last GEMM
                helpers::zeroMatrices(sg, C_block0, C_block1, C_block2, C_block3);

                item.barrier(sycl::access::fence_space::local_space); // wait for B to be loaded

                helpers::MAD(sg, A_sg_start, B_ptr, C_block0, C_block1, C_block2, C_block3);

                // activate and save to slm
                applyActivation<output_activation>(sg, C_block0, A_sg_start);
                applyActivation<output_activation>(sg, C_block1, A_sg_start + TN);
                applyActivation<output_activation>(sg, C_block2, A_sg_start + 2 * TN);
                applyActivation<output_activation>(sg, C_block3, A_sg_start + 3 * TN);

                // save slm to HBM
                helpers::moveMemorySG<TM, WIDTH, T>(sg, A_sg_start,
                                                    global_ptr<T>(intermediate_output + layer_offset_A));
            });
    });

    return {e};
}

} // namespace kernels
} // namespace tinydpcppnn