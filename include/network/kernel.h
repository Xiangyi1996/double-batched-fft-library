// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

// This file lists all the inference, forward, backward and fused (forward+backw)
// functions we have.
//
// In general, there should always be one 'general' implementation which
// ignores performance and then specialized implementations which are optimized
// for their use case.

// The netweork forward_impl, inference_impl, backward_impl functions will then
// decide at runtime which one to choose. May do an abstraction around this?
// The netweok *_impl functions may also have template specializations to
// make the choice quicker.

#pragma once

#include <algorithm>
#include <sycl/sycl.hpp>
#include <vector>

#include "common.h"
#include "kernel_helper.h"

namespace tinydpcppnn {
namespace kernels {

using bf16 = sycl::ext::oneapi::bfloat16;
using namespace sycl::ext::oneapi::experimental::matrix;

template <typename T, typename Tc, int INPUT_WIDTH, int WIDTH, int OUTPUT_WIDTH, size_t TN>
inline std::vector<sycl::event>
batchedGEMM_naive(sycl::queue &q, T *const __restrict__ output_ptr, T const *const __restrict__ intermediate_forward,
                  T const *const __restrict__ intermediate_backward, const int n_hidden_layers, const int M,
                  const std::vector<sycl::event> &deps) {
    constexpr int SG_SIZE = TN;
    auto e = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(deps);

        cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(n_hidden_layers + 1, WIDTH * WIDTH),
                                           sycl::range<2>(1, std::min(1024, WIDTH * WIDTH))),
                         [=](sycl::nd_item<2> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                             const int matrix = item.get_global_id(0);
                             const int element = item.get_global_id(1);
                             const int row = element / WIDTH;
                             const int col = element % WIDTH;

                             Tc tmp_out = static_cast<Tc>(0);
                             T const *intermediate_forward_loc = intermediate_forward + matrix * M * WIDTH + row;
                             T const *intermediate_backward_loc = intermediate_backward + matrix * M * WIDTH + col;
                             for (int i = 0; i < M; i++) {
                                 tmp_out += static_cast<Tc>(*intermediate_forward_loc) *
                                            static_cast<Tc>(*intermediate_backward_loc);
                                 intermediate_forward_loc += WIDTH;
                                 intermediate_backward_loc += WIDTH;
                             }
                             T *const output_ptr_loc = output_ptr + WIDTH * WIDTH * matrix + element;
                             *output_ptr_loc = static_cast<T>(tmp_out);
                         });
    });
    // auto e =
    //     q.parallel_for((n_hidden_layers + 1) * WIDTH * WIDTH, [=](auto item) [[intel::reqd_sub_group_size(SG_SIZE)]]
    //     {
    //         output_ptr[item.get_id()] = static_cast<T>(1.23);
    //     });

    return {e};
}

////////////////////////////GENERAL FUNCTIONS WHICH CAN DO EVERYTHING///////////

// Todo: May want to remove some of the template parameters of these functions and
// make them inputs.

// This is the general forward map which also doubles as inference. We use template
// specialization for all the versions
template <typename T, typename Tc, int INPUT_WIDTH, int WIDTH, int OUTPUT_WIDTH, Activation activation,
          Activation output_activation, bool INFERENCE, size_t TN>
std::vector<sycl::event> forward_impl_general(sycl::queue &q, T const *const __restrict__ weights_ptr,
                                              T const *const __restrict__ inputs_ptr,
                                              T *const __restrict__ intermediate_output, const int n_hidden_layers,
                                              const int M, const std::vector<sycl::event> &deps) {
    constexpr int SG_SIZE = TN;
    constexpr size_t TM = 8;                                             // this may be adjusted in the future
    constexpr size_t TK = 8 * std::min<size_t>(8, 32 / (8 * sizeof(T))); // This depends on the datatype T
    const int SGS_IN_WG =
        std::min(M / TM, q.get_device().get_info<sycl::info::device::max_work_group_size>() / SG_SIZE);
    static_assert(WIDTH % TN == 0);
    constexpr int NC = WIDTH / TN; // number of systolic C matrices in the output

    assert(M % TM == 0); // make sure there is no remainder and no out of bounds accesses
    static_assert(INPUT_WIDTH == WIDTH);
    static_assert(OUTPUT_WIDTH == WIDTH);

    // One Block Row has TM rows an N columns.
    auto e = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(deps);

        sycl::local_accessor<T, 1> B(sycl::range<1>(WIDTH * WIDTH), cgh);
        sycl::local_accessor<T, 1> Atmp(sycl::range<1>(TM * WIDTH * SGS_IN_WG), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(M / TM * SG_SIZE, SGS_IN_WG * SG_SIZE),
            [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                auto sg = item.get_sub_group();

                auto weights_ptr_loc =
                    address_space_cast<access::address_space::global_space, access::decorated::yes>(weights_ptr);
                auto intermediate_output_loc =
                    address_space_cast<access::address_space::global_space, access::decorated::yes>(
                        intermediate_output);
                auto A_sg_start =
                    Atmp.template get_multi_ptr<access::decorated::yes>() + sg.get_group_id()[0] * WIDTH * TM;
                auto B_ptr = B.template get_multi_ptr<access::decorated::yes>();

                // offset in all the data
                const int wg_and_sg_offset_A =
                    item.get_group().get_group_id() * SGS_IN_WG * WIDTH * TM + sg.get_group_id()[0] * WIDTH * TM;
                int layer_offset_A = M * WIDTH + wg_and_sg_offset_A;

                helpers::moveMemory<WIDTH, WIDTH>(item, weights_ptr_loc, B_ptr);
                weights_ptr_loc += WIDTH * WIDTH; // next weight matrix

                // load input in slm
                helpers::moveMemorySG<TM, WIDTH>(
                    sg,
                    address_space_cast<access::address_space::global_space, access::decorated::yes>(inputs_ptr +
                                                                                                    wg_and_sg_offset_A),
                    A_sg_start);

                // if not inference activate and store in intermediate output
                if constexpr (!INFERENCE)
                    helpers::applyActivation<activation, TM, WIDTH>(sg, A_sg_start,
                                                                    intermediate_output_loc + wg_and_sg_offset_A);

                std::array<joint_matrix<sycl::sub_group, Tc, use::accumulator, TM, TN>, NC> Cs;
                for (int layer = 0; layer < n_hidden_layers; layer++) {
                    // reset result matrices
                    helpers::zeroMatrices(sg, Cs);

                    // ensure weight matrix is loaded
                    item.barrier(sycl::access::fence_space::local_space);

                    helpers::MAD<TK>(sg, A_sg_start, B_ptr, Cs);

                    item.barrier(sycl::access::fence_space::local_space);
                    // load next weight matrix

                    helpers::moveMemory<WIDTH, WIDTH>(item, weights_ptr_loc, B_ptr);
                    weights_ptr_loc += WIDTH * WIDTH; // next weight matrix

                    // activate and save
                    helpers::applyActivation<activation>(sg, Cs, A_sg_start);

                    if constexpr (!INFERENCE)
                        helpers::moveMemorySG<TM, WIDTH>(sg, A_sg_start, intermediate_output_loc + layer_offset_A);

                    layer_offset_A += M * WIDTH;
                }

                // generate output, i.e. last GEMM
                helpers::zeroMatrices(sg, Cs);

                // wait for B to be loaded
                item.barrier(sycl::access::fence_space::local_space);

                helpers::MAD<TK>(sg, A_sg_start, B_ptr, Cs);

                // activate and save to slm
                helpers::applyActivation<output_activation>(sg, Cs, A_sg_start);

                // save slm to HBM
                helpers::moveMemorySG<TM, WIDTH>(sg, A_sg_start, intermediate_output_loc + layer_offset_A);
            });
    });

    return {e};
}

template <typename T, typename Tc, int INPUT_WIDTH, int WIDTH, int OUTPUT_WIDTH, Activation activation,
          Activation output_activation, size_t TN>
std::vector<sycl::event> backward_impl_general(queue &q, T const *const __restrict__ weights_ptr,
                                               T const *const __restrict__ inputs_ptr, T *const __restrict__ output_ptr,
                                               T *const __restrict__ intermediate_output,
                                               T const *const __restrict__ forward, const int n_hidden_layers,
                                               const int M, const std::vector<sycl::event> &deps) {

    // make sure there is no remainder and no out of bounds accesses
    static_assert(WIDTH % TN == 0);
    // only works for input_width == width == output_width
    static_assert(INPUT_WIDTH == WIDTH);
    static_assert(OUTPUT_WIDTH == WIDTH);

    constexpr int SG_SIZE = TN;
    const int SGS_IN_WG =
        q.get_device().get_info<sycl::info::device::max_work_group_size>() / SG_SIZE; // maximum number

    constexpr int NC = WIDTH / TN; // number of systolic C matrices in the output
    constexpr size_t TM = 8;       // this may be adjusted in the future
    assert(M % TM == 0);
    constexpr size_t TK = 8 * std::min<size_t>(8, 32 / (8 * sizeof(T))); // This depends on the datatype T

    auto e = q.submit([&](handler &cgh) {
        cgh.depends_on(deps);

        local_accessor<T, 1> B(range<1>(WIDTH * WIDTH), cgh);
        local_accessor<T, 1> Atmp(range<1>(TM * WIDTH * SGS_IN_WG), cgh);

        const int nitems = M / TM * SG_SIZE;
        cgh.parallel_for(
            sycl::nd_range<1>(nitems, std::min(nitems, SGS_IN_WG * SG_SIZE)),
            [=](nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                auto sg = item.get_sub_group();

                auto weights_ptr_loc =
                    address_space_cast<access::address_space::global_space, access::decorated::yes>(weights_ptr) +
                    n_hidden_layers * WIDTH * WIDTH;
                auto intermediate_output_loc =
                    address_space_cast<access::address_space::global_space, access::decorated::yes>(
                        intermediate_output);
                const auto forward_loc =
                    address_space_cast<access::address_space::global_space, access::decorated::yes>(forward);
                auto A_sg_start =
                    Atmp.template get_multi_ptr<access::decorated::yes>() + sg.get_group_id()[0] * WIDTH * TM;
                auto B_ptr = B.template get_multi_ptr<access::decorated::yes>();

                // offset in all the data
                const int wg_and_sg_offset_A =
                    item.get_group().get_group_id() * SGS_IN_WG * WIDTH * TM + sg.get_group_id()[0] * WIDTH * TM;
                /// TODO: check if this is n_hidden_layers or n_hidden_layers+1
                int layer_offset_A = n_hidden_layers * M * WIDTH + wg_and_sg_offset_A;

                // Get B into slm
                helpers::moveMemory<WIDTH, WIDTH>(item, weights_ptr_loc, B_ptr);
                weights_ptr_loc -= WIDTH * WIDTH; // decrease weights pointer by one layer

                // load input in slm
                helpers::moveMemorySG<TM, WIDTH>(
                    sg,
                    address_space_cast<access::address_space::global_space, access::decorated::yes>(inputs_ptr +
                                                                                                    wg_and_sg_offset_A),
                    A_sg_start);

                // store backward activated input to the last intermediate output
                // note that output_activation == ReLU does not need any work since that means
                // forward >= 0
                if constexpr (output_activation != Activation::None && output_activation != Activation::ReLU) {
                    helpers::applyBackwardActivation<output_activation, TM, WIDTH>(
                        sg, A_sg_start, forward_loc + layer_offset_A + M * WIDTH, A_sg_start);
                }

                // store activated slm in intermediate output
                helpers::moveMemorySG<TM, WIDTH>(sg, A_sg_start, intermediate_output_loc + layer_offset_A);

                std::array<joint_matrix<sycl::sub_group, Tc, use::accumulator, TM, TN>, NC> Cs;
                // we are also doing output->last hidden layer
                for (int layer = n_hidden_layers; layer > 0; layer--) {
                    layer_offset_A -= M * WIDTH;
                    helpers::zeroMatrices(sg, Cs);

                    // wait for B to be done storing
                    item.barrier(sycl::access::fence_space::local_space);

                    helpers::MAD<TK>(sg, A_sg_start, B_ptr, Cs);

                    // load B for next iteration into SLM
                    if (layer > 1) {
                        // wait for B to de done in the MAD
                        item.barrier(sycl::access::fence_space::local_space);
                        helpers::moveMemory<WIDTH, WIDTH>(item, weights_ptr_loc, B_ptr);
                        weights_ptr_loc -= WIDTH * WIDTH;
                    }

                    // If forward activation is ReLU we also do not need to do anything since all the values in forward
                    // are >= 0
                    helpers::applyBackwardActivation<activation == Activation::ReLU || activation == Activation::None
                                                         ? Activation::None
                                                         : activation>(sg, Cs, forward_loc + layer_offset_A + M * WIDTH,
                                                                       A_sg_start);

                    // store A slm to HBM
                    helpers::moveMemorySG<TM, WIDTH>(sg, A_sg_start, intermediate_output_loc + layer_offset_A);
                }
            });
    });

    // // NOTE: MKL gemm_batch is slower.
    // std::vector<sycl::event> events(n_hidden_layers + 1);
    // if constexpr (std::is_same<T, bf16>::value) { // need to cast to onemkls bf16 type.
    //     for (int iter = 0; iter < n_hidden_layers + 1; iter++) {
    //         events[iter] = oneapi::mkl::blas::row_major::gemm(
    //             q, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, WIDTH, WIDTH, M, 1.0f,
    //             reinterpret_cast<const oneapi::mkl::bfloat16 *>(forward) + iter * M * WIDTH, WIDTH,
    //             reinterpret_cast<oneapi::mkl::bfloat16 *>(intermediate_output) + iter * M * WIDTH, WIDTH, 1.0f,
    //             reinterpret_cast<oneapi::mkl::bfloat16 *>(output_ptr) + iter * WIDTH * WIDTH, WIDTH, {e});
    //     }
    // } else {
    //     throw std::invalid_argument("Untested code path.");
    //     for (int iter = 0; iter < n_hidden_layers + 1; iter++) {
    //         events[iter] = oneapi::mkl::blas::row_major::gemm(
    //             q, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, WIDTH, WIDTH, M, 1.0,
    //             forward + iter * M * WIDTH, WIDTH, intermediate_output + iter * M * WIDTH, WIDTH, 1.0,
    //             output_ptr + iter * WIDTH * WIDTH, WIDTH, {e});
    //     }
    // }
    // return events;

    return batchedGEMM_naive<T, Tc, INPUT_WIDTH, WIDTH, OUTPUT_WIDTH, TN>(q, output_ptr, forward, intermediate_output,
                                                                          n_hidden_layers, M, {e});
}

} // namespace kernels
} // namespace tinydpcppnn
