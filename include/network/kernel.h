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

////////////////////////////GENERAL FUNCTIONS WHICH CAN DO EVERYTHING///////////

// Todo: May want to remove some of the template parameters of these functions and
// make them inputs.

// This is the general forward map which also doubles as inference. We use template
// specialization for all the versions
template <typename T, typename Tc, int INPUT_WIDTH, int OUTPUT_WIDTH, Activation activation,
          Activation output_activation, bool INFERENCE, std::size_t TN>
std::vector<sycl::event>
forward_impl_4(sycl::queue &q, T const *const __restrict__ weights_ptr, T const *const __restrict__ inputs_ptr,
               T *const __restrict__ intermediate_output, const int n_hidden_layers, const int M,
               const std::vector<sycl::event> &deps) { // reuse of B, this is in subgroups, ONLY works for 64
    // note that large grf mode requires this to be set to 32, but then this code does not work anymore
    constexpr int SG_SIZE = TN;
    const int SGS_IN_WG =
        q.get_device().get_info<sycl::info::device::max_work_group_size>() / SG_SIZE; // maximum number
    constexpr int WIDTH = 4 * TN;
    constexpr size_t TM = 8;                                             // this may be adjusted in the future
    constexpr size_t TK = 8 * std::min<size_t>(8, 32 / (8 * sizeof(T))); // This depends on the datatype T
    assert(M % TM == 0); // make sure there is no remainder and no out of bounds accesses
    static_assert(INPUT_WIDTH == WIDTH);
    static_assert(OUTPUT_WIDTH == WIDTH);

    std::cout << "Starting KERNEL" << std::endl;

    // One Block Row has TM rows an N columns.
    auto e = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(deps);

        sycl::local_accessor<T, 1> B(sycl::range<1>(WIDTH * WIDTH), cgh);
        sycl::local_accessor<T, 1> Atmp(sycl::range<1>(TM * WIDTH * SGS_IN_WG), cgh);

        // number of SGS is given by M / TM, since batch_size is the number of rows in the output
        cgh.parallel_for(
            sycl::nd_range<1>(M / TM * SG_SIZE, SGS_IN_WG * SG_SIZE),
            [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                auto sg = item.get_sub_group();

                T const *weights_ptr_loc = weights_ptr;
                auto A_sg_start =
                    Atmp.template get_multi_ptr<access::decorated::yes>() + sg.get_group_id()[0] * WIDTH * TM;
                auto B_ptr = B.template get_multi_ptr<access::decorated::yes>();

                // offset in all the data
                const int wg_and_sg_offset_A =
                    item.get_group().get_group_id() * SGS_IN_WG * WIDTH * TM + sg.get_group_id()[0] * WIDTH * TM;
                int layer_offset_A = M * WIDTH + wg_and_sg_offset_A;

                helpers::moveMemory<WIDTH, WIDTH>(
                    item,
                    address_space_cast<access::address_space::global_space, access::decorated::yes>(weights_ptr_loc),
                    B_ptr);
                weights_ptr_loc += WIDTH * WIDTH; // next weight matrix

                // load input in slm
                helpers::moveMemorySG<TM, WIDTH>(
                    sg,
                    address_space_cast<access::address_space::global_space, access::decorated::yes>(inputs_ptr +
                                                                                                    wg_and_sg_offset_A),
                    A_sg_start);

                // if not inference activate and store in intermediate output
                if constexpr (!INFERENCE)
                    helpers::applyActivation<activation, TM, WIDTH>(
                        sg, A_sg_start,
                        address_space_cast<access::address_space::global_space, access::decorated::yes>(
                            intermediate_output + wg_and_sg_offset_A));

                // this is the only reason why this is not general
                joint_matrix<sycl::sub_group, Tc, use::accumulator, TM, TN> C_block0, C_block1, C_block2, C_block3;

                for (int layer = 0; layer < n_hidden_layers; layer++) {
                    // reset result matrices
                    helpers::zeroMatrices(sg, C_block0, C_block1, C_block2, C_block3);

                    // ensure weight matrix is loaded
                    item.barrier(sycl::access::fence_space::local_space);

                    helpers::MAD<WIDTH, TK>(sg, A_sg_start, B_ptr, C_block0, C_block1, C_block2, C_block3);

                    item.barrier(sycl::access::fence_space::local_space);
                    // load next weight matrix

                    helpers::moveMemory<WIDTH, WIDTH>(
                        item,
                        address_space_cast<access::address_space::global_space, access::decorated::yes>(
                            weights_ptr_loc),
                        B_ptr);

                    // activate and save
                    helpers::applyActivation<activation, WIDTH>(sg, C_block0, A_sg_start);
                    helpers::applyActivation<activation, WIDTH>(sg, C_block1, A_sg_start + TN);
                    helpers::applyActivation<activation, WIDTH>(sg, C_block2, A_sg_start + 2 * TN);
                    helpers::applyActivation<activation, WIDTH>(sg, C_block3, A_sg_start + 3 * TN);

                    if constexpr (!INFERENCE)
                        helpers::moveMemorySG<TM, WIDTH>(
                            sg, A_sg_start,
                            address_space_cast<access::address_space::global_space, access::decorated::yes>(
                                intermediate_output + layer_offset_A));

                    layer_offset_A += M * WIDTH;
                }

                // generate output, i.e. last GEMM
                helpers::zeroMatrices(sg, C_block0, C_block1, C_block2, C_block3);

                // wait for B to be loaded
                item.barrier(sycl::access::fence_space::local_space);

                helpers::MAD<WIDTH, TK>(sg, A_sg_start, B_ptr, C_block0, C_block1, C_block2, C_block3);

                // activate and save to slm
                helpers::applyActivation<output_activation, WIDTH>(sg, C_block0, A_sg_start);
                helpers::applyActivation<output_activation, WIDTH>(sg, C_block1, A_sg_start + TN);
                helpers::applyActivation<output_activation, WIDTH>(sg, C_block2, A_sg_start + 2 * TN);
                helpers::applyActivation<output_activation, WIDTH>(sg, C_block3, A_sg_start + 3 * TN);

                // save slm to HBM
                helpers::moveMemorySG<TM, WIDTH>(
                    sg, A_sg_start,
                    address_space_cast<access::address_space::global_space, access::decorated::yes>(
                        intermediate_output + layer_offset_A));
            });
    });

    return {e};
}

} // namespace kernels
} // namespace tinydpcppnn
