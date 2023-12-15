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
#include "kernel_helper_esimd.h"
#include "oneapi/mkl.hpp"

namespace tinydpcppnn {
namespace kernels {
namespace esimd {

using bf16 = sycl::ext::oneapi::bfloat16;
using namespace sycl::ext::intel::esimd;
using sycl::ext::intel::experimental::esimd::cache_hint;

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
          Activation output_activation, bool INFERENCE, int TN>
std::vector<sycl::event> forward_impl_general(sycl::queue &q, T const *const __restrict__ weights_ptr,
                                              T const *const __restrict__ inputs_ptr,
                                              T *const __restrict__ intermediate_output, const int n_hidden_layers,
                                              const int M, const std::vector<sycl::event> &deps) {

    // throw std::logic_error("General function should not be called.");
    static_assert(INPUT_WIDTH == WIDTH);
    static_assert(OUTPUT_WIDTH == WIDTH);
    static_assert(WIDTH % TN == 0);

    constexpr int TM = 8;
    // make sure there is no remainder and no out of bounds accesses
    // this may be adjusted in the future
    assert(M % TM == 0);
    // TK depends on the datatype T
    constexpr int TK = 8 * std::min<int>(8, 32 / (8 * sizeof(T)));
    constexpr int number_preload_weights = 128 * 1024 / (sizeof(T) * WIDTH * WIDTH);
    // TODO: 64 depends on the device. It is different for non-PVC hardware
    int ITEMS_IN_WG = std::min<int>(M / TM, 64);
    while (M / TM % ITEMS_IN_WG != 0) {
        ITEMS_IN_WG--;
    }
    if (ITEMS_IN_WG <= 0) throw std::logic_error("Number of SGS per WG cannot be less than 1");

    // One Block Row has TM rows an N columns.
    auto e = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(deps);

        cgh.parallel_for(sycl::nd_range<1>(M / TM, ITEMS_IN_WG), [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
            // b offset = 0;
            // testing if we can preload multiple weight matrices for better perf
            slm_init<number_preload_weights * WIDTH * WIDTH * sizeof(T)>();

            int B_offset = 0;
            int layer_offset_A = item.get_global_linear_id() * WIDTH * TM;

            for (int iter = 0; iter < std::min<int>(number_preload_weights, n_hidden_layers + 1); iter++) {
                helpers::moveToSlmWG<TK, TN, WIDTH>(item, weights_ptr + iter * WIDTH * WIDTH,
                                                    B_offset + iter * WIDTH * WIDTH);
            }

            // we store blocks contiguously
            simd<T, TM * WIDTH> As;
            helpers::loadRow<TM, TK, WIDTH, cache_hint::uncached, cache_hint::uncached>(inputs_ptr + layer_offset_A,
                                                                                        As);

            // if not inference activate and store in intermediate output
            if constexpr (!INFERENCE) {
                simd<T, TM * WIDTH> tmpA;
                helpers::applyActivation<activation, TM, TK, TN>(As, tmpA);
                helpers::storeRow<TM, TK, WIDTH, cache_hint::uncached, cache_hint::uncached>(tmpA, intermediate_output +
                                                                                                       layer_offset_A);
            }

            simd<Tc, TM * WIDTH> Cs;
            layer_offset_A += M * WIDTH;
            item.barrier();
            for (int layer = 0; layer < n_hidden_layers; layer++) {
                // reset result matrices
                Cs = static_cast<Tc>(0);

                helpers::MAD<TM, TK, TN>(As, B_offset, Cs);
                B_offset += WIDTH * WIDTH;

                if ((layer + 1) % number_preload_weights == 0) {
                    item.barrier();
                    B_offset = 0;
                    // load next weight matrix
                    for (int iter = 0; iter < std::min<int>(number_preload_weights, n_hidden_layers - layer); iter++) {
                        helpers::moveToSlmWG<TK, TN, WIDTH>(
                            item, weights_ptr + (layer + 1) * WIDTH * WIDTH + iter * WIDTH * WIDTH,
                            B_offset + iter * WIDTH * WIDTH);
                    }

                    item.barrier();
                }

                // activate and save
                helpers::applyActivation<activation, TM, TK, TN>(Cs, As);

                if constexpr (!INFERENCE)
                    helpers::storeRow<TM, TK, WIDTH, cache_hint::uncached, cache_hint::uncached>(
                        As, intermediate_output + layer_offset_A);

                layer_offset_A += M * WIDTH;
            }

            // generate output, i.e. last GEMM
            Cs = static_cast<Tc>(0);

            helpers::MAD<TM, TK, TN>(As, B_offset, Cs);

            // activate and save to slm
            helpers::applyActivation<output_activation, TM, TK, TN>(Cs, As);

            // save slm to HBM
            helpers::storeRow<TM, TK, WIDTH, cache_hint::uncached, cache_hint::uncached>(As, intermediate_output +
                                                                                                 layer_offset_A);
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
    // this may be adjusted in the future in dpendence of M
    constexpr size_t TM = 8;
    constexpr int number_preload_weights = 128 * 1024 / (sizeof(T) * WIDTH * WIDTH);
    int ITEMS_IN_WG = std::min<int>(M / TM, 64);
    /// TODO: say we use M/TM = 65. Then this results in WG=1 SG and too many slm load of B.
    /// Better: Use max size WGs and return those which are larger than M/TM. But
    /// requires special care for the loading of B
    while (M / TM % ITEMS_IN_WG != 0) {
        ITEMS_IN_WG--;
    }
    if (ITEMS_IN_WG <= 0) throw std::logic_error("Number of SGS per WG cannot be less than 1");

    assert(M % TM == 0);
    // TK depends on the datatype T
    constexpr size_t TK = 8 * std::min<size_t>(8, 32 / (8 * sizeof(T)));

    auto e = q.submit([&](handler &cgh) {
        cgh.depends_on(deps);

        cgh.parallel_for(sycl::nd_range<1>(M / TM, ITEMS_IN_WG), [=](nd_item<1> item) SYCL_ESIMD_KERNEL {
            slm_init<number_preload_weights * WIDTH * WIDTH * sizeof(T)>();

            auto weights_ptr_loc = weights_ptr + n_hidden_layers * WIDTH * WIDTH;
            int B_offset = 0;
            const int item_offset_A = item.get_global_linear_id() * WIDTH * TM;
            int layer_offset_A = n_hidden_layers * M * WIDTH + item_offset_A;

            for (int iter = 0; iter < std::min<int>(number_preload_weights, n_hidden_layers + 1); iter++) {
                helpers::moveToSlmWG<TK, TN, WIDTH>(item, weights_ptr_loc, B_offset + iter * WIDTH * WIDTH);
                weights_ptr_loc -= WIDTH * WIDTH; // next weight matrix
            }

            simd<T, TM * WIDTH> As;
            helpers::loadRow<TM, TK, WIDTH, cache_hint::uncached, cache_hint::uncached>(inputs_ptr + item_offset_A, As);

            // store backward activated input to the last intermediate output
            // note that output_activation == ReLU does not need any work since that means
            // forward >= 0
            if constexpr (output_activation != Activation::None && output_activation != Activation::ReLU) {
                // helpers::applyBackwardActivation<output_activation, TM, WIDTH>(
                //     sg, A_sg_start, forward_loc + layer_offset_A + M * WIDTH, A_sg_start);
            }

            // store activated slm in intermediate output
            helpers::storeRow<TM, TK, WIDTH, cache_hint::uncached, cache_hint::uncached>(As, intermediate_output +
                                                                                                 layer_offset_A);
            // Esnure B is loaded into slm
            item.barrier();
            simd<Tc, TM * WIDTH> Cs;
            // we are also doing output->last hidden layer
            for (int layer = n_hidden_layers; layer > 0; layer--) {
                layer_offset_A -= M * WIDTH;
                Cs = static_cast<Tc>(0);

                helpers::MAD<TM, TK, TN>(As, B_offset, Cs);
                B_offset += WIDTH * WIDTH;

                // load B for next iteration into SLM
                if (B_offset % (number_preload_weights * WIDTH * WIDTH) == 0) {
                    item.barrier();
                    B_offset = 0;
                    // load next weight matrix
                    for (int iter = 0; iter < std::min<int>(number_preload_weights, layer - 1); iter++) {
                        helpers::moveToSlmWG<TK, TN, WIDTH>(item, weights_ptr_loc, B_offset + iter * WIDTH * WIDTH);
                        weights_ptr_loc -= WIDTH * WIDTH;
                    }

                    item.barrier();
                }

                // TODO: Apply correct activation
                helpers::applyActivation<Activation::None, TM, TK, TN>(Cs, As);

                helpers::storeRow<TM, TK, WIDTH, cache_hint::uncached, cache_hint::uncached>(As, intermediate_output +
                                                                                                     layer_offset_A);
            }
        });
    });

    // NOTE: MKL gemm_batch is slower.
    std::vector<sycl::event> events(n_hidden_layers + 1);
    if constexpr (std::is_same<T, bf16>::value) { // need to cast to onemkls bf16 type.
        for (int iter = 0; iter < n_hidden_layers + 1; iter++) {
            events[iter] = oneapi::mkl::blas::row_major::gemm(
                q, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, WIDTH, WIDTH, M, 1.0f,
                reinterpret_cast<const oneapi::mkl::bfloat16 *>(forward) + iter * M * WIDTH, WIDTH,
                reinterpret_cast<oneapi::mkl::bfloat16 *>(intermediate_output) + iter * M * WIDTH, WIDTH, 1.0f,
                reinterpret_cast<oneapi::mkl::bfloat16 *>(output_ptr) + iter * WIDTH * WIDTH, WIDTH, {e});
        }
    } else {
        throw std::invalid_argument("Untested code path.");
        for (int iter = 0; iter < n_hidden_layers + 1; iter++) {
            events[iter] = oneapi::mkl::blas::row_major::gemm(
                q, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, WIDTH, WIDTH, M, 1.0,
                forward + iter * M * WIDTH, WIDTH, intermediate_output + iter * M * WIDTH, WIDTH, 1.0,
                output_ptr + iter * WIDTH * WIDTH, WIDTH, {e});
        }
    }
    return events;

    // return batchedGEMM_naive<T, Tc, INPUT_WIDTH, WIDTH, OUTPUT_WIDTH, TN>(q, output_ptr, forward,
    // intermediate_output,
    //                                                                       n_hidden_layers, M, {e});
}

} // namespace esimd
} // namespace kernels
} // namespace tinydpcppnn
