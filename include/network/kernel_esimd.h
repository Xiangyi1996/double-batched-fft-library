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
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>
#include <vector>

#include "common.h"
#include "oneapi/mkl.hpp"

namespace tinydpcppnn {
namespace kernels {
namespace esimd {

using namespace sycl::ext::intel::esimd;
using sycl::ext::intel::experimental::esimd::cache_hint;

namespace helpers {

using namespace sycl::ext::intel::experimental::esimd;
#define DSZ lsc_data_size::default_size

// #ifdef __SYCL_DEVICE_ONLY__
// #define MY_INLINE inline
// #define MY_STATIC static
// #else
#define MY_INLINE
#define MY_STATIC
// #endif

// using a block major layout in slm
template <int TK, int TN, int WIDTH, typename T>
MY_STATIC MY_INLINE void moveToSlmWG(const sycl::nd_item<1> &item, T const *const ptr, const int offset) {
    constexpr int vnni_factor = std::max<int>(1, 4 / sizeof(T));
    constexpr int nblock_total = WIDTH * WIDTH / (TK * TN);

    for (int blockiter = item.get_local_linear_id(); blockiter < nblock_total; blockiter += item.get_local_range()[0]) {

        const int block_row = blockiter / (WIDTH / TN);
        const int block_col = blockiter % (WIDTH / TN);
        config_2d_mem_access<float, TN, TK / vnni_factor, 1> my_config(
            reinterpret_cast<float const *>(ptr), vnni_factor * WIDTH * sizeof(T) - 1, WIDTH / vnni_factor - 1,
            vnni_factor * WIDTH * sizeof(T) - 1, block_col * TN, block_row * TK / vnni_factor);

        // this loads one block in T
        simd<float, TK / vnni_factor * TN> tmp =
            lsc_load_2d<float, TN, TK / vnni_factor, 1, false, false, cache_hint::cached, cache_hint::cached>(
                my_config);

        slm_block_store<float, TN * TK / vnni_factor>(
            sizeof(float) * (blockiter * TK * TN / vnni_factor) + sizeof(T) * offset, tmp, overaligned_tag<16>());
    }
}

// in register everything is in block major format with blocks of size TMxTK
template <int TM, int TK, int WIDTH, cache_hint L1, cache_hint L3, int TMWIDTH, typename T>
MY_STATIC MY_INLINE void storeRow(simd<T, TMWIDTH> &src, T *const dest) {
    // TODO: minimize the number of calls to the loads by enabling to store multiple rows at once

    constexpr int rows_per_load = std::min<int>(512 / (WIDTH * sizeof(T)), TM);
#pragma unroll
    for (int row = 0; row < TM; row += rows_per_load) {
        simd<T, WIDTH * rows_per_load> tmp;
#pragma collapse 2 unroll
        for (int locrowiter = 0; locrowiter < rows_per_load; locrowiter++) {
            for (int iter = 0; iter < WIDTH / TK; iter++) {
                tmp.template select<TK, 1>(locrowiter * WIDTH + iter * TK) =
                    src.template select<TK, 1>((row + locrowiter) * TK + iter * TM * TK);
            }
        }
        lsc_block_store<T, rows_per_load * WIDTH, DSZ, L1, L3>(dest + row * WIDTH, tmp, overaligned_tag<8>());
    }
}

// // in register everything is in block major format with blocks of size TMxTK
// template <int TM, int TK, int WIDTH, cache_hint L1, cache_hint L3, int TMWIDTH, typename T>
// MY_STATIC MY_INLINE void storeRow(simd<T, TMWIDTH> &src, T *const dest) {
//     // TODO: minimize the number of calls to the loads by enabling to store multiple rows at once

//     constexpr int nblocks = WIDTH / TK;
//     constexpr int blocks_per_load = std::max<int>(1, 4 / sizeof(T));

// #pragma unroll
//     for (int blockiter = 0; blockiter < nblocks; blockiter++) {
//         config_2d_mem_access<float, TK / blocks_per_load, TM, 1> my_config_store(
//             reinterpret_cast<const float *>(dest), WIDTH * sizeof(T) - 1, TM - 1, WIDTH * sizeof(T) - 1,
//             blockiter * TK / blocks_per_load, 0);
//         simd<float, TK / blocks_per_load * TM> tmp =
//             src.template bit_cast_view<float>().template select<TK / blocks_per_load * TM, 1>(blockiter * TK /
//                                                                                               blocks_per_load * TM);
//         lsc_store_2d<float, TK / blocks_per_load, TM, 1, L1, L3>(my_config_store, tmp);
//     }
// }
// template <int TM, int TK, int WIDTH, cache_hint L1, cache_hint L3, int TMWIDTH, typename T>
// MY_STATIC MY_INLINE void storeRow(simd<T, TMWIDTH> &src, T *const dest) {
//     // TODO: minimize the number of calls to the loads by enabling to store multiple rows at once

//     constexpr int nblocks = WIDTH / TK;
//     constexpr int blocks_per_load = std::max<int>(1, 4 / sizeof(T));

// #pragma unroll
//     for (int blockiter = 0; blockiter < nblocks; blockiter += blocks_per_load) {
//         config_2d_mem_access<float, TK, TM, 1> my_config_store(reinterpret_cast<const float *>(dest),
//                                                                WIDTH * sizeof(T) - 1, TM - 1, WIDTH * sizeof(T) - 1,
//                                                                blockiter * TK / blocks_per_load, 0);
//         //Attention: this is wrong. Just to check perf.
//         simd<float, TK * TM> tmp =
//             src.template bit_cast_view<float>().template select<TK * TM, 1>(blockiter * TK / blocks_per_load * TM);
//         lsc_store_2d<float, TK, TM, 1, L1, L3>(my_config_store, tmp);
//     }
// }

// in register everything is in block major format with blocks of size TMxTK
// template <int TM, int TK, int WIDTH, cache_hint L1, cache_hint L3, int TMWIDTH, typename T>
// MY_STATIC MY_INLINE void loadRow(T const *const src, simd<T, TMWIDTH> &dest) {
//     constexpr int rows_per_load = std::min<int>(512 / (WIDTH * sizeof(T)), TM);
// #pragma unroll
//     for (int row = 0; row < TM; row += rows_per_load) {
//         simd<T, WIDTH * rows_per_load> tmp =
//             lsc_block_load<T, WIDTH * rows_per_load, DSZ, L1, L3>(src + row * WIDTH, overaligned_tag<16>());
// #pragma collapse 2 unroll
//         for (int locrowiter = 0; locrowiter < rows_per_load; locrowiter++) {
//             for (int iter = 0; iter < WIDTH / TK; iter++) {
//                 dest.template select<TK, 1>((row + locrowiter) * TK + iter * TM * TK) =
//                     tmp.template select<TK, 1>(iter * TK);
//             }
//         }
//     }
// }

// in register everything is in block major format with blocks of size TMxTK
template <int TM, int TK, int WIDTH, cache_hint L1, cache_hint L3, int TMWIDTH, typename T>
MY_STATIC MY_INLINE void loadRow(T const *const src, simd<T, TMWIDTH> &dest) {

    constexpr int blocks_per_load = std::max<int>(1, 4 / sizeof(T));
    constexpr int nloads = WIDTH / (TK * blocks_per_load);
#pragma unroll
    for (int load_iter = 0; load_iter < nloads; load_iter++) {
        config_2d_mem_access<float, TK / blocks_per_load, TM, blocks_per_load> my_config(
            reinterpret_cast<float const *>(src), WIDTH * sizeof(T) - 1, TM - 1, WIDTH * sizeof(T) - 1, load_iter * TK,
            0);

        simd<float, TK * TM> tmp =
            lsc_load_2d<float, TK / blocks_per_load, TM, blocks_per_load, false, false, L1, L3>(my_config);
        dest.template select<TM * TK * blocks_per_load, 1>(TM * TK * blocks_per_load * load_iter) =
            tmp.template bit_cast_view<T>().template select<TM * TK * blocks_per_load, 1>(0);
    }
}

template <int TM, int TK, int WIDTH, cache_hint L1, cache_hint L3, typename T>
MY_STATIC MY_INLINE void prefetchRow(T const *const src) {
    constexpr int vnni_factor = std::max<int>(1, 4 / sizeof(T));
#pragma unroll
    for (int iter = 0; iter < WIDTH; iter += TK) {
        config_2d_mem_access<float, TK, TM / vnni_factor, 1> my_config(
            reinterpret_cast<float const *>(src), vnni_factor * WIDTH * sizeof(T) - 1, WIDTH / vnni_factor - 1,
            vnni_factor * WIDTH * sizeof(T) - 1, iter, 0);
        lsc_prefetch_2d<float, TK, TM / vnni_factor, 1, false, false, cache_hint::cached, cache_hint::cached>(
            my_config);
    }
}

// we are assuming a block major layout and vnni'd B
template <int TM, int TK, int TN, int TMWIDTH, typename Ta, typename Tc>
MY_STATIC MY_INLINE void MAD(simd<Ta, TMWIDTH> &As, const int B_offset, simd<Tc, TMWIDTH> &Cs) {

    constexpr int WIDTH = TMWIDTH / TM;
#pragma collapse 2 unroll
    for (int iterA = 0; iterA < TMWIDTH; iterA += TM * TK) {
        for (int iterB = 0; iterB < WIDTH; iterB += TN) {
            simd<Ta, TK * TN> row_B =
                slm_block_load<Ta, TK * TN>(sizeof(Ta) * (B_offset + iterA / TM * WIDTH + iterB * TK));
            Cs.template select<TM * TN, 1>(iterB * TM) =
                xmx::dpas<8, TM, Tc>(simd<Tc, TM * TN>(Cs.template select<TM * TN, 1>(iterB * TM)), row_B,
                                     simd<Ta, TM * TK>(As.template select<TM * TK, 1>(iterA)));
        }
    }
}
template <int TM, int TK, int TN, int TMWIDTH, typename Ta, typename Tc>
MY_STATIC MY_INLINE void MAD(simd<Ta, TMWIDTH> &As, Ta const *const __restrict__ B, simd<Tc, TMWIDTH> &Cs) {

    constexpr int WIDTH = TMWIDTH / TM;
    constexpr int vnni_factor = std::max<int>(1, 4 / sizeof(Ta));
#pragma collapse 2 unroll
    for (int iterA = 0; iterA < TMWIDTH; iterA += TM * TK) {
        for (int iterB = 0; iterB < WIDTH; iterB += TN) {
            config_2d_mem_access<float, TN, TK / vnni_factor, 1> my_config(
                reinterpret_cast<float const *>(B), vnni_factor * WIDTH * sizeof(Ta) - 1, WIDTH / vnni_factor - 1,
                vnni_factor * WIDTH * sizeof(Ta) - 1, iterB, iterA / TM / vnni_factor);
            simd<float, TK / vnni_factor * TN> row_B =
                lsc_load_2d<float, TN, TK / vnni_factor, 1, false, false, cache_hint::cached, cache_hint::cached>(
                    my_config);

            Cs.template select<TM * TN, 1>(iterB * TM) = xmx::dpas<8, TM, Tc>(
                simd<Tc, TM * TN>(Cs.template select<TM * TN, 1>(iterB * TM)),
                simd<Ta, TN * TK>(row_B.template bit_cast_view<Ta>().template select<TK * TN, 1>(0)),
                simd<Ta, TM * TK>(As.template select<TM * TK, 1>(iterA)));
        }
    }
}
// template <int TM, int TK, int TN, int TMWIDTH, typename Ta, typename Tc>
// MY_STATIC MY_INLINE void MAD(simd<Ta, TMWIDTH> &As, const int B_offset, simd<Tc, TMWIDTH> &Cs) {

//     constexpr int WIDTH = TMWIDTH / TM;
// #pragma collapse 2 unroll
//     for (int iterB = 0; iterB < WIDTH; iterB += TN) {
//         for (int iterA = 0; iterA < TMWIDTH; iterA += TM * TK) {

//             simd<Ta, TK * TN> row_B =
//                 slm_block_load<Ta, TK * TN>(sizeof(Ta) * (B_offset + iterA / TM * WIDTH + iterB * TK));
//             Cs.template select<TM * TN, 1>(iterB * TM) =
//                 xmx::dpas<8, TM, Tc>(simd<Tc, TM * TN>(Cs.template select<TM * TN, 1>(iterB * TM)), row_B,
//                                      simd<Ta, TM * TK>(As.template select<TM * TK, 1>(iterA)));
//         }
//     }
// }

template <Activation act, int TM, int TK, int TN, typename Tin, typename Tout, int N>
MY_STATIC MY_INLINE void applyActivation(simd<Tin, N> &Src, simd<Tout, N> &Dest) {
    static_assert(TK == TN); // otherwise we would need to reshuffle
    if constexpr (act == Activation::None)
        Dest = convert<Tout, Tin>(Src);
    else if constexpr (act == Activation::ReLU)
        Dest = convert<Tout, Tin>(max<Tin>(Src, simd<Tin, N>(static_cast<Tin>(0))));
}

} // namespace helpers

// template <typename T, typename Tc, int INPUT_WIDTH, int WIDTH, int OUTPUT_WIDTH, size_t TN>
// inline std::vector<sycl::event>
// batchedGEMM_naive(sycl::queue &q, T *const __restrict__ output_ptr, T const *const __restrict__ intermediate_forward,
//                   T const *const __restrict__ intermediate_backward, const int n_hidden_layers, const int M,
//                   const std::vector<sycl::event> &deps) {
//     constexpr int SG_SIZE = TN;
//     auto e = q.submit([&](sycl::handler &cgh) {
//         cgh.depends_on(deps);

//         cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(n_hidden_layers + 1, WIDTH * WIDTH),
//                                            sycl::range<2>(1, std::min(1024, WIDTH * WIDTH))),
//                          [=](sycl::nd_item<2> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
//                              const int matrix = item.get_global_id(0);
//                              const int element = item.get_global_id(1);
//                              const int row = element / WIDTH;
//                              const int col = element % WIDTH;

//                              Tc tmp_out = static_cast<Tc>(0);
//                              T const *intermediate_forward_loc = intermediate_forward + matrix * M * WIDTH + row;
//                              T const *intermediate_backward_loc = intermediate_backward + matrix * M * WIDTH + col;
//                              for (int i = 0; i < M; i++) {
//                                  tmp_out += static_cast<Tc>(*intermediate_forward_loc) *
//                                             static_cast<Tc>(*intermediate_backward_loc);
//                                  intermediate_forward_loc += WIDTH;
//                                  intermediate_backward_loc += WIDTH;
//                              }
//                              T *const output_ptr_loc = output_ptr + WIDTH * WIDTH * matrix + element;
//                              *output_ptr_loc = static_cast<T>(tmp_out);
//                          });
//     });
//     // auto e =
//     //     q.parallel_for((n_hidden_layers + 1) * WIDTH * WIDTH, [=](auto item)
//     [[intel::reqd_sub_group_size(SG_SIZE)]]
//     //     {
//     //         output_ptr[item.get_id()] = static_cast<T>(1.23);
//     //     });

//     return {e};
// }

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
            int layer_offset_A = item.get_global_linear_id() * WIDTH * TM;

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
            for (int layer = 0; layer < n_hidden_layers; layer++) {
                // reset result matrices
                Cs = static_cast<Tc>(0);

                helpers::MAD<TM, TK, TN>(As, weights_ptr + layer * WIDTH * WIDTH, Cs);

                // activate and save
                helpers::applyActivation<activation, TM, TK, TN>(Cs, As);

                if constexpr (!INFERENCE)
                    helpers::storeRow<TM, TK, WIDTH, cache_hint::uncached, cache_hint::uncached>(
                        As, intermediate_output + layer_offset_A);

                layer_offset_A += M * WIDTH;
            }

            // generate output, i.e. last GEMM
            Cs = static_cast<Tc>(0);

            // helpers::MAD<TM, TK, TN>(As, B_offset, Cs);
            helpers::MAD<TM, TK, TN>(As, weights_ptr + n_hidden_layers * WIDTH * WIDTH, Cs);

            // activate and save to slm
            helpers::applyActivation<output_activation, TM, TK, TN>(Cs, As);

            // save slm to HBM
            helpers::storeRow<TM, TK, WIDTH, cache_hint::uncached, cache_hint::write_back>(As, intermediate_output +
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
            const int item_offset_A = item.get_global_linear_id() * WIDTH * TM;
            int layer_offset_A = n_hidden_layers * M * WIDTH + item_offset_A;

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
            simd<Tc, TM * WIDTH> Cs;
            // we are also doing output->last hidden layer
            for (int layer = n_hidden_layers; layer > 0; layer--) {
                layer_offset_A -= M * WIDTH;
                Cs = static_cast<Tc>(0);

                helpers::MAD<TM, TK, TN>(As, weights_ptr + layer * WIDTH * WIDTH, Cs);

                // TODO: Apply correct activation
                helpers::applyActivation<Activation::None, TM, TK, TN>(Cs, As);

                helpers::storeRow<TM, TK, WIDTH, cache_hint::uncached, cache_hint::uncached>(As, intermediate_output +
                                                                                                     layer_offset_A);
            }
        });
    });

    // NOTE: MKL gemm_batch is slower.
    std::vector<sycl::event> events(n_hidden_layers + 1);
    if constexpr (std::is_same<T, sycl::ext::oneapi::bfloat16>::value) { // need to cast to onemkls bf16 type.
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
