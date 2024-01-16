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

#include "DeviceMatrix.h"
#include "common.h"
#include "oneapi/mkl.hpp"

namespace sycl::ext::intel::esimd::xmx {
template <int SystolicDepth, int RepeatCount, typename T, typename CT, typename BT, typename AT,
          dpas_argument_type BPrecision = detail::dpas_precision_from_type<BT>(),
          dpas_argument_type APrecision = detail::dpas_precision_from_type<AT>(), int N, int N_orig, int BN, int AN,
          int AN_orig>
__ESIMD_NS::simd<T, N> dpas(__ESIMD_NS::simd_view<simd<CT, N_orig>, region1d_t<CT, N, 1>> C, __ESIMD_NS::simd<BT, BN> B,
                            __ESIMD_NS::simd_view<simd<AT, AN_orig>, region1d_t<AT, AN, 1>> A) {
    (void)detail::verify_parameters_and_deduce_exec_size<SystolicDepth, RepeatCount, T, CT, BT, AT, BPrecision,
                                                         APrecision, BN, AN>();

    using MsgT = int;
    constexpr int ANCasted = AN * sizeof(AT) / sizeof(MsgT);
    constexpr int BNCasted = BN * sizeof(BT) / sizeof(MsgT);
    __ESIMD_NS::simd<MsgT, ANCasted> ACasted = A.template bit_cast_view<MsgT>();
    __ESIMD_NS::simd<MsgT, BNCasted> BCasted = B.template bit_cast_view<MsgT>();
    using CRawT = typename __ESIMD_NS::simd<CT, N>::raw_element_type;
    using RawT = typename __ESIMD_NS::simd<T, N>::raw_element_type;
    return __esimd_dpas2<BPrecision, APrecision, SystolicDepth, RepeatCount, RawT, CRawT, MsgT, MsgT, N, BNCasted,
                         ANCasted>(C.data(), BCasted.data(), ACasted.data());
}
}; // namespace sycl::ext::intel::esimd::xmx

namespace tinydpcppnn {
namespace kernels {
namespace esimd {

using namespace sycl::ext::intel::esimd;
using sycl::ext::intel::experimental::esimd::cache_hint;

namespace helpers {

using namespace sycl::ext::intel::experimental::esimd;
#define DSZ lsc_data_size::default_size
using bf16 = sycl::ext::oneapi::bfloat16;

// #ifdef __SYCL_DEVICE_ONLY__
// #define MY_INLINE inline
// #define MY_STATIC static
// #else
#define MY_INLINE
#define MY_STATIC
// #endif

template <typename T> struct XMXCType {
    typedef T CType;
};
template <> struct XMXCType<bf16> {
    typedef float CType;
};
template <> struct XMXCType<sycl::half> {
    typedef sycl::half CType;
};

// in register everything is in block major format with blocks of size TMxTK
template <int TM, int TK, int WIDTH, cache_hint L1, cache_hint L3, int TMWIDTH, typename T>
SYCL_ESIMD_FUNCTION MY_STATIC MY_INLINE void storeRow(simd<T, TMWIDTH> &src, T *const dest) {

    static_assert(TM >= 1 && TM <= 8);
    static_assert(WIDTH % TK == 0);
    static_assert(TMWIDTH == TM * WIDTH);
    static_assert(sizeof(T) <= 4);

    constexpr int rows_per_load = std::min<int>(512 / (WIDTH * sizeof(T)), TM);
    auto src_2d = src.template bit_cast_view<T, TMWIDTH / TK, TK>(); // block major

#pragma unroll
    for (int row = 0; row < TM; row += rows_per_load) {
        simd<T, WIDTH * rows_per_load> tmp;
#pragma unroll
        for (int locrowiter = 0; locrowiter < rows_per_load; locrowiter++) {
            tmp.template select<WIDTH, 1>(locrowiter * WIDTH) =
                src_2d.template select<WIDTH / TK, TM, TK, 1>(row + locrowiter, 0);
        }
        lsc_block_store<T, rows_per_load * WIDTH, DSZ, L1, L3>(dest + row * WIDTH, tmp, overaligned_tag<8>());
    }
}

// in register everything is in block major format with blocks of size TMxTK
// template <int TM, int TK, int WIDTH, cache_hint L1, cache_hint L3, int TMWIDTH, typename T>
// SYCL_ESIMD_FUNCTION MY_STATIC MY_INLINE void storeRow(simd<T, TMWIDTH> &src, T *const dest) {
//     constexpr int nblocks = WIDTH / TK;
//     auto src_int = src.template bit_cast_view<int16_t>();
// #pragma unroll
//     for (int blockiter = 0; blockiter < nblocks; blockiter++) {
//         lsc_store_2d<int16_t, TK, TM, L1, L3>(
//             reinterpret_cast<int16_t *const>(dest), WIDTH * sizeof(T) - 1, TM - 1, WIDTH * sizeof(T) - 1,
//             blockiter * TK, 0, simd<int16_t, TK * TM>(src_int.template select<TM * TK, 1>(blockiter * TM * TK)));
//     }
// }

// in register everything is in block major format with blocks of size TMxTK
template <int TM, int TK, int WIDTH, cache_hint L1, cache_hint L3, int TMWIDTH, typename T>
SYCL_ESIMD_FUNCTION MY_STATIC MY_INLINE void loadRow(T const *const src, simd<T, TMWIDTH> &dest) {
    static_assert(TM >= 1 && TM <= 8);
    static_assert(WIDTH % TK == 0);
    static_assert(TMWIDTH == TM * WIDTH);
    static_assert(sizeof(T) <= 4);
    constexpr int elems_per_pos = 4 / sizeof(T);
    constexpr int blocks_per_load = TK * elems_per_pos > WIDTH ? 1 : elems_per_pos;
    constexpr int nloads = WIDTH / (TK * blocks_per_load);
    static_assert(nloads > 0);

    auto dest_int = dest.template bit_cast_view<int32_t>();
#pragma unroll
    for (int load_iter = 0; load_iter < nloads; load_iter++) {
        dest_int.template select<TM * TK / elems_per_pos * blocks_per_load, 1>(TM * TK / elems_per_pos *
                                                                               blocks_per_load * load_iter) =
            lsc_load_2d<int32_t, TK / elems_per_pos, TM, blocks_per_load, false, false, L1, L3>(
                reinterpret_cast<int32_t const *>(src), WIDTH * sizeof(T) - 1, TM - 1, WIDTH * sizeof(T) - 1,
                load_iter * TK, 0);
    }
}

// we are assuming a block major layout and vnni'd B
template <int TM, int TK, int TN, int TMWIDTH, typename Ta, typename Tc>
SYCL_ESIMD_FUNCTION MY_STATIC MY_INLINE void MAD(simd<Ta, TMWIDTH> &As, Ta const *const __restrict__ B,
                                                 simd<Tc, TMWIDTH> &Cs) {
    static_assert(TM >= 1 && TM <= 8);
    static_assert(TN == 16 || TN == 8);
    static_assert(TMWIDTH % TM == 0);
    constexpr int WIDTH = TMWIDTH / TM;
    static_assert(WIDTH % TK == 0 && WIDTH % TN == 0);
    static_assert(sizeof(Ta) <= 4 && sizeof(Tc) <= 4);

    constexpr int vnni_factor = std::max<int>(1, 4 / sizeof(Ta));
#pragma collapse 2 unroll
    for (int iterA = 0; iterA < TMWIDTH; iterA += TM * TK) {
        for (int iterB = 0; iterB < WIDTH; iterB += TN) {
            simd<Ta, TK * TN> BlockB;
            auto BlockB_float = BlockB.template bit_cast_view<float>();
            BlockB_float =
                lsc_load_2d<float, TN, TK / vnni_factor, 1, false, false, cache_hint::cached, cache_hint::cached>(
                    reinterpret_cast<float const *>(B), vnni_factor * WIDTH * sizeof(Ta) - 1, WIDTH / vnni_factor - 1,
                    vnni_factor * WIDTH * sizeof(Ta) - 1, iterB, iterA / TM / vnni_factor);

            Cs.template select<TM * TN, 1>(iterB * TM) = xmx::dpas<8, TM, Tc>(
                Cs.template select<TM * TN, 1>(iterB * TM), BlockB, As.template select<TM * TK, 1>(iterA));
        }
    }
}

template <Activation act, int TM, int TK, int TN, typename Tin, typename Tout, int N>
SYCL_ESIMD_FUNCTION MY_STATIC MY_INLINE void applyActivation(simd<Tin, N> &Src, simd<Tout, N> &Dest) {
    static_assert(TM >= 1 && TM <= 8);
    static_assert(TN == 16 || TN == 8);
    static_assert(TK == TN); // otherwise we would need to reshuffle due to block major format

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
std::vector<sycl::event> forward_impl_general(sycl::queue &q,
                                              DeviceMatricesView<T> weights /*T const *const __restrict__ weights_ptr*/,
                                              /*T const *const __restrict__*/ const DeviceMatrixView<T> &input,
                                              T *const __restrict__ intermediate_output, const int n_hidden_layers,
                                              const std::vector<sycl::event> &deps) {

    // throw std::logic_error("General function should not be called.");
    const size_t M = input.m();
    static_assert(INPUT_WIDTH == WIDTH);
    static_assert(OUTPUT_WIDTH == WIDTH);
    static_assert(WIDTH % TN == 0);

    constexpr int TM = 8;
    // make sure there is no remainder and no out of bounds accesses
    // this may be adjusted in the future
    assert(M % TM == 0);
    // TK depends on the datatype T
    constexpr int TK = 8 * std::min<int>(8, 32 / (8 * sizeof(T)));

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
            const int layer_offset_A = item.get_global_linear_id() * WIDTH * TM;

            // we store blocks contiguously
            simd<T, TM * WIDTH> As;
            helpers::loadRow<TM, TK, WIDTH, cache_hint::uncached, cache_hint::uncached>(
                input.GetPointer(TM * item.get_global_linear_id(), 0) /* + layer_offset_A*/, As);

            // if not inference activate and store in intermediate output
            if constexpr (!INFERENCE) {
                simd<T, TM * WIDTH> tmpA;
                helpers::applyActivation<activation, TM, TK, TN>(As, tmpA);
                helpers::storeRow<TM, TK, WIDTH, cache_hint::uncached, cache_hint::uncached>(tmpA, intermediate_output +
                                                                                                       layer_offset_A);
            }

            simd<Tc, TM * WIDTH> Cs;
            for (int layer = 0; layer < n_hidden_layers; layer++) {
                // reset result matrices
                Cs = static_cast<Tc>(0);

                helpers::MAD<TM, TK, TN>(As, weights.GetMatrixPointer(layer) /*weights_ptr + layer * WIDTH * WIDTH*/,
                                         Cs);

                // activate and save
                helpers::applyActivation<activation, TM, TK, TN>(Cs, As);

                if constexpr (!INFERENCE)
                    helpers::storeRow<TM, TK, WIDTH, cache_hint::uncached, cache_hint::uncached>(
                        As, intermediate_output + (layer + 1) * M * WIDTH + layer_offset_A);
            }

            // generate output, i.e. last GEMM
            Cs = static_cast<Tc>(0);

            // helpers::MAD<TM, TK, TN>(As, B_offset, Cs);
            helpers::MAD<TM, TK, TN>(
                As, weights.GetMatrixPointer(n_hidden_layers) /*+ n_hidden_layers * WIDTH * WIDTH*/, Cs);

            // activate and save to slm
            helpers::applyActivation<output_activation, TM, TK, TN>(Cs, As);

            // save slm to HBM
            if constexpr (!INFERENCE)
                helpers::storeRow<TM, TK, WIDTH, cache_hint::uncached, cache_hint::write_back>(
                    As, intermediate_output + (n_hidden_layers + 1) * M * WIDTH + layer_offset_A);
            else if constexpr (INFERENCE) // storing at the beginning since no intermediate results
                helpers::storeRow<TM, TK, WIDTH, cache_hint::uncached, cache_hint::write_back>(As, intermediate_output +
                                                                                                       layer_offset_A);
        });
    });

    return {e};
}

template <typename T, typename Tc, int INPUT_WIDTH, int WIDTH, int OUTPUT_WIDTH, Activation activation,
          Activation output_activation, size_t TN>
std::vector<sycl::event> backward_impl_general(sycl::queue &q, T const *const __restrict__ weights_ptr,
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

    auto e = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(deps);

        cgh.parallel_for(sycl::nd_range<1>(M / TM, ITEMS_IN_WG), [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
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

                // TODO: Apply correct backward activation
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
    } else if constexpr (!std::is_same<T, sycl::ext::oneapi::bfloat16>::value) {
        // throw std::invalid_argument("Untested code path.");
        for (int iter = 0; iter < n_hidden_layers + 1; iter++) {
            events[iter] = oneapi::mkl::blas::row_major::gemm(
                q, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, WIDTH, WIDTH, M, 1.0,
                forward + iter * M * WIDTH, WIDTH, intermediate_output + iter * M * WIDTH, WIDTH, 1.0,
                output_ptr + iter * WIDTH * WIDTH, WIDTH, {e});
        }
    }
    return events;
}

} // namespace esimd
} // namespace kernels
} // namespace tinydpcppnn
