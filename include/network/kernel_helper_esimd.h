// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

// header file which implements functions commonly used in all the kernels, like
// loading, storing + specializations for different types
// multiplications patterns, etc.

#pragma once

#include <array>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include "common.h"

namespace tinydpcppnn {
namespace kernels {
namespace esimd {
namespace helpers {

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;
#define L1_C cache_hint::cached
#define L3_C cache_hint::cached
#define L1_NC cache_hint::uncached
#define L3_NC cache_hint::uncached
#define DSZ lsc_data_size::default_size

// using a block major layout in slm
template <int TK, int TN, int WIDTH, typename T>
static inline void moveToSlmWG(const sycl::nd_item<1> &item, T const *const ptr, const int offset) {
    constexpr int vnni_factor = std::max<int>(1, 4 / sizeof(T));
    for (int iter = item.get_local_linear_id() * vnni_factor * TN; iter < WIDTH * WIDTH;
         iter += item.get_local_range()[0] * vnni_factor * TN) {
        slm_block_store<T, vnni_factor * TN>(
            sizeof(T) * (offset + iter),
            lsc_block_load<T, vnni_factor * TN, DSZ, cache_hint::streaming, cache_hint::cached>(ptr + iter));
    }
}

// in register everything is in block major format with blocks of size TMxTK
template <int TM, int TK, int WIDTH, int TMWIDTH, typename T>
static inline void storeRow(simd<T, TMWIDTH> &src, T *const dest) {

    for (int blockcol = 0; blockcol < WIDTH / TK; blockcol++) {
        for (int row = 0; row < TM; row++) {
            lsc_block_store<T, TK, DSZ, cache_hint::streaming, cache_hint::write_back>(
                dest + row * WIDTH + blockcol * TK, src.template select<TK, 1>(row * TK + blockcol * TM * TK));
        }
    }
}

// in register everything is in block major format with blocks of size TMxTK
template <int TM, int TK, int WIDTH, int TMWIDTH, typename T>
static inline void loadRow(T const *const src, simd<T, TMWIDTH> &dest) {
    for (int blockcol = 0; blockcol < WIDTH / TK; blockcol++) {
        for (int row = 0; row < TM; row++) {
            dest.template select<TK, 1>(row * TK + blockcol * TM * TK) =
                lsc_block_load<T, TK, DSZ, cache_hint::streaming, cache_hint::uncached>(src + row * WIDTH +
                                                                                        blockcol * TK);
        }
    }
}

// we are assuming a block major layout and vnni'd B
template <int TM, int TK, int TN, int TMWIDTH, typename Ta, typename Tc>
static inline void MAD(simd<Ta, TMWIDTH> &As, const int B_offset, simd<Tc, TMWIDTH> &Cs) {

    constexpr int WIDTH = TMWIDTH / TM;
    constexpr int vnni_factor = std::max<int>(1, 4 / sizeof(Ta));
    for (int iterA = 0; iterA < TMWIDTH; iterA += TM * TK) {
#pragma unroll
        for (int iterB = 0; iterB < WIDTH; iterB += TN) {
            simd<Ta, TK * TN> block_B = static_cast<Ta>(1.23f);
            for (int rowiter = 0; rowiter < TK / vnni_factor; rowiter++) {
                block_B.template select<vnni_factor * TN, 1>(rowiter * vnni_factor * TN) =
                    slm_block_load<Ta, vnni_factor * TN>(sizeof(Ta) * (B_offset + vnni_factor * WIDTH * rowiter +
                                                                       vnni_factor * iterB + iterA / TM * WIDTH));
            }

            Cs.template select<TM * TN, 1>(iterB * TM) =
                xmx::dpas<8, TM, Tc>(simd<Tc, TM * TN>(Cs.template select<TM * TN, 1>(iterB * TM)), block_B,
                                     simd<Ta, TM * TK>(As.template select<TM * TK, 1>(iterA)));
        }
    }
}

template <Activation act, int TM, int TK, int TN, typename Tin, typename Tout, int N>
static inline void applyActivation(simd<Tin, N> &Src, simd<Tout, N> &Dest) {
    static_assert(TK == TN); // otherwise we would need to reshuffle
    if constexpr (act == Activation::None)
        Dest = convert<Tout, Tin>(Src);
    else if constexpr (act == Activation::ReLU)
        Dest = convert<Tout, Tin>(max<Tin>(Src, simd<Tin, N>(static_cast<Tin>(0))));
}

// template <Activation act, int TM, int TK, int TN, typename Tin, typename Tout, int N>
// static inline void applyActivation(const simd<Tin, N> &Src, Tout *const Dest) {
//     constexpr int chacheline_elems = 64 / sizeof(Tout);
//     static_assert(N % chacheline_elems == 0);

//     for (int iter = 0; iter < N; iter += chacheline_elems) {
//         if constexpr (act == Activation::None)
//             lsc_block_store<Tout, chacheline_elems>(Dest + iter,
//                                                     convert<Tout, Tin>(Src.select<cacheline_elems, 1>(iter)));
//         else if constexpr (act == Activation::ReLU)
//             lsc_block_store<Tout, chacheline_elems>(
//                 Dest + iter, convert<Tout, Tin>(max(Src.select<cacheline_elems, 1>(iter), static_cast<Tin>(0))));
//     }
// }

} // namespace helpers
} // namespace esimd
} // namespace kernels
} // namespace tinydpcppnn