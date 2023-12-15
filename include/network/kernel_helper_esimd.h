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

#ifdef __SYCL_DEVICE_ONLY__
#define MY_INLINE inline
#define MY_STATIC static
#else
#define MY_INLINE
#define MY_STATIC
#endif

// using a block major layout in slm
template <int TK, int TN, int WIDTH, typename T>
MY_STATIC MY_INLINE void moveToSlmWG(const sycl::nd_item<1> &item, T const *const ptr, const int offset) {
    constexpr int vnni_factor = std::max<int>(1, 4 / sizeof(T));
    constexpr int elems_per_load = 128; // vnni_factor * TN; this has to be a multiple of TK

    // TODO: Use a block major format . Play a bit with block loads and float data type
    for (int iter = item.get_local_linear_id() * elems_per_load; iter < WIDTH * WIDTH;
         iter += item.get_local_range()[0] * elems_per_load) {
        simd<T, elems_per_load> tmp =
            lsc_block_load<T, elems_per_load, DSZ, cache_hint::uncached, cache_hint::cached>(ptr + iter);

        slm_block_store<T, elems_per_load>(sizeof(T) * (offset + iter), tmp);
    }
}

// in register everything is in block major format with blocks of size TMxTK
template <int TM, int TK, int WIDTH, cache_hint L1, cache_hint L3, int TMWIDTH, typename T>
MY_STATIC MY_INLINE void storeRow(simd<T, TMWIDTH> &src, T *const dest) {
// TODO: minimize the number of calls to the loads by enabling to store multiple rows at once
#pragma unroll
    for (int row = 0; row < TM; row++) {
        simd<T, WIDTH> tmp;
        for (int iter = 0; iter < WIDTH / TK; iter++) {
            tmp.template select<TK, 1>(iter * TK) = src.template select<TK, 1>(row * TK + iter * TM * TK);
        }
        lsc_block_store<T, WIDTH, DSZ, L1, L3>(dest + row * WIDTH, tmp);
    }
}

// in register everything is in block major format with blocks of size TMxTK
template <int TM, int TK, int WIDTH, cache_hint L1, cache_hint L3, int TMWIDTH, typename T>
MY_STATIC MY_INLINE void loadRow(T const *const src, simd<T, TMWIDTH> &dest) {
// TODO: minimize the number of calls to the loads by enabling to load multiple rows at once
#pragma unroll
    for (int row = 0; row < TM; row++) {
        simd<T, WIDTH> tmp = lsc_block_load<T, WIDTH, DSZ, L1, L3>(src + row * WIDTH);
        for (int iter = 0; iter < WIDTH / TK; iter++) {
            dest.template select<TK, 1>(row * TK + iter * TM * TK) = tmp.template select<TK, 1>(iter * TK);
        }
    }
}

// we are assuming a block major layout and vnni'd B
template <int TM, int TK, int TN, int TMWIDTH, typename Ta, typename Tc>
MY_STATIC MY_INLINE void MAD(simd<Ta, TMWIDTH> &As, const int B_offset, simd<Tc, TMWIDTH> &Cs) {

    constexpr int WIDTH = TMWIDTH / TM;
    constexpr int vnni_factor = std::max<int>(1, 4 / sizeof(Ta));
#pragma unroll
    for (int iterA = 0; iterA < TMWIDTH; iterA += TM * TK) {
        simd<Ta, TK * WIDTH> row_B =
            slm_block_load<Ta, TK * WIDTH>(sizeof(Ta) * (B_offset + iterA / TM * WIDTH), overaligned_tag<16>());
        auto row_B2d = row_B.template bit_cast_view<Ta, TK / vnni_factor, vnni_factor * WIDTH>();
#pragma unroll
        for (int iterB = 0; iterB < WIDTH; iterB += TN) {
            Cs.template select<TM * TN, 1>(iterB * TM) = xmx::dpas<8, TM, Tc>(
                simd<Tc, TM * TN>(Cs.template select<TM * TN, 1>(iterB * TM)),
                simd<Ta, TN * TK>(
                    row_B2d.template select<TK / vnni_factor, 1, vnni_factor * TN, 1>(0, vnni_factor * iterB)),
                simd<Ta, TM * TK>(As.template select<TM * TK, 1>(iterA)));
        }
    }
}

// we are assuming a block major layout and vnni'd B
template <int TM, int TK, int TN, int TMWIDTH, typename Ta, typename Tc>
MY_STATIC MY_INLINE void MAD(simd<Ta, TMWIDTH> &As, Ta const *const __restrict__ B, simd<Tc, TMWIDTH> &Cs) {

    constexpr int WIDTH = TMWIDTH / TM;
    constexpr int vnni_factor = std::max<int>(1, 4 / sizeof(Ta));
#pragma unroll
    for (int iterA = 0; iterA < TMWIDTH; iterA += TM * TK) {
        simd<Ta, TK * WIDTH> row_B;
        row_B.copy_from(B + iterA / TM * WIDTH);
        auto row_B2d = row_B.template bit_cast_view<Ta, TK / vnni_factor, vnni_factor * WIDTH>();
#pragma unroll
        for (int iterB = 0; iterB < WIDTH; iterB += TN) {
            Cs.template select<TM * TN, 1>(iterB * TM) = xmx::dpas<8, TM, Tc>(
                simd<Tc, TM * TN>(Cs.template select<TM * TN, 1>(iterB * TM)),
                simd<Ta, TN * TK>(
                    row_B2d.template select<TK / vnni_factor, 1, vnni_factor * TN, 1>(0, vnni_factor * iterB)),
                simd<Ta, TM * TK>(As.template select<TM * TK, 1>(iterA)));
        }
    }
}

template <Activation act, int TM, int TK, int TN, typename Tin, typename Tout, int N>
MY_STATIC MY_INLINE void applyActivation(simd<Tin, N> &Src, simd<Tout, N> &Dest) {
    static_assert(TK == TN); // otherwise we would need to reshuffle
    if constexpr (act == Activation::None)
        Dest = convert<Tout, Tin>(Src);
    else if constexpr (act == Activation::ReLU)
        Dest = convert<Tout, Tin>(max<Tin>(Src, simd<Tin, N>(static_cast<Tin>(0))));
}

} // namespace helpers
} // namespace esimd
} // namespace kernels
} // namespace tinydpcppnn