// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

// header file which implements functions commonly used in all the kernels, like
// loading, storing + specializations for different types
// multiplications patterns, etc.

#pragma once

#include <array>
#include <sycl/sycl.hpp>

#include "common.h"

namespace tinydpcppnn {
namespace kernels {
namespace helpers {

using namespace sycl::ext::oneapi::experimental::matrix;

// load a submatrix row-major piece of size MxN int SLM
template <int M, int N, typename Tsrc, typename Tdest, sycl::access::address_space AddressSpacesrc,
          sycl::access::decorated IsDecoratedsrc, sycl::access::address_space AddressSpacedest,
          sycl::access::decorated IsDecorateddest>
static inline void moveMemory(sycl::nd_item<1> &item, const sycl::multi_ptr<Tsrc, AddressSpacesrc, IsDecoratedsrc> &src,
                              sycl::multi_ptr<Tdest, AddressSpacedest, IsDecorateddest> dest) {

    for (int iter = item.get_local_linear_id(); iter < M * N; iter += item.get_local_range(0)) {
        dest[iter] = static_cast<Tdest>(src[iter]);
    }
}

// load a submatrix row-major piece of size MxN int SLM, sub-group by sub-group
template <int M, int N, typename Tsrc, typename Tdest, typename Group, sycl::access::address_space AddressSpacesrc,
          sycl::access::decorated IsDecoratedsrc, sycl::access::address_space AddressSpacedest,
          sycl::access::decorated IsDecorateddest>
static inline void moveMemorySG(Group sg, const sycl::multi_ptr<Tsrc, AddressSpacesrc, IsDecoratedsrc> &src,
                                sycl::multi_ptr<Tdest, AddressSpacedest, IsDecorateddest> dest) {

    for (int iter = sg.get_local_id()[0]; iter < M * N; iter += sg.get_local_range()[0]) {
        dest[iter] = static_cast<Tdest>(src[iter]);
    }
}

template <typename Group, typename T, use Use, size_t NumRows, size_t NumCols, layout Layout, size_t Nmats>
static inline void zeroMatrices(Group sg,
                                std::array<joint_matrix<Group, T, Use, NumRows, NumCols, Layout>, Nmats> &matrices) {
#pragma unroll
    for (auto &mat : matrices) {
        joint_matrix_fill(sg, mat, static_cast<T>(0));
    }
}

template <size_t K, typename Group, typename Ta, typename Tb, typename Tc, sycl::access::address_space AddressSpaceA,
          sycl::access::decorated IsDecoratedA, sycl::access::address_space AddressSpaceB,
          sycl::access::decorated IsDecoratedB, size_t M, size_t N, size_t nCs>
static inline void MAD_1_ROW(Group sg, const sycl::multi_ptr<Ta, AddressSpaceA, IsDecoratedA> &A,
                             const sycl::multi_ptr<Tb, AddressSpaceB, IsDecoratedB> &B,
                             std::array<joint_matrix<Group, Tc, use::accumulator, M, N>, nCs> &mCs) {

    // WIDTH = nCs*N
    //  A is not vnnied
    constexpr int WIDTH = nCs * N;
    joint_matrix<Group, Ta, use::a, M, K, layout::row_major> mA;
    joint_matrix<Group, Tb, use::b, K, N, layout::ext_intel_packed> mB;
    joint_matrix_load(sg, mA, A, WIDTH);
#pragma unroll
    for (int iter = 0; iter < nCs; iter++) {
        constexpr int vnni_factor = std::max<int>(1, 4 / sizeof(Tb));
        joint_matrix_load(sg, mB, B + iter * vnni_factor * N, vnni_factor * WIDTH);
        joint_matrix_mad(sg, mCs[iter], mA, mB, mCs[iter]);
    }
}

template <size_t K, typename Group, typename Ta, typename Tb, typename Tc, sycl::access::address_space AddressSpaceA,
          sycl::access::decorated IsDecoratedA, sycl::access::address_space AddressSpaceB,
          sycl::access::decorated IsDecoratedB, size_t M, size_t N, size_t nCs>
static inline void MAD(Group sg, const sycl::multi_ptr<Ta, AddressSpaceA, IsDecoratedA> &A,
                       const sycl::multi_ptr<Tb, AddressSpaceB, IsDecoratedB> &B,
                       std::array<joint_matrix<Group, Tc, use::accumulator, M, N>, nCs> &mCs) {

    // WIDTH = nCs*N
    for (int aiter = 0; aiter < nCs * N; aiter += K) {
        MAD_1_ROW<K>(sg, A + aiter, B + aiter * nCs * N, mCs);
    }
}

template <typename Tin, typename Tout, Activation act> inline void activate(const Tin &data_in, Tout &data_out) {
    if constexpr (act == Activation::None)
        data_out = static_cast<Tout>(data_in);
    else if constexpr (act == Activation::ReLU)
        data_out = static_cast<Tout>(std::max<Tin>(static_cast<Tin>(0), data_in));
}

template <typename Tin, typename Tdec, typename Tout, Activation act, sycl::access::address_space AddressSpace,
          sycl::access::decorated IsDecorated>
inline void activateBackward(const Tin &data_in, const sycl::multi_ptr<Tdec, AddressSpace, IsDecorated> data_decision,
                             Tout &data_out) {
    if constexpr (act == Activation::None)
        data_out = static_cast<Tout>(data_in);
    else if constexpr (act == Activation::ReLU)
        data_out = data_decision[0] > static_cast<Tout>(0) ? static_cast<Tout>(data_in) : static_cast<Tout>(0);
}

template <Activation act, typename Group, use Use, typename Tin, typename Tout, size_t NumRows, size_t NumCols,
          layout Layout, sycl::access::address_space AddressSpace, sycl::access::decorated IsDecorated, size_t nMats>
static inline void applyActivation(Group sg,
                                   std::array<joint_matrix<Group, Tin, Use, NumRows, NumCols, Layout>, nMats> &in,
                                   sycl::multi_ptr<Tout, AddressSpace, IsDecorated> dest) {

    // WIDTH = NumCols*nMats;
    for (auto matiter = 0; matiter < nMats; matiter++) {
        auto data_in = sycl::ext::oneapi::detail::get_wi_data(sg, in[matiter]);
        for (int rowiter = 0; rowiter < data_in.length(); rowiter++) {
            activate<Tin, Tout, act>(static_cast<Tin>(data_in[rowiter]),
                                     dest[rowiter * NumCols * nMats + matiter * NumCols + sg.get_local_id()[0]]);
        }
    }
}

template <Activation act, int M, int N, typename Group, typename Tin, typename Tout,
          sycl::access::address_space AddressSpacesrc, sycl::access::decorated IsDecoratedsrc,
          sycl::access::address_space AddressSpacedest, sycl::access::decorated IsDecorateddest>
static inline void applyActivation(Group sg, const sycl::multi_ptr<Tin, AddressSpacesrc, IsDecoratedsrc> &src,
                                   sycl::multi_ptr<Tout, AddressSpacedest, IsDecorateddest> dest) {

    for (int iter = sg.get_local_id()[0]; iter < M * N; iter += sg.get_local_range()[0]) {
        activate<Tin, Tout, act>(static_cast<Tin>(src[iter]), dest[iter]);
    }
}

template <Activation act, typename Group, use Use, typename Tin, typename Tdec, typename Tout, size_t NumRows,
          size_t NumCols, layout Layout, sycl::access::address_space AddressSpacedecision,
          sycl::access::decorated IsDecorateddecision, sycl::access::address_space AddressSpace,
          sycl::access::decorated IsDecorated, size_t nMats>
static inline void
applyBackwardActivation(Group sg, std::array<joint_matrix<Group, Tin, Use, NumRows, NumCols, Layout>, nMats> &in,
                        const sycl::multi_ptr<Tdec, AddressSpacedecision, IsDecorateddecision> decision_values,
                        sycl::multi_ptr<Tout, AddressSpace, IsDecorated> dest) {

    // WIDTH = NumCols*nMats;
    for (auto matiter = 0; matiter < nMats; matiter++) {
        auto data_in = sycl::ext::oneapi::detail::get_wi_data(sg, in[matiter]);
        for (int rowiter = 0; rowiter < data_in.length(); rowiter++) {
            const size_t offset = rowiter * NumCols * nMats + matiter * NumCols + sg.get_local_id()[0];
            activateBackward<Tin, Tdec, Tout, act>(static_cast<Tin>(data_in[rowiter]), decision_values + offset,
                                                   dest[offset]);
        }
    }
}

template <Activation act, int M, int N, typename Group, typename Tin, typename Tdec, typename Tout,
          sycl::access::address_space AddressSpacesrc, sycl::access::decorated IsDecoratedsrc,
          sycl::access::address_space AddressSpacedecision, sycl::access::decorated IsDecorateddecision,
          sycl::access::address_space AddressSpacedest, sycl::access::decorated IsDecorateddest>
static inline void
applyBackwardActivation(Group sg, const sycl::multi_ptr<Tin, AddressSpacesrc, IsDecoratedsrc> &src,
                        const sycl::multi_ptr<Tout, AddressSpacedecision, IsDecorateddecision> decision_values,
                        sycl::multi_ptr<Tout, AddressSpacedest, IsDecorateddest> dest) {

    for (int iter = sg.get_local_id()[0]; iter < M * N; iter += sg.get_local_range()[0]) {
        activateBackward<Tin, Tdec, Tout, act>(static_cast<Tin>(src[iter]), decision_values + iter, dest[iter]);
    }
}

} // namespace helpers
} // namespace kernels
} // namespace tinydpcppnn