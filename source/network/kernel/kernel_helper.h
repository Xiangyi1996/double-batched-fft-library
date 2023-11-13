// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

// header file which implements functions commonly used in all the kernels, like
// loading, storing + specializations for different types
// multiplications patterns, etc.

#pragma once

#include <SYCL/sycl.hpp>
#include <vector>

#include "common.h"

namespace tinydpcppnn {
namespace kernels {
namespace helpers {

using namespace tinydpcppnn::builtin;
using namespace sycl::ext::oneapi::experimental::matrix;

// load a submatrix row-major piece of size MxN int SLM
template <int M, int N, typename T, access::address_space AddressSpacesrc, access::decorated IsDecoratedsrc,
          access::address_space AddressSpacedest, access::decorated IsDecorateddest>
static inline void moveMemory(sycl::nd_item<1> &item, const multi_ptr<T, AddressSpacesrc, IsDecoratedsrc> &src,
                              multi_ptr<T, AddressSpacedest, IsDecorateddest> &dest) {

    for (int iter = item.get_local_linear_id(); iter < M * N; iter += item.get_local_linear_range()) {
        dest[iter] = src[iter];
    }
}

// load a submatrix row-major piece of size MxN int SLM, sub-group by sub-group
template <int M, int N, typename T, typename Group, access::address_space AddressSpacesrc,
          access::decorated IsDecoratedsrc, access::address_space AddressSpacedest, access::decorated IsDecorateddest>
static inline void moveMemorySG(Group sg, const multi_ptr<T, AddressSpacesrc, IsDecoratedsrc> &src,
                                multi_ptr<T, AddressSpacedest, IsDecorateddest> &dest) {

    for (int iter = sg.get_local_id()[0]; iter < M * N; iter += sg.get_local_range()) {
        dest[iter] = src[iter];
    }
}

template <ypename Group, typename T, size_t NumRows, size_t NumCols, matrix_layout Layout, typename... Args>
static inline void zeroMatrices(Group sg, joint_matrix<T, NumRows, NumCols, Layout, Group> &matrix, Args... args) {
    joint_matrix_fill(sg, matrix, static_cast<T>(0));
    zeroMatrices(sg, args...);
}

template <int WIDTH, typename Group, typename Ta, typename Tb, typename Tc, std::size_t M, std::size_t K, std::size_t N,
          layout LayoutA, typename... AllC>
static inline void
MAD_1_ROW(Group sg, const joint_matrix<Group, Ta, use::a, M, K, layout::row_major> &mA, const local_ptr<Tb> &B,
          joint_matrix<Group, Tc, use::accumulator, M, N, sycl::ext::oneapi::experimental::matrix::layout::dynamic> &mC,
          AllC &...Cs) {

    constexpr int vnni_factor = std::max<int>(4 / sizeof(Tb), 1); // how many rows are consecutive elements
    joint_matrix<Group, Tb, use::b, K, N, sycl::ext::intel::experimental::matrix::layout::packed> mB;
    joint_matrix_load(
        sg, mB, B, vnni_factor * WIDTH); // B is vnnied, which only has on impact if Tb is a datatype sized less than 4
    MAD_1_ROW(sg, mA, B + vnni_factor * N,
              Cs...); // increment B to get next block col and call recursively on remaining C's
}

template <int WIDTH, typename Group, typename Ta, typename Tb, typename Tc, std::size_t M, std::size_t K, std::size_t N,
          typename... AllC>
static inline void
MAD_1_ROW(Group sg, const local_ptr<Ta> &A, const local_ptr<Tb> &B,
          joint_matrix<Group, Tc, use::accumulator, M, N, sycl::ext::oneapi::experimental::matrix::layout::dynamic> &mC,
          AllC &...Cs) {

    // First load the matrix A from slm, then call the jount matrix based version above
    joint_matrix<Group, Ta, use::a, M, K, layout::row_major> mA;
    // A is not vnnied
    joint_matrix_load(sg, mA, A, WIDTH);
    MAD_1_ROW(sg, mA, B, mC,
              Cs...); // increment B to get next block col and call recursively on remaining C's
}

template <int WIDTH, typename Group, typename Ta, typename Tb, typename Tc, std::size_t M, std::size_t K, std::size_t N,
          typename... AllC>
static inline void MAD(Group sg, const local_ptr<Ta> &A, const local_ptr<Tb> &B, AllC &...Cs) {

    for (int aiter = 0; aiter < WIDTH; aiter += K) {
        helpers::MAD_1_ROW(sg, local_ptr<T>(&Atmp[aiter]), local_ptr<T>(&B[aiter * WIDTH]), Cs...);
    }
}

template <int WIDTH, typename Group, typename T, std::size_t NumRows, std::size_t NumCols, layout Layout,
          typename... MatrixTypes, access::address_space AddressSpace, access::decorated IsDecorated>
static inline void load1Row(Group sg, const multi_ptr<T, AddressSpace, IsDecorated> &src,
                            joint_matrix<T, NumRows, NumCols, Layout, Group> &matrix, MatrixTypes &...matrices) {
    joint_matrix_load(sg, matrix, src, WIDTH);
    load1Row(sg, src + NumCols, matrices...);
}

template <int WIDTH, typename Group, typename T, std::size_t NumRows, std::size_t NumCols, layout Layout,
          typename... MatrixTypes, access::address_space AddressSpace, access::decorated IsDecorated>
static inline void store1Row(Group sg, multi_ptr<T, AddressSpace, IsDecorated> &dest,
                             joint_matrix<T, NumRows, NumCols, Layout, Group> &matrix, MatrixTypes &...matrices) {
    sycl::ext::intel::experimental::matrix::joint_matrix_store(sg, matrix, dest, WIDTH);
    store1Row(sg, dest + NumCols, matrices...);
}

template <typename Tin, typename Tout, Activation act> inline void activate(const Tin &data_in, Tout &data_out) {
    if constexpr (act == Activation::None)
        data_out == static_cast<Tout>(data_in);
    else if constexpr (act == Activation::ReLU)
        data_out == static_cast<Tout>(std::max<Tin>(static_cast<Tin>(0), data_in));
}

template <Activation act, typename Group, typename Tout, typename Tin, std::size_t NumRows, std::size_t NumCols,
          layout Layout>
static inline void applyActivation(Group sg, const joint_matrix<Tin, NumRows, NumCols, Layout, Group> &in,
                                   joint_matrix<Tout, NumRows, NumCols, Layout, Group> &out) {
    auto data_in = sycl::ext::intel::experimental::matrix::get_wi_data(sg, in);
    auto data_out = sycl::ext::intel::experimental::matrix::get_wi_data(sg, out);

    for (int rowiter = 0; rowiter < data_in.length(); rowiter++) // should be TM in length
    {
        activate<Tin, Tout, act>(static_cast<Tin>(data_in[rowiter]), static_cast<Tout>(data_out[rowiter]));
    }
}

template <Activation act, typename Group, typename Tout, typename Tin, std::size_t NumRows, std::size_t NumCols,
          layout Layout, int WIDTH, access::address_space AddressSpace, access::decorated IsDecorated>
static inline void applyActivation(Group sg, const joint_matrix<Tin, NumRows, NumCols, Layout, Group> &in,
                                   multi_ptr<Tout, AddressSpace, IsDecorated> &dest) {

    auto data_in = sycl::ext::intel::experimental::matrix::get_wi_data(sg, in);
    for (int rowiter = 0; rowiter < data_in.length(); rowiter++) // should be TM in length
    {
        activate<Tin, Tout, act>(static_cast<Tin>(data_in[rowiter]), dest[rowiter * WIDTH + sg.get_local_id()[0]]);
    }
}

// row stride >= NumCols!
template <Activation act, typename Group, typename Tout, typename Tin, int M, int N,
          access::address_space AddressSpacesrc, access::decorated IsDecoratedsrc,
          access::address_space AddressSpacedest, access::decorated IsDecorateddest>
static inline void applyActivation(Group sg, const multi_ptr<Tin, AddressSpacesrc, IsDecoratedsrc> &src,
                                   multi_ptr<Tout, AddressSpacedest, IsDecorateddest> &dest) {

    for (int iter = sg.get_local_id()[0]; iter < M * N; iter += sg.get_local_range()) // should be TM in length
    {
        activate<Tin, Tout, act>(static_cast<Tin>(src[iter]), dest[iter]);
    }
}

} // namespace helpers
} // namespace kernels
} // namespace tinydpcppnn