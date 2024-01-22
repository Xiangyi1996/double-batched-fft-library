/**
 * @file doctest_kernel_esimd.cpp
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Class which tests the esimd kernels.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "doctest/doctest.h"

#include "kernel_esimd.h"

#include <sycl/sycl.hpp>

using bf16 = sycl::ext::oneapi::bfloat16;
using sycl::ext::intel::experimental::esimd::cache_hint;
using namespace tinydpcppnn::kernels::esimd;
using namespace sycl::ext::intel::esimd;

/// TODO: EVERYTHING

template <int M, int N, int TK, typename T> void TestLoadRow(sycl::queue &q) {
    constexpr int nElems = M * N;
    T *in = sycl::malloc_shared<T>(nElems, q);
    T *out = sycl::malloc_shared<T>(nElems, q);
    for (int iter = 0; iter < nElems; iter++) {
        in[iter] = static_cast<T>(iter);
    }

    q.parallel_for(sycl::nd_range<1>(1, 1), [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
         simd<T, nElems> tmp;
         EsimdKernels<T, N, N, N, Activation::ReLU, Activation::None, 16>::template loadRow<M, TK, cache_hint::none,
                                                                                            cache_hint::none>(in, tmp);
         tmp.copy_to(out);
     }).wait();

    for (int iter = 0; iter < N; iter++) {
        CHECK(out[iter] == static_cast<T>(iter + (iter / (M * TK)) * (M * TK)));
    }

    sycl::free(in, q);
    sycl::free(out, q);
}

TEST_CASE("MoveMemory") {

    sycl::queue q(sycl::gpu_selector_v);

    SUBCASE("load row bf16 N64 M8 TK16") { TestLoadRow<8, 64, 16, bf16>(q); }
    SUBCASE("load row half N64 M8 TK16") { TestLoadRow<8, 64, 16, sycl::half>(q); }
    SUBCASE("load row bf16 N16 M8 TK16") { TestLoadRow<8, 16, 16, bf16>(q); }
    SUBCASE("load row bf16 N32 M8 TK16") { TestLoadRow<8, 32, 16, bf16>(q); }
    SUBCASE("load row bf16 N128 M8 TK16") { TestLoadRow<8, 128, 16, bf16>(q); }
    SUBCASE("load row bf16 N64 M1 TK16") { TestLoadRow<1, 64, 16, bf16>(q); }
    SUBCASE("load row bf16 N64 M2 TK16") { TestLoadRow<2, 64, 16, bf16>(q); }
    SUBCASE("load row bf16 N64 M3 TK16") { TestLoadRow<3, 64, 16, bf16>(q); }
    SUBCASE("load row bf16 N64 M4 TK16") { TestLoadRow<4, 64, 16, bf16>(q); }
    SUBCASE("load row bf16 N64 M5 TK16") { TestLoadRow<5, 64, 16, bf16>(q); }
    SUBCASE("load row bf16 N64 M6 TK16") { TestLoadRow<6, 64, 16, bf16>(q); }
    SUBCASE("load row bf16 N64 M7 TK16") { TestLoadRow<7, 64, 16, bf16>(q); }
}
