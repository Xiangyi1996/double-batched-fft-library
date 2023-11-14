// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "doctest/doctest.h"

#include "kernel_helper.h"

#include <sycl/sycl.hpp>

TEST_CASE("joint_matrix_mad") {

    using namespace sycl::ext::oneapi::experimental::matrix;
    using bf16 = sycl::ext::oneapi::bfloat16;
    using namespace sycl;

    sycl::queue Q(sycl::gpu_selector_v);
    float *out = sycl::malloc_shared<float>(8 * 16, Q);

    Q.parallel_for(sycl::nd_range<1>(16, 16), [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(16)]] {
        joint_matrix<sub_group, bf16, use::a, 8, 16, layout::row_major> mA;
        joint_matrix<sub_group, bf16, use::b, 16, 16, layout::ext_intel_packed> mB;
        joint_matrix<sub_group, float, use::accumulator, 8, 16> mC;

        joint_matrix_fill(item.get_sub_group(), mC, 0);
        joint_matrix_fill(item.get_sub_group(), mA, 1);
        joint_matrix_fill(item.get_sub_group(), mB, 0.5);
        joint_matrix_mad(item.get_sub_group(), mC, mA, mB, mC);
        auto sg = item.get_sub_group();
        joint_matrix_store(sg, mC, address_space_cast<access::address_space::global_space, access::decorated::yes>(out),
                           16, layout::row_major);
    });

    Q.wait();
    std::vector<float> result(8 * 16);
    Q.memcpy(result.data(), out, 8 * 16 * sizeof(float)).wait();

    for (int i = 0; i < 8 * 16; i++) {
        CHECK((result[i] - 8.0f) < 1e-4);
    }

    sycl::free(out, Q);
}

TEST_CASE("tinydpcppnn::kernels::helpers::MAD_1_ROW 1") {
    using namespace sycl::ext::oneapi::experimental::matrix;
    using bf16 = sycl::ext::oneapi::bfloat16;
    using namespace sycl;

    sycl::queue Q(sycl::gpu_selector_v);
    float *out = sycl::malloc_device<float>(8 * 16, Q);
    bf16 *in = sycl::malloc_device<bf16>(16 * 16, Q);

    Q.parallel_for(sycl::nd_range<1>(16, 16), [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(16)]] {
        joint_matrix<sub_group, bf16, use::a, 8, 16, layout::row_major> mA;
        joint_matrix<sub_group, float, use::accumulator, 8, 16> mC;

        joint_matrix_fill(item.get_sub_group(), mC, 0);
        joint_matrix_fill(item.get_sub_group(), mA, 1);
        for (int i = 0; i < 16; i++) {
            in[item.get_global_linear_id() + 16 * i] = 0.5f;
        }

        auto sg = item.get_sub_group();
        tinydpcppnn::kernels::helpers::MAD_1_ROW<16, 16>(
            sg, mA, address_space_cast<access::address_space::global_space, access::decorated::yes>(in), mC);

        joint_matrix_store(sg, mC, address_space_cast<access::address_space::global_space, access::decorated::yes>(out),
                           16, layout::row_major);
    });

    Q.wait();
    std::vector<float> result(8 * 16);
    Q.memcpy(result.data(), out, 8 * 16 * sizeof(float)).wait();

    for (int i = 0; i < 8 * 16; i++) {
        CHECK((result[i] - 8.0f) < 1e-4);
    }

    sycl::free(out, Q);
}

TEST_CASE("tinydpcppnn::kernels::helpers::MAD_1_ROW 2") {
    using namespace sycl::ext::oneapi::experimental::matrix;
    using bf16 = sycl::ext::oneapi::bfloat16;
    using namespace sycl;

    sycl::queue Q(sycl::gpu_selector_v);
    float *out = sycl::malloc_device<float>(8 * 16, Q);
    bf16 *inA = sycl::malloc_device<bf16>(8 * 16, Q);
    bf16 *inB = sycl::malloc_device<bf16>(16 * 16, Q);

    Q.parallel_for(sycl::nd_range<1>(16, 16), [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(16)]] {
        joint_matrix<sub_group, bf16, use::a, 8, 16, layout::row_major> mA;
        joint_matrix<sub_group, float, use::accumulator, 8, 16> mC;

        joint_matrix_fill(item.get_sub_group(), mC, 0);
        joint_matrix_fill(item.get_sub_group(), mA, 1);
        for (int i = 0; i < 16; i++) {
            inB[item.get_global_linear_id() + 16 * i] = 0.5f;
        }

        for (int i = 0; i < 8; i++) {
            inB[item.get_global_linear_id() + 16 * i] = 1.0f;
        }

        auto sg = item.get_sub_group();
        tinydpcppnn::kernels::helpers::MAD_1_ROW<16, 16>(
            sg, address_space_cast<access::address_space::global_space, access::decorated::yes>(inA),
            address_space_cast<access::address_space::global_space, access::decorated::yes>(inB), mC);

        joint_matrix_store(sg, mC, address_space_cast<access::address_space::global_space, access::decorated::yes>(out),
                           16, layout::row_major);
    });

    Q.wait();
    std::vector<float> result(8 * 16);
    Q.memcpy(result.data(), out, 8 * 16 * sizeof(float)).wait();

    for (int i = 0; i < 8 * 16; i++) {
        CHECK((result[i] - 8.0f) < 1e-4);
    }

    sycl::free(out, Q);
}