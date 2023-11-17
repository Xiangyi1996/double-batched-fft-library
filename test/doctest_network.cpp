// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "doctest/doctest.h"

#include "kernel_helper.h"

#include <sycl/sycl.hpp>

using bf16 = sycl::ext::oneapi::bfloat16;
using namespace tinydpcppnn::kernels::helpers;

///TODO: consolidate the MAD_1_ROW cases into 1 TEST_CASE with multiple SUBCASES

TEST_CASE("MoveMemory") {

    sycl::queue Q;
    try {
        Q = sycl::queue(sycl::gpu_selector_v);
    } catch (...) {
        std::cout << "No device of requested type found" << std::endl;
        return;
    }

    SUBCASE("global to slm") {
        constexpr int M = 8;
        constexpr int N = 64;
        constexpr int nElems = M * N;
        constexpr int SG_SIZE = 16;
        const bf16 val = static_cast<bf16>(1.23);
        bf16 *in = sycl::malloc_device<bf16>(nElems, Q);
        bf16 *out = sycl::malloc_device<bf16>(nElems, Q);

        Q.fill(in, val, nElems).wait();

        Q.submit([&](sycl::handler &cgh) {
             sycl::local_accessor<bf16, 1> slm(sycl::range<1>(nElems), cgh);

             cgh.parallel_for(
                 sycl::nd_range<1>(2 * SG_SIZE, SG_SIZE),
                 [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                     moveMemory<M, N>(
                         item, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(in),
                         slm.template get_multi_ptr<sycl::access::decorated::yes>());
                     moveMemory<M, N>(
                         item, slm.template get_multi_ptr<sycl::access::decorated::yes>(),
                         sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(out));
                 });
         }).wait();

        std::vector<bf16> out_host(nElems);
        Q.memcpy(out_host.data(), out, out_host.size() * sizeof(bf16)).wait();

        for (int iter = 0; iter < nElems; iter++) {
            CHECK(out_host[iter] == val);
        }

        sycl::free(in, Q);
        sycl::free(out, Q);
    }
}

TEST_CASE("MoveMemorySG") {
    using namespace sycl::ext::oneapi::experimental::matrix;
    

    sycl::queue Q;
    try {
        Q = sycl::queue(sycl::gpu_selector_v);
    } catch (...) {
        std::cout << "No device of requested type found" << std::endl;
        return;
    }

    SUBCASE("global to slm to global") {
        constexpr int M = 16;
        constexpr int TM = 8;
        constexpr int N = 64;
        constexpr int nElems = M * N;
        constexpr int SG_SIZE = 16;
        const bf16 val = static_cast<bf16>(1.23);
        bf16 *in = sycl::malloc_device<bf16>(nElems, Q);
        bf16 *out = sycl::malloc_device<bf16>(nElems, Q);

        Q.fill(in, val, nElems).wait();

        Q.submit([&](sycl::handler &cgh) {
             sycl::local_accessor<bf16, 1> slm(sycl::range<1>(nElems), cgh);

             cgh.parallel_for(
                 sycl::nd_range<1>(2 * SG_SIZE, SG_SIZE),
                 [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                     auto sg = item.get_sub_group();
                     const int offset = TM * N * sg.get_group_id()[0] + TM * N * sg.get_group_range()[0] * item.get_group().get_group_id();
                     moveMemorySG<TM, N>(
                         sg,
                         sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(
                             in + offset),
                         slm.template get_multi_ptr<sycl::access::decorated::yes>() + offset);

                     moveMemorySG<TM, N>(
                         sg, slm.template get_multi_ptr<sycl::access::decorated::yes>() + offset,
                         sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(
                             out + offset));
                 });
         }).wait();

        std::vector<bf16> out_host(nElems);
        Q.memcpy(out_host.data(), out, out_host.size() * sizeof(bf16)).wait();

        for (int iter = 0; iter < nElems; iter++) {
            CHECK(std::abs(static_cast<float>(out_host[iter]) - static_cast<float>(val)) < 0.001 );
        }

        sycl::free(in, Q);
        sycl::free(out, Q);
    }
}

TEST_CASE("activate") {
    
    SUBCASE("host none") {
        const bf16 inval = -0.12;
        float outval = 0;
        activate<bf16, float, Activation::None>(inval, outval);
        CHECK(outval == inval);
    }

    SUBCASE("host relu") {
        bf16 inval = -0.12;
        float outval = 1;
        activate<bf16, float, Activation::ReLU>(inval, outval);
        CHECK(outval == 0);

        inval = 0.12;
        activate<bf16, float, Activation::ReLU>(inval, outval);
        CHECK(outval == inval);
    }

    sycl::queue Q;
    try {
        Q = sycl::queue(sycl::gpu_selector_v);
    } catch (...) {
        std::cout << "No device of requested type found" << std::endl;
        return;
    }

    float * out = sycl::malloc_device<float>(1, Q);
    float out_host;

    SUBCASE("device none") {
        Q.parallel_for(1, [=](auto item) {

            const bf16 inval = -0.12;
            float outval = 0;
            activate<bf16, float, Activation::None>(inval, outval);
            out[0] = outval;

        }).wait();

        Q.memcpy(&out_host, out, sizeof(float)).wait();

        CHECK(out_host == doctest::Approx(-0.12).epsilon(0.01));
    }


    SUBCASE("device relu") {
        Q.parallel_for(1, [=](auto item) {

            bf16 inval = -0.12;
            float outval = 1;
            activate<bf16, float, Activation::ReLU>(inval, outval);
            out[0] = outval;

        }).wait();

        Q.memcpy(&out_host, out, sizeof(float)).wait();

        CHECK(out_host == 0);

        Q.parallel_for(1, [=](auto item) {

            bf16 inval = 0.12;
            float outval = 1;
            activate<bf16, float, Activation::ReLU>(inval, outval);
            out[0] = outval;

        }).wait();

        Q.memcpy(&out_host, out, sizeof(float)).wait();

        CHECK(out_host == doctest::Approx(0.12).epsilon(0.01));
    }

    sycl::free(out, Q);
}

TEST_CASE("applyActivation") {
    SUBCASE("slm to global") {}

    SUBCASE("joint_matrices to slm") {}
}

TEST_CASE("joint_matrix_mad") {

    using namespace sycl::ext::oneapi::experimental::matrix;

    sycl::queue Q;
    try {
        Q = sycl::queue(sycl::gpu_selector_v);
    } catch (...) {
        std::cout << "No device of requested type found" << std::endl;
        return;
    }
    float *out = sycl::malloc_shared<float>(8 * 16, Q);

    Q.parallel_for(sycl::nd_range<1>(16, 16), [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(16)]] {
        joint_matrix<sycl::sub_group, bf16, use::a, 8, 16, layout::row_major> mA;
        joint_matrix<sycl::sub_group, bf16, use::b, 16, 16, layout::ext_intel_packed> mB;
        joint_matrix<sycl::sub_group, float, use::accumulator, 8, 16> mC;

        joint_matrix_fill(item.get_sub_group(), mC, 0);
        joint_matrix_fill(item.get_sub_group(), mA, 1);
        joint_matrix_fill(item.get_sub_group(), mB, 0.5);
        joint_matrix_mad(item.get_sub_group(), mC, mA, mB, mC);
        auto sg = item.get_sub_group();
        joint_matrix_store(sg, mC, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(out),
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

TEST_CASE("tinydpcppnn::kernels::helpers::zeroMatrices 1") {
    using namespace sycl::ext::oneapi::experimental::matrix;

    sycl::queue Q;
    try {
        Q = sycl::queue(sycl::gpu_selector_v);
    } catch (...) {
        std::cout << "No device of requested type found" << std::endl;
        return;
    }
    float *out = sycl::malloc_device<float>(8 * 16 * 2, Q);

    Q.parallel_for(sycl::nd_range<1>(16, 16), [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(16)]] {
        std::array<joint_matrix<sycl::sub_group, float, use::accumulator, 8, 16>, 2> mCs;

        auto sg = item.get_sub_group();
        joint_matrix_fill(item.get_sub_group(), mCs[0], 1.0f);
        joint_matrix_fill(item.get_sub_group(), mCs[1], 1.0f);

        joint_matrix_store(sg, mCs[0],
                           sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(out), 32,
                           layout::row_major);
        joint_matrix_store(sg, mCs[1],
                           sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(out + 16),
                           32, layout::row_major);

        // now everything in out should be 1

        zeroMatrices(sg, mCs);

        joint_matrix_store(sg, mCs[0],
                           sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(out), 32,
                           layout::row_major);

        // now the first one in out should be 0
    });

    Q.wait();
    std::vector<float> result(8 * 16 * 2);
    Q.memcpy(result.data(), out, 8 * 16 * 2 * sizeof(float)).wait();

    for (int i = 0; i < 8 * 16 * 2; i++) {
        if (i / 16 % 2 == 0)
            CHECK((result[i] - 0.0f) < 1e-5);
        else
            CHECK((result[i] - 1.0f) < 1e-5);
    }

    sycl::free(out, Q);
}

TEST_CASE("tinydpcppnn::kernels::helpers::MAD_1_ROW 1") {
    using namespace sycl::ext::oneapi::experimental::matrix;

    sycl::queue Q;
    try {
        Q = sycl::queue(sycl::gpu_selector_v);
    } catch (...) {
        std::cout << "No device of requested type found" << std::endl;
        return;
    }
    float *out = sycl::malloc_device<float>(8 * 16, Q);
    bf16 *in = sycl::malloc_device<bf16>(16 * 16, Q);

    Q.parallel_for(sycl::nd_range<1>(16, 16), [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(16)]] {
        joint_matrix<sycl::sub_group, bf16, use::a, 8, 16, layout::row_major> mA;
        std::array<joint_matrix<sycl::sub_group, float, use::accumulator, 8, 16>, 1> mCs;

        joint_matrix_fill(item.get_sub_group(), mCs[0], 0);
        for (int i = 0; i < 16; i++) {
            in[item.get_global_linear_id() + 16 * i] = 0.5f;
        }

        auto sg = item.get_sub_group();
        MAD_1_ROW<16>(
            sg, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(in),
            sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(in), mCs);

        joint_matrix_store(sg, mCs[0],
                           sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(out), 16,
                           layout::row_major);
    });

    Q.wait();
    std::vector<float> result(8 * 16);
    Q.memcpy(result.data(), out, 8 * 16 * sizeof(float)).wait();

    for (int i = 0; i < 8 * 16; i++) {
        CHECK((result[i] - 4.0f) < 1e-4);
    }

    sycl::free(out, Q);
    sycl::free(in, Q);
}

TEST_CASE("tinydpcppnn::kernels::helpers::MAD_1_ROW 2") {
    using namespace sycl::ext::oneapi::experimental::matrix;
    

    sycl::queue Q;
    try {
        Q = sycl::queue(sycl::gpu_selector_v);
    } catch (...) {
        std::cout << "No device of requested type found" << std::endl;
        return;
    }
    float *out = sycl::malloc_device<float>(8 * 16 * 2, Q);
    bf16 *inA = sycl::malloc_device<bf16>(8 * 16, Q);
    bf16 *inB = sycl::malloc_device<bf16>(16 * 16 * 2, Q);

    Q.parallel_for(sycl::nd_range<1>(16, 16), [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(16)]] {
        std::array<joint_matrix<sycl::sub_group, float, use::accumulator, 8, 16>, 2> mCs;

        joint_matrix_fill(item.get_sub_group(), mCs[0], 0);
        joint_matrix_fill(item.get_sub_group(), mCs[1], 0);
        for (int i = 0; i < 16 * 2; i++) {
            inB[item.get_global_linear_id() + 16 * i] = 0.5f;
        }

        for (int i = 0; i < 8; i++) {
            inA[item.get_global_linear_id() + 16 * i] = 1.0f;
        }

        auto sg = item.get_sub_group();
        MAD_1_ROW<16>(
            sg, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(inA),
            sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(inB), mCs);

        joint_matrix_store(sg, mCs[0],
                           sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(out), 32,
                           layout::row_major);
        joint_matrix_store(sg, mCs[1],
                           sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(out + 16),
                           32, layout::row_major);
    });

    Q.wait();
    std::vector<float> result(8 * 16 * 2);
    Q.memcpy(result.data(), out, 8 * 16 * 2 * sizeof(float)).wait();

    for (int i = 0; i < 8 * 16 * 2; i++) {
        CHECK((result[i] - 8.0f) < 1e-4);
    }

    sycl::free(out, Q);
    sycl::free(inA, Q);
    sycl::free(inB, Q);
}

TEST_CASE("tinydpcppnn::kernels::helpers::MAD 1") {
    using namespace sycl::ext::oneapi::experimental::matrix;

    // Multiply a matrix of size 8x16 consistinf of 1's with a matrix of size
    // 16x32 consisting of 0.5's for an output of size 8x32 consisting of 8's
    // only works on PVC

    sycl::queue Q;
    try {
        Q = sycl::queue(sycl::gpu_selector_v);
    } catch (...) {
        std::cout << "No device of requested type found" << std::endl;
        return;
    }
    float *out = sycl::malloc_device<float>(8 * 16 * 2, Q);
    bf16 *inA = sycl::malloc_device<bf16>(8 * 16, Q);
    bf16 *inB = sycl::malloc_device<bf16>(16 * 16 * 2, Q);

    Q.parallel_for(sycl::nd_range<1>(16, 16), [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(16)]] {
        std::array<joint_matrix<sycl::sub_group, float, use::accumulator, 8, 16>, 2> mCs;

        joint_matrix_fill(item.get_sub_group(), mCs[0], 0);
        joint_matrix_fill(item.get_sub_group(), mCs[1], 0);
        for (int i = 0; i < 16 * 2; i++) {
            inB[item.get_global_linear_id() + 16 * i] = 0.5f;
        }

        for (int i = 0; i < 8; i++) {
            inA[item.get_global_linear_id() + 16 * i] = 1.0f;
        }

        auto sg = item.get_sub_group();
        MAD<16>(
            sg, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(inA),
            sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(inB), mCs);

        joint_matrix_store(sg, mCs[0],
                           sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(out), 32,
                           layout::row_major);
        joint_matrix_store(sg, mCs[1],
                           sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(out + 16),
                           32, layout::row_major);
    });

    Q.wait();
    std::vector<float> result(8 * 16 * 2);
    Q.memcpy(result.data(), out, 8 * 16 * 2 * sizeof(float)).wait();

    for (int i = 0; i < 8 * 16 * 2; i++) {
        CHECK((result[i] - 8.0f) < 1e-4);
    }

    sycl::free(out, Q);
    sycl::free(inA, Q);
    sycl::free(inB, Q);
}