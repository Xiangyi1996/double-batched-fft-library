// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "doctest/doctest.h"
#include "result_check.h"
#include <iostream>
#include <vector>

#include "encoding_factory.h"
using bf16 = sycl::ext::oneapi::bfloat16;
using tinydpcppnn::encodings::grid::GridEncoding;

template <typename T> void initialize_arange(std::vector<T> &vec) {

    // Repeat the col_vector and perform the operations
    double offset = (double)vec.size() / 2.0;

    for (long long i = 0; i < vec.size(); i++) {
        vec[i] = static_cast<T>(i / offset - 1.0);
    }
}

TEST_CASE("tinydpcppnn::encoding Identity") {
    SUBCASE("Not padded") {
        const int batch_size = 2;

        const int input_width = 3;
        const int output_width = 3;

        sycl::queue q;
        DeviceMatrix<float> input(batch_size, input_width, q);
        input.fill(1.23f).wait();

        DeviceMatrix<float> output_float(batch_size, output_width, q);
        output_float.fill(0.0f).wait();

        // Define the parameters for creating IdentityEncoding
        std::unordered_map<std::string, std::string> encoding = {
            {"n_dims_to_encode", std::to_string(input_width)}, {"scale", "1.0"}, {"offset", "0.0"}};
        std::shared_ptr<Encoding<float>> network = create_encoding<float>("Identity", encoding);
        network->set_padded_output_width(output_width);

        std::unique_ptr<Context> model_ctx = network->forward_impl(&q, input, &output_float);
        q.wait();

        CHECK(input == output_float);
    }
}

TEST_CASE("tinydpcppnn::encoding Spherical Harmonics") {
    SUBCASE("Not padded") {
        const int batch_size = 2;

        const int input_width = 3;
        const int output_width = 3;
        const int DEGREE = 1;

        sycl::queue q;
        std::vector<float> input_float(input_width * batch_size);
        initialize_arange(input_float);
        DeviceMatrix<float> input(batch_size, input_width, q);
        input.copy_from_host(input_float).wait();

        DeviceMatrix<float> output_float(batch_size, output_width, q);
        output_float.fill(0.0f).wait();

        std::unordered_map<std::string, std::string> encoding = {{"n_dims_to_encode", std::to_string(input_width)},
                                                                 {"degree", std::to_string(DEGREE)}};
        std::shared_ptr<Encoding<float>> network = create_encoding<float>("SphericalHarmonics", encoding);
        network->set_padded_output_width(output_float.n());
        std::unique_ptr<Context> model_ctx = network->forward_impl(&q, input, &output_float);
        q.wait();

        std::vector<float> out = output_float.copy_to_host();
        const std::vector<float> reference_out = {0.2821, 1.0, 1.0, 0.2821, 1.0, 1.0};

        const double epsilon = 1e-3;

        // Check if the actual vector is equal to the expected vector within the tolerance
        for (size_t i = 0; i < out.size(); i++) {
            CHECK(out[i] == doctest::Approx(reference_out[i]).epsilon(epsilon));
        }
    }
}

TEST_CASE("tinydpcppnn::encoding Grid Encoding") {
    SUBCASE("Not padded") {
        // SWIFTNET
        const int input_width = 3;
        const int batch_size = 1;
        const int padded_output_width = 32;
        sycl::queue q;

        DeviceMatrix<float> input(batch_size, input_width, q);
        input.fill(1.0f).wait();

        DeviceMatrix<float> output_float(batch_size, padded_output_width, q);
        output_float.fill(1.23f).wait(); // fill with something to check if it is written to

        json encoding_json = {
            {"n_dims_to_encode", std::to_string(input_width)},
            {"otype", "Grid"},
            {"type", "Hash"},
            {"n_levels", 16},
            {"n_features_per_level", 2},
            {"log2_hashmap_size", 19},
            {"base_resolution", 16},
            {"per_level_scale", 2.0},
        };

        std::shared_ptr<GridEncoding<float>> network =
            tinydpcppnn::encodings::grid::create_grid_encoding<float>(input_width, encoding_json);
        q.wait();
        network->set_padded_output_width(output_float.n());

        std::vector<float> tmp_params_host(network->n_params(), 1.0f);
        initialize_arange(tmp_params_host);
        DeviceMem<float> params_full_precision(network->n_params(), q);
        params_full_precision.copy_from_host(tmp_params_host).wait();

        network->set_params(params_full_precision.data(), params_full_precision.data(), nullptr);

        std::unique_ptr<Context> model_ctx = network->forward_impl(&q, input, &output_float);
        q.wait();

        const std::vector<float> out = output_float.copy_to_host();
        const std::vector<float> reference_out = {
            -1,      -1,      -0.9985, -0.9985, -0.98,   -0.98,   -0.8076, -0.8076, -0.6606, -0.6606, -0.5107,
            -0.5107, -0.4202, -0.4202, -0.2527, -0.2527, -0.1031, -0.1031, 0.06964, 0.06964, 0.1893,  0.1893,
            0.2996,  0.2996,  0.4565,  0.4565,  0.6128,  0.6128,  0.7783,  0.7783,  0.9258,  0.9258};

        const double epsilon = 1e-4; // Set the tolerance for floating-point comparisons

        // Check if the actual vector is equal to the expected vector within the tolerance
        for (size_t i = 0; i < out.size(); ++i) {
            CHECK(out[i] == doctest::Approx(reference_out[i]).epsilon(epsilon));
            std::cout << out[i] << ", ";
        }
        std::cout << std::endl;
    }

    SUBCASE("Check results loaded") {
        // SWIFTNET
        const int input_width = 2;
        const int batch_size = 8;
        const int padded_output_width = 32;
        sycl::queue q;

        DeviceMatrix<float> input(batch_size, input_width, q);
        input.fill(0.0f).wait();

        DeviceMatrix<float> output_float(batch_size, padded_output_width, q);
        output_float.fill(0.0f);

        json encoding_json = {
            {"n_dims_to_encode", std::to_string(input_width)},
            {"otype", "Grid"},
            {"type", "Hash"},
            {"n_levels", 16},
            {"n_features_per_level", 2},
            {"log2_hashmap_size", 15},
            {"base_resolution", 16},
            {"per_level_scale", 1.5},
        };

        std::shared_ptr<GridEncoding<float>> network =
            tinydpcppnn::encodings::grid::create_grid_encoding<float>(input_width, encoding_json);
        network->set_padded_output_width(output_float.n());
        DeviceMem<float> params_full_precision(network->n_params(), q);

        std::vector<float> params =
            loadVectorFromCSV<float>("../../test/ref_values/network_with_grid_encoding/full/encoding_params.csv");
        std::vector<float> input_ref =
            loadVectorFromCSV<float>("../../test/ref_values/network_with_grid_encoding/full/input.csv");
        std::cout << "Loaded n_params: " << params.size() << ", encoding n_params: " << network->n_params()
                  << std::endl;
        CHECK(params.size() == network->n_params());

        params_full_precision.copy_from_host(params).wait();
        std::vector<float> input_ref_cut(input_ref.begin(), input_ref.begin() + batch_size * input_width);
        input.copy_from_host(input_ref_cut).wait();

        network->set_params(params_full_precision.data(), params_full_precision.data(), nullptr);

        std::unique_ptr<Context> model_ctx = network->forward_impl(&q, input, &output_float);
        q.wait();

        std::vector<float> out = output_float.copy_to_host();
        std::vector<float> reference_out =
            loadVectorFromCSV<float>("../../test/ref_values/network_with_grid_encoding/full/encoding_output.csv");

        const double epsilon = 1e-4; // Set the tolerance for floating-point comparisons

        // Check if the actual vector is equal to the expected vector within the tolerance
        for (size_t i = 0; i < out.size(); ++i) {
            CHECK(out[i] == doctest::Approx(reference_out[i]).epsilon(epsilon));
            // std::cout << out[i] << ", " << reference_out[i] << std::endl;
        }
        std::cout << std::endl;
    }
}