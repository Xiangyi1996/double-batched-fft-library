// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "doctest/doctest.h"
#include "result_check.h"
#include <iostream>
#include <vector>

#include "encoding_factory.h"
using bf16 = sycl::ext::oneapi::bfloat16;

template <typename T> void initialize_arange(std::vector<T> &vec) {

    // Repeat the col_vector and perform the operations
    for (int i = 0; i < vec.size(); i++) {
        vec[i] = static_cast<T>((i - vec.size() / 2)) / static_cast<T>(vec.size() / 2);
    }
}

template <typename T> bool check_output_non_zero(GPUMatrix<T> &matrix) {

    // Check actual data
    std::vector<T> data = matrix.copy_to_host();

    // Compare each element
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i] == 0.0) return false;
    }

    return true;
}

TEST_CASE("tinydpcppnn::encoding Identity") {
    SUBCASE("Not padded") {
        const int batch_size = 2;

        const int input_width = 3;
        const int output_width = 3;

        sycl::queue q;
        GPUMatrix<float> input(batch_size, input_width, q);
        input.fill(1.23f).wait();

        GPUMatrix<float> output_float(batch_size, output_width, q);
        output_float.fill(0.0f).wait();

        // Define the parameters for creating IdentityEncoding
        std::unordered_map<std::string, std::string> encoding = {
            {"n_dims_to_encode", std::to_string(input_width)}, {"scale", "1.0"}, {"offset", "0.0"}};
        Encoding<float> *network = create_encoding<float>("Identity", encoding);
        network->set_padded_output_width(output_width);

        std::unique_ptr<Context> model_ctx = network->forward_impl(&q, input, &output_float);
        q.wait();

        CHECK(input == output_float);

        delete network;
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
        GPUMatrix<float> input(batch_size, input_width, q);
        input.copy_from_host(input_float).wait();

        GPUMatrix<float> output_float(batch_size, output_width, q);
        output_float.fill(0.0f).wait();

        std::unordered_map<std::string, std::string> encoding = {{"n_dims_to_encode", std::to_string(input_width)},
                                                                 {"degree", std::to_string(DEGREE)}};
        Encoding<float> *network = create_encoding<float>("SphericalHarmonics", encoding);
        network->set_padded_output_width(output_float.n());
        std::unique_ptr<Context> model_ctx = network->forward_impl(&q, input, &output_float);
        q.wait();
        CHECK(check_output_non_zero(output_float));

        std::vector<float> out = output_float.copy_to_host();
        const std::vector<float> reference_out = {0.2821, 1.0, 1.0, 0.2821, 1.0, 1.0};

        const double epsilon = 1e-3;

        // Check if the actual vector is equal to the expected vector within the tolerance
        for (size_t i = 0; i < out.size(); i++) {
            CHECK(out[i] == doctest::Approx(reference_out[i]).epsilon(epsilon));
        }

        delete network;
    }
}

TEST_CASE("tinydpcppnn::encoding Grid Encoding") {
    SUBCASE("Not padded") {
        // SWIFTNET
        const int input_width = 3;
        const int batch_size = 1;
        sycl::queue q;

        GPUMatrix<float> input(batch_size, input_width, q);
        input.fill(1.0f).wait();

        GPUMatrix<float> output_float(32, batch_size, q);
        output_float.fill(0.0f).wait();

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

        GridEncoding<float> *network = create_grid_encoding<float>(input_width, encoding_json);

        std::vector<float> tmp_params_host(network->n_params());
        initialize_arange(tmp_params_host);
        DeviceMem<float> params_full_precision(network->n_params(), q);
        params_full_precision.copy_from_host(tmp_params_host).wait();

        network->set_params(params_full_precision.data(), params_full_precision.data(), nullptr);

        std::unique_ptr<Context> model_ctx = network->forward_impl(&q, input, &output_float);
        q.wait();

        CHECK(check_output_non_zero(output_float));

        const std::vector<float> out = output_float.copy_to_host();
        const std::vector<float> reference_out = {
            -1,      -1,      -0.9985, -0.9985, -0.98,   -0.98,   -0.8076, -0.8076, -0.6606, -0.6606, -0.5107,
            -0.5107, -0.4202, -0.4202, -0.2527, -0.2527, -0.1031, -0.1031, 0.06964, 0.06964, 0.1893,  0.1893,
            0.2996,  0.2996,  0.4565,  0.4565,  0.6128,  0.6128,  0.7783,  0.7783,  0.9258,  0.9258};

        const double epsilon = 1e-4; // Set the tolerance for floating-point comparisons

        // Check if the actual vector is equal to the expected vector within the tolerance
        for (size_t i = 0; i < out.size(); ++i) {
            CHECK(out[i] == doctest::Approx(reference_out[i]).epsilon(epsilon));
        }

        delete network;
    }

    SUBCASE("Check results loaded") {
        // SWIFTNET
        const int input_width = 2;
        const int batch_size = 8;
        sycl::queue q;

        GPUMatrix<float> input(batch_size, input_width, q);
        input.fill(0.0f).wait();

        GPUMatrix<float> output_float(batch_size, 32, q);
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

        GridEncoding<float> *network = create_grid_encoding<float>(input_width, encoding_json);
        DeviceMem<float> params_full_precision(network->n_params(), q);

        std::vector<float> params =
            loadVectorFromCSV<float>("../../test/ref_values/network_with_grid_encoding/full/encoding_params.csv");
        std::vector<float> input_ref =
            loadVectorFromCSV<float>("../../test/ref_values/network_with_grid_encoding/full/input.csv");
        std::cout << "Loaded n_params: " << params.size() << ", encoding n_params: " << network->n_params()
                  << std::endl;
        CHECK(params.size() == network->n_params());

        params_full_precision.copy_from_host(params).wait();
        input.copy_from_host(input_ref);

        network->set_params(params_full_precision.data(), params_full_precision.data(), nullptr);

        std::unique_ptr<Context> model_ctx = network->forward_impl(&q, input, &output_float);
        q.wait();

        CHECK(check_output_non_zero(output_float));

        std::vector<float> out = output_float.copy_to_host();
        q.wait();
        std::vector<float> reference_out =
            loadVectorFromCSV<float>("../../test/ref_values/network_with_grid_encoding/full/encoding_output.csv");

        const double epsilon = 1e-4; // Set the tolerance for floating-point comparisons

        // Check if the actual vector is equal to the expected vector within the tolerance
        for (size_t i = 0; i < out.size(); ++i) {
            CHECK(out[i] == doctest::Approx(reference_out[i]).epsilon(epsilon));
        }

        delete network;
    }
}