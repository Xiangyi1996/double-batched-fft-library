// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "doctest/doctest.h"
#include "result_check.h"
#include <iostream>
#include <vector>

#include "SwiftNetMLP.h"
#include "encoding_factory.h"
#include "mkl.h"
#include "mkl_omp_offload.h"
#include "oneapi/mkl.hpp"
using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
using bf16 = sycl::ext::oneapi::bfloat16;

template <typename T> bool areGPUMatricesEqual(GPUMatrix<T> &matrix1, GPUMatrix<T> &matrix2) {
    // Check dimensions
    if (matrix1.rows() != matrix2.rows() || matrix1.cols() != matrix2.cols()) {
        return false;
    }

    // Check layout
    if (matrix1.layout() != matrix2.layout()) {
        return false;
    }

    // Check actual data
    std::vector<T> data1 = matrix1.to_cpu_vector();
    std::vector<T> data2 = matrix2.to_cpu_vector();

    // Compare each element
    for (size_t i = 0; i < data1.size(); ++i) {
        if (data1[i] != data2[i]) {
            return false;
        }
    }

    return true;
}

template <typename T> bool check_output_non_zero(GPUMatrix<T> &matrix) {

    // Check actual data
    std::vector<T> data = matrix.to_cpu_vector();

    // Compare each element
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i] == 0.0) {
            return false;
        }
    }

    return true;
}

TEST_CASE("tinydpcppnn::encoding Identity") {
    SUBCASE("Not padded") {
        const int batch_size = 2;

        int input_width = 3;
        int output_width = 3;

        queue q;
        DeviceMem<float> input_float(input_width * batch_size, q);
        input_float.initialize_arange(q);
        GPUMatrix<float> input(input_float.data(), input_width, batch_size);

        GPUMatrix<float> output_float(output_width, batch_size);

        output_float.initialize_constant(0.00f);

        // std::cout << "In" << std::endl;
        // input.print();
        // std::cout << "Out" << std::endl;
        // output_float.print();

        // Define the parameters for creating IdentityEncoding
        std::unordered_map<std::string, std::string> encoding = {
            {"n_dims_to_encode", std::to_string(input_width)}, {"scale", "1.0"}, {"offset", "0.0"}};
        Encoding<float> *network = create_encoding<float>("Identity", encoding);
        network->set_padded_output_width(output_width);

        std::unique_ptr<Context> model_ctx = network->forward_impl(&q, input, &output_float);
        // std::cout << "out width: " << network->output_width() << std::endl;
        // std::cout << "Out2" << std::endl;
        // output_float.print();

        CHECK(areGPUMatricesEqual(input, output_float));
    }
}

TEST_CASE("tinydpcppnn::encoding Spherical Harmonics") {
    SUBCASE("Not padded") {
        const int batch_size = 2;

        int input_width = 3;
        int output_width = 3;
        int DEGREE = 4;

        queue q;
        DeviceMem<float> input_float(input_width * batch_size, q);
        input_float.initialize_arange(q);
        GPUMatrix<float> input(input_float.data(), input_width, batch_size);

        GPUMatrix<float> output_float(output_width, batch_size);

        output_float.initialize_constant(0.00f);

        // std::cout << "In Spherical" << std::endl;
        // input.print();
        // std::cout << "Out" << std::endl;
        // output_float.print();

        std::unordered_map<std::string, std::string> encoding = {{"n_dims_to_encode", std::to_string(input_width)},
                                                                 {"degree", std::to_string(DEGREE)}};
        Encoding<float> *network = create_encoding<float>("SphericalHarmonics", encoding);

        std::unique_ptr<Context> model_ctx = network->forward_impl(&q, input, &output_float);
        // std::cout << "out width: " << network->output_width() << std::endl;
        // std::cout << "Out Spherical" << std::endl;
        // output_float.print();
        q.wait();
        CHECK(check_output_non_zero(output_float));

        std::vector<float> out = output_float.to_cpu_vector();
        std::vector<float> reference_out = {0.2821, 1.14, -0.8143, 1.466, 7.648, -4.249};

        double epsilon = 1e-3; // Set the tolerance for floating-point comparisons

        // Check if the actual vector is equal to the expected vector within the tolerance
        for (size_t i = 0; i < out.size(); ++i) {
            // std::cout << reference_out[i] << "/" << out[i] << std::endl;
            CHECK(std::abs(reference_out[i] - out[i]) < epsilon);
        }
    }
}

TEST_CASE("tinydpcppnn::encoding Grid Encoding") {
    SUBCASE("Not padded") {
        // SWIFTNET
        int input_width = 3;
        const int batch_size = 1;
        queue q;
        DeviceMem<float> input_float(input_width * batch_size, q);
        //   input_float.initialize_arange(q);
        input_float.initialize_constant(1.0f, q);
        GPUMatrix<float> input(input_float.data(), input_width, batch_size);

        GPUMatrix<float> output_float(32, batch_size);

        output_float.initialize_constant(0.0f);

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

        DeviceMem<float> params_full_precision(network->n_params(), q);

        params_full_precision.initialize_arange(q);

        network->set_params(params_full_precision.data(), params_full_precision.data(), nullptr);

        std::unique_ptr<Context> model_ctx = network->forward_impl(&q, input, &output_float);
        q.wait();

        CHECK(check_output_non_zero(output_float));

        std::vector<float> out = output_float.to_cpu_vector();
        std::vector<float> reference_out = {-1,      -1,      -0.9985, -0.9985, -0.98,   -0.98,   -0.8076, -0.8076,
                                            -0.6606, -0.6606, -0.5107, -0.5107, -0.4202, -0.4202, -0.2527, -0.2527,
                                            -0.1031, -0.1031, 0.06964, 0.06964, 0.1893,  0.1893,  0.2996,  0.2996,
                                            0.4565,  0.4565,  0.6128,  0.6128,  0.7783,  0.7783,  0.9258,  0.9258};

        double epsilon = 1e-4; // Set the tolerance for floating-point comparisons

        // Check if the actual vector is equal to the expected vector within the tolerance
        for (size_t i = 0; i < out.size(); ++i) {
            CHECK(std::abs(reference_out[i] - out[i]) < epsilon);
        }
    }
    SUBCASE("Check results loaded") {
        // SWIFTNET
        int input_width = 2;
        const int batch_size = 8;
        queue q;
        DeviceMem<float> input_float(input_width * batch_size, q);
        input_float.initialize_constant(0.0f, q);
        GPUMatrix<float> input(input_float.data(), input_width, batch_size);

        GPUMatrix<float> output_float(32, batch_size);

        output_float.initialize_constant(0.0f);

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
        // Encoding<float> *network = create_encoding<float>("Grid", encoding);
        // std::cout << "N params: " << network->n_params() << std::endl;
        DeviceMem<float> params_full_precision(network->n_params(), q);

        std::vector<float> params =
            loadVectorFromCSV<float>("../test/ref_values/network_with_grid_encoding/full/encoding_params.csv");
        std::vector<float> input_ref =
            loadVectorFromCSV<float>("../test/ref_values/network_with_grid_encoding/full/input.csv");
        std::cout << "Loaded n_params: " << params.size() << ", encoding n_params: " << network->n_params()
                  << std::endl;
        CHECK(params.size() == network->n_params());

        params_full_precision.copy_from_host(params, q);
        input_float.copy_from_host(input_ref, q);

        network->set_params(params_full_precision.data(), params_full_precision.data(), nullptr);

        // std::cout << "In" << std::endl;
        // input.print();
        std::unique_ptr<Context> model_ctx = network->forward_impl(&q, input, &output_float);
        q.wait();

        // std::cout << "Out" << std::endl;
        // output_float.print();

        CHECK(check_output_non_zero(output_float));

        std::vector<float> out = output_float.to_cpu_vector();
        std::vector<float> reference_out =
            loadVectorFromCSV<float>("../test/ref_values/network_with_grid_encoding/full/encoding_output.csv");

        double epsilon = 1e-4; // Set the tolerance for floating-point comparisons

        // Check if the actual vector is equal to the expected vector within the tolerance
        for (size_t i = 0; i < out.size(); ++i) {
            // std::cout << i << ", ref:" << reference_out[i] << ", val: " << out[i] << std::endl;
            CHECK(std::abs(reference_out[i] - out[i]) < epsilon);
        }
    }
}