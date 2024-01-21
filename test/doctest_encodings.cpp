/**
 * @file doctest_encodings.cpp
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief File with test of the encodings.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "doctest/doctest.h"
#include "result_check.h"
#include <iostream>
#include <vector>

#include "encoding_factory.h"

using bf16 = sycl::ext::oneapi::bfloat16;
using tinydpcppnn::encodings::grid::GridEncoding;
using json = nlohmann::json;

json loadJsonConfig(const std::string &filename) {
    std::ifstream file{filename};
    if (!file) {
        throw std::runtime_error("Error: Unable to open file '" + filename + "'");
    }
    return json::parse(file, nullptr, true, /*skip_comments=*/true);
}

template <typename T> void initialize_arange(std::vector<T> &vec) {

    // Repeat the col_vector and perform the operations
    double offset = (double)vec.size() / 2.0;

    for (long long i = 0; i < vec.size(); i++) {
        vec[i] = static_cast<T>(i / offset - 1.0);
    }
}

template <typename T>
void test_encoding_from_loaded_file(const int batch_size, const int input_width, const int output_width,
                                    std::string filepath, sycl::queue &q) {
    DeviceMatrix<float> input(batch_size, input_width, q);
    input.fill(0.0f).wait();

    DeviceMatrix<T> output(batch_size, output_width, q);
    output.fill(0.0f).wait();
    json config = loadJsonConfig(filepath + "/encoding_config.json");
    config[EncodingParams::N_DIMS_TO_ENCODE] = input_width;

    std::shared_ptr<Encoding<T>> encoding = create_encoding<T>(config);
    encoding->set_padded_output_width(output_width);

    std::vector<T> params = loadVectorFromCSV<T>(filepath + "encoding_params.csv");
    std::vector<float> input_ref = loadVectorFromCSV<float>(filepath + "input_encoding.csv");
    std::vector<T> output_ref = loadVectorFromCSV<T>(filepath + "output_encoding.csv");

    DeviceMem<T> params_full_precision(params.size(), q);

    if (params.size()) {
        CHECK(params.size() == encoding->n_params());

        params_full_precision.copy_from_host(params).wait();

        encoding->set_params(params_full_precision.data(), params_full_precision.data(), nullptr);
    }

    CHECK(input_ref.size() == input.size());
    CHECK(output_ref.size() == output.size());

    input.copy_from_host(input_ref).wait();

    std::unique_ptr<Context> model_ctx = encoding->forward_impl(&q, input, &output);
    q.wait();

    std::vector<T> out = output.copy_to_host();
    const T epsilon = 1e-2; // Set the tolerance for floating-point comparisons

    // Check if the actual vector is equal to the expected vector within the tolerance
    for (size_t i = 0; i < out.size(); ++i) {
        CHECK(static_cast<float>(out[i]) == doctest::Approx(output_ref[i]).epsilon(epsilon));
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
        const json encoding_config{{EncodingParams::N_DIMS_TO_ENCODE, input_width},
                                   {EncodingParams::SCALE, 1.0},
                                   {EncodingParams::OFFSET, 0.0},
                                   {EncodingParams::ENCODING, EncodingNames::IDENTITY}};
        std::shared_ptr<Encoding<float>> network = create_encoding<float>(encoding_config);
        network->set_padded_output_width(output_width);

        std::unique_ptr<Context> model_ctx = network->forward_impl(&q, input, &output_float);
        q.wait();

        std::vector<float> in = input.copy_to_host();
        std::vector<float> out = output_float.copy_to_host();

        const float epsilon = 1e-3; // Set the tolerance for floating-point comparisons

        // Check if the actual vector is equal to the expected vector within the tolerance
        for (size_t i = 0; i < out.size(); ++i) {
            CHECK(static_cast<float>(in[i]) == doctest::Approx(out[i]).epsilon(epsilon));
        }
    }

    SUBCASE("Check results loaded float") {
        // SWIFTNET
        const int input_width = 3;
        const int batch_size = 64;
        const int output_width = 3;
        sycl::queue q;

        std::string filepath = "../test/ref_values/encoding/identity/";
        test_encoding_from_loaded_file<float>(batch_size, input_width, output_width, filepath, q);
    }

    SUBCASE("Check results loaded bf16") {
        // SWIFTNET
        const int input_width = 3;
        const int batch_size = 64;
        const int output_width = 3;
        sycl::queue q;

        std::string filepath = "../test/ref_values/encoding/identity/";
        test_encoding_from_loaded_file<bf16>(batch_size, input_width, output_width, filepath, q);
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

        const json encoding_config{{EncodingParams::N_DIMS_TO_ENCODE, input_width},
                                   {EncodingParams::DEGREE, DEGREE},
                                   {EncodingParams::ENCODING, EncodingNames::SPHERICALHARMONICS}};
        std::shared_ptr<Encoding<float>> network = create_encoding<float>(encoding_config);
        network->set_padded_output_width(output_float.n());
        std::unique_ptr<Context> model_ctx = network->forward_impl(&q, input, &output_float);
        q.wait();

        std::vector<float> out = output_float.copy_to_host();
        const std::vector<float> reference_out = {0.2821, 1.0, 1.0, 0.2821, 1.0, 1.0};

        const double epsilon = 1e-3;
        // Check if the actual vector is equal to the expected vector within the tolerance
        CHECK(areVectorsWithinTolerance(out, reference_out, epsilon));
    }
    SUBCASE("Check results loaded float") {
        // SWIFTNET
        const int input_width = 3;
        const int batch_size = 64;
        const int output_width = 16;
        sycl::queue q;

        std::string filepath = "../test/ref_values/encoding/spherical/";
        test_encoding_from_loaded_file<float>(batch_size, input_width, output_width, filepath, q);
    }
}

TEST_CASE("tinydpcppnn::encoding Grid Encoding") {
    SUBCASE("Test grid encoding using create_grid_encoding instead of factory") {
        // SWIFTNET
        const int input_width = 3;
        const int batch_size = 1;
        const int padded_output_width = 32;
        sycl::queue q;

        DeviceMatrix<float> input(batch_size, input_width, q);
        input.fill(1.0f).wait();

        DeviceMatrix<float> output_float(batch_size, padded_output_width, q);
        output_float.fill(1.23f).wait(); // fill with something to check if it is written to

        const json encoding_config{{EncodingParams::N_DIMS_TO_ENCODE, input_width},
                                   {EncodingParams::ENCODING, EncodingNames::GRID},
                                   {EncodingParams::GRID_TYPE, "Hash"},
                                   {EncodingParams::N_LEVELS, 16},
                                   {EncodingParams::N_FEATURES_PER_LEVEL, 2},
                                   {EncodingParams::LOG2_HASHMAP_SIZE, 19},
                                   {EncodingParams::BASE_RESOLUTION, 16},
                                   {EncodingParams::PER_LEVEL_SCALE, 2.0}};

        std::shared_ptr<GridEncoding<float>> network =
            tinydpcppnn::encodings::grid::create_grid_encoding<float>(encoding_config);
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

        // Check if the actual vector is equal to the expected vector within the tolerance
        CHECK(areVectorsWithinTolerance(out, reference_out, 1.0e-3));
    }

    SUBCASE("Check results loaded float small grid") {
        // SWIFTNET
        const int input_width = 2;
        const int batch_size = 64;
        const int output_width = 32;
        sycl::queue q;

        std::string filepath = "../test/ref_values/encoding/grid/";

        // Check if the file exists
        if (!std::filesystem::exists(filepath + "encoding_params.csv")) {
            // TODO: any good solution here? E.g., a download script or sth
            std::cout << "encoding_params.csv doesn't exist as it's quite large for grid encoding." << std::endl;
        } else {
            test_encoding_from_loaded_file<float>(batch_size, input_width, output_width, filepath, q);
        }
    }
    SUBCASE("Check results loaded float. Large grid") {
        // SWIFTNET
        const int input_width = 2;
        const int batch_size = 128;
        const int output_width = 32;
        sycl::queue q;

        std::string filepath = "../test/ref_values/network_with_grid_encoding/HashGrid/";

        // Check if the file exists
        if (!std::filesystem::exists(filepath + "encoding_params.csv")) {
            // TODO: any good solution here? E.g., a download script or sth
            std::cout << "encoding_params.csv doesn't exist as it's quite large for grid encoding." << std::endl;
        } else {
            test_encoding_from_loaded_file<float>(batch_size, input_width, output_width, filepath, q);
        }
    }
}