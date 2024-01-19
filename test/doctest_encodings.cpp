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
        const json encoding_config{{EncodingParams::N_DIMS_TO_ENCODE, input_width},
                                   {EncodingParams::SCALE, 1.0},
                                   {EncodingParams::OFFSET, 0.0},
                                   {EncodingParams::ENCODING, EncodingNames::IDENTITY}};
        std::shared_ptr<Encoding<float>> network = create_encoding<float>(encoding_config);
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

        const json encoding_config{
            {EncodingParams::N_DIMS_TO_ENCODE, input_width}, {EncodingParams::ENCODING, EncodingNames::GRID},
            {EncodingParams::GRID_TYPE, GridType::Hash},     {EncodingParams::N_LEVELS, 16},
            {EncodingParams::N_FEATURES_PER_LEVEL, 2},       {EncodingParams::LOG2_HASHMAP_SIZE, 19},
            {EncodingParams::BASE_RESOLUTION, 16},           {EncodingParams::PER_LEVEL_SCALE, 2.0}};

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

        const json encoding_config{
            {EncodingParams::N_DIMS_TO_ENCODE, input_width}, {EncodingParams::ENCODING, EncodingNames::GRID},
            {EncodingParams::GRID_TYPE, GridType::Hash},     {EncodingParams::N_LEVELS, 16},
            {EncodingParams::N_FEATURES_PER_LEVEL, 2},       {EncodingParams::LOG2_HASHMAP_SIZE, 15},
            {EncodingParams::BASE_RESOLUTION, 16},           {EncodingParams::PER_LEVEL_SCALE, 1.5}};

        std::shared_ptr<GridEncoding<float>> network =
            tinydpcppnn::encodings::grid::create_grid_encoding<float>(encoding_config);
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

        std::vector<float> reference_out =
            loadVectorFromCSV<float>("../../test/ref_values/network_with_grid_encoding/full/encoding_output.csv");

        // Check if the actual vector is equal to the expected vector within the tolerance
        CHECK(areVectorsWithinTolerance(output_float.copy_to_host(), reference_out, 1.0e-3));
    }
}