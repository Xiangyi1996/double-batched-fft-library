/**
 * @file doctest_network_with_encodings.cpp
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Tests for the network with encodings class.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cmath>
#include <iostream>
#include <vector>

#include "SwiftNetMLP.h"
#include "doctest/doctest.h"
#include "json.hpp"
#include "network_with_encodings.h"
#include "result_check.h"

using namespace sycl;
using bf16 = sycl::ext::oneapi::bfloat16;
using tinydpcppnn::encodings::grid::GridEncoding;

/// Function which applies a grid encoding to a R2 vector, resulting in a vector of size
/// network_input_width, then applies the network and the output is the network_output_width
template <typename T, int WIDTH = 64> void test_network_with_encoding(sycl::queue &q) {

    static_assert(WIDTH == 64);
    constexpr int n_hidden_layers = 1;
    constexpr int batch_size = 8;
    constexpr int input_width = WIDTH;
    constexpr int output_width = WIDTH;
    constexpr int unpadded_output_width = 1;
    constexpr int encoding_input_width = 2;
    constexpr int encoding_output_width = input_width;

    const json encoding_config{{EncodingParams::N_DIMS_TO_ENCODE, encoding_input_width},
                               {EncodingParams::ENCODING, EncodingNames::GRID},
                               {EncodingParams::GRID_TYPE, GridType::Hash},
                               {EncodingParams::N_LEVELS, 16},
                               {EncodingParams::N_FEATURES_PER_LEVEL, 2},
                               {EncodingParams::LOG2_HASHMAP_SIZE, 15},
                               {EncodingParams::BASE_RESOLUTION, 16},
                               {EncodingParams::PER_LEVEL_SCALE, 1.5}};
    // encoding output size has to be at least n_levels*n_features_per_level =32
    static_assert(encoding_output_width >= 32); // TODO: generalize min encoding output size

    SwiftNetMLP<T, WIDTH> network(q, input_width, unpadded_output_width, n_hidden_layers, Activation::ReLU,
                                  Activation::None, Network<T>::WeightInitMode::constant_pos);
    q.wait();
    assert(input_width == network.get_input_width());
    assert(output_width == network.get_output_width());

    DeviceMatrix<float> input_encoding(batch_size, encoding_input_width, q);
    input_encoding.fill(1.0f).wait();

    DeviceMatrix<float> output_encoding(batch_size, encoding_output_width, q);
    output_encoding.fill(1.0f).wait();

    std::shared_ptr<GridEncoding<float>> encoding =
        tinydpcppnn::encodings::grid::create_grid_encoding<float>(encoding_config);

    std::vector<float> params =
        loadVectorFromCSV<float>("../../test/ref_values/network_with_grid_encoding/full/encoding_params.csv");
    assert(params.size() == encoding->n_params());
    DeviceMem<float> params_full_precision(encoding->n_params(), q);
    params_full_precision.copy_from_host(params).wait();
    encoding->set_params(params_full_precision.data(), params_full_precision.data(), nullptr);

    encoding->set_padded_output_width(encoding_output_width);

    std::unique_ptr<Context> model_ctx = encoding->forward_impl(&q, input_encoding, &output_encoding);
    q.wait();

    DeviceMatrix<T> inputs_network(batch_size, input_width, q);
    inputs_network.copy_from_device(output_encoding.data());

    DeviceMatrix<T> output_network(batch_size, output_width, q);
    network.inference(inputs_network, output_network, {});
    q.wait();

    std::vector<float> output_ref =
        loadVectorFromCSV<float>("../../test/ref_values/network_with_grid_encoding/full/network_output.csv");
    std::vector<float> out_ref_cut(output_ref.begin(), output_ref.begin() + output_network.size());

    CHECK(areVectorsWithinTolerance(output_network.copy_to_host(), out_ref_cut, 1.0e-3));
}

TEST_CASE("tinydpcppnn::network_with_encoding step-by-step") {
    sycl::queue q(gpu_selector_v);
    test_network_with_encoding<bf16, 64>(q);
}

TEST_CASE("tinydpcppnn::network_with_encoding class") {
    sycl::queue q(gpu_selector_v);
    constexpr int n_hidden_layers = 1;
    constexpr int WIDTH = 64;
    constexpr int batch_size = 8;
    constexpr int input_width = WIDTH;
    constexpr int output_width = WIDTH;
    constexpr int encoding_input_width = 64;
    constexpr int encoding_output_width = input_width;

    // Define the parameters for creating IdentityEncoding
    const json encoding_config{{EncodingParams::N_DIMS_TO_ENCODE, encoding_input_width},
                               {EncodingParams::SCALE, 1.0},
                               {EncodingParams::OFFSET, 0.0},
                               {EncodingParams::ENCODING, EncodingNames::IDENTITY}};
    auto Net = create_network_with_encoding<bf16, WIDTH>(q, input_width, output_width, n_hidden_layers,
                                                         Activation::ReLU, Activation::None, encoding_config);

    const bf16 weight_val = 0.01;
    std::vector<bf16> new_weights(Net->get_network()->get_weights_matrices().nelements(), weight_val);
    Net->get_network()->set_weights_matrices(new_weights);

    constexpr float input_val = 1.0f;
    DeviceMatrix<float> input_encoding(batch_size, encoding_input_width, q);
    input_encoding.fill(input_val).wait();

    DeviceMatrix<bf16> output_encoding = Net->GenerateEncodingOutputMatrix(batch_size);
    output_encoding.fill(0.0f).wait();

    DeviceMatrix<bf16> output_network = Net->GenerateForwardOutputMatrix(batch_size);
    output_network.fill(1.234f).wait();

    Net->inference(input_encoding, output_encoding, output_network, {});
    q.wait();

    CHECK(isVectorWithinTolerance(output_network.copy_to_host(),
                                  input_val * std::pow(WIDTH * (double)weight_val, n_hidden_layers + 1), 1.0e-3));
}