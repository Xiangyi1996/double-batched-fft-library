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
#include "io.h"
#include "network_with_encodings.h"
#include "result_check.h"

using namespace sycl;
using bf16 = sycl::ext::oneapi::bfloat16;
using tinydpcppnn::encodings::grid::GridEncoding;
using json = nlohmann::json;

/// Function which applies a grid encoding to a R2 vector, resulting in a vector of size
/// network_input_width, then applies the network and the output is the network_output_width
// ATTENTION: currently only works for WIDTH=64
template <typename T_enc, typename T_net, int WIDTH = 64>
void test_network_with_encoding_loaded(sycl::queue &q, std::string filepath, const int n_hidden_layers,
                                       const int batch_size, const int unpadded_output_width,
                                       const int encoding_input_width) {
    static_assert(WIDTH == 64);
    constexpr int input_width = WIDTH;
    constexpr int output_width = WIDTH;
    constexpr int encoding_output_width = input_width;

    json encoding_config = loadJsonConfig(filepath + "encoding_config.json");
    encoding_config[EncodingParams::N_DIMS_TO_ENCODE] = encoding_input_width;

    auto Net = create_network_with_encoding<T_enc, T_net, WIDTH>(q, input_width, unpadded_output_width, n_hidden_layers,
                                                                 Activation::ReLU, Activation::None, encoding_config);

    std::vector<T_enc> encoding_params = loadVectorFromCSV<T_enc>(filepath + "encoding_params.csv");

    DeviceMem<T_enc> params_full_precision(Net->get_encoding()->n_params(), q);
    if (encoding_params.size()) {
        assert(encoding_params.size() == Net->get_encoding()->n_params());
        params_full_precision.copy_from_host(encoding_params).wait();
        Net->get_encoding()->set_params(params_full_precision.data(), params_full_precision.data(), nullptr);
    }

    Net->get_encoding()->set_padded_output_width(encoding_output_width);
    std::vector<T_net> network_weights_ref = load_weights_as_packed_from_file<T_net, WIDTH>(
        filepath + "network_params.csv", n_hidden_layers, input_width, output_width);
    Net->get_network()->set_weights_matrices(network_weights_ref);
    DeviceMatrix<float> input_encoding(batch_size, encoding_input_width, q);
    std::vector<float> input_encoding_ref = loadVectorFromCSV<float>(filepath + "input_encoding.csv");
    input_encoding.copy_from_host(input_encoding_ref);
    q.wait();

    DeviceMatrix<T_enc> output_encoding(batch_size, encoding_output_width, q);
    CHECK(encoding_output_width == input_width);
    DeviceMatrix<T_net> input_network(batch_size, input_width, q);

    DeviceMatrix<T_net> output_network = Net->GenerateForwardOutputMatrix(batch_size);

    Net->inference(input_encoding, input_network, output_encoding, output_network, {});
    q.wait();
    std::vector<T_net> output_network_vec(output_network.size());
    std::vector<T_enc> output_encoding_vec(output_encoding.size());

    std::vector<T_enc> encoding_output_ref = loadVectorFromCSV<T_enc>(filepath + "output_encoding.csv");
    output_encoding.copy_to_host(output_encoding_vec).wait();
    CHECK(areVectorsWithinTolerance(output_encoding_vec, encoding_output_ref, 2.0e-2));

    std::vector<T_net> network_output_ref = loadVectorFromCSV<T_net>(filepath + "output_network.csv");
    output_network.copy_to_host(output_network_vec).wait();
    CHECK(areVectorsWithinTolerance(output_network_vec, network_output_ref, 2.0e-2));
}

void test_network_with_encoding_identity_inference(sycl::queue &q) {
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
    auto Net = create_network_with_encoding<float, bf16, WIDTH>(q, input_width, output_width, n_hidden_layers,
                                                                Activation::ReLU, Activation::None, encoding_config);

    const bf16 weight_val = 0.01;
    std::vector<bf16> new_weights(Net->get_network()->get_weights_matrices().nelements(), weight_val);

    Net->get_network()->set_weights_matrices(new_weights);

    constexpr float input_val = 1.0f;
    DeviceMatrix<float> input_encoding(batch_size, encoding_input_width, q);
    input_encoding.fill(input_val).wait();

    DeviceMatrix<float> output_encoding = Net->GenerateEncodingOutputMatrix(batch_size);
    output_encoding.fill(0.0f).wait();
    DeviceMatrix<bf16> input_network(batch_size, input_width, q);

    DeviceMatrix<bf16> output_network = Net->GenerateForwardOutputMatrix(batch_size);
    output_network.fill(1.234f).wait();

    Net->inference(input_encoding, input_network, output_encoding, output_network, {});
    q.wait();

    CHECK(isVectorWithinTolerance(output_network.copy_to_host(),
                                  input_val * std::pow(WIDTH * (double)weight_val, n_hidden_layers + 1), 1.0e-3));
}

void test_network_with_encoding_identity_forward_backward(sycl::queue &q) {
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
    auto Net = create_network_with_encoding<float, bf16, WIDTH>(q, input_width, output_width, n_hidden_layers,
                                                                Activation::ReLU, Activation::None, encoding_config);

    const bf16 weight_val = 0.01;
    std::vector<bf16> new_weights(Net->get_network()->get_weights_matrices().nelements(), weight_val);

    Net->get_network()->set_weights_matrices(new_weights);

    constexpr float input_val = 1.0f;
    DeviceMatrix<float> input_encoding(batch_size, encoding_input_width, q);
    input_encoding.fill(input_val).wait();

    DeviceMatrix<float> output_encoding = Net->GenerateEncodingOutputMatrix(batch_size);
    output_encoding.fill(0.0f).wait();
    DeviceMatrix<bf16> input_network(batch_size, input_width, q);

    DeviceMatrices<bf16> interm_forw(
        Net->get_network()->get_n_hidden_layers() + 2, batch_size, Net->get_network()->get_input_width(), batch_size,
        Net->get_network()->get_network_width(), batch_size, Net->get_network()->get_output_width(), q);

    Net->forward_pass(input_encoding, input_network, output_encoding, interm_forw, {});
    q.wait();
    std::vector<bf16> interm_forw_vec = interm_forw.copy_to_host();
    std::vector<bf16> output_network(interm_forw_vec.end() - (batch_size * output_width), interm_forw_vec.end());

    CHECK(isVectorWithinTolerance(output_network, input_val * std::pow(WIDTH * (double)weight_val, n_hidden_layers + 1),
                                  1.0e-3));
}
// Create a shared pointer of network with encoding using create_network_with_encoding
template <typename T_enc, typename T_net, int WIDTH = 64>
std::shared_ptr<NetworkWithEncoding<T_enc, T_net>>
test_create_network_with_encoding_as_shared_ptr(sycl::queue &q, const int encoding_input_width,
                                                const json &encoding_config) {

    static_assert(WIDTH == 64);
    constexpr int n_hidden_layers = 1;
    constexpr int input_width = WIDTH;
    constexpr int output_width = WIDTH;
    constexpr int unpadded_output_width = 1;
    Activation activation = Activation::ReLU;
    Activation output_activation = Activation::None;

    std::shared_ptr<NetworkWithEncoding<T_enc, T_net>> network_with_encoding_shared_ptr =
        create_network_with_encoding<T_enc, T_net, WIDTH>(q, encoding_input_width, unpadded_output_width,
                                                          n_hidden_layers, activation, output_activation,
                                                          encoding_config);
    q.wait();

    assert(input_width == network_with_encoding_shared_ptr->get_network()->get_input_width());
    assert(output_width == network_with_encoding_shared_ptr->get_network()->get_output_width());
    return network_with_encoding_shared_ptr;
}

TEST_CASE("tinydpcppnn::network_with_encoding step-by-step") {
    sycl::queue q(gpu_selector_v);
    SUBCASE("Create network_with_encoding as shared_ptr") {
        const int encoding_input_width = 64;

        const json encoding_config{{EncodingParams::N_DIMS_TO_ENCODE, encoding_input_width},
                                   {EncodingParams::SCALE, 1.0},
                                   {EncodingParams::OFFSET, 0.0},
                                   {EncodingParams::ENCODING, EncodingNames::IDENTITY}};
        test_create_network_with_encoding_as_shared_ptr<float, bf16, 64>(q, encoding_input_width, encoding_config);
    }
    SUBCASE("Identity encoding inference") { test_network_with_encoding_identity_inference(q); }
    SUBCASE("Identity encoding fwd") { test_network_with_encoding_identity_forward_backward(q); }

    // #ifdef TEST_PATH

    //     SUBCASE("Grid encoding inference, loaded data") {
    //         std::string filepath =
    //             std::string(TEST_PATH) + "/tiny-dpcpp-data/ref_values/network_with_grid_encoding/HashGrid/";
    //         const int n_hidden_layers = 2;
    //         const int batch_size = 128;
    //         const int unpadded_output_width = 1;
    //         const int encoding_input_width = 2;
    //         test_network_with_encoding_loaded<float, bf16, 64>(q, filepath, n_hidden_layers, batch_size,
    //                                                            unpadded_output_width, encoding_input_width);
    //     }
    //     SUBCASE("Identity encoding inference, loaded data") {
    //         std::string filepath =
    //             std::string(TEST_PATH) + "/tiny-dpcpp-data/ref_values/network_with_grid_encoding/Identity/";
    //         const int n_hidden_layers = 2;
    //         const int batch_size = 128;
    //         const int unpadded_output_width = 64;
    //         const int encoding_input_width = 64;
    //         test_network_with_encoding_loaded<float, bf16, 64>(q, filepath, n_hidden_layers, batch_size,
    //                                                            unpadded_output_width, encoding_input_width);
    //     }
    // #endif
}