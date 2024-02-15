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
#include "l2.h"
#include "mlp.h"
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

    json encoding_config = io::loadJsonConfig(filepath + "encoding_config.json");
    encoding_config[EncodingParams::N_DIMS_TO_ENCODE] = encoding_input_width;

    auto Net = create_network_with_encoding<T_enc, T_net, WIDTH>(q, input_width, unpadded_output_width, n_hidden_layers,
                                                                 Activation::ReLU, Activation::None, encoding_config);

    std::vector<T_enc> encoding_params = io::loadVectorFromCSV<T_enc>(filepath + "encoding_params.csv");

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
    std::vector<float> input_encoding_ref = io::loadVectorFromCSV<float>(filepath + "input_encoding.csv");
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

    std::vector<T_enc> encoding_output_ref = io::loadVectorFromCSV<T_enc>(filepath + "output_encoding.csv");
    output_encoding.copy_to_host(output_encoding_vec).wait();
    CHECK(areVectorsWithinTolerance(output_encoding_vec, encoding_output_ref, 2.0e-2));

    std::vector<T_net> network_output_ref = io::loadVectorFromCSV<T_net>(filepath + "output_network.csv");
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

template <typename T, int WIDTH>
void test_network_with_encoding_identity_forward(sycl::queue &q, const int input_width, const int output_width,
                                                 const int n_hidden_layers, const int batch_size,
                                                 std::string activation, std::string weight_init_mode) {
    // main functionalities of backward and forward are tested in doctest_swifnet
    // here, we test only if the combination of encoding (tested in doctest_encoding) and swifnet works
    constexpr int encoding_input_width = WIDTH;
    constexpr int encoding_output_width = WIDTH;

    const float input_val = 1.0f;
    const float target_val = 0.1;
    std::vector<float> input_ref(input_width, input_val);

    CHECK(input_width == output_width); // this is not a hard requirement, but currently the loop over the
                                        // mlp reference (batch size = 1) assumes this. if this is changed, ensure the
                                        // checks are still correct
    CHECK(input_width == WIDTH);
    Activation network_activation;
    if (activation == "relu") {
        network_activation = Activation::ReLU;
    } else if (activation == "linear") {
        network_activation = Activation::None;
    }
    // Define the parameters for creating IdentityEncoding
    const json encoding_config{{EncodingParams::N_DIMS_TO_ENCODE, encoding_input_width},
                               {EncodingParams::SCALE, 1.0},
                               {EncodingParams::OFFSET, 0.0},
                               {EncodingParams::ENCODING, EncodingNames::IDENTITY}};

    auto Net = create_network_with_encoding<float, bf16, WIDTH>(q, input_width, output_width, n_hidden_layers,
                                                                network_activation, Activation::None, encoding_config);

    MLP<float> mlp(input_width, WIDTH, output_width, n_hidden_layers + 1, batch_size, activation, "linear",
                   weight_init_mode);
    std::vector<T> unpacked_weights = convert_vector<float, T>(mlp.getUnpackedWeights());

    Net->get_network()->set_weights_matrices(
        io::get_packed_weights<T, WIDTH>(unpacked_weights, n_hidden_layers, input_width, output_width));

    DeviceMatrix<float> input_encoding(batch_size, encoding_input_width, q);

    // Repeat source vector N times
    std::vector<float> input_full = stack_vector(input_ref, batch_size);
    input_encoding.copy_from_host(input_full).wait();

    DeviceMatrix<float> output_encoding = Net->GenerateEncodingOutputMatrix(batch_size);
    output_encoding.fill(0.0f).wait();
    DeviceMatrix<T> input_network(batch_size, input_width, q);

    DeviceMatrices<T> interm_forw(
        Net->get_network()->get_n_hidden_layers() + 2, batch_size, Net->get_network()->get_input_width(), batch_size,
        Net->get_network()->get_network_width(), batch_size, Net->get_network()->get_output_width(), q);

    DeviceMatrices<T> interm_backw(
        Net->get_network()->get_n_hidden_layers() + 1, batch_size, Net->get_network()->get_network_width(), batch_size,
        Net->get_network()->get_network_width(), batch_size, Net->get_network()->get_output_width(), q);

    DeviceMatrices<T> network_backward_output(Net->get_network()->get_n_hidden_layers() + 1,
                                              Net->get_network()->get_network_width(), WIDTH, WIDTH, WIDTH, WIDTH,
                                              Net->get_network()->get_output_width(), q);

    Net->forward_pass(input_encoding, input_network, output_encoding, interm_forw, {});
    q.wait();

    std::vector<std::vector<float>> fwd_result_ref = mlp.forward(input_ref, true);
    auto interm_forw_ref = repeat_inner_vectors<float>(fwd_result_ref, batch_size);
    std::vector<T> interm_forw_vec = interm_forw.copy_to_host();

    CHECK(interm_forw_vec.size() == interm_forw_ref.size());
    CHECK(areVectorsWithinTolerance(interm_forw_vec, interm_forw_ref, 1.0e-2));
}

template <typename T, int WIDTH>
void test_network_with_encoding_identity_backward(sycl::queue &q, const int input_width, const int output_width,
                                                  const int n_hidden_layers, const int batch_size,
                                                  std::string activation, std::string weight_init_mode) {
    // main functionalities of backward and forward are tested in doctest_swifnet
    // here, we test only if the combination of encoding (tested in doctest_encoding) and swifnet works

    constexpr int encoding_input_width = WIDTH;
    constexpr int encoding_output_width = WIDTH;

    const float input_val = 1.0f;
    const float target_val = 0.1;
    std::vector<float> input_ref(input_width, input_val);
    std::vector<float> target_ref(output_width, target_val);

    CHECK(input_width == output_width); // this is not a hard requirement, but currently the loop over the
    // mlp reference (batch size = 1) assumes this. if this is changed, ensure checks are still correct
    CHECK(input_width == WIDTH);
    Activation network_activation;
    if (activation == "relu") {
        network_activation = Activation::ReLU;
    } else if (activation == "linear") {
        network_activation = Activation::None;
    }
    // Define the parameters for creating IdentityEncoding
    const json encoding_config{{EncodingParams::N_DIMS_TO_ENCODE, encoding_input_width},
                               {EncodingParams::SCALE, 1.0},
                               {EncodingParams::OFFSET, 0.0},
                               {EncodingParams::ENCODING, EncodingNames::IDENTITY}};

    MLP<float> mlp(input_width, WIDTH, output_width, n_hidden_layers + 1, batch_size, activation, "linear", "random");
    auto Net = create_network_with_encoding<float, T, WIDTH>(q, input_width, output_width, n_hidden_layers,
                                                             network_activation, Activation::None, encoding_config);

    std::vector<T> unpacked_weights = convert_vector<float, T>(mlp.getUnpackedWeights());
    Net->get_network()->set_weights_matrices(
        io::get_packed_weights<T, WIDTH>(unpacked_weights, n_hidden_layers, input_width, output_width));

    std::vector<Matrix<float>> grad_matrices_ref(n_hidden_layers + 1, Matrix<float>(1, 1));
    std::vector<std::vector<float>> loss_grads_ref;
    std::vector<float> loss_ref;

    mlp.backward(input_ref, target_ref, grad_matrices_ref, loss_grads_ref, loss_ref, 1.0);

    DeviceMatrices<T> interm_forw(
        Net->get_network()->get_n_hidden_layers() + 2, batch_size, Net->get_network()->get_input_width(), batch_size,
        Net->get_network()->get_network_width(), batch_size, Net->get_network()->get_output_width(), q);

    DeviceMatrices<T> interm_backw(
        Net->get_network()->get_n_hidden_layers() + 1, batch_size, Net->get_network()->get_network_width(), batch_size,
        Net->get_network()->get_network_width(), batch_size, Net->get_network()->get_output_width(), q);

    DeviceMatrices<T> network_backward_output(Net->get_network()->get_n_hidden_layers() + 1,
                                              Net->get_network()->get_network_width(), WIDTH, WIDTH, WIDTH, WIDTH,
                                              Net->get_network()->get_output_width(), q);
    DeviceMatrix<T> dL_doutput(batch_size, Net->get_network()->get_output_width(), q);

    network_backward_output.fill(0.0).wait();
    dL_doutput.fill(0.0).wait();

    std::vector<std::vector<float>> fwd_result_ref = mlp.forward(input_ref, true);
    std::vector<float> interm_forw_ref = repeat_inner_vectors<float>(fwd_result_ref, batch_size);
    interm_forw.copy_from_host(convert_vector<float, T>(interm_forw_ref)).wait();

    std::vector<float> stacked_loss_grads_back_ref = stack_vector(loss_grads_ref.back(), batch_size);
    dL_doutput.copy_from_host(convert_vector<float, T>(stacked_loss_grads_back_ref)).wait();

    Net->get_network()->backward_pass(dL_doutput, network_backward_output, interm_backw, interm_forw, {});
    q.wait();
    std::vector<T> interm_backw_vec = interm_backw.copy_to_host();
    q.wait();

    for (int i = 0; i < loss_grads_ref.size(); i++) {
        std::vector<float> interm_backw_ref;
        std::vector<T> interm_backw_sliced_actual(interm_backw_vec.begin() + i * batch_size * WIDTH,
                                                  interm_backw_vec.begin() + i * batch_size * WIDTH +
                                                      batch_size * WIDTH);
        auto inner_stacked = stack_vector(loss_grads_ref[i], batch_size);
        for (T value : inner_stacked) {
            interm_backw_ref.push_back(value); // Add each element to the flattened vector
        }

        bool interm_backw_within_tolerance =
            areVectorsWithinTolerance(interm_backw_sliced_actual, interm_backw_ref, 1.0e-2);
        if (!interm_backw_within_tolerance) {
            printVector("interm_backw_ref: ", interm_backw_ref);
            printVector("interm_backw_vec: ", interm_backw_sliced_actual);
        }
        CHECK(interm_backw_within_tolerance);
    }
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

TEST_CASE("Network with Identity Encoding - test fwd") {
    sycl::queue q(sycl::gpu_selector_v);
    const int n_hidden_layers = 1;
    auto test_function = [=](sycl::queue &q, const int width, const int batch_size, std::string activation,
                             std::string weight_init_mode) {
        typedef sycl::ext::oneapi::bfloat16 T;
        if (width == 16)
            test_network_with_encoding_identity_forward<T, 16>(q, 16, 16, n_hidden_layers, batch_size, activation,
                                                               weight_init_mode);
        else if (width == 32)
            test_network_with_encoding_identity_forward<T, 32>(q, 32, 32, n_hidden_layers, batch_size, activation,
                                                               weight_init_mode);
        else if (width == 64)
            test_network_with_encoding_identity_forward<T, 64>(q, 64, 64, n_hidden_layers, batch_size, activation,
                                                               weight_init_mode);
        else if (width == 128)
            test_network_with_encoding_identity_forward<T, 128>(q, 128, 128, n_hidden_layers, batch_size, activation,
                                                                weight_init_mode);
        else
            throw std::invalid_argument("Unsupported width");
    };
    const int widths[] = {16, 32, 64, 128};
    const int batch_sizes[] = {8, 16, 32, 64};
    std::string activations[] = {"linear", "relu"};
    std::string weight_init_modes[] = {"constant", "random"};

    for (int batch_size : batch_sizes) {
        for (int width : widths) {
            for (std::string activation : activations) {
                for (std::string weight_init_mode : weight_init_modes) {
                    std::string testName = "Testing grad WIDTH " + std::to_string(width) +
                                           " - activation: " + activation + " - weight_init_mode: " + weight_init_mode +
                                           " - Batch size: " + std::to_string(batch_size);
                    SUBCASE(testName.c_str()) {
                        CHECK_NOTHROW(test_function(q, width, batch_size, activation, weight_init_mode));
                    }
                }
            }
        }
    }
}

TEST_CASE("Network with Identity Encoding - test bwd") {
    sycl::queue q(sycl::gpu_selector_v);
    const int n_hidden_layers = 1;
    auto test_function = [=](sycl::queue &q, const int width, const int batch_size, std::string activation,
                             std::string weight_init_mode) {
        typedef sycl::ext::oneapi::bfloat16 T;
        if (width == 16)
            test_network_with_encoding_identity_backward<T, 16>(q, 16, 16, n_hidden_layers, batch_size, activation,
                                                                weight_init_mode);
        else if (width == 32)
            test_network_with_encoding_identity_backward<T, 32>(q, 32, 32, n_hidden_layers, batch_size, activation,
                                                                weight_init_mode);
        else if (width == 64)
            test_network_with_encoding_identity_backward<T, 64>(q, 64, 64, n_hidden_layers, batch_size, activation,
                                                                weight_init_mode);
        else if (width == 128)
            test_network_with_encoding_identity_backward<T, 128>(q, 128, 128, n_hidden_layers, batch_size, activation,
                                                                 weight_init_mode);
        else
            throw std::invalid_argument("Unsupported width");
    };
    const int widths[] = {16, 32, 64, 128};
    const int batch_sizes[] = {8, 16, 32, 64};
    std::string activations[] = {"linear", "relu"};
    std::string weight_init_modes[] = {"constant", "random"};

    for (int batch_size : batch_sizes) {
        for (int width : widths) {
            for (std::string activation : activations) {
                for (std::string weight_init_mode : weight_init_modes) {
                    std::string testName = "Testing grad WIDTH " + std::to_string(width) +
                                           " - activation: " + activation + " - weight_init_mode: " + weight_init_mode +
                                           " - Batch size: " + std::to_string(batch_size);
                    SUBCASE(testName.c_str()) {
                        CHECK_NOTHROW(test_function(q, width, batch_size, activation, weight_init_mode));
                    }
                }
            }
        }
    }
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