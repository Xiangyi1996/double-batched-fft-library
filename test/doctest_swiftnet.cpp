// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "doctest/doctest.h"
#include <filesystem>

#include "SwiftNetMLP.h"
#include "result_check.h"

template <typename T, int WIDTH>
void test_inference_1layer(sycl::queue &q, const int input_width, const int output_width, const int batch_size) {

    constexpr int n_hidden_layers = 1;
    constexpr float input_val = 1.0f;
    // setting Network<T>::WeightInitMode::constant_pos sets the weights to this value
    constexpr float weight_val = 0.01f;

    SwiftNetMLP<T, WIDTH> network(q, input_width, output_width, n_hidden_layers, Activation::ReLU, Activation::None,
                                  Network<T>::WeightInitMode::constant_pos);

    DeviceMatrix<T> network_output(batch_size, network.get_output_width(), q);
    DeviceMatrix<T> network_input(batch_size, network.get_input_width(), q);

    network_input.fill(input_val);

    network.inference(network_input, network_output, {});

    q.wait();

    std::vector<double> out_ref(network_output.size());
    for (int output_idx = 0; output_idx < out_ref.size(); output_idx++) {

        const int nonzero_value = (output_idx % network.get_output_width()) < output_width ? 1 : 0;
        out_ref[output_idx] =
            nonzero_value * weight_val * input_width * input_val * network.get_network_width() * weight_val;
    }
    CHECK(areVectorsWithinTolerance(network_output.copy_to_host(), out_ref, 1e-2));
}

template <typename T, int WIDTH>
void test_forward_1layer(sycl::queue &q, const int input_width, const int output_width, const int batch_size) {

    constexpr int n_hidden_layers = 1;
    constexpr float input_val = 1.0f;
    // setting Network<T>::WeightInitMode::constant_pos sets the weights to this value
    constexpr float weight_val = 0.01f;

    SwiftNetMLP<T, WIDTH> network(q, input_width, output_width, n_hidden_layers, Activation::ReLU, Activation::None,
                                  Network<T>::WeightInitMode::constant_pos);

    DeviceMatrices<T> network_interm_forw(network.get_n_hidden_layers() + 2, batch_size, network.get_input_width(),
                                          batch_size, network.get_network_width(), batch_size,
                                          network.get_output_width(), q);
    DeviceMatrix<T> network_input(batch_size, network.get_input_width(), q);

    network_input.fill(input_val);

    network.forward_pass(network_input, network_interm_forw, {});

    q.wait();

    std::vector<T> fwd_host = network_interm_forw.copy_to_host();

    // input
    CHECK(areVectorsWithinTolerance(
        std::vector<T>(fwd_host.begin(), fwd_host.begin() + batch_size * network.get_input_width()),
        std::vector<T>(batch_size * network.get_input_width(), input_val), 1e-3));

    // intermediate
    const double ref_result = weight_val * input_width * input_val;
    CHECK(areVectorsWithinTolerance(
        std::vector<T>(fwd_host.begin() + batch_size * network.get_input_width(),
                       fwd_host.begin() + batch_size * (network.get_input_width() + network.get_network_width())),
        std::vector<T>(batch_size * network.get_network_width(), ref_result), 1e-3));

    // output
    for (int i = batch_size * (network.get_input_width() + network.get_network_width()); i < fwd_host.size(); i++) {
        const int output_idx = i - batch_size * (network.get_input_width() + network.get_network_width());
        const int nonzero_value = (output_idx % network.get_output_width()) < output_width ? 1 : 0;
        const double ref_result =
            nonzero_value * weight_val * input_width * input_val * network.get_network_width() * weight_val;
        CHECK(static_cast<double>(fwd_host[i]) == doctest::Approx(ref_result).epsilon(1e-2));
    }
}

/*
void test_backward() {

    const int batch_size = 512;
    constexpr int WIDTH = 64;
    constexpr int OUTPUT_WIDTH = WIDTH;
    constexpr int INPUT_WIDTH = WIDTH;
    constexpr int HIDDEN_LAYERS = 4;

    const size_t out_inter_forw_size = batch_size * (INPUT_WIDTH + OUTPUT_WIDTH + WIDTH * HIDDEN_LAYERS);
    const size_t inputs_size = INPUT_WIDTH * batch_size;
    const size_t backward_inputs_size = batch_size * OUTPUT_WIDTH;
    const size_t out_inter_backw_size = batch_size * WIDTH * (HIDDEN_LAYERS + 1);
    const size_t backward_out_size = WIDTH * WIDTH * (HIDDEN_LAYERS + 1);

    sycl::queue Q;
    try {
        Q = sycl::queue(sycl::gpu_selector_v);
    } catch (...) {
        std::cout << "No device of requested type found" << std::endl;
        return;
    }

    DeviceMem<bf16> inputs = DeviceMem<bf16>(INPUT_WIDTH * batch_size, Q);
    DeviceMem<bf16> backward_inputs = DeviceMem<bf16>(batch_size * OUTPUT_WIDTH, Q);

    SwiftNetMLP<64> network =
        SwiftNetMLP<64>(Q, INPUT_WIDTH, OUTPUT_WIDTH, HIDDEN_LAYERS, Activation::ReLU, Activation::None);

    float *out_inter_forw = sycl::malloc_device<float>(out_inter_forw_size, Q);
    float *out_inter_backw = sycl::malloc_device<float>(out_inter_backw_size, Q);

    network.initialize_params(1);

    inputs.initialize_constant(0.1f, Q);
    backward_inputs.initialize_arange(Q);

    Q.wait();

    std::vector<sycl::event> es = network.inference(inputs, out_inter_forw, batch_size, {});

    network.backward_pass(backward_inputs, out_inter_backw, out_inter_forw, batch_size, es);

    Q.wait();

    // Allocate host memory
    std::vector<bf16> out_inter_forw_vec(out_inter_forw_size);
    // std::vector<bf16> inputs_vec(inputs_size);
    std::vector<bf16> backward_inputs_vec(backward_inputs_size);
    // // check if the activated backward inputs are in m_out_inter at the right place
    // std::vector<bf16> out_inter_backward_inputs(backward_inputs_size);
    std::vector<bf16> out_inter_backw_vec(out_inter_backw_size);
    std::vector<bf16> backward_outputs_vec(backward_out_size);

    // Copy data from device to host
    Q.memcpy(out_inter_forw_vec.data(), out_inter_forw, out_inter_forw_size * sizeof(bf16)).wait();
    // inputs.copy_to_host(inputs_vec, Q);
    backward_inputs.copy_to_host(backward_inputs_vec, Q);

    Q.memcpy(out_inter_backw_vec.data(), out_inter_backw, out_inter_backw_size * sizeof(bf16)).wait();

    // Q.memcpy(out_inter_backward_inputs.data(),
    //          reinterpret_cast<bf16 *>(out_inter_backw) + HIDDEN_LAYERS * batch_size * WIDTH,
    //          backward_inputs_size * sizeof(bf16))
    //     .wait();

    Q.memcpy(backward_outputs_vec.data(), network.m_grads_matrices.data(), backward_out_size * sizeof(bf16)).wait();

    // Load the CSV files into vectors
    std::vector<float> forward_vec_ref = loadVectorFromCSV<float>("../test/ref_values/bwd_matrices/m_forward.csv");
    // std::vector<bf16> inputs_vec_ref = loadVectorFromCSV<bf16>("../bwd_matrices/inputs.csv");
    std::vector<bf16> backward_inputs_vec_ref = loadVectorFromCSV<bf16>("../test/ref_values/bwd_matrices/grads.csv");
    std::vector<float> out_inter_vec_ref = loadVectorFromCSV<float>("../test/ref_values/bwd_matrices/out_inter.csv");
    std::vector<bf16> backward_outputs_vec_ref =
        loadVectorFromCSV<bf16>("../test/ref_values/bwd_matrices/grads_matrices.csv");

    Q.wait();

    const double tolerance = 1e-2;

    CHECK(areVectorsWithinTolerance(out_inter_forw_vec, forward_vec_ref, tolerance));
    // areVectorsWithinTolerance(inputs_vec, inputs_vec_ref, tolerance);
    CHECK(areVectorsWithinTolerance(backward_inputs_vec, backward_inputs_vec_ref, tolerance));
    CHECK(areVectorsWithinTolerance(out_inter_backw_vec, out_inter_vec_ref, tolerance));
    // areVectorsWithinTolerance(out_inter_backward_inputs, backward_inputs_vec_ref, tolerance);
    CHECK(areVectorsWithinTolerance(backward_outputs_vec, backward_outputs_vec_ref, tolerance));
}
*/

TEST_CASE("Swiftnet - Constructor") {

    sycl::queue q;
    typedef sycl::ext::oneapi::bfloat16 T;

    // No need to test width template parameter since it is statically asserted in swiftnetmlp class
    // No need to test type template parameter since it is statically asserted in Network class
    SUBCASE("Supported 1") { CHECK_NOTHROW(SwiftNetMLP<T, 16>(q, 16, 16, 4, Activation::ReLU, Activation::None)); }
    SUBCASE("Supported 2") { CHECK_NOTHROW(SwiftNetMLP<T, 32>(q, 32, 32, 4, Activation::ReLU, Activation::None)); }
    SUBCASE("Supported 3") { CHECK_NOTHROW(SwiftNetMLP<T, 64>(q, 64, 64, 4, Activation::ReLU, Activation::None)); }
    SUBCASE("Supported 4") { CHECK_NOTHROW(SwiftNetMLP<T, 128>(q, 128, 128, 4, Activation::ReLU, Activation::None)); }
    SUBCASE("Pad input 1") { CHECK_NOTHROW(SwiftNetMLP<T, 64>(q, 16, 64, 4, Activation::ReLU, Activation::None)); }
    SUBCASE("Pad input 2") { CHECK_NOTHROW(SwiftNetMLP<T, 64>(q, 1, 64, 4, Activation::ReLU, Activation::None)); }
    SUBCASE("Pad output 1") { CHECK_NOTHROW(SwiftNetMLP<T, 64>(q, 64, 1, 4, Activation::ReLU, Activation::None)); }
    SUBCASE("Pad output 2") { CHECK_NOTHROW(SwiftNetMLP<T, 64>(q, 64, 16, 4, Activation::ReLU, Activation::None)); }
    SUBCASE("Unsupported layers 1") {
        CHECK_THROWS(SwiftNetMLP<T, 64>(q, 64, 64, 0, Activation::ReLU, Activation::None));
    }
    SUBCASE("Unsupported layers 2") {
        CHECK_THROWS(SwiftNetMLP<T, 64>(q, 64, 64, -1, Activation::ReLU, Activation::None));
    }

    SUBCASE("Unsupported input width 1") {
        CHECK_THROWS(SwiftNetMLP<T, 64>(q, -1, 64, 4, Activation::ReLU, Activation::None));
    }

    SUBCASE("Unsupported output width 1") {
        CHECK_THROWS(SwiftNetMLP<T, 64>(q, 64, -1, 4, Activation::ReLU, Activation::None));
    }
    SUBCASE("Unsupported activation 1") {
        CHECK_THROWS(SwiftNetMLP<T, 64>(q, 64, 64, 4, Activation::None, Activation::None));
    }
    SUBCASE("Unsupported activation 2") {
        CHECK_THROWS(SwiftNetMLP<T, 64>(q, 64, 64, 4, Activation::Tanh, Activation::None));
    }
    SUBCASE("Unsupported activation 3") {
        CHECK_THROWS(SwiftNetMLP<T, 64>(q, 64, 64, 4, Activation::Sigmoid, Activation::None));
    }
    SUBCASE("Unsupported output activation 1") {
        CHECK_THROWS(SwiftNetMLP<T, 64>(q, 64, 64, 4, Activation::ReLU, Activation::ReLU));
    }
    SUBCASE("Unsupported output activation 2") {
        CHECK_THROWS(SwiftNetMLP<T, 64>(q, 64, 64, 4, Activation::ReLU, Activation::Tanh));
    }
    SUBCASE("Unsupported output activation 3") {
        CHECK_THROWS(SwiftNetMLP<T, 64>(q, 64, 64, 4, Activation::ReLU, Activation::Sigmoid));
    }
}

/// TODO: check if the weights are actually 0 whereever they should be.
TEST_CASE("Swiftnet - Zero Padding") {
    sycl::queue q;
    typedef sycl::ext::oneapi::bfloat16 T;
    SUBCASE("Input 1-64") {
        SwiftNetMLP<T, 64> network(q, 1, 64, 4, Activation::ReLU, Activation::None);
        CHECK(network.get_input_width() == 64);
        CHECK(network.get_network_width() == 64);
        CHECK(network.get_output_width() == 64);
    }
    SUBCASE("Input 1-16") {
        SwiftNetMLP<T, 16> network(q, 1, 16, 4, Activation::ReLU, Activation::None);
        CHECK(network.get_input_width() == 16);
        CHECK(network.get_network_width() == 16);
        CHECK(network.get_output_width() == 16);
    }
    SUBCASE("Input 17-32") {
        SwiftNetMLP<T, 32> network(q, 17, 32, 4, Activation::ReLU, Activation::None);
        CHECK(network.get_input_width() == 32);
        CHECK(network.get_network_width() == 32);
        CHECK(network.get_output_width() == 32);
    }
    SUBCASE("Input 17-128") {
        SwiftNetMLP<T, 128> network(q, 17, 128, 4, Activation::ReLU, Activation::None);
        CHECK(network.get_input_width() == 128);
        CHECK(network.get_network_width() == 128);
        CHECK(network.get_output_width() == 128);
    }
    SUBCASE("Output 1-64") {
        SwiftNetMLP<T, 64> network(q, 64, 1, 4, Activation::ReLU, Activation::None);
        CHECK(network.get_input_width() == 64);
        CHECK(network.get_network_width() == 64);
        CHECK(network.get_output_width() == 64);
    }
    SUBCASE("Output 1-16") {
        SwiftNetMLP<T, 16> network(q, 16, 1, 4, Activation::ReLU, Activation::None);
        CHECK(network.get_input_width() == 16);
        CHECK(network.get_network_width() == 16);
        CHECK(network.get_output_width() == 16);
    }
    SUBCASE("Output 17-32") {
        SwiftNetMLP<T, 32> network(q, 32, 17, 4, Activation::ReLU, Activation::None);
        CHECK(network.get_input_width() == 32);
        CHECK(network.get_network_width() == 32);
        CHECK(network.get_output_width() == 32);
    }
    SUBCASE("Output 17-128") {
        SwiftNetMLP<T, 128> network(q, 128, 17, 4, Activation::ReLU, Activation::None);
        CHECK(network.get_input_width() == 128);
        CHECK(network.get_network_width() == 128);
        CHECK(network.get_output_width() == 128);
    }
}

TEST_CASE("Swiftnet - weights init") {
    sycl::queue q(sycl::gpu_selector_v);
    typedef sycl::ext::oneapi::bfloat16 T;

    SUBCASE("Default positive, No Pad") {
        SwiftNetMLP<T, 64> network(q, 64, 64, 4, Activation::ReLU, Activation::None,
                                   Network<T>::WeightInitMode::constant_pos);
        CHECK_NOTHROW(network.get_weights_matrices());
        CHECK(network.get_weights_matrices().GetNumberOfMatrices() == 5);
        for (int iter = 0; iter < 5; iter++) {
            CHECK(network.get_weights_matrices().GetView(iter).m() == 64);
            CHECK(network.get_weights_matrices().GetView(iter).n() == 64);
        }

        CHECK(areVectorsWithinTolerance(network.get_weights_matrices().copy_to_host(),
                                        std::vector<T>(network.get_weights_matrices().nelements(), 0.01), 1e-3));
    }

    SUBCASE("Default positive, Output Pad") {
        SwiftNetMLP<T, 64> network(q, 64, 63, 4, Activation::ReLU, Activation::None,
                                   Network<T>::WeightInitMode::constant_pos);
        CHECK_NOTHROW(network.get_weights_matrices());
        CHECK(network.get_weights_matrices().GetNumberOfMatrices() == 5);

        for (int iter = 0; iter < 4; iter++) {
            CHECK(network.get_weights_matrices().GetView(iter).m() == 64);
            CHECK(network.get_weights_matrices().GetView(iter).n() == 64);
        }
        CHECK(network.get_weights_matrices().Back().m() == 64);
        CHECK(network.get_weights_matrices().Back().n() == 64);
    }

    SUBCASE("Overwrite, No Pad") {
        SwiftNetMLP<T, 64> network(q, 64, 64, 4, Activation::ReLU, Activation::None,
                                   Network<T>::WeightInitMode::constant_pos);
        CHECK_NOTHROW(network.get_weights_matrices());
        CHECK(network.get_weights_matrices().GetNumberOfMatrices() == 5);
        std::vector<T> new_weights(network.get_weights_matrices().nelements(), 1.23);
        network.set_weights_matrices(new_weights);
        for (int iter = 0; iter < 5; iter++) {
            CHECK(network.get_weights_matrices().GetView(iter).m() == 64);
            CHECK(network.get_weights_matrices().GetView(iter).n() == 64);
        }

        CHECK(areVectorsWithinTolerance(network.get_weights_matrices().copy_to_host(),
                                        std::vector<T>(network.get_weights_matrices().nelements(), 1.23), 1e-3));
    }
}

TEST_CASE("Swiftnet - zero pad forward_pass WIDTH 64") {
    sycl::queue q(sycl::gpu_selector_v);

    auto test_function = [=](const int input_width, const int output_width, sycl::queue &q) {
        typedef sycl::ext::oneapi::bfloat16 T;
        constexpr int WIDTH = 64;
        test_forward_1layer<T, WIDTH>(q, input_width, output_width, 8);
    };

    SUBCASE("No Pad") {
        constexpr int input_width = 64;
        constexpr int output_width = 64;
        test_function(input_width, output_width, q);
    }
    SUBCASE("Input Pad") {
        constexpr int input_width = 3;
        constexpr int output_width = 64;
        test_function(input_width, output_width, q);
    }
    SUBCASE("Output Pad") {
        constexpr int input_width = 64;
        constexpr int output_width = 7;
        test_function(input_width, output_width, q);
    }
    SUBCASE("Input and Output Pad") {
        constexpr int input_width = 3;
        constexpr int output_width = 5;
        test_function(input_width, output_width, q);
    }
}

TEST_CASE("Swiftnet - zero pad inference WIDTH 64") {
    sycl::queue q(sycl::gpu_selector_v);

    auto test_function = [=](const int input_width, const int output_width, sycl::queue &q) {
        typedef sycl::ext::oneapi::bfloat16 T;
        constexpr int WIDTH = 64;
        test_inference_1layer<T, WIDTH>(q, input_width, output_width, 8);
    };

    SUBCASE("No Pad") {
        constexpr int input_width = 64;
        constexpr int output_width = 64;
        test_function(input_width, output_width, q);
    }
    SUBCASE("Input Pad") {
        constexpr int input_width = 3;
        constexpr int output_width = 64;
        test_function(input_width, output_width, q);
    }
    SUBCASE("Output Pad") {
        constexpr int input_width = 64;
        constexpr int output_width = 7;
        test_function(input_width, output_width, q);
    }
    SUBCASE("Input and Output Pad") {
        constexpr int input_width = 3;
        constexpr int output_width = 5;
        test_function(input_width, output_width, q);
    }
}

TEST_CASE("Swiftnet - Batch Sizes") {
    sycl::queue q(sycl::gpu_selector_v);

    auto test_function = [=](const int batch_size, sycl::queue &q) {
        typedef sycl::ext::oneapi::bfloat16 T;
        constexpr int WIDTH = 64;
        test_forward_1layer<T, WIDTH>(q, WIDTH, WIDTH, batch_size);
    };

    SUBCASE("Batch size 8") { CHECK_NOTHROW(test_function(8, q)); }
    SUBCASE("Batch size 512") { CHECK_NOTHROW(test_function(512, q)); }
    SUBCASE("Batch size 16") { CHECK_NOTHROW(test_function(16, q)); }
    SUBCASE("Batch size 1") { CHECK_THROWS(test_function(1, q)); }
    SUBCASE("Batch size 13") { CHECK_THROWS(test_function(13, q)); }
}

TEST_CASE("Swiftnet - Net Widths") {
    // only testing constructor. values tested later
    sycl::queue q(sycl::gpu_selector_v);

    auto test_function = [=](const int width, sycl::queue &q) {
        typedef sycl::ext::oneapi::bfloat16 T;
        if (width == 16)
            test_forward_1layer<T, 16>(q, 16, 16, 8);
        else if (width == 32)
            test_forward_1layer<T, 32>(q, 32, 32, 8);
        else if (width == 64)
            test_forward_1layer<T, 64>(q, 64, 64, 8);
        else if (width == 128)
            test_forward_1layer<T, 128>(q, 128, 128, 8);
        else
            throw std::invalid_argument("Unsupported width");
    };

    SUBCASE("WIDTH 16") { CHECK_NOTHROW(test_function(16, q)); }
    SUBCASE("WIDTH 32") { CHECK_NOTHROW(test_function(32, q)); }
    SUBCASE("WIDTH 64") { CHECK_NOTHROW(test_function(64, q)); }
    SUBCASE("WIDTH 128") { CHECK_NOTHROW(test_function(128, q)); }
}