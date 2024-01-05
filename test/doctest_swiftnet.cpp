// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "doctest/doctest.h"
#include <filesystem>

#include "SwiftNetMLP.h"
#include "result_check.h"
float calculateMAPE(float prediction, float reference) {
    if (reference == 0.0) {
        return 0.0;
    } else {
        float absoluteError = std::abs(prediction - reference);
        return (absoluteError / std::abs(reference)) * 100.0;
    }
}

template <typename T, int WIDTH>
void test_inference_1layer(sycl::queue &q, const int input_width, const int output_width, const int batch_size) {

    constexpr int n_hidden_layers = 1;
    constexpr float input_val = 1.0f;
    // setting Network<T>::WeightInitMode::constant_pos sets the weights to this value
    constexpr float weight_val = 0.01f;

    SwiftNetMLP<T, WIDTH> network(q, input_width, output_width, n_hidden_layers, Activation::ReLU, Activation::None,
                                  Network<T>::WeightInitMode::constant_pos);

    DeviceMem<T> network_output(batch_size * network.get_output_width(), q);
    DeviceMem<T> network_input(network.get_inputs_width() * batch_size, q);

    network_input.fill(input_val);

    network.forward_pass(network_input, network_output, batch_size, {});

    q.wait();

    std::vector<T> out_host(network_output.size());
    network_output.copy_to_host(out_host).wait();

    for (int output_idx = 0; output_idx < out_host.size(); output_idx++) {

        const int nonzero_value = (output_idx % network.get_output_width()) < output_width ? 1 : 0;
        const double ref_result =
            nonzero_value * weight_val * input_width * input_val * network.get_network_width() * weight_val;
        CHECK(static_cast<double>(out_host[output_idx]) == doctest::Approx(ref_result).epsilon(1e-3));
    }
}

template <typename T, int WIDTH>
void test_forward_1layer(sycl::queue &q, const int input_width, const int output_width, const int batch_size) {

    constexpr int n_hidden_layers = 1;
    constexpr float input_val = 1.0f;
    // setting Network<T>::WeightInitMode::constant_pos sets the weights to this value
    constexpr float weight_val = 0.01f;

    SwiftNetMLP<T, WIDTH> network(q, input_width, output_width, n_hidden_layers, Activation::ReLU, Activation::None,
                                  Network<T>::WeightInitMode::constant_pos);

    DeviceMem<T> network_interm_forw(batch_size * (network.get_inputs_width() + network.get_output_width() +
                                                   network.get_network_width() * network.get_n_hidden_layers()),
                                     q);
    DeviceMem<T> network_input(network.get_inputs_width() * batch_size, q);

    network_input.fill(input_val);

    network.forward_pass(network_input, network_interm_forw, batch_size, {});

    q.wait();

    std::vector<T> fwd_host(network_interm_forw.size());
    network_interm_forw.copy_to_host(fwd_host).wait();

    // Check result of fwd_host. First block = input, second block is the
    // hidden layer, last block is output.
    for (int i = 0; i < fwd_host.size(); i++) {
        if (i < batch_size * network.get_inputs_width()) {
            // std::cout << static_cast<double>(fwd_host[i]) << ": ";
            CHECK(static_cast<double>(fwd_host[i]) == doctest::Approx(input_val).epsilon(1e-3));
        } else if ((i >= batch_size * network.get_inputs_width()) &&
                   (i < batch_size * (network.get_inputs_width() + network.get_network_width()))) {
            const double ref_result = weight_val * input_width * input_val;
            CHECK(static_cast<double>(fwd_host[i]) == doctest::Approx(ref_result).epsilon(1e-3));
            // std::cout << static_cast<double>(fwd_host[i]) << ", ";
        } else {
            const int output_idx = i - batch_size * (network.get_inputs_width() + network.get_network_width());
            const int nonzero_value = (output_idx % network.get_output_width()) < output_width ? 1 : 0;
            const double ref_result =
                nonzero_value * weight_val * input_width * input_val * network.get_network_width() * weight_val;
            CHECK(static_cast<double>(fwd_host[i]) == doctest::Approx(ref_result).epsilon(1e-3));
            // std::cout << static_cast<double>(fwd_host[i]) << "; ";
        }
    }
}

/*
template <typename T, int WIDTH>
void test_forward(sycl::queue &q, const int input_width, const int output_width, const int n_hidden_layers,
                  const int net_width, const int batch_size, const int init_mode,
                  Activation activation = Activation::ReLU, Activation output_activation = Activation::None,
                  const int load_weights = 0, std::string filetype = "full") {

    const float input_val = 1.0f;
    const float weight_val = 0.01f; // setting initialize_params(2) sets the weights to this value

    constexpr int input_width_padded = WIDTH;
    constexpr int output_width_padded = WIDTH;

    SwiftNetMLP<T, WIDTH> network(q, input_width, output_width, n_hidden_layers, activation, output_activation);
    if (load_weights) {
        network.load_from_file("../test/ref_values/network_with_grid_encoding/" + filetype + "/network_params.csv");
    } else {
        network.initialize_params(init_mode);
    }

    DeviceMem<T> network_interm_forw(
        batch_size * (input_width_padded + output_width_padded + net_width * n_hidden_layers), q);
    DeviceMem<T> network_input(input_width_padded * batch_size, q);
    std::vector<sycl::event> deps;
    std::vector<T> input_ref;
    std::string filepath;
    if (load_weights) {
        filepath = "../test/ref_values/network_with_grid_encoding/" + filetype + "/";
        std::string filename =
            std::filesystem::exists(filepath + "encoding_output.csv") ? "encoding_output.csv" : "input.csv";
        input_ref = loadVectorFromCSV<T>(filepath + filename);

        CHECK(input_ref.size() == network_input.size());
        network_input.copy_from_host(input_ref, q);
        q.wait();
    } else {
        network_input.initialize_constant(input_val, q);
    }

    /// TODO: move this outside in separate tests
    if (batch_size % 8 == 0) { // CHECK_THROWS here and return, otherwise continue normally
        network.forward_pass(network_input, network_interm_forw, batch_size, deps);
    } else {
        CHECK_THROWS(network.forward_pass(network_input, network_interm_forw, batch_size, deps));
        return;
    }

    q.wait();

    std::vector<T> fwd_host(batch_size * (input_width_padded + output_width_padded + net_width * n_hidden_layers));
    network_interm_forw.copy_to_host(fwd_host).wait();

    if (load_weights) {

        std::vector<float> output_ref = loadVectorFromCSV<float>("../test/ref_values/network_with_grid_encoding/" +
                                                                 filetype + "/network_output.csv");
        CHECK(output_ref.size() == (batch_size * output_width_padded));
        // std::cout << "Loaded size: " << output_ref.size() << "ref: " << (batch_size * output_width_padded) <<
        // std::endl;

        int check_fwd_values = std::filesystem::exists(filepath + "network_forward.csv");

        float totalMAPE = 0.0;
        int mapeCount = 0;
        if (check_fwd_values) {

            std::vector<float> fwd_ref = loadVectorFromCSV<float>("../test/ref_values/network_with_grid_encoding/" +
                                                                  filetype + "/network_forward.csv");
            for (int i = 0; i < fwd_host.size(); i++) {
                if ((i == batch_size * input_width_padded) | (i == batch_size * (input_width_padded + net_width)) |
                    (i == batch_size * (input_width_padded + net_width + net_width))) {

                    double averageMAPE = totalMAPE / mapeCount;
                    std::cout << "Average MAPE: " << averageMAPE << "%" << std::endl;

                    std::cout << "==========================================" << std::endl;
                }
                float ref_result = fwd_ref[i];
                float mape = calculateMAPE(fwd_host[i], ref_result);
                totalMAPE += mape;
                mapeCount++;
                if (i > batch_size * (input_width_padded + net_width + net_width) && (ref_result != 0)) {
                    // CHECK(mape < 10.0); // MAPE is in per cent
                    // if (mape > 10.0) {

                    std::cout << i << ": " << fwd_host[i] << " / " << ref_result << ", mape: " << mape << std::endl;
                    // }
                }
            }
        } else {
            for (int i = 0; i < fwd_host.size(); i++) {
                if ((i >= batch_size * (input_width_padded + net_width * n_hidden_layers)) &&
                    (i < batch_size * (input_width_padded + net_width * n_hidden_layers + output_width_padded))) {
                    int output_idx = i - batch_size * (input_width_padded + net_width * n_hidden_layers);
                    double ref_result = 0.0;
                    if ((output_idx % output_width_padded) < output_width) {
                        ref_result = output_ref[output_idx];
                        double mape = calculateMAPE(fwd_host[i], ref_result);
                        totalMAPE += mape;
                        mapeCount++;

                        CHECK(mape < 1.0); // MAPE is in per cent

                    } else {
                        // this checks that the amount of inputs multiplied by weight is correct after the first two
                        // layers
                        // Check that  output padded correctly
                        ref_result = 0.0;
                        CHECK(static_cast<double>(fwd_host[i]) == doctest::Approx(ref_result).epsilon(1e-3));
                    }
                } else {
                    // std::cout << i << ": " << fwd_host[i] << std::endl;
                }
            }
        }

        // // Calculate and print the average MAPE
        // double averageMAPE = totalMAPE / mapeCount;
        // CHECK(averageMAPE < 1.0); // MAPE is in per cent

    } else {
        // this has only one layer, thus net_width instead of net_width * n_hidden_layer = net_width
        for (int i = 0; i < fwd_host.size(); i++) {
            if (i < batch_size * input_width_padded) {
                // no checks for input
                // std::cout << i << ", Input: " << fwd_host[i] << std::endl;
            } else if ((i >= batch_size * input_width_padded) &&
                       (i < batch_size * (input_width_padded + net_width * n_hidden_layers)) &&
                       (n_hidden_layers == 1)) {
                // 1st layer output. no test if the layer amount is higher.
                double ref_result = weight_val * input_width * input_val;
                // this checks that the amount of inputs multiplied by weight is correct after the first layer
                CHECK(static_cast<double>(fwd_host[i]) == doctest::Approx(ref_result).epsilon(1e-3));
                // std::cout << i << ", 1st layer: " << fwd_host[i] << std::endl;

            } else if ((i >= batch_size * (input_width_padded + net_width * n_hidden_layers)) &&
                       (i < batch_size * (input_width_padded + net_width * n_hidden_layers + output_width_padded))) {
                int output_idx = i - batch_size * (input_width_padded + net_width * n_hidden_layers);
                double ref_result = 0.0;
                if ((output_idx % output_width_padded) < output_width) {
                    ref_result = weight_val * input_width * input_val * net_width * weight_val;
                    if (n_hidden_layers == 1) {
                        CHECK(static_cast<double>(fwd_host[i]) == doctest::Approx(ref_result).epsilon(1e-3));
                    } else {
                        CHECK(static_cast<double>(fwd_host[i]) != 0.0);
                    }
                } else {
                    // this checks that the amount of inputs multiplied by weight is correct after the first two layers
                    // Check that  output padded correctly
                    ref_result = 0.0;
                    CHECK(static_cast<double>(fwd_host[i]) == doctest::Approx(ref_result).epsilon(1e-3));
                }
                // std::cout << i << ", last layer: " << fwd_host[i] << "(" << ref_result << ")" << std::endl;
            }
        }
    }
}
*/
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
        CHECK(network.get_inputs_width() == 64);
        CHECK(network.get_network_width() == 64);
        CHECK(network.get_output_width() == 64);
    }
    SUBCASE("Input 1-16") {
        SwiftNetMLP<T, 16> network(q, 1, 16, 4, Activation::ReLU, Activation::None);
        CHECK(network.get_inputs_width() == 16);
        CHECK(network.get_network_width() == 16);
        CHECK(network.get_output_width() == 16);
    }
    SUBCASE("Input 17-32") {
        SwiftNetMLP<T, 32> network(q, 17, 32, 4, Activation::ReLU, Activation::None);
        CHECK(network.get_inputs_width() == 32);
        CHECK(network.get_network_width() == 32);
        CHECK(network.get_output_width() == 32);
    }
    SUBCASE("Input 17-128") {
        SwiftNetMLP<T, 128> network(q, 17, 128, 4, Activation::ReLU, Activation::None);
        CHECK(network.get_inputs_width() == 128);
        CHECK(network.get_network_width() == 128);
        CHECK(network.get_output_width() == 128);
    }
    SUBCASE("Output 1-64") {
        SwiftNetMLP<T, 64> network(q, 64, 1, 4, Activation::ReLU, Activation::None);
        CHECK(network.get_inputs_width() == 64);
        CHECK(network.get_network_width() == 64);
        CHECK(network.get_output_width() == 64);
    }
    SUBCASE("Output 1-16") {
        SwiftNetMLP<T, 16> network(q, 16, 1, 4, Activation::ReLU, Activation::None);
        CHECK(network.get_inputs_width() == 16);
        CHECK(network.get_network_width() == 16);
        CHECK(network.get_output_width() == 16);
    }
    SUBCASE("Output 17-32") {
        SwiftNetMLP<T, 32> network(q, 32, 17, 4, Activation::ReLU, Activation::None);
        CHECK(network.get_inputs_width() == 32);
        CHECK(network.get_network_width() == 32);
        CHECK(network.get_output_width() == 32);
    }
    SUBCASE("Output 17-128") {
        SwiftNetMLP<T, 128> network(q, 128, 17, 4, Activation::ReLU, Activation::None);
        CHECK(network.get_inputs_width() == 128);
        CHECK(network.get_network_width() == 128);
        CHECK(network.get_output_width() == 128);
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

/*TEST_CASE("Swiftnet Forward - Activation") {
    // only testing constructor. values tested later
    int input_width = 64;
    int init_mode = 2;
    int batch_size = 512;
    int n_hidden_layers = 1;
    int output_width = 64;
    int net_width = 64;

    SUBCASE("Activation sigmoid") {
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode, Activation::Sigmoid,
                     Activation::None);
    }

    SUBCASE("Output activation sigmoid") {
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode, Activation::ReLU,
                     Activation::Sigmoid);
    }
}

// SUBCASE("Output activation None") {
//     test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode, Activation::ReLU,
//                  Activation::None);
// }
// }

// TEST_CASE("Swiftnet Forward - load weights") {
//     int net_width = 64;
//     int n_hidden_layers = 2;

    SUBCASE("Simple full") {
        int input_width = 64;
        int output_width = 64;
        int init_mode = 0;
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode, Activation::ReLU,
                     Activation::None, 1, "simple_full");
    }
    SUBCASE("Simple padded") {
        int input_width = 32;
        int output_width = 16;
        int init_mode = 2;
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode, Activation::ReLU,
                     Activation::None, 1, "simple_padded");
    }
    SUBCASE("Simple random arange") {
        int input_width = 64;
        int output_width = 64;
        int init_mode = 2;
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode, Activation::ReLU,
                     Activation::None, 1, "simple_arange");
    }
    SUBCASE("Full network") {
        int input_width = 32;
        int output_width = 1;
        int init_mode = 2;
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode, Activation::ReLU,
                     Activation::None, 1, "full_network");
    }
    SUBCASE("Simple MLP") {
        int input_width = 32;
        int output_width = 1;
        int init_mode = 2;
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode, Activation::ReLU,
                     Activation::None, 1, "simple_mlp");
    }
    SUBCASE("Full (encoding output as input)") {
        int input_width = 32;
        int output_width = 1;
        int init_mode = 2;
        batch_size = 512 * 1;
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode, Activation::ReLU,
                     Activation::None, 1, "full");
    }
}
*/
// TEST_CASE("Swiftnet Backward") {
//     SUBCASE("") { test_backward(); }
// }