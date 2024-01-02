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

void test_forward(const int input_width, const int output_width, const int n_hidden_layers, const int net_width,
                  const int batch_size, const int init_mode, Activation activation = Activation::ReLU,
                  Activation output_activation = Activation::None, int load_weights = 0,
                  std::string filetype = "full") {

    const float input_val = 1.0f;
    const float weight_val = 0.01f; // setting initialize_params(2) sets the weights to this value

    REQUIRE(net_width == 64);
    int input_width_padded = net_width;
    int output_width_padded = net_width;

    std::vector<bf16> out(batch_size * (output_width_padded));
    std::vector<bf16> fwd(batch_size * (input_width_padded + output_width_padded + net_width * n_hidden_layers));

    queue q = queue();

    if ((input_width <= 0) || (output_width <= 0) || (n_hidden_layers <= 0) || (activation != Activation::ReLU) ||
        (output_activation != Activation::None)) { // CHECK_THROWS here and return, otherwise continue normally
        CHECK_THROWS(SwiftNetMLP<64>(q, input_width, output_width, n_hidden_layers, activation, output_activation));
        return;
    }

    SwiftNetMLP<64> network =
        SwiftNetMLP<64>(q, input_width, output_width, n_hidden_layers, activation, output_activation);
    if (load_weights) {
        network.load_from_file("../test/ref_values/network_with_grid_encoding/" + filetype + "/network_params.csv");
    } else {
        network.initialize_params(init_mode);
    }
    // std::vector<bf16> weights = network.get_weights_matrices_as_vector();
    float *forward =
        malloc_device<float>(batch_size * (input_width_padded + output_width_padded + net_width * n_hidden_layers), q);
    DeviceMem<bf16> network_input = DeviceMem<bf16>(input_width_padded * batch_size, q);
    std::vector<sycl::event> deps;
    std::vector<bf16> input_ref;

    std::string filepath;
    if (load_weights) {
        filepath = "../test/ref_values/network_with_grid_encoding/" + filetype + "/";
        std::string filename =
            std::filesystem::exists(filepath + "encoding_output.csv") ? "encoding_output.csv" : "input.csv";
        input_ref = loadVectorFromCSV<bf16>(filepath + filename);

        CHECK(input_ref.size() == network_input.size());
        network_input.copy_from_host(input_ref, q);
        q.wait();
        // std::cout << "Input: " << std::endl;
        // for (int i = 0; i < input_ref.size(); i++) {
        //     std::cout << i << ": " << input_ref[i] << std::endl;
        // }
    } else {
        network_input.initialize_constant(input_val, q);
    }

    // q.parallel_for<>(range<1>(network_input.size()), [=](id<1> idx) {
    //      if ((idx % input_width_padded) >= input_width) network_input[idx] = (bf16)(0.0f);
    //  }).wait();

    if (batch_size % 8 == 0) { // CHECK_THROWS here and return, otherwise continue normally
        network.inference(network_input, forward, batch_size, deps);
    } else {
        CHECK_THROWS(network.inference(network_input, forward, batch_size, deps));
        return;
    }

    q.wait();

    q.memcpy(out.data(), network.GetOutput(forward, batch_size), sizeof(bf16) * out.size()).wait();
    q.memcpy(fwd.data(), reinterpret_cast<bf16 const *const>(forward), sizeof(bf16) * fwd.size()).wait();

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
            for (int i = 0; i < fwd.size(); i++) {
                if ((i == batch_size * input_width_padded) | (i == batch_size * (input_width_padded + net_width)) |
                    (i == batch_size * (input_width_padded + net_width + net_width))) {

                    double averageMAPE = totalMAPE / mapeCount;
                    std::cout << "Average MAPE: " << averageMAPE << "%" << std::endl;

                    std::cout << "==========================================" << std::endl;
                }
                float ref_result = fwd_ref[i];
                float mape = calculateMAPE(fwd[i], ref_result);
                totalMAPE += mape;
                mapeCount++;
                if (i > batch_size * (input_width_padded + net_width + net_width) && (ref_result != 0)) {
                    // CHECK(mape < 10.0); // MAPE is in per cent
                    // if (mape > 10.0) {

                    std::cout << i << ": " << fwd[i] << " / " << ref_result << ", mape: " << mape << std::endl;
                    // }
                }
                // std::cout << i << ": " << fwd[i] << " / " << ref_result << std::endl;
                // CHECK(static_cast<double>(fwd[i]) == doctest::Approx(ref_result).epsilon(1e-3));
            }
        } else {
            for (int i = 0; i < fwd.size(); i++) {
                if ((i >= batch_size * (input_width_padded + net_width * n_hidden_layers)) &&
                    (i < batch_size * (input_width_padded + net_width * n_hidden_layers + output_width_padded))) {
                    int output_idx = i - batch_size * (input_width_padded + net_width * n_hidden_layers);
                    double ref_result = 0.0;
                    if ((output_idx % output_width_padded) < output_width) {
                        ref_result = output_ref[output_idx];
                        double mape = calculateMAPE(fwd[i], ref_result);
                        totalMAPE += mape;
                        mapeCount++;
                        // std::cout << i << ", last layer: " << fwd[i] << "should be :" << ref_result << " MAPE: " <<
                        // mape
                        //           << "%" << std::endl;
                        // CHECK(static_cast<double>(fwd[i]) == doctest::Approx(ref_result).epsilon(1e-3));
                        CHECK(mape < 1.0); // MAPE is in per cent

                    } else {
                        // this checks that the amount of inputs multiplied by weight is correct after the first two
                        // layers
                        // Check that  output padded correctly
                        ref_result = 0.0;
                        CHECK(static_cast<double>(fwd[i]) == doctest::Approx(ref_result).epsilon(1e-3));
                        // std::cout << i << ", padded, last layer: " << fwd[i] << "should be :" << ref_result <<
                        // std::endl;
                    }
                } else {
                    // std::cout << i << ": " << fwd[i] << std::endl;
                }
            }
        }

        // // Calculate and print the average MAPE
        // double averageMAPE = totalMAPE / mapeCount;
        // CHECK(averageMAPE < 1.0); // MAPE is in per cent

    } else {
        // this has only one layer, thus net_width instead of net_width * n_hidden_layer = net_width
        for (int i = 0; i < fwd.size(); i++) {
            if (i < batch_size * input_width_padded) {
                // no checks for input
                // std::cout << i << ", Input: " << fwd[i] << std::endl;
            } else if ((i >= batch_size * input_width_padded) &&
                       (i < batch_size * (input_width_padded + net_width * n_hidden_layers)) &&
                       (n_hidden_layers == 1)) {
                // 1st layer output. no test if the layer amount is higher.
                double ref_result = weight_val * input_width * input_val;
                // this checks that the amount of inputs multiplied by weight is correct after the first layer
                CHECK(static_cast<double>(fwd[i]) == doctest::Approx(ref_result).epsilon(1e-3));
                // std::cout << i << ", 1st layer: " << fwd[i] << std::endl;

            } else if ((i >= batch_size * (input_width_padded + net_width * n_hidden_layers)) &&
                       (i < batch_size * (input_width_padded + net_width * n_hidden_layers + output_width_padded))) {
                int output_idx = i - batch_size * (input_width_padded + net_width * n_hidden_layers);
                double ref_result = 0.0;
                if ((output_idx % output_width_padded) < output_width) {
                    ref_result = weight_val * input_width * input_val * net_width * weight_val;
                    if (n_hidden_layers == 1) {
                        CHECK(static_cast<double>(fwd[i]) == doctest::Approx(ref_result).epsilon(1e-3));
                    } else {
                        CHECK(static_cast<double>(fwd[i]) != 0.0);
                    }
                } else {
                    // this checks that the amount of inputs multiplied by weight is correct after the first two layers
                    // Check that  output padded correctly
                    ref_result = 0.0;
                    CHECK(static_cast<double>(fwd[i]) == doctest::Approx(ref_result).epsilon(1e-3));
                }
                // std::cout << i << ", last layer: " << fwd[i] << "(" << ref_result << ")" << std::endl;
            }
        }
    }
}

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

TEST_CASE("Swiftnet Forward - zero pad input") {
    int net_width = 64;
    int output_width = 64;
    int n_hidden_layers = 1;
    int init_mode = 2;

    int batch_size = 512;

    SUBCASE("Input 64 (no pad)") {
        int input_width = 64;
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
    }
    SUBCASE("Input 1 (63 pad)") {
        int input_width = 1;
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
    }
    SUBCASE("Input 2 (62 pad)") {
        int input_width = 2;
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
    }
    SUBCASE("Input 16 (48 pad)") {
        int input_width = 16;
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
    }
    SUBCASE("Input 32 (32 pad)") {
        int input_width = 32;
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
    }
    SUBCASE("Input 63 (1 pad)") {
        int input_width = 63;
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
    }

    SUBCASE("Input -1 (failure)") {
        int input_width = -1;
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
    }
    SUBCASE("Input 0 (failure)") {
        int input_width = 0;
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
    }
}

// TEST_CASE("Swiftnet Forward - zero pad output") {
//     int input_width = 64;
//     int net_width = 64;
//     int init_mode = 2;
//     int n_hidden_layers = 1;

//     int batch_size = 8;

//     SUBCASE("Output 64 (no pad)") {
//         int output_width = 64;
//         test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
//     }
//     SUBCASE("Output 63 (1 pad)") {
//         int output_width = 63;
//         test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
//     }
//     SUBCASE("Output 32 (32 pad)") {
//         int output_width = 32;
//         test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
//     }
//     SUBCASE("Output 16 (48 pad)") {
//         int output_width = 16;
//         test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
//     }
//     SUBCASE("Output 8 (56 pad)") {
//         int output_width = 8;
//         test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
//     }
//     SUBCASE("Output 4 (60 pad)") {
//         int output_width = 4;
//         test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
//     }
//     SUBCASE("Output 2 (62 pad)") {
//         int output_width = 2;
//         test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
//     }
//     SUBCASE("Output 1 (63 pad)") {
//         int output_width = 1;
//         test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
//     }
//     SUBCASE("Output -1 (failure)") {
//         int output_width = -1;
//         test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
//     }
//     SUBCASE("Output 0 (failure)") {
//         int output_width = 0;
//         test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
//     }
//     n_hidden_layers = 4;

//     SUBCASE("Output 64 (no pad) - multiple layers") {
//         int output_width = 64;
//         test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
//     }
//     SUBCASE("Output 63 (1 pad) - multiple layers") {
//         int output_width = 63;
//         test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
//     }
//     SUBCASE("Output 32 (32 pad) - multiple layers") {
//         int output_width = 32;
//         test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
//     }
//     SUBCASE("Output 16 (48 pad) - multiple layers") {
//         int output_width = 16;
//         test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
//     }
//     SUBCASE("Output 8 (56 pad) - multiple layers") {
//         int output_width = 8;
//         test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
//     }
//     SUBCASE("Output 4 (60 pad) - multiple layers") {
//         int output_width = 4;
//         test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
//     }
//     SUBCASE("Output 2 (62 pad) - multiple layers") {
//         int output_width = 2;
//         test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
//     }
//     SUBCASE("Output 1 (63 pad) - multiple layers") {
//         int output_width = 1;
//         test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
//     }
//     SUBCASE("Output -1 (failure) - multiple layers") {
//         int output_width = -1;
//         test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
//     }
//     SUBCASE("Output 0 (failure) - multiple layers") {
//         int output_width = 0;
//         test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
//     }
// }

// TEST_CASE("Swiftnet Forward - Batch Sizes") {
//     int input_width = 64;
//     int net_width = 64;
//     int output_width = 64;
//     int n_hidden_layers = 1;
//     int init_mode = 2;

//     SUBCASE("Batch size 512") {
//         int batch_size = 512;
//         test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
//     }
//     SUBCASE("Batch size 8") {
//         int batch_size = 8;
//         test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
//     }
//     SUBCASE("Batch size 16") {
//         int batch_size = 16;
//         test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
//     }
//     SUBCASE("Batch size 32") {
//         int batch_size = 32;
//         test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
//     }
//     SUBCASE("Batch size 64") {
//         int batch_size = 64;
//         test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
//     }
//     SUBCASE("Batch size 128") {
//         int batch_size = 128;
//         test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
//     }
//     SUBCASE("Batch size 1") {
//         int batch_size = 1;
//         test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
//     }

//     SUBCASE("Batch size 13") {
//         int batch_size = 13;
//         test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
//     }
//     SUBCASE("Batch size 513") {
//         int batch_size = 513;
//         test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
//     }
//     SUBCASE("Batch size 1024") {
//         int batch_size = 1024;
//         test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
//     }
//     SUBCASE("Batch size 1025") {
//         int batch_size = 1025;
//         test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
//     }
// }

// TEST_CASE("Swiftnet Forward - Layer amount") {
//     // only testing constructor. values tested later
//     int input_width = 64;
//     int net_width = 64;
//     int output_width = 64;
//     int init_mode = 2;
//     int batch_size = 512;

//     SUBCASE("Layer amount: 1 (success)") {
//         int n_hidden_layers = 1;
//         test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
//     }
//     SUBCASE("Layer amount: 0 (failure)") {
//         int n_hidden_layers = 0;
//         test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
//     }
//     SUBCASE("Layer amount: -1 (failure)") {
//         int n_hidden_layers = -1;
//         test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
//     }
// }

// TEST_CASE("Swiftnet Forward - Net Widths") {
//     // only testing constructor. values tested later
//     int input_width = 64;
//     int init_mode = 2;
//     int batch_size = 512;
//     int n_hidden_layers = 1;
//     int output_width = 64;

//     SUBCASE("Layer amount: 64 (success)") {
//         int net_width = 64;
//         int input_width_padded = net_width;
//         test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
//     }
//     // SUBCASE("Layer amount: 32 (failure)") {
//     //     int net_width = 32;
//     //     test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
//     // }
//     // SUBCASE("Layer amount: 16 (failure)") {
//     //     int net_width = 16;
//     //     test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
//     // }
// }

// TEST_CASE("Swiftnet Forward - Activation") {
//     // only testing constructor. values tested later
//     int input_width = 64;
//     int init_mode = 2;
//     int batch_size = 512;
//     int n_hidden_layers = 1;
//     int output_width = 64;
//     int net_width = 64;

//     SUBCASE("Activation relu") {
//         test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
//         //, Activation::ReLU,
//         //           Activation::None);
//     }
// SUBCASE("Activation tanh") {
//     test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode, Activation::Tanh,
//                  Activation::Tanh);
// }

// SUBCASE("Output activation None") {
//     test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode, Activation::ReLU,
//                  Activation::None);
// }
// }

// TEST_CASE("Swiftnet Forward - load weights") {
//     int net_width = 64;
//     int n_hidden_layers = 2;

//     int batch_size = 8;

// SUBCASE("Simple full") {
//     int input_width = 64;
//     int output_width = 64;
//     int init_mode = 0;
//     test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode, Activation::ReLU,
//                  Activation::None, 1, "simple_full");
// }
// SUBCASE("Simple padded") {
//     int input_width = 32;
//     int output_width = 16;
//     int init_mode = 2;
//     test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode, Activation::ReLU,
//                  Activation::None, 1, "simple_padded");
// }
// SUBCASE("Simple random arange") {
//     int input_width = 64;
//     int output_width = 64;
//     int init_mode = 2;
//     test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode, Activation::ReLU,
//                  Activation::None, 1, "simple_arange");
// }
// SUBCASE("Full network") {
//     int input_width = 32;
//     int output_width = 1;
//     int init_mode = 2;
//     test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode, Activation::ReLU,
//                  Activation::None, 1, "full_network");
// }
// SUBCASE("Simple MLP") {
//     int input_width = 32;
//     int output_width = 1;
//     int init_mode = 2;
//     test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode, Activation::ReLU,
//                  Activation::None, 1, "simple_mlp");
// }
// SUBCASE("Full (encoding output as input)") {
//     int input_width = 32;
//     int output_width = 1;
//     int init_mode = 2;
//     batch_size = 512 * 1;
//     test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode, Activation::ReLU,
//                  Activation::None, 1, "full");
// }
// }

// TEST_CASE("Swiftnet Backward") {
//     SUBCASE("") { test_backward(); }
// }