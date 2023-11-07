// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "doctest/doctest.h"

#include "SwiftNetMLP.h"
#include "activation.h"

void test_forward(int input_width, int output_width, int n_hidden_layers, int net_width, int batch_size, int init_mode,
                  Activation activation = Activation::ReLU, Activation output_activation = Activation::None) {

    float input_val = 1.0f;
    float weight_val = 0.01f; // setting initialize_params(2) sets the weights to this value

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
    network.initialize_params(init_mode);
    // std::vector<bf16> weights = network.get_weights_matrices_as_vector();
    float *forward =
        malloc_device<float>(batch_size * (input_width_padded + output_width_padded + net_width * n_hidden_layers), q);
    DeviceMem<bf16> network_input = DeviceMem<bf16>(input_width_padded * batch_size, q);
    std::vector<sycl::event> deps;

    network_input.initialize_constant(input_val, q);
    q.parallel_for<>(range<1>(network_input.size()), [=](id<1> idx) {
         if ((idx % input_width_padded) >= input_width) network_input[idx] = (bf16)(0.0f);
     }).wait();

    if (((batch_size % 16) == 0) && (batch_size >= 512)) { // CHECK_THROWS here and return, otherwise continue normally
        network.forward_pass(network_input, forward, batch_size, deps);
    } else {
        CHECK_THROWS(network.forward_pass(network_input, forward, batch_size, deps));
        return;
    }

    q.memcpy(out.data(), network.GetOutput(forward, batch_size), sizeof(bf16) * out.size()).wait();
    q.memcpy(fwd.data(), reinterpret_cast<bf16 const *const>(forward), sizeof(bf16) * fwd.size()).wait();

    // this has only one layer, thus net_width instead of net_width * n_hidden_layer = net_width
    for (int i = 0; i < fwd.size(); i++) {
        if (i < batch_size * input_width_padded) {
            // no checks for input
        } else if ((i >= batch_size * input_width_padded) && (i < batch_size * (input_width_padded + net_width))) {
            // 1st layer output
            bf16 ref_result = (bf16)weight_val * (bf16)input_width * (bf16)input_val;
            // this checks that the amount of inputs multiplied by weight is correct after the first layer
            CHECK(ref_result == fwd[i]);
        } else if ((i >= batch_size * (input_width_padded + net_width)) &&
                   (i < batch_size * (input_width_padded + net_width + output_width_padded))) {
            int output_idx = i - batch_size * (input_width_padded + net_width);
            bf16 ref_result;
            if ((output_idx % output_width_padded) < output_width) {
                ref_result =
                    (bf16)weight_val * (bf16)input_width * (bf16)input_val * (bf16)net_width * (bf16)weight_val;
            } else {
                // this checks that the amount of inputs multiplied by weight is correct after the first two layers
                // Check that  output padded correctly
                ref_result = (bf16)0.0f;
            }
            CHECK(ref_result == fwd[i]);
        } else { // there shouldn't be anything here
            throw std::runtime_error("This case shouldn't exist. i:" + std::to_string(i));
        }
    }

    free(forward, q);
    network_input.free_mem(q);
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

TEST_CASE("Swiftnet Forward - zero pad output") {
    int input_width = 64;
    int net_width = 64;
    int n_hidden_layers = 1;
    int init_mode = 2;

    int batch_size = 512;

    SUBCASE("Output 64 (no pad)") {
        int output_width = 64;
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
    }
    SUBCASE("Output 63 (1 pad)") {
        int output_width = 63;
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
    }
    SUBCASE("Output 32 (32 pad)") {
        int output_width = 32;
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
    }
    SUBCASE("Output 16 (48 pad)") {
        int output_width = 16;
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
    }
    SUBCASE("Output 8 (56 pad)") {
        int output_width = 8;
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
    }
    SUBCASE("Output 4 (60 pad)") {
        int output_width = 4;
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
    }
    SUBCASE("Output 2 (62 pad)") {
        int output_width = 2;
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
    }
    SUBCASE("Output 1 (63 pad)") {
        int output_width = 1;
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
    }
    SUBCASE("Output -1 (failure)") {
        int output_width = -1;
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
    }
    SUBCASE("Output 0 (failure)") {
        int output_width = 0;
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
    }
}

TEST_CASE("Swiftnet Forward - Batch Sizes") {
    int input_width = 64;
    int net_width = 64;
    int output_width = 64;
    int n_hidden_layers = 1;
    int init_mode = 2;

    SUBCASE("Batch size 512") {
        int batch_size = 512;
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
    }
    SUBCASE("Batch size 16") {
        int batch_size = 16;
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
    }
    SUBCASE("Batch size 32") {
        int batch_size = 32;
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
    }
    SUBCASE("Batch size 64") {
        int batch_size = 64;
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
    }
    SUBCASE("Batch size 128") {
        int batch_size = 128;
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
    }
    SUBCASE("Batch size 1") {
        int batch_size = 1;
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
    }
    SUBCASE("Batch size 13") {
        int batch_size = 13;
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
    }
    SUBCASE("Batch size 513") {
        int batch_size = 513;
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
    }
    SUBCASE("Batch size 1024") {
        int batch_size = 1024;
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
    }
    SUBCASE("Batch size 1025") {
        int batch_size = 1025;
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
    }
}

TEST_CASE("Swiftnet Forward - Layer amount") {
    // only testing constructor. values tested later
    int input_width = 64;
    int net_width = 64;
    int output_width = 64;
    int init_mode = 2;
    int batch_size = 512;

    SUBCASE("Layer amount: 1 (success)") {
        int n_hidden_layers = 1;
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
    }
    SUBCASE("Layer amount: 0 (failure)") {
        int n_hidden_layers = 0;
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
    }
    SUBCASE("Layer amount: -1 (failure)") {
        int n_hidden_layers = -1;
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
    }
}

TEST_CASE("Swiftnet Forward - Net Widths") {
    // only testing constructor. values tested later
    int input_width = 64;
    int init_mode = 2;
    int batch_size = 512;
    int n_hidden_layers = 1;
    int output_width = 64;

    SUBCASE("Layer amount: 64 (success)") {
        int net_width = 64;
        int input_width_padded = net_width;
        test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
    }
    // SUBCASE("Layer amount: 32 (failure)") {
    //     int net_width = 32;
    //     test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
    // }
    // SUBCASE("Layer amount: 16 (failure)") {
    //     int net_width = 16;
    //     test_forward(input_width, output_width, n_hidden_layers, net_width, batch_size, init_mode);
    // }
}

TEST_CASE("Swiftnet Forward - Activation") {
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