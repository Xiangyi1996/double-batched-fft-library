// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "doctest/doctest.h"
#include <iostream>
#include <vector>

#include "activation.h"
#include "network_with_encodings.h"

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
using bf16 = sycl::ext::oneapi::bfloat16;

#define INPUT_WIDTH 64
#define OUTPUT_WIDTH_PADDED 64
#define HIDDEN_LAYERS 1
#define NET_WIDTH 64
void test_network_with_encoding() {
    // SWIFTNET
    const int batch_size = 512;
    const int output_width = 64;
    const int m_n_hidden_layers = HIDDEN_LAYERS;

    GPUMatrix<float> input(INPUT_WIDTH, batch_size);
    //   GPUMatrix<float> input(batch_size, INPUT_WIDTH);
    input.initialize_constant(0.1f);

    //   Define the parameters for creating IdentityEncoding
    std::unordered_map<std::string, std::string> encoding = {
        {"n_dims_to_encode", std::to_string(INPUT_WIDTH)}, {"scale", "1.0"}, {"offset", "0.0"}};
    std::string encoding_name = "Identity";

    //   std::unordered_map<std::string, std::string> encoding = {
    //       {"n_dims_to_encode", std::to_string(INPUT_WIDTH)},
    //       {"degree", std::to_string(4)}};
    //   std::string encoding_name = "SphericalHarmonics";
    NetworkWithEncoding network = NetworkWithEncoding(INPUT_WIDTH, output_width, m_n_hidden_layers, Activation::ReLU,
                                                      Activation::None, encoding_name, encoding);
    network.initialize_params(2);
    sycl::queue q;

    float *forward =
        malloc_device<float>(batch_size * (INPUT_WIDTH + OUTPUT_WIDTH_PADDED + NET_WIDTH * m_n_hidden_layers), q);
    DeviceMem<bf16> network_output = DeviceMem<bf16>(OUTPUT_WIDTH_PADDED * batch_size, q);

    network.forward_pass(input, 0, network_output, forward);

    std::vector<bf16> fwd(batch_size * (INPUT_WIDTH + OUTPUT_WIDTH_PADDED + NET_WIDTH * m_n_hidden_layers));
    q.memcpy(fwd.data(), reinterpret_cast<bf16 const *const>(forward), sizeof(bf16) * fwd.size()).wait();

    for (int j = 0; j < fwd.size(); j++) {
        if (j % (512 * 64) == 0) {
            std::cout << "__________________________" << std::endl;
        }
        std::cout << "fwd, " << j << ": " << fwd[j] << std::endl;
    }

    std::vector<bf16> out(batch_size * (OUTPUT_WIDTH_PADDED));

    network_output.copy_to_host(out, q);
    for (int j = 0; j < out.size(); j++) {
        std::cout << "out, " << j << ": " << static_cast<float>(out[j]) << std::endl;
    }
    network.free_memory();
}

TEST_CASE("tinydpcppnn::network_with_encoding Fwd test") { test_network_with_encoding(); }