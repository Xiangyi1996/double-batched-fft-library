// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "doctest/doctest.h"
#include <iostream>
#include <vector>

#include "activation.h"
#include "network_with_encodings.h"
#include "result_check.h"

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
using bf16 = sycl::ext::oneapi::bfloat16;

#define INPUT_WIDTH 2
#define OUTPUT_WIDTH_PADDED 64
#define HIDDEN_LAYERS 1
#define NET_WIDTH 64

void test_network_with_encoding() {
    int use_encoding = 1;
    // const int WIDTH = 64;
    const int input_width = 64;
    const int output_width = 1;

    const int output_width_padded = WIDTH; // we pad the remainder to 0
    const int m_n_hidden_layers = 1;
    int batch_size = 8;

    int encoding_input_width = 2;
    json encoding_json = {
        {"n_dims_to_encode", std::to_string(encoding_input_width)},
        {"otype", "Grid"},
        {"type", "Hash"},
        {"n_levels", 16},
        {"n_features_per_level", 2},
        {"log2_hashmap_size", 15},
        {"base_resolution", 16},
        {"per_level_scale", 1.5},
    };
    sycl::queue q;
    sycl::queue q1;

    DeviceMem<bf16> inputs(input_width * batch_size, q);
    inputs.initialize_constant(0.01f, q);

    // std::cout << "Batch size 2^" << std::log2(batch_size) << std::endl;

    // need a factory here for different widths
    SwiftNetMLP<64> network =
        SwiftNetMLP<64>(q, input_width, output_width, m_n_hidden_layers, Activation::ReLU, Activation::None);
    q.wait(); // wait for init netweork
    // network.load_from_file("network_params.csv");
    network.initialize_params(2);

    q.wait(); // wait for init vals.
    const size_t out_inter_forw_size = batch_size * (input_width + output_width_padded + WIDTH * m_n_hidden_layers);

    float *out_inter_forw = sycl::malloc_device<float>(out_inter_forw_size, q);
    // load weights, infer image first before benchmark
    DeviceMem<float> input_encoding_dm(encoding_input_width * batch_size, q);
    input_encoding_dm.initialize_constant(1.0f, q);
    GPUMatrix<float> input_encoding(input_encoding_dm.data(), encoding_input_width, batch_size);

    DeviceMem<float> output_encoding_dm(input_width * batch_size, q);
    output_encoding_dm.initialize_constant(1.0f, q);
    GPUMatrix<float> output_encoding(output_encoding_dm.data(), input_width, batch_size);
    GridEncoding<float> *encoding = create_grid_encoding<float>(encoding_input_width, encoding_json);

    std::vector<float> params =
        loadVectorFromCSV<float>("../test/ref_values/network_with_grid_encoding/encoding_params.csv");
    std::vector<float> input_ref = loadVectorFromCSV<float>("../test/ref_values/network_with_grid_encoding/input.csv");
    DeviceMem<float> params_full_precision(encoding->n_params(), q);
    if (use_encoding) {

        std::cout << "Params ref size: " << params.size() << ", grid size: " << encoding->n_params() << std::endl;
        // std::cout << "Input ref size: " << input_ref.size() << ", input size: " << inputs.size() << std::endl;
        params_full_precision.copy_from_host(params, q);
        // input_encoding_dm.copy_from_host(input_ref, q);
        // std::cout << "Input: " << std::endl;
        input_encoding.print();

        // params_full_precision.initialize_arange(q);

        // encoding->set_padded_output_width(input_width);
        encoding->set_params(params_full_precision.data(), params_full_precision.data(), nullptr);

        std::unique_ptr<Context> model_ctx = encoding->forward_impl(&q1, input_encoding, &output_encoding);
        q1.wait();
        std::cout << "Output: " << output_encoding.n_elements() << std::endl;
        output_encoding.print();

        inputs.set_values(output_encoding.n_elements(), output_encoding.data(), q);
        std::cout << "Input: " << inputs.size() << std::endl;

        std::vector<bf16> inputs_vec(input_width * batch_size);

        q.memcpy(inputs_vec.data(), inputs.data(), sizeof(bf16) * inputs_vec.size()).wait();

        for (int i = 0; i < inputs_vec.size(); i++) {
            std::cout << "i: " << inputs_vec[i] << std::endl;
        }

        q1.wait();
    }
    std::cout << "inference: " << std::endl;

    network.inference(inputs, out_inter_forw, batch_size, {});

    std::cout << "inference done: " << std::endl;

    std::vector<bf16> out_vec(out_inter_forw_size);
    q.memcpy(out_vec.data(), reinterpret_cast<bf16 const *const>(out_inter_forw), out_vec.size() * sizeof(bf16));
    q.wait();
    std::cout << "Output: " << out_vec.size() << std::endl;

    for (int i = 0; i < out_vec.size(); i++) {
        if (i >= batch_size * (input_width + WIDTH * m_n_hidden_layers) && (out_vec[i] != 0)) {
            std::cout << i << ": " << out_vec[i] << std::endl;
        }
    }

    // saveCSV("output.csv", out_vec);
}
// void test_network_with_encoding() {
//     // // const int WIDTH = 64;
//     // const int input_width = 64;
//     // const int output_width = 1;
//     // const int m_n_hidden_layers = 4;
//     // int encoding_input_width = 2;
//     // json encoding_json = {
//     //     {"n_dims_to_encode", std::to_string(encoding_input_width)},
//     //     {"otype", "Grid"},
//     //     {"type", "Hash"},
//     //     {"n_levels", 16},
//     //     {"n_features_per_level", 2},
//     //     {"log2_hashmap_size", 15},
//     //     {"base_resolution", 16},
//     //     {"per_level_scale", 1.5},
//     // };
//     // sycl::queue q;

//     // // std::cout << "Batch size 2^" << std::log2(batch_size) << std::endl;
//     // uint32_t batch_size = 8;

//     // DeviceMem<bf16> inputs(input_width * batch_size, q);
//     // DeviceMem<float> output(batch_size * output_width, q);
//     // DeviceMem<bf16> target(batch_size * output_width, q);
//     // DeviceMem<bf16> grads(batch_size * output_width, q);
//     // DeviceMem<bf16> losses(batch_size * output_width, q);

//     // // need a factory here for different widths
//     // SwiftNetMLP<64> network =
//     //     SwiftNetMLP<64>(q, input_width, output_width, m_n_hidden_layers, Activation::ReLU, Activation::None);

//     // q.wait(); // wait for init netweork

//     // Trainer train(network);

//     // train.initialize_params(1);

//     // inputs.initialize_constant(0.001f, q);
//     // output.initialize_constant(0.0f, q);
//     // target.initialize_constant(0.1f, q);
//     // grads.initialize_constant(0, q);
//     // losses.initialize_constant(0, q);

//     // const size_t out_inter_forw_size = batch_size * (input_width + output_width + WIDTH * m_n_hidden_layers);

//     // const size_t out_inter_backw_size = batch_size * WIDTH * (m_n_hidden_layers + 1);

//     // float *out_inter_forw = sycl::malloc_device<float>(out_inter_forw_size, q);
//     // float *out_inter_backw = sycl::malloc_device<float>(out_inter_backw_size, q);

//     // q.wait(); // wait for init vals.

//     // // load weights, infer image first before benchmark
//     // DeviceMem<float> input_encoding_dm(encoding_input_width * batch_size, q);
//     // input_encoding_dm.initialize_constant(1.0f, q);
//     // GPUMatrix<float> input_encoding(input_encoding_dm.data(), encoding_input_width, batch_size);

//     // DeviceMem<float> output_encoding_dm(input_width * batch_size, q);
//     // output_encoding_dm.initialize_constant(1.0f, q);
//     // GPUMatrix<float> output_encoding(output_encoding_dm.data(), input_width, batch_size);

//     // GridEncoding<float> *encoding = create_grid_encoding<float>(encoding_input_width, encoding_json);

//     // std::vector<float> params = loadVectorFromCSV("encoding_params.csv");
//     // // std::vector<float> input_ref = loadVectorFromCSV("input.csv");
//     // DeviceMem<float> params_full_precision(encoding->n_params(), q);

//     // std::cout << "Params ref size: " << params.size() << ", grid size: " << encoding->n_params() << std::endl;
//     // // std::cout << "Input ref size: " << input_ref.size() << ", input size: " << inputs.size() << std::endl;
//     // params_full_precision.copy_from_host(params, q);
//     // // input_encoding_dm.copy_from_host(input_ref, q);
//     // // std::cout << "Input: " << std::endl;
//     // input_encoding.print();

//     // // params_full_precision.initialize_arange(q);

//     // encoding->set_params(params_full_precision.data(), params_full_precision.data(), nullptr);

//     // std::unique_ptr<Context> model_ctx = encoding->forward_impl(&q, input_encoding, &output_encoding);
//     // q.wait();
//     // std::cout << "Output: " << output_encoding.n_elements() << std::endl;
//     // output_encoding.print();

//     // inputs.set_values(output_encoding.n_elements(), output_encoding.data(), q);
//     // std::cout << "Input: " << inputs.size() << std::endl;

//     // std::vector<bf16> inputs_vec(inputs.size());

//     // q.memcpy(inputs_vec.data(), inputs.data(), sizeof(bf16) * inputs_vec.size()).wait();

//     // for (int i = 0; i << input_width * batch_size; i++) {
//     //     std::cout << "i: " << i << std::endl;
//     // }
//     // std::cout << "inf: " << std::endl;

//     // network.inference(inputs, out_inter_forw, batch_size, {});
//     // std::cout << "inf done: " << std::endl;

//     // std::vector<float> out_vec(batch_size * output_width);
//     // q.memcpy(out_vec.data(), out_inter_forw + batch_size * (input_width + WIDTH * m_n_hidden_layers),
//     //          out_vec.size() * sizeof(float));
//     // q.wait();
//     // std::cout << "Output: " << out_vec.size() << std::endl;

//     // for (int i = 0; i << 8; i++) {
//     //     std::cout << "i: " << i << std::endl;
//     // }
//     // saveCSV("output.csv", out_vec);

//     // SWIFTNET
//     const int batch_size = 8;
//     const int output_width = 64;
//     const int m_n_hidden_layers = HIDDEN_LAYERS;

//     GPUMatrix<float> input(INPUT_WIDTH, batch_size);
//     //   GPUMatrix<float> input(batch_size, INPUT_WIDTH);
//     input.initialize_constant(0.1f);

//     //   Define the parameters for creating IdentityEncoding
//     std::unordered_map<std::string, std::string> encoding = {
//         {"n_dims_to_encode", std::to_string(INPUT_WIDTH)}, {"scale", "1.0"}, {"offset", "0.0"}};
//     std::string encoding_name = "Identity";

//     //   std::unordered_map<std::string, std::string> encoding = {
//     //       {"n_dims_to_encode", std::to_string(INPUT_WIDTH)},
//     //       {"degree", std::to_string(4)}};
//     //   std::string encoding_name = "SphericalHarmonics";
//     NetworkWithEncoding network = NetworkWithEncoding(INPUT_WIDTH, output_width, m_n_hidden_layers, Activation::ReLU,
//                                                       Activation::None, encoding_name, encoding);
//     network.initialize_params(2);
//     sycl::queue q;

//     float *forward =
//         malloc_device<float>(batch_size * (INPUT_WIDTH + OUTPUT_WIDTH_PADDED + NET_WIDTH * m_n_hidden_layers), q);
//     DeviceMem<bf16> network_output = DeviceMem<bf16>(OUTPUT_WIDTH_PADDED * batch_size, q);

//     network.forward_pass(input, 0, network_output, forward);

//     std::vector<bf16> fwd(batch_size * (INPUT_WIDTH + OUTPUT_WIDTH_PADDED + NET_WIDTH * m_n_hidden_layers));
//     q.memcpy(fwd.data(), reinterpret_cast<bf16 const *const>(forward), sizeof(bf16) * fwd.size()).wait();

//     for (int j = 0; j < fwd.size(); j++) {
//         if (j % (512 * 64) == 0) {
//             std::cout << "__________________________" << std::endl;
//         }
//         std::cout << "fwd, " << j << ": " << fwd[j] << std::endl;
//     }

//     // std::vector<bf16> out(batch_size * (OUTPUT_WIDTH_PADDED));

//     // network_output.copy_to_host(out, q);
//     // for (int j = 0; j < out.size(); j++) {
//     //     std::cout << "out, " << j << ": " << static_cast<float>(out[j]) << std::endl;
//     // }
//     network.free_memory();
// }

TEST_CASE("tinydpcppnn::network_with_encoding Fwd test") { test_network_with_encoding(); }