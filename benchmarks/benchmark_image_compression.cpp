// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include <CL/sycl.hpp>
#include <chrono>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <string>
#include <thread>
#include <vector>

#include "SwiftNetMLP.h"
#include "activation.h"
#include "common.h"
#include "encoding_factory.h"
#include "mkl.h"
#include "mkl_omp_offload.h"
#include "mpi.h"
#include "oneapi/mkl.hpp"
#include "result_check.h"
#include "trainer.h"

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
using bf16 = sycl::ext::oneapi::bfloat16;

template <typename T> std::vector<uint8_t> to_ldr(const std::vector<T> &in_vec, size_t n_channels, size_t stride) {
    std::vector<uint8_t> out_vec;

    for (size_t i = 0; i < in_vec.size(); ++i) {
        const size_t pixel = i / n_channels;
        const size_t channel = i - pixel * n_channels;

        // Clamp the input value between 0.0 and 1.0
        float clamped_value = std::min(std::max(static_cast<float>(in_vec[pixel * stride + channel]), 0.0f), 1.0f);

        // Apply gamma correction (1/2.2) and scale to the range [0, 255]
        out_vec.push_back(static_cast<uint8_t>(std::pow(clamped_value, 1.0f / 2.2f) * 255.0f + 0.5f));
    }

    return out_vec;
}

// Function to sample the mesh grid and create a vector
std::vector<float> sampleMeshGrid(int resolutionX, int resolutionY) {
    std::vector<float> meshVector(resolutionX * resolutionY * 2);

    for (int y = 0; y < resolutionY; ++y) {
        for (int x = 0; x < resolutionX; ++x) {
            int idx = (y * resolutionX + x) * 2;
            meshVector[idx + 0] = (float)(x + 0.5) / (float)resolutionX;
            meshVector[idx + 1] = (float)(y + 0.5) / (float)resolutionY;
        }
    }
    return meshVector;
}

void generate_image(std::string iter) {
    const int WIDTH = 64;
    const int input_width = 32;
    const int output_width = 1;

    const int output_width_padded = WIDTH; // we pad the remainder to 0
    const int input_width_padded = WIDTH;  // we pad the remainder to 0
    const int m_n_hidden_layers = 2;
    int x_res = 384 * 6;
    int y_res = 512 * 6;

    std::vector<float> params =
        loadVectorFromCSV<float>("../test/ref_values/network_with_grid_encoding/full/encoding_params" + iter + ".csv");
    std::vector<float> input_ref = sampleMeshGrid(x_res, y_res);
    int batch_size = x_res * y_res;

    // std::vector<float> input_ref =
    //     loadVectorFromCSV<float>("../test/ref_values/network_with_grid_encoding/full/input.csv");
    // int batch_size = 512 * 1;
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

    DeviceMem<bf16> inputs(input_width_padded * batch_size, q);
    std::cout << "Batch size: " << batch_size << ", with input ref size: " << input_ref.size()
              << ", input size: " << inputs.size() << std::endl;
    // if (inputs.size() != input_ref.size()) {
    //     throw std::invalid_argument("Input ref (loaded) and defined not the same size");
    // };
    // inputs.copy_from_host(input_ref, q);

    inputs.initialize_constant(0.0f, q);
    q.wait();
    // need a factory here for different widths
    SwiftNetMLP<64> network =
        SwiftNetMLP<64>(q, input_width, output_width, m_n_hidden_layers, Activation::ReLU, Activation::None);
    q.wait(); // wait for init netweork
    network.load_from_file("../test/ref_values/network_with_grid_encoding/full/network_params" + iter + ".csv");
    // network.initialize_params(2);

    q.wait(); // wait for init vals.
    const size_t out_inter_forw_size =
        batch_size * (input_width_padded + output_width_padded + WIDTH * m_n_hidden_layers);

    float *out_inter_forw = sycl::malloc_device<float>(out_inter_forw_size, q);
    // load weights, infer image first before benchmark
    DeviceMem<float> input_encoding_dm(encoding_input_width * batch_size, q);
    input_encoding_dm.initialize_constant(1.0f, q);
    GPUMatrix<float> input_encoding(input_encoding_dm.data(), encoding_input_width, batch_size);

    DeviceMem<float> output_encoding_dm(input_width_padded * batch_size, q);
    output_encoding_dm.initialize_constant(0.0f, q);
    GPUMatrix<float> output_encoding(output_encoding_dm.data(), input_width_padded, batch_size);
    GridEncoding<float> *encoding = create_grid_encoding<float>(encoding_input_width, encoding_json);

    DeviceMem<float> params_full_precision(encoding->n_params(), q);

    std::cout << "Params ref size: " << params.size() << ", grid size: " << encoding->n_params() << std::endl;
    std::cout << "Input ref size: " << input_ref.size() << ", input size: " << inputs.size() << std::endl;
    params_full_precision.copy_from_host(params, q);
    q.wait();
    input_encoding_dm.copy_from_host(input_ref, q);
    q.wait();
    // std::cout << "Input encoding: " << std::endl;
    // input_encoding.print();

    encoding->set_params(params_full_precision.data(), params_full_precision.data(), nullptr);
    q1.wait();
    std::unique_ptr<Context> model_ctx = encoding->forward_impl(&q1, input_encoding, &output_encoding);
    q1.wait();
    q.wait();

    // // std::cout << "Output encoding: " << output_encoding.n_elements() << std::endl;
    // // output_encoding.print();

    // std::vector<float> encoding_output(input_width_padded * batch_size);
    // output_encoding_dm.copy_to_host(encoding_output, q);
    // q1.wait();
    // q.wait();

    // std::vector<float> encoding_output_ref =
    //     loadVectorFromCSV<float>("../test/ref_values/network_with_grid_encoding/full/encoding_output.csv");
    // std::cout << "Size ref: " << encoding_output_ref.size() << ", encoding output size: " << encoding_output.size()
    //           << std::endl;
    // for (int i = 0; i < encoding_output_ref.size(); i++) {
    //     std::cout << i << ", Ref: " << encoding_output_ref[i] << " / " << encoding_output[i] << std::endl;
    // }
    // std::cout << "Output encoding: " << output_encoding.n_elements() << std::endl;
    // output_encoding.print();
    // return;

    // output_encoding_dm.copy_from_host(encoding_output_ref, q);
    // q1.wait();
    // q.wait();
    inputs.set_values(output_encoding.n_elements(), output_encoding.data(), q);
    // std::vector<bf16> encoding_output_ref =
    //     loadVectorFromCSV<bf16>("../test/ref_values/network_with_grid_encoding/full/encoding_output.csv");

    // inputs.copy_from_host(encoding_output_ref, q);

    q1.wait();
    q.wait();

    // std::cout << "Input to network with size: " << inputs.size() << std::endl;

    // std::vector<bf16> inputs_vec(input_width_padded * batch_size);

    // q.memcpy(inputs_vec.data(), inputs.data(), sizeof(bf16) * inputs_vec.size()).wait();

    // q1.wait();
    // q.wait();

    // for (int i = 0; i < inputs_vec.size(); i++) {
    //     std::cout << i << ": " << inputs_vec[i] << std::endl;
    // }

    // std::cout << "Sleeping for 5 seconds..." << std::endl;

    // // Sleep for 5 seconds
    // std::this_thread::sleep_for(std::chrono::seconds(5));

    std::cout << "inference: " << std::endl;

    // network.inference(inputs, out_inter_forw, batch_size, {});
    network.forward_pass(inputs, out_inter_forw, batch_size, {});

    std::cout << "inference done: " << std::endl;

    std::vector<bf16> out_vec(out_inter_forw_size);
    q.memcpy(out_vec.data(), reinterpret_cast<bf16 const *const>(out_inter_forw), out_vec.size() * sizeof(bf16));
    q.wait();
    std::vector<unsigned char> image(batch_size);

    // int counter = 0;
    // // for (size_t i = 0; i < out_vec.size(); i++) {
    // for (size_t i = batch_size * (input_width_padded + WIDTH * m_n_hidden_layers); i < out_vec.size(); i++) {
    //     // for (size_t i = batch_size * (input_width_padded + WIDTH * m_n_hidden_layers); i < out_vec.size(); i +=
    //     // WIDTH) { for (int j = 0; j < output_width; ++j) { image[counter] = static_cast<unsigned char>(out_vec[i +
    //     j]
    //     // * 255);
    //     // if ((i == batch_size * input_width_padded) | (i == batch_size * (input_width_padded + WIDTH)) |
    //     //     (i == batch_size * (input_width_padded + WIDTH + WIDTH))) {

    //     //     std::cout << "==========================================" << std::endl;
    //     // }
    //     counter++;
    //     // std::cout << i + j << "- Out: " << out_vec[i + j] << std::endl;
    //     if (i > batch_size * (input_width_padded + WIDTH + WIDTH) && (out_vec[i] != 0)) {
    //         // CHECK(mape < 10.0); // MAPE is in per cent
    //         // if (mape > 10.0) {
    //         std::cout << counter << " - Out: " << out_vec[i] << std::endl;

    //         // std::cout << i << ": " << fwd[i] << " / " << ref_result << ", mape: " << mape << std::endl;
    //         // }
    //     }
    //     // }
    // }

    int counter = 0;
    // for (size_t i = batch_size * (input_width_padded + WIDTH * m_n_hidden_layers); i < out_vec.size(); i++) {
    for (size_t i = batch_size * (input_width_padded + WIDTH * m_n_hidden_layers); i < out_vec.size(); i += WIDTH) {
        for (int j = 0; j < output_width; ++j) {
            image[counter] = static_cast<unsigned char>(out_vec[i + j] * 255);
            // std::cout << i + j << " - Out: " << out_vec[i + j] * 255 << std::endl;
            if ((image[counter] < 0) || (image[counter] > 255)) {
                std::cout << counter << ": " << image[counter] << std::endl;
            }
            counter++;
        }
    }

    // Set the filename for the saved image
    std::string filename = "output_image" + iter + ".pgm";
    // auto image_postprocessed = to_ldr(image, 1, 1);
    // Assuming x_res and y_res represent the dimensions of the image
    // saveImageToPGM(filename, x_res, y_res, image_postprocessed);
    saveImageToPGM(filename, x_res, y_res, image);
}

int main() {
    std::cout << "generating final" << std::endl;
    generate_image("");
    std::cout << "generating 0" << std::endl;
    generate_image("0");
    std::cout << "generating 10" << std::endl;
    generate_image("10");
    std::cout << "generating 100" << std::endl;
    generate_image("100");
    std::cout << "generating 1000" << std::endl;
    generate_image("1000");
    return 0;
}
