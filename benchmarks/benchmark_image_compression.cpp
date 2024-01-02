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

// #define INCLUDE_COOLDOWN
// #define TEST_TRAINING
// #define CHECK_RESULTS
// #define TEST_INFERENCE
// #define DEBUG_OUTPUT
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

void start_training(const int WIDTH = 64, const int input_width = 32, const int output_width = 1,
                    const int m_n_hidden_layers = 4) {

    // SWIFTNET
    MPI_Init(NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // Fourth step: train the model by sampling the above image and optimizing
    // relative squared error using Adam.
    // Experimental not sure how many batch sizes are possible

    std::string method = "SwiftNet";
    nlohmann::json bench_result;
    bench_result[method] = nlohmann::json::array();

    const float scale = 1.0f;

    queue Q = queue(sycl::gpu_selector_v);
    std::vector<sycl::device> SubDevices;
    try {
        SubDevices = Q.get_device().create_sub_devices<sycl::info::partition_property::partition_by_affinity_domain>(
            sycl::info::partition_affinity_domain::numa);
    } catch (...) {
        SubDevices.resize(1);
        SubDevices[0] = Q.get_device();
    }

    auto C = sycl::context(SubDevices);

    std::cout << "Running on " << Q.get_device().get_info<sycl::info::device::name>() << " which has "
              << SubDevices.size() << " subdevices\n";

    double global_throughput = 0.0;
#pragma omp parallel num_threads(SubDevices.size()) shared(C, SubDevices)
    {
        const int dev_id = omp_get_thread_num();
        sycl::queue q(C, SubDevices[dev_id]);

        // std::cout << "Batch size 2^" << std::log2(batch_size) << std::endl;
        uint32_t batch_size = 8;

        DeviceMem<bf16> inputs(input_width * batch_size, q);
        DeviceMem<float> output(batch_size * output_width, q);
        DeviceMem<bf16> target(batch_size * output_width, q);
        DeviceMem<bf16> grads(batch_size * output_width, q);
        DeviceMem<bf16> losses(batch_size * output_width, q);

        // need a factory here for different widths
        SwiftNetMLP<64> network =
            SwiftNetMLP<64>(q, input_width, output_width, m_n_hidden_layers, Activation::ReLU, Activation::None);

        q.wait(); // wait for init netweork

        Trainer train(network);

        train.initialize_params(1);

        inputs.initialize_constant(0.001f, q);
        output.initialize_constant(0.0f, q);
        target.initialize_constant(0.1f, q);
        grads.initialize_constant(0, q);
        losses.initialize_constant(0, q);

        const size_t out_inter_forw_size = batch_size * (input_width + output_width + WIDTH * m_n_hidden_layers);

        const size_t out_inter_backw_size = batch_size * WIDTH * (m_n_hidden_layers + 1);

        float *out_inter_forw = sycl::malloc_device<float>(out_inter_forw_size, Q);
        float *out_inter_backw = sycl::malloc_device<float>(out_inter_backw_size, Q);

        q.wait(); // wait for init vals.

        // Various constants for the network and optimization
        uint32_t n_iterations = 1000;
        uint32_t n_iterations_warmup = n_iterations / 2;

        auto begin = std::chrono::steady_clock::now();
        auto begin_total = std::chrono::steady_clock::now();

        float tmp_loss = 0;
        uint32_t tmp_loss_counter = 0;

        uint32_t print_interval = n_iterations / 10;
        const uint32_t STEPS_INCREMENT = 5;

        double mean_training_throughput = 0;
        double mean_inference_throughput = 0;
        size_t mean_counter = 0;
        const float tolerance = 0.001f;
        std::vector<std::vector<float>> targetVectors = readTargetVectorsFromFile("../python/torch.csv", ',');
        std::vector<sycl::event> dependencies;
#ifdef TEST_TRAINING
        std::cout << "Iterations: " << n_iterations << ", steps increment: " << STEPS_INCREMENT << std::endl;

        for (uint32_t i = 0; i < n_iterations; i += STEPS_INCREMENT) {
            bool print_loss = i % print_interval == 0;

            for (uint32_t j = 0; j < STEPS_INCREMENT; ++j) {
                dependencies = train.training_step(inputs, losses, 0, out_inter_forw, out_inter_backw, batch_size);

                if (j == STEPS_INCREMENT - 1) {
                    tmp_loss += 0;
                    ++tmp_loss_counter;
                }
            }

            // Debug outputs
            if (print_loss) {
                q.wait();
                std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
                double throughput = print_interval * batch_size / ((double)microseconds / 1000000.0);
                std::cout << "Iteration#" << i << ": "
                          << "loss=" << tmp_loss / (float)tmp_loss_counter << " time=" << microseconds
                          << "[µs] thp=" << throughput << "/s" << std::endl;

                tmp_loss = 0;
                tmp_loss_counter = 0;

                if (i >= n_iterations_warmup) {
                    mean_training_throughput += throughput;
                    ++mean_counter;
                }

                begin = std::chrono::steady_clock::now();
            }
        }
        q.wait();

        std::chrono::steady_clock::time_point end_total = std::chrono::steady_clock::now();
        auto microseconds_total =
            std::chrono::duration_cast<std::chrono::microseconds>(end_total - begin_total).count();
        std::cout << "Training Time for " << n_iterations << " iterations: " << microseconds_total << "microseconds."
                  << std::endl;

        mean_training_throughput /= (double)mean_counter;

#pragma omp critical
        global_throughput += mean_training_throughput;

#pragma omp barrier

        double tmp_throughput = 0;
        MPI_Reduce(&global_throughput, &tmp_throughput, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (world_rank == 0) {
#pragma omp single
            {
                std::cout << "Finished training benchmark. Mean throughput is " << tmp_throughput
                          << "/s. Waiting 10 seconds for GPU to cool down." << std::endl;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

#pragma omp barrier

#pragma omp single
        global_throughput = 0.0;

#pragma omp barrier

        // Sanity check: we run with aranged weights and 4 layers and 0.001 as
        // input. Values generated from test_compare_torch_dpcpp.py and saved in
        // python/dpcpp.csv (bf16 vals). python/torch.csv is float vals
        std::vector<float> fwd(out_inter_forw_size);
        q.memcpy(fwd.data(), out_inter_forw, fwd.size() * sizeof(float));
        q.wait();

        std::vector<bf16> grads_vec(network.m_grads_matrices.size());
        network.m_grads_matrices.copy_to_host(grads_vec, q);
        q.wait();

#ifdef DEBUG_OUTPUT
        std::ofstream grad_csv("grads.csv");
        std::vector<double> grads_checksum_per_matrix(m_n_hidden_layers + 1, 0);
        for (int grads_mat_iter = 0; grads_mat_iter < m_n_hidden_layers + 1; grads_mat_iter++) {
            for (int grads_iter = 0; grads_iter < WIDTH * WIDTH; grads_iter++) {
                double val = (double)grads_vec[grads_mat_iter * WIDTH * WIDTH + grads_iter];
                grads_checksum_per_matrix[grads_mat_iter] += val;
                grad_csv << val << ",";
                if (std::isnan(val) || std::isinf(val))
                    std::cout << "Found nan or inf in matrix " << grads_mat_iter << " , i = " << grads_iter / WIDTH
                              << " , j = " << grads_iter % WIDTH << std::endl;
            }
            grad_csv << "\n";
        }
        grad_csv.close();

        for (int grads_mat_iter = 0; grads_mat_iter < m_n_hidden_layers + 1; grads_mat_iter++) {
            std::cout << "grads_mat checksum " << grads_mat_iter << " = " << grads_checksum_per_matrix[grads_mat_iter]
                      << std::endl;
        }

        // std::vector<bf16> backward_inter_host((m_n_hidden_layers+1)*batch_size*WIDTH);
        // q.memcpy(backward_inter_host.data(), network.m_out_inter,
        // sizeof(bf16)*(m_n_hidden_layers+1)*batch_size*WIDTH).wait(); std::ofstream
        // problems_in_out_inter_csv("problems_in_out_inter.csv"); for (int matiter = 0; matiter <
        // m_n_hidden_layers+1; matiter++) {
        //     for (int elemiter = 0; elemiter < batch_size*WIDTH; elemiter++) {
        //         if (std::isnan((float)backward_inter_host[matiter*batch_size*WIDTH + elemiter]) ||
        //             std::isinf((float)backward_inter_host[matiter*batch_size*WIDTH + elemiter]))
        //         {
        //             problems_in_out_inter_csv << matiter << "," << elemiter/WIDTH << "," << elemiter%WIDTH <<
        //             "\n";
        //         }
        //     }
        // }
        // problems_in_out_inter_csv.close();

        // //now look at the grads after the loss evaluate
        // std::vector<bf16> grads_after_loss(grads.size());
        // grads.copy_to_host(grads_after_loss, q);
        // std::cout << "Copy grads after loss eval to host done. N elements = " << grads_after_loss.size() <<
        // std::endl; q.wait(); std::ofstream grads_after_loss_csv("grads_after_loss.csv"); for (int iter = 0; iter
        // < grads_after_loss.size(); iter++) {
        //     grads_after_loss_csv << (double)grads_after_loss[iter] << ",";
        // }
        // grads_after_loss_csv.close();
#endif // DEBUG_OUTPUT

        // for (int i_g = 0; i_g < grads_vec.size(); i_g++) {
        //   std::cout << i_g << ": " << grads_vec[i_g] << std::endl;
        // }
#ifdef CHECK_RESULTS
        std::vector<std::vector<float>> targetGrads = readTargetVectorsFromFile("../python/torch_grads.csv", ',');

        // get every layer from fwd:
        for (int i = 0; i < m_n_hidden_layers + 2; i++) {
            size_t layer_size;
            size_t grads_layer_size;

            size_t start_idx;
            size_t end_idx;

            size_t start_idx_grads;
            size_t end_idx_grads;

            if (i == 0) { // input bf16? or float?
                layer_size = input_width;
                start_idx = 0;
                end_idx = input_width * batch_size;

                grads_layer_size = WIDTH * input_width;
                start_idx_grads = 0;
                end_idx_grads = WIDTH * input_width;
            } else if (i == m_n_hidden_layers + 1) { // output in float
                std::cout << "For output layer (out): ";
                // through layers between bf16 and float)
                std::vector<bf16> out(batch_size * WIDTH);
                q.memcpy(out.data(), network.GetOutput(out_inter_forw, batch_size), sizeof(bf16) * out.size()).wait();
                areVectorsWithinTolerance(out, targetVectors[i], 2 * tolerance, output_width);

                continue;
            } else {
                layer_size = WIDTH;
                start_idx = (input_width + WIDTH * (i - 1)) * batch_size;
                end_idx = (input_width + WIDTH * (i)) * batch_size;

                grads_layer_size = WIDTH * WIDTH;
                start_idx_grads = WIDTH * input_width + (WIDTH * WIDTH) * (i - 1);
                end_idx_grads = WIDTH * input_width + (WIDTH * WIDTH) * (i);
            }

            std::vector<bf16> layer(batch_size * layer_size);
            std::copy(fwd.begin() + start_idx, fwd.begin() + end_idx, layer.begin());
            for (size_t iter = start_idx, iter_shift = 0; iter < end_idx; iter++, iter_shift++) {
                layer[iter_shift] = reinterpret_cast<bf16 *>(fwd.data())[iter];
            }

            std::vector<bf16> layer_grads(grads_layer_size);
            std::copy(grads_vec.begin() + start_idx_grads, grads_vec.begin() + end_idx_grads, layer_grads.begin());

            std::cout << "Layer " << i << ", start_idx: " << start_idx << ", end_idx: " << end_idx << ". ";

            areVectorsWithinTolerance(layer, targetVectors[i], tolerance, output_width);

            //   std::cout << "Grads Layer " << i << ", start_idx: " <<
            //   start_idx_grads
            //             << ", end_idx: " << end_idx_grads;

            // areVectorsWithinTolerance(layer_grads, targetGrads[i], tolerance);
        }
#endif // CHECK_RESULTS

#endif // TEST_TRAINING

#ifdef INCLUDE_COOLDOWN
        std::this_thread::sleep_for(std::chrono::seconds{10});
#endif
#ifdef TEST_INFERENCE
#pragma omp barrier

        begin = std::chrono::steady_clock::now();
        begin_total = std::chrono::steady_clock::now();

        // Inference benchmark
        mean_counter = 0;

        print_interval *= 5;
        n_iterations *= 5;
        n_iterations_warmup *= 5;
        auto begin_time_inference = std::chrono::steady_clock::now();
        dependencies = {};
        for (uint32_t i = 0; i < n_iterations; ++i) {
            if (i == n_iterations_warmup)
                begin_time_inference = std::chrono::steady_clock::now(); // start measuring after warmup

            // Inference step
            // dependencies =
            // train.training_step(inputs, output, target, grads, losses, scale, WIDTH, 1, dependencies);
            dependencies = train.training_step(inputs, losses, 1, out_inter_forw, out_inter_backw, batch_size);

            // Debug outputs
            if (i % print_interval == 0) {
                q.wait();
                auto end = std::chrono::steady_clock::now();
                auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
                double throughput = print_interval * batch_size / ((double)microseconds / 1000000.0);
                std::cout << "Iteration#" << i << ": "
                          << "time=" << microseconds << "[µs] thp=" << throughput << "/s" << std::endl;

                if (i >= n_iterations_warmup) {
                    mean_inference_throughput += throughput;
                    ++mean_counter;
                }

                begin = std::chrono::steady_clock::now();
            }
        }
        q.wait();

        end_total = std::chrono::steady_clock::now();
        microseconds_total = std::chrono::duration_cast<std::chrono::microseconds>(end_total - begin_total).count();
        std::cout << "Inference Time for " << n_iterations << " iterations: " << microseconds_total << "microseconds."
                  << std::endl;

        auto end_time_inference = std::chrono::steady_clock::now();
        double elapsed_time_inference =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end_time_inference - begin_time_inference).count();
        std::cout << "Elapse time inference = " << int(elapsed_time_inference / 1000.0)
                  << "[µs] \tAverage time inference for plot interval = "
                  << int(print_interval * elapsed_time_inference / 1000.0 / (n_iterations - n_iterations_warmup))
                  << " [µs]" << std::endl;
        const double flops_inference = 2.0 * (double)batch_size * (double)WIDTH * (double)WIDTH *
                                       (m_n_hidden_layers + 1) * (n_iterations - n_iterations_warmup);
        const double gflops_per_s_inference = flops_inference / elapsed_time_inference; // flops / ns = gflops/s
        const double bytes_loaded_and_stored_inference =
            (2.0 * batch_size * WIDTH + WIDTH * WIDTH) * 2 *
            (n_iterations - n_iterations_warmup); // load A once, store A once per iteration. load B once. Times 2
                                                  // byte since everything is bf16
        const double hbm_bandwidth_GB_per_s_inference = bytes_loaded_and_stored_inference / elapsed_time_inference;

        mean_inference_throughput /= (double)mean_counter;

#pragma omp barrier

#pragma omp critical
        global_throughput += mean_inference_throughput;

#pragma omp barrier

        tmp_throughput = 0;
        MPI_Reduce(&global_throughput, &tmp_throughput, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (world_rank == 0) {
#pragma omp single
            { std::cout << "Finished inference benchmark. Mean throughput/s is " << tmp_throughput << std::endl; }
        }

#pragma omp single
        global_throughput = 0.0;

        if (world_rank == 0) {
#pragma omp critical
            std::cout << " GFlops/s = " << gflops_per_s_inference
                      << ", Bandwidth GB/s = " << hbm_bandwidth_GB_per_s_inference
                      << "\nWaiting 10 seconds for GPU to cool down." << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);

#pragma omp barrier

#ifdef CHECK_RESULTS
        std::vector<bf16> outbf16(batch_size * WIDTH, 0);
        q.memcpy(outbf16.data(),
                 reinterpret_cast<bf16 *>(network.m_forward) + WIDTH * batch_size * (m_n_hidden_layers + 1),
                 sizeof(bf16) * batch_size * WIDTH)
            .wait();
        areVectorsWithinTolerance(outbf16, targetVectors.back(), tolerance * 2);
        // accumulated tolerance for last layer due to
        // accumulated error between bf16 and float
#endif // CHECK_RESULTS
#endif // TEST_INFERENCE
        q.wait();
        inputs.free_mem(q);
        output.free_mem(q);
        target.free_mem(q);
        grads.free_mem(q);
        losses.free_mem(q);

#ifdef INCLUDE_COOLDOWN
        std::this_thread::sleep_for(std::chrono::seconds{10});
#endif

        bench_result[method].push_back({
            {"batch_size", batch_size},
            {"training_throughput", mean_training_throughput},
            {"inference_throughput", mean_inference_throughput},
        });
        q.wait();

        if (world_rank == 0) {
            std::string json_string = bench_result.dump(4);
            std::ofstream out{std::string("../benchmarks/results/benchmark_image_compression.json")};
            out << json_string;
        }
    }

    MPI_Finalize();
}

int main() {
    // start_training();
    std::cout << "generating final" << std::endl;
    generate_image("");
    // std::cout << "generating 0" << std::endl;
    // generate_image("0");
    // std::cout << "generating 10" << std::endl;
    // generate_image("10");
    // std::cout << "generating 100" << std::endl;
    // generate_image("100");
    // std::cout << "generating 1000" << std::endl;
    // generate_image("1000");
    return 0;
}
