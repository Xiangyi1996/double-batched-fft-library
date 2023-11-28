// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include <CL/sycl.hpp>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "SwiftNetMLP.h"
#include "activation.h"
#include "common.h"
#include "mkl.h"
#include "mpi.h"
#include "oneapi/mkl.hpp"
#include "result_check.h"
#include "trainer.h"

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
using bf16 = sycl::ext::oneapi::bfloat16;

// #define INCLUDE_COOLDOWN
#define TEST_TRAINING
// #define CHECK_RESULTS
#define TEST_INFERENCE
// #define DEBUG_OUTPUT

void start_training(const int WIDTH = 64, const int input_width = 64, const int output_width = 64,
                    const int m_n_hidden_layers = 4) {

    // SWIFTNET
    MPI_Init(NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // Fourth step: train the model by sampling the above image and optimizing
    // relative squared error using Adam.
    // Experimental not sure how many batch sizes are possible
    std::vector<uint32_t> batch_sizes = {/*1 << 29, 1 << 28, 1 << 27, 1 << 26, 1 << 25, 1 << 24, 1 << 23,*/
                                         1 << 22, 1 << 21, 1 << 20, 1 << 19, 1 << 18, 1 << 17, 1 << 16,
                                         1 << 15, 1 << 14, 1 << 13, 1 << 12, 1 << 11, 1 << 10};
    std::string method = "SwiftNet";
    nlohmann::json bench_result;
    bench_result[method] = nlohmann::json::array();

    const float scale = 1.0f;

    sycl::queue q(sycl::gpu_selector_v);

    std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    double global_throughput = 0.0;

    for (uint32_t batch_size : batch_sizes) {
        std::cout << "Batch size 2^" << std::log2(batch_size) << std::endl;

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

        float *out_inter_forw = sycl::malloc_device<float>(out_inter_forw_size, q);
        float *out_inter_backw = sycl::malloc_device<float>(out_inter_backw_size, q);

        q.wait(); // wait for init vals.

        // Various constants for the network and optimization
        uint32_t n_iterations = std::max(1000 * (1 << 18) / batch_size, 250u);
        uint32_t n_iterations_warmup = n_iterations / 2;

        auto begin = std::chrono::steady_clock::now();

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
        double tmp_throughput = 0;
#ifdef TEST_TRAINING
        std::cout << "Iterations: " << n_iterations << ", steps increment: " << STEPS_INCREMENT << std::endl;

        for (uint32_t i = 0; i < n_iterations; i += STEPS_INCREMENT) {
            bool print_loss = i % print_interval == 0;

            for (uint32_t j = 0; j < STEPS_INCREMENT; ++j) {
                dependencies =
                    train.training_step(inputs, losses, 0, out_inter_forw, out_inter_backw, batch_size, dependencies);

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

        mean_training_throughput /= (double)mean_counter;

        global_throughput += mean_training_throughput;

        MPI_Reduce(&global_throughput, &tmp_throughput, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (world_rank == 0) {
            std::cout << "Finished training benchmark. Mean throughput is " << tmp_throughput
                      << "/s. Waiting 10 seconds for GPU to cool down." << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);

        global_throughput = 0.0;

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

        begin = std::chrono::steady_clock::now();

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
            dependencies =
                train.training_step(inputs, losses, 1, out_inter_forw, out_inter_backw, batch_size, dependencies);

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

        global_throughput += mean_inference_throughput;

        tmp_throughput = 0;
        MPI_Reduce(&global_throughput, &tmp_throughput, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (world_rank == 0) {
            std::cout << "Finished inference benchmark. Mean throughput/s is " << tmp_throughput << std::endl;
        }
        global_throughput = 0.0;

        if (world_rank == 0) {
            std::cout << " GFlops/s = " << gflops_per_s_inference
                      << ", Bandwidth GB/s = " << hbm_bandwidth_GB_per_s_inference
                      << "\nWaiting 10 seconds for GPU to cool down." << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);

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
    }

    if (world_rank == 0) {
        std::string json_string = bench_result.dump(4);
        std::ofstream out{std::string("../benchmarks/results/benchmark_training.json")};
        out << json_string;
    }

    MPI_Finalize();
}

int main() {
    start_training();
    return 0;
}
