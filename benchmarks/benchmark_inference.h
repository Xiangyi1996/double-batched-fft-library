// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

#include "SwiftNetMLP.h"
#include "common.h"
#include "common_benchmarks.h"
#include "common_host.h"
#include "mpi.h"
#include "result_check.h"

/// benchmarking function with input width = width = output width
template <typename T, int WIDTH>
void benchmark_inference(const size_t batch_size, const int n_hidden_layers, const int n_iterations, sycl::queue &q) {

    tinydpcppnn::benchmarks::common::WriteBenchmarkHeader("Inference", batch_size, WIDTH, n_hidden_layers, sizeof(T),
                                                          type_to_string<T>(), q);

    constexpr int input_width = WIDTH;
    constexpr int output_width = WIDTH;

    DeviceMatrix<T> inputs(batch_size, input_width, q);
    DeviceMatrix<T> output(batch_size, output_width, q);

    const T input_val = static_cast<T>(0.1);
    inputs.fill(input_val);
    output.fill(0);

    // need a factory here for different widths
    SwiftNetMLP<T, WIDTH> network(q, input_width, output_width, n_hidden_layers, Activation::ReLU, Activation::None,
                                  Network<T>::WeightInitMode::constant_pos);
    std::vector<T> new_weights(network.get_weights_matrices().size(), 1.0 / WIDTH);
    network.set_weights_matrices(new_weights);

    constexpr int n_iterations_warmup = 5;
    // Do a warmup loop, not benched.
    for (int iter = 0; iter < n_iterations_warmup; iter++) {
        network.inference(inputs, output, {});
        q.wait();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto begin_time = std::chrono::steady_clock::now();
    std::vector<sycl::event> dependencies;
    for (int iter = 0; iter < n_iterations; iter++) {
        dependencies = network.inference(inputs, output, dependencies);
    }
    q.wait();
    MPI_Barrier(MPI_COMM_WORLD);
    const auto end_time = std::chrono::steady_clock::now();

    tinydpcppnn::benchmarks::common::WritePerformanceDataInference(begin_time, end_time, batch_size, WIDTH,
                                                                   n_hidden_layers, n_iterations, sizeof(T));

    MPI_Barrier(MPI_COMM_WORLD);
    std::vector<T> expected_result(batch_size * output_width,
                                   /*std::pow(WIDTH * 0.01, n_hidden_layers + 1) **/ input_val);
    std::vector<T> out_host = output.copy_to_host();
    areVectorsWithinTolerance(out_host, expected_result, 0.01f);
    std::cout << std::endl;
}
