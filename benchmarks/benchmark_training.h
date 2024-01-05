// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

#include "SwiftNetMLP.h"
#include "common.h"
#include "common_benchmarks.h"
#include "mpi.h"
#include "result_check.h"
#include "trainer.h"

/// benchmarking function with input width = width = output width
/// Note that this is not meant to test the correctness, only perf.
/// Correctness is checked with the tests in the 'test' directory
template <typename T, int WIDTH>
void benchmark_training(const size_t batch_size, const int n_hidden_layers, const int n_iterations, sycl::queue &q) {

    tinydpcppnn::benchmarks::common::WriteBenchmarkHeader("Training (forw+backw, no opt, no loss)", batch_size, WIDTH,
                                                          n_hidden_layers, sizeof(T), q);

    constexpr int input_width = WIDTH;
    constexpr int output_width = WIDTH;

    DeviceMem<T> inputs(input_width * batch_size, q);
    DeviceMem<T> outputs_backw(input_width * WIDTH + output_width * WIDTH + (n_hidden_layers - 1) * WIDTH * WIDTH, q);
    DeviceMem<T> losses(batch_size * output_width, q);
    const size_t out_inter_forw_size =
        batch_size * (input_width + output_width +
                      WIDTH * n_hidden_layers); // includes input and output (thus +1 of the back interm)
    const size_t out_inter_backw_size = batch_size * (output_width + WIDTH * n_hidden_layers);
    DeviceMem<T> out_inter_forw(out_inter_forw_size, q);
    DeviceMem<T> out_inter_backw(out_inter_backw_size, q);

    const T input_val = static_cast<T>(0.1);
    inputs.fill(input_val);
    outputs_backw.fill(0);
    losses.fill(0);

    // need a factory here for different widths
    SwiftNetMLP<T, WIDTH> network(q, input_width, output_width, n_hidden_layers, Activation::ReLU, Activation::None,
                                  Network<T>::WeightInitMode::constant_pos);

    Trainer<T> train(&network);

    constexpr int n_iterations_warmup = 5;
    // Do a warmup loop, not benched.
    for (int iter = 0; iter < n_iterations_warmup; iter++) {
        train.training_step(inputs, outputs_backw, losses, out_inter_forw, out_inter_backw, batch_size, {});
        q.wait();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto begin_time = std::chrono::steady_clock::now();
    std::vector<sycl::event> dependencies;
    for (int iter = 0; iter < n_iterations; iter++) {
        dependencies = train.training_step(inputs, outputs_backw, losses, out_inter_forw, out_inter_backw, batch_size,
                                           dependencies);
    }
    q.wait();
    MPI_Barrier(MPI_COMM_WORLD);
    const auto end_time = std::chrono::steady_clock::now();

    tinydpcppnn::benchmarks::common::WritePerformanceDataTraining(begin_time, end_time, batch_size, WIDTH,
                                                                  n_hidden_layers, n_iterations, sizeof(T));

    MPI_Barrier(MPI_COMM_WORLD);

    // Now do a simple correctness check
    // TODO: check all the elements in the interm forw array.
    // expect that the output of the backward pass is 0 since losses are set to 0
    std::vector<T> expected_result(outputs_backw.size(), 0);

    std::vector<T> out_host(expected_result.size(), 0);
    outputs_backw.copy_to_host(out_host).wait();

    areVectorsWithinTolerance(out_host, expected_result, 0.01f);
}
