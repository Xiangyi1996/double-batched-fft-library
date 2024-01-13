// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include <iostream>
#include <sycl/sycl.hpp>

#include "benchmark_inference.h"
#include "benchmark_training.h"
#include "mpi.h"

using bf16 = sycl::ext::oneapi::bfloat16;

template <typename T, int WIDTH>
void benchmark_training_and_inference(const size_t batch_size, const int n_hidden_layers, const int n_iterations,
                                      sycl::queue &q) {
    benchmark_training<T, WIDTH>(batch_size, n_hidden_layers, n_iterations, q);
    q.wait();
    benchmark_inference<T, WIDTH>(batch_size, n_hidden_layers, n_iterations, q);
    q.wait();
}

template <typename T, int WIDTH> void benchmark_all(sycl::queue &q, int test_over_batch_size = 0) {
    int n_hidden_layers = 4;
    int iterations = 100;
    int batch_size;

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if (test_over_batch_size) {

        // benchmark training over all batch_size
        if (world_rank == 0) {
            std::cout << "=========================Benchmark throughput over batch sizes========================="
                      << std::endl;
        }
        for (int power = 10; power < 23; power++) {
            batch_size = 1 << power;
            benchmark_training_and_inference<T, WIDTH>(batch_size, n_hidden_layers, iterations, q);
        }
    }

    else {

        iterations = 1000; // all benchmarks run 1000 iters

        n_hidden_layers = 11;
        batch_size = 1 << 16; // batch size one less, because MPI does 2 tiles, thus half batch size.
        if (world_rank == 0) {
            std::cout
                << "=================================Benchmark of n_hidden_layers 11================================="
                << std::endl;
        }
        benchmark_training_and_inference<T, WIDTH>(batch_size, n_hidden_layers, iterations, q);

        // Image compression
        n_hidden_layers = 2;
        // batch_size = {2304 * 3072}; // resolution of image
        batch_size = 1 << 22; // batch size one less, because MPI does 2 tiles, thus half batch size.
        if (world_rank == 0) {
            std::cout << "=================================Image compression================================="
                      << std::endl;
        }
        benchmark_training_and_inference<T, WIDTH>(batch_size, n_hidden_layers, iterations, q);

        // PINNs
        n_hidden_layers = 5;
        if (world_rank == 0) {
            std::cout << "=================================PINNs=================================" << std::endl;
        }
        batch_size = 1 << 16; // batch size one less, because MPI does 2 tiles, thus half batch size.
        benchmark_training_and_inference<T, WIDTH>(batch_size, n_hidden_layers, iterations, q);

        // NeRF
        n_hidden_layers = 4;
        if (world_rank == 0) {
            std::cout << "=================================NeRF=================================" << std::endl;
        }
        batch_size = 1 << 19; // batch size one less, because MPI does 2 tiles, thus half batch size.
        benchmark_training_and_inference<T, WIDTH>(batch_size, n_hidden_layers, iterations, q);
    }
}
int main() {
    try {
        MPI_Init(NULL, NULL);
        sycl::queue q(sycl::gpu_selector_v);

        // std::cout << "Bf16, width 64" << std::endl;
        // benchmark_all<bf16, 64>(q, 0);

        // std::cout << "Sycl::half, width 64" << std::endl;
        // benchmark_all<sycl::half, 64>(q, 0);

        // ----------Benchmark for different workloads----------

        // std::cout << "Bf16, width 16" << std::endl;
        // benchmark_all<bf16, 16>(q, 1);

        // std::cout << "Sycl::half, width 16" << std::endl;
        // benchmark_all<sycl::half, 16>(q, 1);

        // std::cout << "Bf16, width 32" << std::endl;
        // benchmark_all<bf16, 32>(q, 1);

        // std::cout << "Sycl::half, width 32" << std::endl;
        // benchmark_all<sycl::half, 32>(q, 1);

        // std::cout << "Bf16, width 64" << std::endl;
        // benchmark_all<bf16, 64>(q, 1);

        std::cout << "Sycl::half, width 64" << std::endl;
        benchmark_all<sycl::half, 64>(q, 1);

        // std::cout << "Bf16, width 128" << std::endl;
        // benchmark_all<bf16, 128>(q, 1);

        // std::cout << "Sycl::half, width 128" << std::endl;
        // benchmark_all<sycl::half, 128>(q, 1);

        MPI_Finalize();

    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    } catch (...) {
        std::cout << "Caught some undefined exception." << std::endl;
        return 2;
    }

    return 0;
}