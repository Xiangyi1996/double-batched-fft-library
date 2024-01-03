#pragma once

#include "mpi.h"
#include <chrono>
#include <iostream>

/// Put all the common functionalities for the performance benchmarks here.
void WriteBenchmarkHeader(const std::string &str, const size_t batch_size, const int WIDTH, const int n_hidden_layers,
                          const int typesize) {

    int world_rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_rank == 0) {
        std::cout << str << std::endl;
        std::cout << "n_hidden_layers = " << n_hidden_layers << ", WIDTH = " << WIDTH << ", batch_size = " << batch_size
                  << ", type size = " << typesize << " bytes" << std::endl
                  << "MPI world_size = " << world_size << std::endl
                  << std::endl;
    }
}

void WritePerformanceDataInference(std::chrono::time_point<std::chrono::steady_clock> begin,
                                   std::chrono::time_point<std::chrono::steady_clock> end, const size_t batch_size,
                                   const int WIDTH, const int n_hidden_layers, const int n_iterations,
                                   const int typesize) {

    int world_rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_rank == 0) {
        const double elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
        const double flops = world_size * 2.0 * (double)batch_size * (double)WIDTH * (double)WIDTH *
                             (n_hidden_layers + 1) * n_iterations;
        // flops / ns = gflops/s
        const double gflops_per_s = flops / elapsed_time;
        // load A once, store A once, load B for every layer per iteration.
        const double bytes_loaded_and_stored =
            typesize * world_size * (2.0 * batch_size * WIDTH + WIDTH * WIDTH * (n_hidden_layers + 1)) * n_iterations;
        const double hbm_bandwidth_GB_per_s = bytes_loaded_and_stored / elapsed_time;
        // arithmetic intensity
        const double oi = flops / bytes_loaded_and_stored;

        std::cout << "Finished inference benchmark." << std::endl;
        std::cout << "#Iterations = " << n_iterations << std::endl;
        std::cout << "Time = " << elapsed_time * 1.0e-9 << " s" << std::endl;
        std::cout << "AI = " << oi << " flops/byte" << std::endl;
        std::cout << "BW = " << hbm_bandwidth_GB_per_s << " GB/s" << std::endl;
        std::cout << "Throughput = " << gflops_per_s << " Gflops/s" << std::endl << std::endl;
    }
}

void WritePerformanceDataTraining(std::chrono::time_point<std::chrono::steady_clock> begin,
                                  std::chrono::time_point<std::chrono::steady_clock> end, const size_t batch_size,
                                  const int WIDTH, const int n_hidden_layers, const int n_iterations,
                                  const int typesize) {

    int world_rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_rank == 0) {
        const double elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
        // we have three matrix multiplications of size M*K*N per layer per iteration in the training
        const double flops = 3 * world_size * 2.0 * (double)batch_size * (double)WIDTH * (double)WIDTH *
                             (n_hidden_layers + 1) * n_iterations;
        // flops / ns = gflops/s
        const double gflops_per_s = flops / elapsed_time;
        const double bytes_loaded_and_stored =
            typesize * world_size * n_iterations *
            (batch_size * WIDTH + 2 * (WIDTH * WIDTH + batch_size * WIDTH) * (n_hidden_layers + 1) +
             (2 * batch_size * WIDTH + WIDTH * WIDTH) * (n_hidden_layers + 1));
        const double hbm_bandwidth_GB_per_s = bytes_loaded_and_stored / elapsed_time;
        // arithmetic intensity
        const double oi = flops / bytes_loaded_and_stored;

        std::cout << "Finished training benchmark." << std::endl;
        std::cout << "#Iterations = " << n_iterations << std::endl;
        std::cout << "Time = " << elapsed_time * 1.0e-9 << " s" << std::endl;
        std::cout << "AI = " << oi << " flops/byte" << std::endl;
        std::cout << "BW = " << hbm_bandwidth_GB_per_s << " GB/s" << std::endl;
        std::cout << "Throughput = " << gflops_per_s << " Gflops/s" << std::endl << std::endl;
    }
}