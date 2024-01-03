// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include <iostream>
#include <sycl/sycl.hpp>

#include "benchmark_inference.h"
#include "mpi.h"

using bf16 = sycl::ext::oneapi::bfloat16;

int main() {
    try {
        MPI_Init(NULL, NULL);
        sycl::queue q(gpu_selector_v);

        const int batch_size = 1 << 22;
        benchmark_inference<bf16, 64>(batch_size, 4, 1000, q);
        q.wait();
        benchmark_inference<sycl::half, 64>(batch_size, 4, 1000, q);
        q.wait();
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
