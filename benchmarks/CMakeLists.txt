# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

find_package(IntelSYCL REQUIRED)
find_package(MPI REQUIRED)

add_executable(benchmark-inference benchmark_inference.cpp)
add_sycl_to_target(TARGET benchmark-inference SOURCES benchmark_inference.cpp)
target_link_libraries(benchmark-inference PRIVATE network ${MPI_CXX_LIBRARIES})

add_executable(benchmark-training benchmark_training.cpp)
add_sycl_to_target(TARGET benchmark-training SOURCES benchmark_training.cpp)
target_link_libraries(benchmark-training PRIVATE network ${MPI_CXX_LIBRARIES})
target_include_directories(benchmark-training PRIVATE "${PROJECT_SOURCE_DIR}/include/")


add_executable(benchmark-all benchmark_all.cpp)
add_sycl_to_target(TARGET benchmark-all SOURCES benchmark_all.cpp)
target_link_libraries(benchmark-all PRIVATE network ${MPI_CXX_LIBRARIES})
target_include_directories(benchmark-all PRIVATE "${PROJECT_SOURCE_DIR}/include/")