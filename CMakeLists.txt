# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.23)

#the whole thing only works with icpx due to using joint_matrix
set(CMAKE_CXX_COMPILER "mpiicpx")
set(default_build_type "Release")

project(tiny-dpcpp-nn VERSION 0.1.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

include(CMakeDependentOption)

option(BUILD_TEST "Build tests in 'test' directory" ON)
option(BUILD_PYBIND "Build pybindings in 'dpcpp_bindings' directory" OFF)
option(BUILD_BENCHMARK "Build benchmarks in 'benchmark' directory" OFF)
option(BUILD_REF_TEST "Use reference tests from downloaded data for test" ON)
option(BUILD_DOCUMENTATION "Build the documentation which is online available" OFF)
option(BUILD_EXAMPLE "Build the examples" OFF)

message(WARNING "cmake build system is still WIP.")

add_subdirectory(source)

enable_testing()

if(BUILD_TEST)
    add_subdirectory(test)
endif()

if(BUILD_BENCHMARK)
    add_subdirectory(benchmarks)
endif()

if(BUILD_PYBIND)
    # Include Pybind11
    add_subdirectory(extern/pybind11)
    add_subdirectory(dpcpp_bindings)
endif()