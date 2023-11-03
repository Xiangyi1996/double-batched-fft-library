# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.23)

project(tiny-dpcpp-nn VERSION 0.1.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

include(CMakeDependentOption)

option(BUILD_TEST "Build tests in 'test' directory" ON)

message(WARNING "cmake build system is still WIP.")

enable_testing()
if(BUILD_TEST)
    add_subdirectory(test)
endif()


