# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

include(CommonOptions)

find_package(IntelSYCL REQUIRED)

set(SOURCES
    SwiftNetMLP.cpp
)

add_library(network ${SOURCES})
add_library(tinydpcppnn::network ALIAS network)
set_common_options(network)
add_sycl_to_target(TARGET network SOURCES ${SOURCES})
target_compile_options(network PUBLIC -qmkl)
target_link_options(network PUBLIC -qmkl)
target_compile_definitions(network PRIVATE -DSYCL2020_CONFORMANT_APIS)
target_link_libraries(network PUBLIC common)
target_include_directories(network PUBLIC 
    "${PROJECT_SOURCE_DIR}/include/network/"
    "${PROJECT_SOURCE_DIR}/include/"
    )