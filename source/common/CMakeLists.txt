# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

include(CommonOptions)

find_package(IntelSYCL REQUIRED)
find_package(LevelZero REQUIRED)

set(SOURCES
    common.cpp
    common_host.cpp
    DeviceMem.cpp
)

add_library(common ${SOURCES})
add_library(tinydpcppnn::common ALIAS common)
set_common_options(common)
add_sycl_to_target(TARGET common SOURCES ${SOURCES})
target_link_libraries(common PUBLIC ${LevelZero_LIBRARY})
target_compile_definitions(common PRIVATE -DSYCL2020_CONFORMANT_APIS)
target_include_directories(common PUBLIC "${PROJECT_SOURCE_DIR}/include/common/")