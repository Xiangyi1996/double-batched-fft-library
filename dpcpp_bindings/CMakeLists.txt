# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# Include headers

find_package(IntelSYCL REQUIRED)
find_package(Torch REQUIRED PATHS ${TORCH_CMAKE_PREFIX_PATH})

include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

# Set source files for the Python bindings
set(PYBIND_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/pybind_module.cpp
)

# Create the Python module
pybind11_add_module(tiny_nn ${PYBIND_SOURCES})
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_CMAKE_PREFIX_PATH}/../../lib")

# Link any required libraries
target_link_libraries(tiny_nn PRIVATE network ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

add_sycl_to_target(TARGET tiny_nn SOURCES ${PYBIND_SOURCES})

target_compile_definitions(tiny_nn
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})