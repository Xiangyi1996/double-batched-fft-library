# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# Include headers

find_package(Torch REQUIRED PATHS ${TORCH_CMAKE_PREFIX_PATH})
include_directories(${TORCH_INCLUDE_DIRS})

# Set source files for the Python bindings
set(PYBIND_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/pybind_module.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../source/tnn_api.cpp
)

# Create the Python module
pybind11_add_module(tiny_nn ${PYBIND_SOURCES})
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_CMAKE_PREFIX_PATH}/../../lib")

# Link any required libraries (replace with actual libraries)
target_link_libraries(tiny_nn PRIVATE swiftnetmlp ${LevelZero_LIBRARY}  ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

target_compile_definitions(tiny_nn
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})