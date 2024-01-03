# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

find_package(IntelSYCL REQUIRED)
find_package(MPI REQUIRED)

add_executable(benchmark-inference benchmark_inference.cpp)
add_sycl_to_target(TARGET benchmark-inference SOURCES benchmark_inference.cpp)
target_compile_options(benchmark-inference PUBLIC -qmkl)
target_link_options(benchmark-inference PUBLIC -qmkl)
target_link_libraries(benchmark-inference PUBLIC common network ${MPI_CXX_LIBRARIES})
# target_include_directories(benchmark-inference PUBLIC
#     "${PROJECT_SOURCE_DIR}/include/network/"
#     "${PROJECT_SOURCE_DIR}/include/common/"
# )

# add_executable(benchmark-image benchmark_image_compression.cpp)
# add_sycl_to_target(TARGET benchmark-image SOURCES benchmark_image_compression.cpp)
# target_link_libraries(benchmark-image PRIVATE swiftnetmlp )

# add_executable(benchmark-nerf benchmark_nerf.cpp)
# add_sycl_to_target(TARGET benchmark-nerf SOURCES benchmark_nerf.cpp)
# target_link_libraries(benchmark-nerf PRIVATE swiftnetmlp)
# target_compile_options(benchmark-nerf PRIVATE -qmkl -fsycl-targets=spir64_gen)
# target_link_options(benchmark-nerf PRIVATE -qmkl -fsycl-targets=spir64_gen -Xsycl-target-backend "-device pvc")

# # add_executable(benchmark-pinns benchmark_pinns.cpp)
# # add_sycl_to_target(TARGET benchmark-pinns SOURCES benchmark_pinns.cpp)
# # target_link_libraries(benchmark-pinns PRIVATE swiftnetmlp)
# target_compile_options(benchmark-pinns PRIVATE -qmkl -fsycl-targets=spir64_gen)
# target_link_options(benchmark-pinns PRIVATE -qmkl -fsycl-targets=spir64_gen -Xsycl-target-backend "-device pvc")
