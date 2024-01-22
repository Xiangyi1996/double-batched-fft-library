# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

find_package(IntelSYCL REQUIRED)

include(CTest)
include(${PROJECT_SOURCE_DIR}/extern/doctest/cmake/doctest.cmake)
include(CommonOptions)

# Pass a preprocessor variable to the C++ code based on the option
if(BUILD_MORE_TEST)
    message(STATUS "Using downloaded data")
    add_definitions(-DUSE_REFERENCE_TEST)
endif()

add_library(test-lib STATIC main.cpp)
target_include_directories(test-lib PUBLIC ${PROJECT_SOURCE_DIR}/extern)
target_compile_features(test-lib PUBLIC cxx_std_20)
add_common_flags(test-lib)

add_executable(test-common doctest_common.cpp)
add_sycl_to_target(TARGET test-common SOURCES doctest_common.cpp)
target_link_libraries(test-common PRIVATE test-lib common)
doctest_discover_tests(test-common)

add_executable(test-network-with-encoding doctest_network_with_encodings.cpp)
add_sycl_to_target(TARGET test-network-with-encoding SOURCES doctest_network_with_encodings.cpp)
target_link_libraries(test-network-with-encoding PRIVATE test-lib network)
doctest_discover_tests(test-network-with-encoding)

add_executable(test-matrix doctest_devicematrix.cpp)
add_sycl_to_target(TARGET test-matrix SOURCES doctest_devicematrix.cpp)
target_link_libraries(test-matrix PRIVATE test-lib common)
doctest_discover_tests(test-matrix)

add_executable(test-encodings doctest_encodings.cpp)
add_sycl_to_target(TARGET test-encodings SOURCES doctest_encodings.cpp)
target_link_libraries(test-encodings PRIVATE test-lib network)
doctest_discover_tests(test-encodings)

add_executable(test-swiftnet doctest_swiftnet.cpp)
add_sycl_to_target(TARGET test-swiftnet SOURCES doctest_swiftnet.cpp)
target_link_libraries(test-swiftnet PRIVATE test-lib network)
doctest_discover_tests(test-swiftnet)

add_executable(test-kernel-esimd doctest_kernel_esimd.cpp)
add_sycl_to_target(TARGET test-kernel-esimd SOURCES doctest_kernel_esimd.cpp)
target_link_libraries(test-kernel-esimd PRIVATE test-lib network)
doctest_discover_tests(test-kernel-esimd)

# add_executable(test-devicemem doctest_devicemem.cpp)
# add_sycl_to_target(TARGET test-devicemem SOURCES doctest_devicemem.cpp)
# target_link_libraries(test-devicemem PRIVATE test-lib swiftnetmlp)
# doctest_discover_tests(test-devicemem)

# add_executable(test-losses doctest_losses.cpp)
# add_sycl_to_target(TARGET test-losses SOURCES doctest_losses.cpp)
# target_link_libraries(test-losses PRIVATE test-lib swiftnetmlp)
# doctest_discover_tests(test-losses)

# add_executable(test-kernelhelper doctest_kernelhelper.cpp)
# add_sycl_to_target(TARGET test-kernelhelper SOURCES doctest_kernelhelper.cpp)
# target_link_libraries(test-kernelhelper PRIVATE test-lib network)
# doctest_discover_tests(test-kernelhelper)
# target_compile_options(test-kernelhelper PUBLIC -fsycl-targets=spir64_gen)
# target_link_options(test-kernelhelper PUBLIC -fsycl-targets=spir64_gen -Xsycl-target-backend "-device pvc")



# add_executable(test-save-image doctest_image_save.cpp)
# add_sycl_to_target(TARGET test-save-image SOURCES doctest_image_save.cpp)
# target_link_libraries(test-save-image PRIVATE test-lib swiftnetmlp)
# doctest_discover_tests(test-save-image)