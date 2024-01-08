# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

find_package(IntelSYCL REQUIRED)

include(CTest)
include(${PROJECT_SOURCE_DIR}/extern/doctest/cmake/doctest.cmake)
include(CommonOptions)


add_library(test-lib STATIC main.cpp)
target_include_directories(test-lib PUBLIC ${PROJECT_SOURCE_DIR}/extern)
target_compile_features(test-lib PUBLIC cxx_std_20)
add_common_flags(test-lib)

# add_executable(test-common doctest_common.cpp)
# add_sycl_to_target(TARGET test-common SOURCES doctest_common.cpp)
# target_link_libraries(test-common PRIVATE test-lib common)
# doctest_discover_tests(test-common)

add_executable(test-network-with-encoding doctest_network_with_encodings.cpp)
add_sycl_to_target(TARGET test-network-with-encoding SOURCES doctest_network_with_encodings.cpp)
target_link_libraries(test-network-with-encoding PRIVATE test-lib network)
doctest_discover_tests(test-network-with-encoding)

add_executable(test-encodings doctest_encodings.cpp)
add_sycl_to_target(TARGET test-encodings SOURCES doctest_encodings.cpp)
target_link_libraries(test-encodings PRIVATE test-lib network)
doctest_discover_tests(test-encodings)

add_executable(test-swiftnet doctest_swiftnet.cpp)
add_sycl_to_target(TARGET test-swiftnet SOURCES doctest_swiftnet.cpp)
target_link_libraries(test-swiftnet PRIVATE test-lib network)
doctest_discover_tests(test-swiftnet)

# add_executable(test-devicemem doctest_devicemem.cpp)
# add_sycl_to_target(TARGET test-devicemem SOURCES doctest_devicemem.cpp)
# target_link_libraries(test-devicemem PRIVATE test-lib swiftnetmlp)
# doctest_discover_tests(test-devicemem)

# add_executable(test-inference doctest_inference.cpp)
# add_sycl_to_target(TARGET test-inference SOURCES doctest_inference.cpp)
# target_link_libraries(test-inference PRIVATE test-lib swiftnetmlp)
# doctest_discover_tests(test-inference)

# add_executable(test-training doctest_training.cpp)
# add_sycl_to_target(TARGET test-training SOURCES doctest_training.cpp)
# target_link_libraries(test-training PRIVATE test-lib swiftnetmlp)
# doctest_discover_tests(test-training)

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

add_executable(test-kernelhelper-esimd doctest_kernelhelper_esimd.cpp)
add_sycl_to_target(TARGET test-kernelhelper-esimd SOURCES doctest_kernelhelper_esimd.cpp)
target_link_libraries(test-kernelhelper-esimd PRIVATE test-lib network)
doctest_discover_tests(test-kernelhelper-esimd)

# add_executable(test-save-image doctest_image_save.cpp)
# add_sycl_to_target(TARGET test-save-image SOURCES doctest_image_save.cpp)
# target_link_libraries(test-save-image PRIVATE test-lib swiftnetmlp)
# doctest_discover_tests(test-save-image)