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


add_executable(test-inference inference.cpp)
target_link_libraries(test-inference PRIVATE test-lib)
doctest_discover_tests(test-inference)


add_executable(test-common common.cpp)
add_sycl_to_target(TARGET test-common SOURCES common.cpp)
target_link_libraries(test-common PRIVATE test-lib common)
doctest_discover_tests(test-common)


add_executable(test-network-with-encoding doctest_network_with_encodings.cpp)
add_sycl_to_target(TARGET test-network-with-encoding SOURCES doctest_network_with_encodings.cpp)
target_link_libraries(test-network-with-encoding PRIVATE test-lib swiftnetmlp ${LevelZero_LIBRARY}) #add libraries you need here

add_executable(test-encodings doctest_encodings.cpp)
add_sycl_to_target(TARGET test-encodings SOURCES doctest_encodings.cpp)
target_link_libraries(test-encodings PRIVATE test-lib swiftnetmlp ${LevelZero_LIBRARY}) #add libraries you need here

add_executable(test-swiftnet doctest_swiftnet.cpp)
add_sycl_to_target(TARGET test-swiftnet SOURCES doctest_swiftnet.cpp)
target_link_libraries(test-swiftnet PRIVATE test-lib swiftnetmlp ${LevelZero_LIBRARY}) #add libraries you need here

# add_executable(test-training doctest_training.cpp)
# add_sycl_to_target(TARGET test-training SOURCES doctest_training.cpp)
# target_link_libraries(test-training PRIVATE test-lib swiftnetmlp ${LevelZero_LIBRARY}) #add libraries you need here

add_executable(test-devicemem doctest_devicemem.cpp)
add_sycl_to_target(TARGET test-devicemem SOURCES doctest_devicemem.cpp)
target_link_libraries(test-devicemem PRIVATE test-lib swiftnetmlp ${LevelZero_LIBRARY}) #add libraries you need here
