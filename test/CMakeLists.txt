# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

include(CTest)
include(${PROJECT_SOURCE_DIR}/extern/doctest/cmake/doctest.cmake)
include(CommonOptions)

add_library(test-lib STATIC main.cpp)
target_include_directories(test-lib PUBLIC ${PROJECT_SOURCE_DIR}/extern)
target_compile_features(test-lib PUBLIC cxx_std_17)
add_common_flags(test-lib)

add_executable(test-inference inference.cpp)
target_link_libraries(test-inference PRIVATE test-lib) #add libraries you need here
doctest_discover_tests(test-inference)
