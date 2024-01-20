# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause


include(CommonOptions)
#include(InstallLib)

add_subdirectory(common)

find_package(IntelSYCL REQUIRED)
find_package(oneDPL REQUIRED)
find_package(MPI REQUIRED)

# set(SOURCES
#     network_with_encodings.cpp
# )
# set(PUBLIC_HEADERS
#     ${PROJECT_SOURCE_DIR}/include/
# )
# list(TRANSFORM PUBLIC_HEADERS PREPEND "${PROJECT_SOURCE_DIR}/include/")
# list(TRANSFORM PUBLIC_HEADERS PREPEND "${PROJECT_SOURCE_DIR}/include/Network/")
# list(TRANSFORM PUBLIC_HEADERS PREPEND "${PROJECT_SOURCE_DIR}/include/Losses/")

add_library(encodings INTERFACE)
target_include_directories(encodings INTERFACE "${PROJECT_SOURCE_DIR}/include/encodings/"
    "${PROJECT_SOURCE_DIR}/extern/json/single_include/nlohmann")
target_link_libraries(encodings INTERFACE common)
target_compile_options(encodings INTERFACE -qmkl)
target_link_options(encodings INTERFACE -qmkl)

add_library(network INTERFACE)
target_include_directories(network INTERFACE "${PROJECT_SOURCE_DIR}/include/network/")
target_link_libraries(network INTERFACE encodings)
# target_compile_options(network INTERFACE -qmkl)
# target_link_options(network INTERFACE -qmkl)


# add_library(swiftnetmlp ${SOURCES})
# add_library(tinydpcppnn::swiftnetmlp ALIAS swiftnetmlp)
# set_common_options(swiftnetmlp)
# #add_sycl_to_target(TARGET swiftnetmlp SOURCES ${SOURCES})
# target_compile_definitions(swiftnetmlp PRIVATE -DSYCL2020_CONFORMANT_APIS)
# target_compile_options(swiftnetmlp PUBLIC -qmkl -O2)
# target_link_options(swiftnetmlp PUBLIC -qmkl)
# target_link_libraries(swiftnetmlp PUBLIC common network ${MPI_CXX_LIBRARIES})
# target_include_directories(swiftnetmlp PUBLIC "${PROJECT_SOURCE_DIR}/include/"
#     "${PROJECT_SOURCE_DIR}/include/losses/"
#     "${PROJECT_SOURCE_DIR}/include/optimizers/"
#     "${PROJECT_SOURCE_DIR}/include/encodings/"
# )
#target_include_directories(swiftnetmlp PUBLIC "${PROJECT_SOURCE_DIR}/include/common/")
# target_sources(swiftnetmlp PUBLIC FILE_SET HEADERS
#     BASE_DIRS ${PROJECT_SOURCE_DIR}/include
#     FILES ${PUBLIC_HEADERS})

#install_lib(swiftnetmlp swiftnet)