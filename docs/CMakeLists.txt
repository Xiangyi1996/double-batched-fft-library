# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

find_package(Sphinx REQUIRED COMPONENTS furo)

set(SPHINX_SRC ${CMAKE_CURRENT_SOURCE_DIR})
set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/_build)
set(SPHINX_INDEX ${SPHINX_BUILD}/index.html)

set(RST_DOC_FILES
    index.rst
    api/index.rst
    manual/build.rst
    manual/contributing.rst
    manual/encoding.rst
    manual/example.rst
    manual/index.rst
    manual/networks.rst
    manual/pytorch.rst
)

add_custom_command(OUTPUT ${SPHINX_INDEX}
    MAIN_DEPENDENCY ${SPHINX_SRC}/conf.py
    DEPENDS ${RST_DOC_FILES}
    COMMAND ${Sphinx_BUILD_COMMAND}
        ${SPHINX_SRC} ${SPHINX_BUILD}
    COMMAND_EXPAND_LISTS
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_custom_target(docs ALL DEPENDS ${SPHINX_INDEX})

install(DIRECTORY ${SPHINX_BUILD}/ DESTINATION ${CMAKE_INSTALL_DOCDIR})