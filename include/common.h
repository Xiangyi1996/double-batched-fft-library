#pragma once

#include <iostream>
#include <vector>
#include <CL/sycl.hpp>

/**
 * @brief Convert index from original matrix layout to packed layout
 *
 * @param idx Index in packed layout
 * @param rows Number of rows in original matrix
 * @param cols Number of columns in original matrix
 * @return Index in packed matrix layout
 */
extern SYCL_EXTERNAL int toPackedLayoutCoord(int idx, int rows, int cols);

/**
 * @brief Convert index from packed layout to original matrix layout
 *
 * @param idx Index in original matrix layout
 * @param rows Number of rows in original matrix
 * @param cols Number of columns in original matrix
 * @return Index in original matrix layout
 */
extern SYCL_EXTERNAL int fromPackedLayoutCoord(int idx, int rows, int cols);

/**
 * @brief Compare two strings case-insensitively
 *
 * @param str1 First string
 * @param str2 Second string
 * @return True if the strings are equal, false otherwise
 */
extern SYCL_EXTERNAL bool isequalstring(const std::string& str1, const std::string& str2);

