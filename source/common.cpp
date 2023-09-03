#include "common.h"

/**
 * Convert an index to a packed layout coordinate for interleaved even and odd
 * rows.
 *
 * This function calculates the packed layout coordinate for an interleaved
 * layout where consecutive even and odd rows are interleaved. It takes an
 * index, the number of rows, and the number of columns as inputs and returns
 * the corresponding packed layout coordinate.
 *
 * @param idx   The index to convert to a packed layout coordinate.
 * @param rows  The number of rows in the layout.
 * @param cols  The number of columns in the layout.
 * @return      The packed layout coordinate for the given index.
 */
int toPackedLayoutCoord(int idx, int rows, int cols) {
  int i = idx / cols;
  int j = idx % cols;
  if (i % 2 == 0) {
    return i * cols + 2 * j;
  } else {
    return (i - 1) * cols + 2 * j + 1;
  }
}

/**
 * Convert a packed layout coordinate to an index for interleaved even and odd
 * rows.
 *
 * This function calculates the original index for an interleaved layout
 * where consecutive even and odd rows are interleaved. It takes a packed layout
 * coordinate, the number of rows, and the number of columns as inputs and
 * returns the corresponding index.
 *
 * @param idx   The packed layout coordinate to convert to an index.
 * @param rows  The number of rows in the layout.
 * @param cols  The number of columns in the layout.
 * @return      The index corresponding to the given packed layout coordinate.
 */
int fromPackedLayoutCoord(int idx, int rows, int cols) {
  int i = idx / (cols * 2);
  int j = idx % (cols * 2);
  if (j % 2 == 0) {
    return (i * 2) * cols + j / 2;
  } else {
    return (i * 2 + 1) * cols + (j - 1) / 2;
  }
}

/**
 * Check if two strings are equal while ignoring the case of characters.
 *
 * This function compares two input strings for equality, considering the case
 * of characters. It returns true if the strings are equal (ignoring case),
 * and false otherwise.
 *
 * @param str1  The first string to compare.
 * @param str2  The second string to compare.
 * @return      True if the strings are equal (ignoring case), false otherwise.
 */
bool isequalstring(const std::string& str1, const std::string& str2) {
  if (str1.length() != str2.length()) {
    return false;
  }
  for (int i = 0; i < str1.length(); i++) {
    if (std::tolower(str1[i]) != std::tolower(str2[i])) {
      return false;
    }
  }
  return true;
}
