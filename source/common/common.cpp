#include "common.h"
#include <algorithm> //std::equal
#include <cctype>    //std::tolower

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
unsigned toPackedLayoutCoord(const unsigned idx, const unsigned rows, const unsigned cols) {
    assert(idx < rows * cols);
    const int i = idx / cols;
    const int j = idx % cols;
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
unsigned fromPackedLayoutCoord(const unsigned idx, const unsigned rows, const unsigned cols) {
    const int i = idx / (cols * 2);
    const int j = idx % (cols * 2);
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
bool isequalstring(const std::string &str1, const std::string &str2) {

    return str1.size() == str2.size() &&
           std::equal(str1.begin(), str1.end(), str2.begin(), str2.end(), [&](char a, char b) {
               return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b));
           });
}
