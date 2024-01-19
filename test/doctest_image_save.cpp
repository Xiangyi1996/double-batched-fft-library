/**
 * @file doctest_image_save.cpp
 * @author Kai Yuan
 * @brief File which tests the image save functionalities.
 * TODO: put this either together with results check or consolidate with common.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "doctest/doctest.h"
#include "result_check.h"
void generateRandomNoiseImage(int width, int height, std::vector<unsigned char> &image) {
    // Seed the random number generator
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Resize the image vector to hold all pixels
    image.resize(width * height);

    // Generate random noise and set it in the vector
    for (int i = 0; i < width * height; ++i) {
        // Generate random intensity value
        image[i] = std::rand() % 256;
    }
}
void generateVisualPattern(int width, int height, std::vector<unsigned char> &image) {
    // Resize the image vector to hold all pixels
    image.resize(width * height);

    // Generate a visual pattern (linear gradient along the x-axis)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Set intensity based on x-coordinate
            image[y * width + x] = static_cast<unsigned char>(255 * x / static_cast<float>(width - 1));
        }
    }
}
void save_image() {
    // Set the resolution of the image
    int width = 800;
    int height = 600;

    // Set the filename for the saved image
    std::string filename = "random_noise_image.pgm";

    // Create a vector to store the image data
    std::vector<unsigned char> image;

    // Generate random noise and store it in the vector
    generateRandomNoiseImage(width, height, image);

    // Save the image vector to a PGM file
    saveImageToPGM(filename, width, height, image);
}
int save_pattern() {
    // Set the resolution of the image
    int width = 800;
    int height = 600;

    // Set the filename for the saved image
    std::string filename = "visual_pattern_image.pgm";

    // Create a vector to store the image data
    std::vector<unsigned char> image;

    // Generate a visual pattern and store it in the vector
    generateVisualPattern(width, height, image);

    // Save the image vector to a PGM file
    saveImageToPGM(filename, width, height, image);
}
TEST_CASE("Save noise") { save_image(); }
TEST_CASE("Save pattern") { save_pattern(); }
