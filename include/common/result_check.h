/**
 * @file result_check.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Basic comparison, load, store functionalities to check correctness of results.
 * TODO: move the load/store functionalities in a different file and put everything in a namespace.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

void saveImageToPGM(const std::string &filename, const int width, const int height,
                    const std::vector<unsigned char> &image) {
    // Create and open the output file
    std::ofstream outputFile(filename, std::ios::out | std::ios::binary);

    // Write PGM header
    outputFile << "P5\n" << width << " " << height << "\n255\n";

    // Write the image data to the file
    outputFile.write(reinterpret_cast<const char *>(image.data()), image.size());

    // Close the file
    outputFile.close();
}

template <typename Tval, typename Ttarget> double getRelError(const Tval value, const Ttarget target) {
    double diff = 0.0;
    double tol = 1e-5;
    if (value < tol || target < tol) {
        // Add a safety check for small values
        if (std::abs(value - target) < tol) {
            return diff; // Arbitrarily return 0 in this case
        }
    }

    if (!std::isfinite(value) || !std::isfinite(target)) throw std::invalid_argument("Infinite numbers");

    if (target == 0) {
        throw std::invalid_argument("Error: Division by zero");
    }

    return std::abs((double)value - (double)target) / std::abs((double)target);
    // if ((double)value != 0.0 || (double)target != 0.0) {
    //     diff = std::abs((double)value - (double)target) /
    //            std::max<double>(std::abs((double)value), std::abs((double)target));
    // }

    // return diff;
}

template <typename Tval, typename Ttarget>
bool isVectorWithinTolerance(const std::vector<Tval> &value, const Ttarget target, const double tolerance) {

    long long count = 0;
    bool is_same = true;
    double max_diff = 0.0;
    for (size_t i = 0; i < value.size(); ++i) {
        const double diff = getRelError(value.at(i), target);
        max_diff = std::max(diff, max_diff);

        if (diff > tolerance) {
            is_same = false;
            count++;
        }
    }
    if (count) {
        std::cout << count << "/" << value.size() << " are wrong. Max diff = " << max_diff << std::endl;
    }

    return is_same;
}

template <typename Tval, typename Ttarget>
bool areVectorsWithinTolerance(const std::vector<Tval> &value, const std::vector<Ttarget> &target,
                               const double tolerance) {

    long long count = 0;
    bool is_same = true;
    double max_diff = 0.0;
    for (size_t i = 0; i < value.size(); ++i) {
        const double diff = getRelError(value.at(i), target.at(i));

        max_diff = std::max(diff, max_diff);

        if (diff > tolerance) {
            std::cout << "Diff: " << diff << ". i: " << i << ". Val: " << value.at(i) << "/" << target.at(i)
                      << std::endl;
            is_same = false;
            count++;
        }
    }
    if (count) {
        std::cout << count << "/" << target.size() << " are wrong. Max diff = " << max_diff << std::endl;
    }

    return is_same;
}

template <typename T> std::vector<T> loadVectorFromCSV(const std::string &filename) {
    std::vector<T> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::invalid_argument("Failed to open the file for reading: " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ',')) {
            data.push_back(static_cast<T>(std::stof(token)));
        }
    }

    return data;
}

template <typename T> void saveCSV(const std::string &filename, const std::vector<T> &data) {
    std::ofstream file(filename);

    if (file.is_open()) {
        for (const auto &value : data) {
            file << (double)value << std::endl;
        }
        file.close();
    }
}

std::vector<float> loadCSV(const std::string &filename) {
    std::vector<float> data;
    std::ifstream file(filename);

    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            float value;
            std::istringstream iss(line);
            iss >> value;
            data.push_back(value);
        }
        file.close();
    }
    return data;
}

// Function to read target vectors from a file with a specified delimiter
std::vector<std::vector<float>> readTargetVectorsFromFile(const std::string &filename, char delimiter) {
    std::vector<std::vector<float>> targetVectors;

    std::ifstream inputFile(filename);
    if (!inputFile.is_open()) {
        std::cerr << "Error opening " << filename << std::endl;
        return targetVectors; // Return an empty vector in case of an error
    }

    std::string line;

    while (std::getline(inputFile, line)) {
        std::vector<float> vectorFromCSV;
        std::istringstream lineStream(line);
        std::string valueStr;

        while (std::getline(lineStream, valueStr, delimiter)) {
            float value = std::stod(valueStr); // Convert the string to a double
            vectorFromCSV.push_back(value);
        }

        targetVectors.push_back(vectorFromCSV);
    }

    inputFile.close();

    return targetVectors;
}
