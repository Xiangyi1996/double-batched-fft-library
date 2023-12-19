#include "gpu_matrix.h"
// #include <cstdlib>
// #include <ctime>
// #include <fstream>
// #include <iostream>
#include <vector>

void saveImageToPGM(const std::string &filename, int width, int height, const std::vector<unsigned char> &image) {
    // Create and open the output file
    std::ofstream outputFile(filename, std::ios::out | std::ios::binary);

    // Write PGM header
    outputFile << "P5\n" << width << " " << height << "\n255\n";

    // Write the image data to the file
    outputFile.write(reinterpret_cast<const char *>(image.data()), image.size());

    // Close the file
    outputFile.close();
}

template <typename Tval, typename Ttarget>
bool areVectorsWithinTolerance(const std::vector<Tval> &value, const std::vector<Ttarget> &target,
                               const double tolerance) {

    long long count = 0;
    bool is_same = true;
    double max_diff = 0.0;
    for (size_t i = 0; i < target.size(); ++i) {
        double diff = 0.0;
        if ((double)value[i] != 0.0 || (double)target[i] != 0.0)
            diff = std::abs((double)value[i] - (double)target[i]) /
                   std::max<double>(std::abs((double)value[i]), std::abs((double)target[i]));

        max_diff = std::max(diff, max_diff);

        if (diff > tolerance) {
            is_same = false;
            count++;
            // std::cout << "At " << i << ", Val: " << (double)value[i] << ", target: " << (double)target[i] <<
            // std::endl;
        }
    }
    if (!is_same) std::cout << count << "/" << target.size() << " are wrong. Max diff = " << max_diff << std::endl;

    // CHECK(is_same);
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

void saveCSV(const std::string &filename, const std::vector<float> &data) {
    std::ofstream file(filename);

    if (file.is_open()) {
        for (const auto &value : data) {
            file << value << std::endl;
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

bool areVectorsWithinTolerance(const std::vector<bf16> &value, const std::vector<float> &target, float tolerance,
                               int output_width) {
    //   assert(a.size() == b.size());  // Ensure vectors have the same length

    int total_values_checked = 0;
    bool allWithinTolerance = true;

    double max_diff = 0.0f;
    for (size_t i = 0; i < value.size(); ++i) {
        const double diff = std::abs((double)value[i] - target[i % output_width]);
        max_diff = std::max<double>(diff, max_diff);
        // std::cout << "Checking idx: " << i << std::endl;
        total_values_checked++;
        if (diff > tolerance) {
            allWithinTolerance = false;
            // std::cout << "Element at index " << i << " is not within tolerance. Value: " << value[i]
            //           << ", Target: " << target[i % output_width] << ". Diff: " << diff << std::endl;
        }
    }

    if (allWithinTolerance) {

        std::cout << "All elements are within tolerance. Total values checked: " << total_values_checked << std::endl;
    } else {
        std::cout << "Not all elements are within tolerance.. Total values checked: " << total_values_checked
                  << " , max diff = " << max_diff << std::endl;
    }

    return allWithinTolerance;
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
