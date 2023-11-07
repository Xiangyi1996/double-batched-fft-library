#include "gpu_matrix.h"

bool areVectorsWithinTolerance(const std::vector<float> &value, const std::vector<float> &target, float tolerance,
                               int output_width) {
    //   assert(a.size() == b.size());  // Ensure vectors have the same length

    int total_values_checked = 0;
    bool allWithinTolerance = true;

    for (size_t i = 0; i < value.size(); ++i) {
        float diff = std::abs(value[i] - target[i % output_width]);
        // std::cout << "Checking idx: " << i << std::endl;
        total_values_checked++;
        if (diff > tolerance) {
            allWithinTolerance = false;
            std::cout << "Element at index " << i << " is not within tolerance. Value: " << value[i]
                      << ", Target: " << target[i % output_width] << ". Diff: " << diff << std::endl;
        }
    }

    if (allWithinTolerance) {

        std::cout << "All elements are within tolerance. Total values checked: " << total_values_checked << std::endl;
    } else {
        std::cout << "Not all elements are within tolerance.. Total values checked: " << total_values_checked
                  << std::endl;
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
