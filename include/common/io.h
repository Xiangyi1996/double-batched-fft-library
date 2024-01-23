#include "encoding.h"
#include "json.hpp"
#include <string>
#include <unordered_map>

using json = nlohmann::json;
// Define the enum classes

// Map to convert string to enum for each enum class
const std::unordered_map<std::string, GridType> gridTypeMap{
    {"Hash", GridType::Hash}, {"Dense", GridType::Dense}, {"Tiled", GridType::Tiled}};

const std::unordered_map<std::string, HashType> hashTypeMap{{"Prime", HashType::Prime},
                                                            {"CoherentPrime", HashType::CoherentPrime},
                                                            {"ReversedPrime", HashType::ReversedPrime},
                                                            {"Rng", HashType::Rng}};

const std::unordered_map<std::string, InterpolationType> interpolationTypeMap{
    {"Nearest", InterpolationType::Nearest},
    {"Linear", InterpolationType::Linear},
    {"Smoothstep", InterpolationType::Smoothstep}};

// Helper function to convert string to enum
template <typename T> T stringToEnum(const std::string &value, const std::unordered_map<std::string, T> &enumMap) {
    auto it = enumMap.find(value);
    if (it != enumMap.end()) {
        return it->second;
    }
    // Handle error or default case here
    // For example, you can throw an exception if the value is not found
    throw std::runtime_error("Invalid enum value");
}

json loadJsonConfig(const std::string &filename) {
    std::ifstream file{filename};
    if (!file) {
        throw std::runtime_error("Error: Unable to open file '" + filename + "'");
    }
    json config = json::parse(file, nullptr, true, /*skip_comments=*/true);

    if (config.contains(EncodingParams::GRID_TYPE)) {
        config[EncodingParams::GRID_TYPE] = stringToEnum(config[EncodingParams::GRID_TYPE], gridTypeMap);
    } else if (config.contains(EncodingParams::HASH)) {
        config[EncodingParams::HASH] = stringToEnum(config[EncodingParams::HASH], hashTypeMap);
    } else if (config.contains(EncodingParams::INTERPOLATION_METHOD)) {
        config[EncodingParams::INTERPOLATION_METHOD] =
            stringToEnum(config[EncodingParams::INTERPOLATION_METHOD], interpolationTypeMap);
    }
    return config;
}