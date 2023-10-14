#include <string>
#include <unordered_map>

#include "Encodings/identity.h"
#include "Encodings/spherical_harmonics.h"
#include "encoding.h"

// Base EncodingFactory class
template <typename T>
class EncodingFactory {
 public:
  virtual Encoding<T>* create(
      const std::unordered_map<std::string, std::string>& params) const = 0;
};

// EncodingFactory for IdentityEncoding
template <typename T>
class IdentityEncodingFactory : public EncodingFactory<T> {
 public:
  Encoding<T>* create(const std::unordered_map<std::string, std::string>&
                          params) const override {
    uint32_t n_dims_to_encode = std::stoi(params.at("n_dims_to_encode"));
    float scale = std::stof(params.at("scale"));
    float offset = std::stof(params.at("offset"));
    return new IdentityEncoding<T>(n_dims_to_encode, scale, offset);
  }
};

// EncodingFactory for IdentityEncoding
template <typename T>
class SphericalHarmonicsEncodingFactory : public EncodingFactory<T> {
 public:
  Encoding<T>* create(const std::unordered_map<std::string, std::string>&
                          params) const override {
    uint32_t degree = std::stoi(params.at("degree"));
    uint32_t n_dims_to_encode = std::stoi(params.at("n_dims_to_encode"));
    return new SphericalHarmonicsEncoding<T>(degree, n_dims_to_encode);
  }
};
// // EncodingFactory for GridEncodingTemplated
// template <typename T>
// class GridEncodingFactory : public EncodingFactory<T> {
//  public:
//   std::unique_ptr<Encoding<T>> create(
//       const std::unordered_map<std::string, std::string>& params)
//       const override {
//     uint32_t n_features = std::stoi(params.at("n_features"));
//     uint32_t log2_hashmap_size = std::stoi(params.at("log2_hashmap_size"));
//     uint32_t base_resolution = std::stoi(params.at("base_resolution"));
//     float per_level_scale = std::stof(params.at("per_level_scale"));
//     bool stochastic_interpolation =
//         params.at("stochastic_interpolation") == "true";
//     InterpolationType interpolation_type = static_cast<InterpolationType>(
//         std::stoi(params.at("interpolation_type")));
//     GridType grid_type =
//         static_cast<GridType>(std::stoi(params.at("grid_type")));

//     return std::make_unique<GridEncodingTemplated<T>>(
//         n_features, log2_hashmap_size, base_resolution, per_level_scale,
//         stochastic_interpolation, interpolation_type, grid_type);
//   }
// };

// Create a map to associate encoding names with their factories
template <typename T>
class EncodingFactoryRegistry {
 public:
  void registerFactory(const std::string& name,
                       std::unique_ptr<EncodingFactory<T>> factory) {
    factories_[name] = std::move(factory);
  }

  Encoding<T>* create(
      const std::string& name,
      const std::unordered_map<std::string, std::string>& params) const {
    auto it = factories_.find(name);
    if (it != factories_.end()) {
      return it->second->create(params);
    } else {
      // Handle the case where the encoding name is not registered
      throw std::runtime_error("Unknown encoding type: " + name);
    }
  }

 private:
  std::unordered_map<std::string, std::unique_ptr<EncodingFactory<T>>>
      factories_;
};

template <typename T>
Encoding<T>* create_encoding(
    uint32_t n_dims_to_encode, std::string name,
    std::unordered_map<std::string, std::string> encoding_config,
    uint32_t alignment = 8) {
  //   std::cout << "Contents of encoding config:" << std::endl;
  //   for (const auto& pair : encoding_config) {
  //     std::cout << pair.first << ": " << pair.second << std::endl;
  //   }
  // Create a registry for encoding factories
  EncodingFactoryRegistry<T> encodingRegistry;

  if (name == "Identity") {
    // Register the IdentityEncoding factory
    encodingRegistry.registerFactory(
        "Identity", std::make_unique<IdentityEncodingFactory<T>>());

    // Create an IdentityEncoding instance using the factory and parameters
    Encoding<T>* identityEncoding =
        encodingRegistry.create("Identity", encoding_config);
    return identityEncoding;

  } else if (name == "SphericalHarmonics") {
    // Register the IdentityEncoding factory
    encodingRegistry.registerFactory(
        "SphericalHarmonics",
        std::make_unique<SphericalHarmonicsEncodingFactory<T>>());

    // Create an SphericalHarmonicsEncoding instance using the factory and
    // parameters
    Encoding<T>* sphericalHarmonicsEncoding =
        encodingRegistry.create("SphericalHarmonics", encoding_config);
    return sphericalHarmonicsEncoding;
  } else {
    std::cout << name << " not implemented. Exiting." << std::endl;
    exit(0);
  }
}
