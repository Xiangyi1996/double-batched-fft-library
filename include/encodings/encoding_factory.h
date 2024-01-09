#include <string>
#include <unordered_map>

#include "encoding.h"
#include "grid.h"
#include "identity.h"
#include "spherical_harmonics.h"

// Base EncodingFactory class
template <typename T> class EncodingFactory {
  public:
    virtual ~EncodingFactory() {}
    virtual std::shared_ptr<Encoding<T>> create(const std::unordered_map<std::string, std::string> &params) const = 0;
};

// EncodingFactory for IdentityEncoding
template <typename T> class IdentityEncodingFactory : public EncodingFactory<T> {
  public:
    std::shared_ptr<Encoding<T>> create(const std::unordered_map<std::string, std::string> &params) const override {
        uint32_t n_dims_to_encode = std::stoi(params.at("n_dims_to_encode"));
        float scale = std::stof(params.at("scale"));
        float offset = std::stof(params.at("offset"));
        return std::make_shared<IdentityEncoding<T>>(n_dims_to_encode, scale, offset);
    }
};

// EncodingFactory for SphericalHarmonicsEncoding
template <typename T> class SphericalHarmonicsEncodingFactory : public EncodingFactory<T> {
  public:
    std::shared_ptr<Encoding<T>> create(const std::unordered_map<std::string, std::string> &params) const override {
        uint32_t degree = std::stoi(params.at("degree"));
        uint32_t n_dims_to_encode = std::stoi(params.at("n_dims_to_encode"));
        return std::make_shared<SphericalHarmonicsEncoding<T>>(degree, n_dims_to_encode);
    }
};

template <typename T> class GridEncodingFactory;

// Specialization for T = bf16 (exclude implementation)
template <> class GridEncodingFactory<bf16> : public EncodingFactory<bf16> {
  public:
    std::shared_ptr<Encoding<bf16>> create(const std::unordered_map<std::string, std::string> &params) const override {
        // Throw an error or handle the unsupported case for bf16
        throw std::runtime_error("GridEncodingFactory does not support bf16");
    }
};

// Specialization for T != bf16 (include implementation)
// EncodingFactory for GridEncodingTemplated
template <typename T> class GridEncodingFactory : public EncodingFactory<T> {
  public:
    std::shared_ptr<Encoding<T>> create(const std::unordered_map<std::string, std::string> &params) const override {
        uint32_t n_levels = params.count("n_levels") ? std::stoi(params.at("n_levels")) : 16u;
        uint32_t n_features_per_level =
            params.count("n_features_per_level") ? std::stoi(params.at("n_features_per_level")) : 2u;
        uint32_t n_dims_to_encode = params.count("n_dims_to_encode") ? std::stoi(params.at("n_dims_to_encode")) : 2u;

        uint32_t log2_hashmap_size =
            params.count("log2_hashmap_size") ? std::stoi(params.at("log2_hashmap_size")) : 19u;
        uint32_t base_resolution = params.count("base_resolution") ? std::stoi(params.at("base_resolution")) : 16u;
        float per_level_scale = params.count("per_level_scale") ? std::stof(params.at("per_level_scale")) : 2.0f;

        std::string type = params.count("type") ? params.at("type") : "Hash";
        json encoding_json = {
            {"n_dims_to_encode", n_dims_to_encode},
            {"otype", "Grid"},
            {"type", type},
            {"n_levels", n_levels},
            {"n_features_per_level", n_features_per_level},
            {"log2_hashmap_size", log2_hashmap_size},
            {"base_resolution", base_resolution},
            {"per_level_scale", per_level_scale},
        };

        return create_grid_encoding<T>(n_dims_to_encode, encoding_json);
    }
};

// Create a map to associate encoding names with their factories
template <typename T> class EncodingFactoryRegistry {
  public:
    void registerFactory(const std::string &name, std::unique_ptr<EncodingFactory<T>> factory) {
        factories_[name] = std::move(factory);
    }

    std::shared_ptr<Encoding<T>> create(const std::string &name,
                                        const std::unordered_map<std::string, std::string> &params) const {
        auto it = factories_.find(name);
        if (it != factories_.end()) {
            return it->second->create(params);
        } else {
            // Handle the case where the encoding name is not registered
            throw std::runtime_error("Unknown encoding type: " + name);
        }
    }

  private:
    std::unordered_map<std::string, std::unique_ptr<EncodingFactory<T>>> factories_;
};

template <typename T>
std::shared_ptr<Encoding<T>> create_encoding(std::string name,
                                             std::unordered_map<std::string, std::string> encoding_config) {
    //   std::cout << "Contents of encoding config:" << std::endl;
    //   for (const auto& pair : encoding_config) {
    //     std::cout << pair.first << ": " << pair.second << std::endl;
    //   }
    // Create a registry for encoding factories
    EncodingFactoryRegistry<T> encodingRegistry;

    if (name == "Identity") {
        // Register the IdentityEncoding factory
        encodingRegistry.registerFactory("Identity", std::make_unique<IdentityEncodingFactory<T>>());

        // Create an IdentityEncoding instance using the factory and parameters
        return encodingRegistry.create("Identity", encoding_config);

    } else if (name == "SphericalHarmonics") {
        // Register the SphericalHarmonicsEncoding factory
        encodingRegistry.registerFactory("SphericalHarmonics",
                                         std::make_unique<SphericalHarmonicsEncodingFactory<T>>());

        // Create an SphericalHarmonicsEncoding instance using the factory and
        // parameters
        return encodingRegistry.create("SphericalHarmonics", encoding_config);
    } else if (name.find("Grid") != std::string::npos) {
        // Register the GridEncodings factory
        encodingRegistry.registerFactory("Grid", std::make_unique<GridEncodingFactory<T>>());

        return encodingRegistry.create("Grid", encoding_config);
        ;
    } else {
        throw std::invalid_argument("Encoding name unknown");
    }
}
