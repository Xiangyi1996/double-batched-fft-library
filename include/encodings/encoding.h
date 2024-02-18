/**
 * @file encoding.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Implementation of an absract base class for the encodings.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <common.h>
#include <type_traits> // for std::is_same

#include <cstdint>
#include <sycl/sycl.hpp>

#include "DeviceMatrix.h"
#include "DeviceMem.h"
#include "common.h"
#include "json.hpp"

using json = nlohmann::json;

struct EncodingParams {
    inline static const std::string ENCODING = "otype";                            // EncodingNames
    inline static const std::string N_DIMS_TO_ENCODE = "n_dims_to_encode";         // uint32_t
    inline static const std::string GRID_TYPE = "type";                            // GridType Hash, Dense, Tiled
    inline static const std::string N_LEVELS = "n_levels";                         // uint32_t
    inline static const std::string N_FEATURES = "n_features";                     // uint32_t
    inline static const std::string N_FEATURES_PER_LEVEL = "n_features_per_level"; // uint32_t
    inline static const std::string LOG2_HASHMAP_SIZE = "log2_hashmap_size";       // uint32_t
    inline static const std::string BASE_RESOLUTION = "base_resolution";           // uint32_t
    inline static const std::string PER_LEVEL_SCALE = "per_level_scale";           // float
    inline static const std::string DEGREE = "degree";                             // uint32_t
    inline static const std::string SCALE = "scale";                               // float
    inline static const std::string OFFSET = "offset";                             // float
    inline static const std::string HASH = "hash";                                 // HashType
    inline static const std::string INTERPOLATION_METHOD = "interpolation_method"; // InterpolationType
    inline static const std::string USE_STOCHASTIC_INTERPOLATION = "stochastic_interpolation"; // bool
};

struct EncodingNames {
    inline static const std::string IDENTITY = "Identity";
    inline static const std::string SPHERICALHARMONICS = "SphericalHarmonics";
    inline static const std::string GRID = "Grid";
};

enum class GradientMode {
    Ignore,
    Overwrite,
    Accumulate,
};

enum class GridType {
    Hash,
    Dense,
    Tiled,
};

enum class HashType {
    Prime,
    CoherentPrime,
    ReversedPrime,
    Rng,
};

enum class InterpolationType {
    Nearest,
    Linear,
    Smoothstep,
};

enum class ReductionType {
    Concatenation,
    Sum,
    Product,
};

// Type trait to detect bf16 type
template <typename T> struct is_bf16 : std::false_type {};

// Specialization for actual bf16 type, if bf16 type is available
template <> struct is_bf16<sycl::ext::oneapi::bfloat16> : std::true_type {};

template <typename T> class Encoding {
  public:
    Encoding() : m_n_params(0) {}
    virtual ~Encoding() {}

    virtual std::unique_ptr<Context> forward_impl(sycl::queue *const q, const DeviceMatrix<float> &input,
                                                  DeviceMatrix<T> *output = nullptr, bool use_inference_params = false,
                                                  bool prepare_input_gradients = false) = 0;

    virtual void backward_impl(sycl::queue *const q, const Context &ctx, const DeviceMatrix<float> &input,
                               const DeviceMatrix<T> &output, const DeviceMatrix<T> &dL_doutput,
                               DeviceMatrix<float> *dL_dinput = nullptr, bool use_inference_params = false,
                               GradientMode param_gradients_mode = GradientMode::Overwrite) = 0;

    virtual void set_padded_output_width(uint32_t padded_output_width) = 0;

    virtual void initialize_params(float *params_full_precision, float scale = 1) = 0;

    virtual uint32_t input_width() const = 0;

    virtual uint32_t padded_output_width() const = 0;

    virtual uint32_t output_width() const = 0;

    // TODO: Remove; should be inherited from object.h at some point
    // These are the weights
    T *params() const { return m_params; }

    T *inference_params() const { return m_inference_params; }

    T *gradients() const { return m_gradients; }

    size_t n_params() const { return m_n_params; }

    void set_params(T *params, T *inference_params, T *gradients) {
        m_params = params;
        m_inference_params = inference_params;
        m_gradients = gradients;
    }

    void set_params_helper(DeviceMem<T> &params_full_precision, DeviceMem<T> &gradients,
                           std::vector<T> *params = nullptr) {

        if constexpr (!is_bf16<T>::value) {

            if (params == nullptr) {
                initialize_params(params_full_precision.data());
            } else {
                CHECK(params->size() == this->n_params());
                params_full_precision.copy_from_host(*params).wait();
            }

            this->set_params(params_full_precision.data(), params_full_precision.data(), gradients.data());
        }
    }

  private:
    T *m_params = nullptr;
    T *m_inference_params = nullptr;
    T *m_gradients = nullptr;

    struct ForwardContext : public Context {
        DeviceMatrix<T> network_input;
        std::unique_ptr<Context> encoding_ctx;
        std::unique_ptr<Context> network_ctx;
    };

  protected:
    uint32_t m_n_params;
};
