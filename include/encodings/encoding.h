#pragma once

#include <common.h>

#include <cstdint>
#include <sycl/sycl.hpp>

#include "DeviceMatrix.h"
#include "common_host.h"
#include "json.hpp"

using json = nlohmann::json;
using bf16 = sycl::ext::oneapi::bfloat16;
enum class GradientMode {
    Ignore,
    Overwrite,
    Accumulate,
};

template <typename T> class Encoding {
  public:
    Encoding() {}
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

    // TODO: Remove; should be inherited from object.h at soe point
    // These are the weights
    T *params() const { return m_params; }

    T *inference_params() const { return m_inference_params; }

    T *gradients() const { return m_gradients; }

    void set_params(T *params, T *inference_params, T *gradients) {
        // std::cout << "Set params got called" << std::endl;
        m_params = params;
        m_inference_params = inference_params;
        m_gradients = gradients;
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
};
