#pragma once

#include <common.h>

#include <cstdint>
#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>

#include "activation.h"
#include "common_host.h"
#include "gpu_matrix.h"
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

    // void inference_mixed_precision_impl(
    // 	dpct::queue_ptr stream, const GPUMatrixDynamic<float> &input,
    // 	GPUMatrixDynamic<T> &output,
    // 	bool use_inference_params = true) override {
    // 		this->forward(stream, input, &output, use_inference_params,
    // false);
    // }

    virtual std::unique_ptr<Context> forward_impl(dpct::queue_ptr stream, const GPUMatrixDynamic<float> &input,
                                                  GPUMatrixDynamic<T> *output = nullptr,
                                                  bool use_inference_params = false,
                                                  bool prepare_input_gradients = false) = 0;

    virtual void backward_impl(dpct::queue_ptr stream, const Context &ctx, const GPUMatrixDynamic<float> &input,
                               const GPUMatrixDynamic<T> &output, const GPUMatrixDynamic<T> &dL_doutput,
                               GPUMatrixDynamic<float> *dL_dinput = nullptr, bool use_inference_params = false,
                               GradientMode param_gradients_mode = GradientMode::Overwrite) = 0;

    // Get the SYCL queue associated with the network
    queue get_queue() { return m_q; }

    virtual void set_padded_output_width(uint32_t padded_output_width) = 0;
    virtual uint32_t required_output_alignment() const = 0;

    virtual MatrixLayout preferred_output_layout() const = 0;

    virtual size_t n_nested() const { return 0; }
    virtual const std::shared_ptr<Encoding<T>> &nested(size_t idx = 0) const {
        throw std::runtime_error{"Encoding does not support nesting."};
    }

    // By default, an encoding has no parameters
    // void set_params_impl(T* params, T* inference_params, T* gradients) override
    // { } void initialize_params(pcg32& rnd, float* params_full_precision, float
    // scale = 1) override { }
    void initialize_params() {}

    virtual void initialize_params(float *params_full_precision, float scale = 1) = 0;
    // size_t n_params() const override { return 0; }
    // std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override {
    // return {}; }

    // TODO: Do we require lcm / gcm as well?
    void set_alignment(uint32_t alignment) {
        // this->set_padded_output_width(
        //     tinydpcppnn::math::next_multiple(this->output_width(),
        //                   lcm(alignment, this->required_output_alignment())));
        this->set_padded_output_width(this->output_width());
    }

    virtual uint32_t input_width() const = 0;

    virtual uint32_t padded_output_width() const = 0;

    virtual uint32_t output_width() const = 0;

    virtual uint32_t required_input_alignment() const = 0;

    // TODO: Remove; should be inherited from object.h at soe point
    T *params() const { return m_params; }

    T *inference_params() const { return m_inference_params; }

    T *gradients() const { return m_gradients; }

    size_t n_params() const { return 0; };

    void set_params(T *params, T *inference_params, T *gradients) {
        std::cout << "Set params got called" << std::endl;
        m_params = params;
        m_inference_params = inference_params;
        m_gradients = gradients;
    }

    // Data members
    float *m_forward;
    int m_shmem_size;
    size_t m_alignment;

    bf16 *m_act_mem;
    float *m_act_mem_temp;

    float *m_A_forward;
    float *m_B_forward;
    float *m_C_forward;

    float *m_out_inter;
    float *m_deltas_temp;
    DeviceMem<bf16> m_deltas;

    float *m_A_backward;
    float *m_B_backward;
    float *m_C_backward;

    float *m_A_backward_last_layer;
    float *m_B_backward_last_layer;
    float *m_C_backward_last_layer;
    float *m_D_backward_last_layer;
    float *m_E_backward_last_layer;
    float *m_F_backward_last_layer;

    float *m_A_dgemm;
    float *m_B_dgemm;
    float *m_C_dgemm;

    queue m_q;
    DeviceMem<bf16> m_grads_matrices;
    DeviceMem<bf16> m_weights_matrices;
    DeviceMem<bf16> m_weightsT_matrices;

  private:
    int m_n_hidden_layers;
    int m_n_hidden_matrices;
    int m_inputs_width;
    int m_net_width;
    int m_output_width;
    int m_padded_output_width;
    int m_batch_size;

    Activation m_activation;
    Activation m_output_activation;

    DeviceMem<bf16> m_weights_matrices_inferences;

    int m_total_n_params;
    T *m_params = nullptr;
    T *m_inference_params = nullptr;
    T *m_gradients = nullptr;

    struct ForwardContext : public Context {
        GPUMatrixDynamic<T> network_input;
        std::unique_ptr<Context> encoding_ctx;
        std::unique_ptr<Context> network_ctx;
    };
};
