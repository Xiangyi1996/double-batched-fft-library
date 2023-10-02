/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TOR
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** @file   encoding.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  API interface for input encodings
 */

#pragma once

#include <common.h>

#include <cstdint>
#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>

#include "activation.h"
#include "gpu_matrix.h"

enum class GradientMode {
  Ignore,
  Overwrite,
  Accumulate,
};

// Base classs for Encoding

template <typename T>
class Encoding {
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

  virtual std::unique_ptr<Context> forward_impl(
      dpct::queue_ptr stream, const GPUMatrixDynamic<float>& input,
      GPUMatrixDynamic<T>* output = nullptr, bool use_inference_params = false,
      bool prepare_input_gradients = false) = 0;

  virtual void backward_impl(
      dpct::queue_ptr stream, const Context& ctx,
      const GPUMatrixDynamic<float>& input, const GPUMatrixDynamic<T>& output,
      const GPUMatrixDynamic<T>& dL_doutput,
      GPUMatrixDynamic<float>* dL_dinput = nullptr,
      bool use_inference_params = false,
      GradientMode param_gradients_mode = GradientMode::Overwrite) = 0;

  // Perform forward pass through the encoding
  // tncc: inference_mixed_precision_impl() -> forward()
  virtual void forward_pass(
      const DeviceMem<bf16>& input, float* forward, float* A, float* B,
      float* C,
      DeviceMem<float>& output) = 0;  // inference_mixed_precision_impl
                                      // -> forward

  // Perform backward pass through the encoding
  // tncc: inference_mixed_precision_impl() -> forward()
  virtual void backward_pass(
      const DeviceMem<bf16>& input, DeviceMem<bf16>& grads, float* out_inter,
      float* delta_temp, DeviceMem<bf16> loss, float* A, float* B, float* C,
      float* A_backward_last_layer, float* B_backward_last_layer,
      float* C_backward_last_layer, float* D_backward_last_layer,
      float* E_backward_last_layer, float* F_backward_last_layer,
      float* A_dgemm, float* B_dgemm, float* C_dgemm, float* forward) = 0;

  // Get the SYCL queue associated with the network
  queue get_queue() { return m_q; }

  virtual void set_padded_output_width(uint32_t padded_output_width) = 0;
  virtual uint32_t required_output_alignment() const = 0;

  virtual MatrixLayout preferred_output_layout() const = 0;

  virtual size_t n_nested() const { return 0; }
  virtual const std::shared_ptr<Encoding<T>>& nested(size_t idx = 0) const {
    throw std::runtime_error{"Encoding does not support nesting."};
  }

  // By default, an encoding has no parameters
  // void set_params_impl(T* params, T* inference_params, T* gradients) override
  // { } void initialize_params(pcg32& rnd, float* params_full_precision, float
  // scale = 1) override { }
  void initialize_params() {}
  // size_t n_params() const override { return 0; }
  // std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override {
  // return {}; }

  // TODO: Do we require lcm / gcm as well?
  void set_alignment(uint32_t alignment) {
    this->set_padded_output_width(
        next_multiple(this->output_width(),
                      lcm(alignment, this->required_output_alignment())));
  }

  virtual uint32_t input_width() const = 0;

  virtual uint32_t padded_output_width() const = 0;

  virtual uint32_t output_width() const = 0;

  virtual uint32_t required_input_alignment() const = 0;

  // Data members
  float* m_forward;
  int m_shmem_size;
  size_t m_alignment;

  bf16* m_act_mem;
  float* m_act_mem_temp;

  float* m_A_forward;
  float* m_B_forward;
  float* m_C_forward;

  float* m_out_inter;
  float* m_deltas_temp;
  DeviceMem<bf16> m_deltas;

  float* m_A_backward;
  float* m_B_backward;
  float* m_C_backward;

  float* m_A_backward_last_layer;
  float* m_B_backward_last_layer;
  float* m_C_backward_last_layer;
  float* m_D_backward_last_layer;
  float* m_E_backward_last_layer;
  float* m_F_backward_last_layer;

  float* m_A_dgemm;
  float* m_B_dgemm;
  float* m_C_dgemm;

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

 private:
  struct ForwardContext : public Context {
    GPUMatrixDynamic<T> network_input;
    std::unique_ptr<Context> encoding_ctx;
    std::unique_ptr<Context> network_ctx;
  };
};