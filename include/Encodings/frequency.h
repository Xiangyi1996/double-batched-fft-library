/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** @file   frequency.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Implementation of the frequency encoding of NeRF [Mildenhall et al. 2020].
 */

#pragma once

#include <common.h>
#include <common_device.h>
#include <dpct/dpct.hpp>
#include <encoding.h>
#include <gpu_memory.h>
#include <sycl/sycl.hpp>

#include <numeric>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

namespace tcnn {

template <typename T>
void frequency_encoding(const uint32_t num_elements, const uint32_t n_frequencies, const uint32_t num_to_encode,
                        const uint32_t num_to_pad, MatrixView<const float> data_in, MatrixView<T> data_out,
                        float *__restrict__ dy_dx, const sycl::nd_item<3> &item_ct1) {
    const uint32_t encoded_index = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range(2);
    if (encoded_index >= num_elements) return;

    const uint32_t fan_out_encoded = num_to_encode * n_frequencies * 2;
    const uint32_t fan_out = fan_out_encoded + num_to_pad;

    const uint32_t i = encoded_index / fan_out;
    const uint32_t j = encoded_index - i * fan_out;

    if (j >= fan_out_encoded) {
        data_out(j, i) = 1;
    } else {
        const uint32_t encoded_input_feature_i = j / (n_frequencies * 2);
        const uint32_t log2_frequency = (j / 2) % n_frequencies;

        const float phase_shift = (j % 2) * (PI / 2);

        /*
        DPCT1017:195: The sycl::exp2 call is used instead of the scalbnf
        call. These two calls do not provide exactly the same
        functionality. Check the potential precision and/or performance
        issues for the generated code.
        */
        const float x = data_in(encoded_input_feature_i, i) * (2 << log2_frequency);
        const float input = x * PI + phase_shift;
        data_out(j, i) = (T)sycl::sin((float)input);
        if (dy_dx != nullptr) {
            /*
            DPCT1017:196: The sycl::exp2 call is used instead of the
            scalbnf call. These two calls do not provide exactly the
            same functionality. Check the potential precision and/or
            performance issues for the generated code.
            */
            dy_dx[i * fan_out_encoded + j] = 1.0f * (2 << log2_frequency) * PI * sycl::cos((float)input);
        }
    }
}

template <typename T>
void frequency_encoding_backward(const uint32_t num_elements, const uint32_t n_dims_to_encode,
                                 const uint32_t n_frequencies, MatrixView<const T> dL_dy, const float *dy_dx,
                                 MatrixView<float> dL_dx, const sycl::nd_item<3> &item_ct1) {
    const uint32_t encoded_index = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range(2);
    if (encoded_index >= num_elements) return;

    const uint32_t i = encoded_index / n_dims_to_encode;
    const uint32_t j = encoded_index - i * n_dims_to_encode;

    const uint32_t outputs_per_input = n_frequencies * 2;

    float result = 0;
    for (int k = 0; k < outputs_per_input; ++k) {
        result += (float)dL_dy(j * outputs_per_input + k, i) *
                  dy_dx[i * n_dims_to_encode * outputs_per_input + j * outputs_per_input + k];
    }
    dL_dx(j, i) = result;
}

template <typename T> class FrequencyEncoding : public Encoding<T> {
  public:
    FrequencyEncoding(uint32_t n_frequencies, uint32_t n_dims_to_encode)
        : m_n_frequencies{n_frequencies}, m_n_dims_to_encode{n_dims_to_encode} {
        m_n_output_dims = m_n_dims_to_encode * m_n_frequencies * 2;
    }

    std::unique_ptr<Context> forward_impl(dpct::queue_ptr stream, const GPUMatrixDynamic<float> &input,
                                          GPUMatrixDynamic<T> *output = nullptr, bool use_inference_params = false,
                                          bool prepare_input_gradients = false) override {
        auto forward = std::make_unique<ForwardContext>();

        if (!output || padded_output_width() == 0) {
            return forward;
        }

        if (prepare_input_gradients) {
            forward->dy_dx = GPUMatrix<float>{m_n_dims_to_encode * m_n_frequencies * 2, input.n(), stream};
        }

        linear_kernel(frequency_encoding<T>, 0, stream, input.n() * padded_output_width(), m_n_frequencies,
                      m_n_dims_to_encode, m_n_to_pad, input.view(), output->view(), forward->dy_dx.data());

        return forward;
    }

    void backward_impl(dpct::queue_ptr stream, const Context &ctx, const GPUMatrixDynamic<float> &input,
                       const GPUMatrixDynamic<T> &output, const GPUMatrixDynamic<T> &dL_doutput,
                       GPUMatrixDynamic<float> *dL_dinput = nullptr, bool use_inference_params = false,
                       GradientMode param_gradients_mode = GradientMode::Overwrite) override {
        if (!dL_dinput) {
            return;
        }

        const auto &forward = dynamic_cast<const ForwardContext &>(ctx);

        linear_kernel(frequency_encoding_backward<T>, 0, stream, input.n() * m_n_dims_to_encode, m_n_dims_to_encode,
                      m_n_frequencies, dL_doutput.view(), forward.dy_dx.data(), dL_dinput->view());
    }

    uint32_t input_width() const override { return m_n_dims_to_encode; }

    uint32_t padded_output_width() const override { return m_n_output_dims + m_n_to_pad; }

    uint32_t output_width() const override { return padded_output_width(); }

    uint32_t required_input_alignment() const override { return 1; }

    void set_padded_output_width(uint32_t padded_output_width) override {
        CHECK_THROW(padded_output_width >= m_n_output_dims);
        m_n_to_pad = padded_output_width - m_n_output_dims;
    }

    uint32_t required_output_alignment() const override { return 1; }

    MatrixLayout preferred_output_layout() const override { return MatrixLayout::AoS; }

    json hyperparams() const override {
        return {
            {"otype", "Frequency"},
            {"n_frequencies", m_n_frequencies},
        };
    }

  private:
    struct ForwardContext : public Context {
        GPUMatrix<float> dy_dx;
    };

    uint32_t m_n_frequencies;
    uint32_t m_n_dims_to_encode;

    // derived sizes
    uint32_t m_n_output_dims;
    uint32_t m_n_to_pad = 0;
};

} // namespace tcnn
