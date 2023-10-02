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

/** @file   oneblob.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Implementation of the oneblob encoding [Mueller et al. 2019].
 *          The Gaussian kernel was replaced by a quartic kernel for performance.
 */

#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <common.h>
#include <encoding.h>
#include <gpu_memory.h>
#include <common_device.h>

#include <numeric>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

namespace tcnn {

template <typename F>
inline float one_blob_subwarp_aligned(F kernel, MatrixView<const float> data_in, const uint32_t elem_index, const uint32_t encoded_index, const uint32_t num_bins_log2,
                                      const sycl::nd_item<3> &item_ct1) {
	const uint32_t n_bins = 1 << num_bins_log2;
	const uint32_t bin_index = encoded_index & (n_bins - 1);
	const float x = data_in(encoded_index >> num_bins_log2, elem_index);

        /*
        DPCT1017:199: The sycl::exp2 call is used instead of the scalbnf call.
        These two calls do not provide exactly the same functionality. Check the
        potential precision and/or performance issues for the generated code.
        */
        const float left_boundary = bin_index * (2 << -num_bins_log2);
        float left_cdf = kernel(left_boundary - x, n_bins) + kernel(left_boundary - x - 1.0f, n_bins) + kernel(left_boundary - x + 1.0f, n_bins);

	// OneBlob needs an evaluation for both the left and the right boundary.
	// Compute cost can be saved by computing just one boundary and shuffling the result of the next from the neighboring lane.
	// Threadblocks are arranged such that bin counts are powers of two that never span across multiple warps.
	// Note that this procedure necessitates making the OneBlob encoding wrap around (hence also the 3 kernel calls above),
	// which may not always be desired.
	// If not desired, use the slower implementation without wraparound below.
        /*
        DPCT1023:35: The SYCL sub-group does not support mask options for
        dpct::select_from_sub_group.
        */
        /*
        DPCT1096:209: The right-most dimension of the work-group used in the
        SYCL kernel that calls this function may be less than "32". The function
        "dpct::select_from_sub_group" may return an unexpected result on the CPU
        device. Modify the size of the work-group to ensure that the value of
        the right-most dimension is a multiple of "32".
        */
        float right_cdf = dpct::select_from_sub_group(
            item_ct1.get_sub_group(), left_cdf, bin_index + 1, n_bins);
        if (bin_index == n_bins - 1) {
		right_cdf += 1; // The right CDF must gain a 1 due to wrapping from right to left (it lost one (hopefully) saturated CDF)
	}

	return right_cdf - left_cdf;
}

template <typename F>
inline float one_blob(F kernel, const float* __restrict__ data_in, const uint32_t encoded_index, const uint32_t num_bins_log2) {
	const uint32_t n_bins = 1 << num_bins_log2;
	const uint32_t bin_index = encoded_index & (n_bins - 1);
	const float x = data_in[encoded_index >> num_bins_log2];

        /*
        DPCT1017:200: The sycl::exp2 call is used instead of the scalbnf call.
        These two calls do not provide exactly the same functionality. Check the
        potential precision and/or performance issues for the generated code.
        */
        const float left_boundary = bin_index * (2 << -num_bins_log2);
        const float left_cdf = kernel(left_boundary - x, n_bins);

        /*
        DPCT1017:201: The sycl::exp2 call is used instead of the scalbnf call.
        These two calls do not provide exactly the same functionality. Check the
        potential precision and/or performance issues for the generated code.
        */
        const float right_boundary = bin_index + 1 * (2 << -num_bins_log2);
        const float right_cdf = kernel(right_boundary - x, n_bins);

	return right_cdf - left_cdf;
}

template <typename T>
void kernel_one_blob(
	const uint32_t num_elements,
	const uint32_t num_bins_log2,
	MatrixView<const float> data_in,
	PitchedPtr<T> data_out
,
	const sycl::nd_item<3> &item_ct1) {
        const uint32_t i = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                           item_ct1.get_local_id(1);
        const uint32_t j = item_ct1.get_local_id(2);
        if (i >= num_elements) return;

        data_out(i)[j] = (T)one_blob_subwarp_aligned(quartic_cdf, data_in, i, j,
                                                     num_bins_log2, item_ct1);
}

template <typename T>
void kernel_one_blob_soa(
	const uint32_t num_elements,
	const uint32_t num_bins_log2,
	const uint32_t num_to_encode,
	MatrixView<const float> data_in,
	T* __restrict__ data_out
,
	const sycl::nd_item<3> &item_ct1) {
        const uint32_t i = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                           item_ct1.get_local_id(1);
        const uint32_t j = item_ct1.get_local_id(2);
        const uint32_t to_encode_index = j + i * item_ct1.get_local_range(2);
        if (to_encode_index >= num_elements * num_to_encode) return;

	const float x = data_in(j, i);

	const uint32_t n_bins = 1 << num_bins_log2;
	T* out = (data_out + i + j * n_bins * num_elements);

	float left_cdf = quartic_cdf(-x, n_bins) + quartic_cdf(-x - 1.0f, n_bins) + quartic_cdf(-x + 1.0f, n_bins);

	for (uint32_t k = 0; k < n_bins; ++k) {
                /*
                DPCT1017:202: The sycl::exp2 call is used instead of the scalbnf
                call. These two calls do not provide exactly the same
                functionality. Check the potential precision and/or performance
                issues for the generated code.
                */
                const float right_boundary = k + 1 * (2 << -num_bins_log2);
                const float right_cdf = quartic_cdf(right_boundary - x, n_bins) + quartic_cdf(right_boundary - x - 1.0f, n_bins) + quartic_cdf(right_boundary - x + 1.0f, n_bins);

		*out = (T)(right_cdf - left_cdf);

		left_cdf = right_cdf;
		out += num_elements;
	}
}

template <typename T>
void kernel_one_blob_backward(
	const uint32_t num_elements,
	const uint32_t n_dims_to_encode,
	const uint32_t num_bins_log2,
	MatrixView<const T> dL_dy,
	MatrixView<const float> data_in,
	MatrixView<float> dL_dx
,
	const sycl::nd_item<3> &item_ct1) {
        const uint32_t i = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                           item_ct1.get_local_id(1);
        const uint32_t j = item_ct1.get_local_id(2);
        const uint32_t to_encode_index = j + i * item_ct1.get_local_range(2);
        if (to_encode_index >= num_elements * n_dims_to_encode) return;

	const float x = data_in(j, i);

	const uint32_t n_bins = 1 << num_bins_log2;

	float result = 0;

	float left_cdf = quartic_cdf_deriv(-x, n_bins) + quartic_cdf_deriv(-x - 1.0f, n_bins) + quartic_cdf_deriv(-x + 1.0f, n_bins);

	for (uint32_t k = 0; k < n_bins; ++k) {
                /*
                DPCT1017:203: The sycl::exp2 call is used instead of the scalbnf
                call. These two calls do not provide exactly the same
                functionality. Check the potential precision and/or performance
                issues for the generated code.
                */
                const float right_boundary = k + 1 * (2 << -num_bins_log2);
                const float right_cdf = quartic_cdf_deriv(right_boundary - x, n_bins) + quartic_cdf_deriv(right_boundary - x - 1.0f, n_bins) + quartic_cdf_deriv(right_boundary - x + 1.0f, n_bins);

		float deriv = left_cdf - right_cdf;

		left_cdf = right_cdf;

		uint32_t encoded_dim = j * n_bins + k;
		result += (float)dL_dy(encoded_dim, i) * deriv;
	}

	dL_dx(j, i) = result;
}

template <typename T>
class OneBlobEncoding : public Encoding<T> {
public:
	OneBlobEncoding(uint32_t n_bins, uint32_t n_dims_to_encode)
	: m_n_bins{n_bins}, m_n_dims_to_encode{n_dims_to_encode} {
		m_n_output_dims = m_n_dims_to_encode * m_n_bins;

		// Make sure the number of bins is a power of 2---this is required for certain optimizations
		// in our compute kernel.
		if ((n_bins & (n_bins - 1)) != 0) {
			throw std::runtime_error{"Number of bins must be a power of 2"};
		}
	}

    std::unique_ptr<Context>
    forward_impl(dpct::queue_ptr stream,
                     const GPUMatrixDynamic<float> &input,
                     GPUMatrixDynamic<T> *output = nullptr,
                     bool use_inference_params = false,
                     bool prepare_input_gradients = false) override {
        if (!output || padded_output_width() == 0) {
			return std::make_unique<Context>();
		}

		const uint32_t num_bins_log2 = (uint32_t)std::log2(m_n_bins);

		if (output->layout() == AoS) {
			const uint32_t min_n_threads = N_THREADS_LINEAR;
                        const sycl::range<3> threads = {
                            1, div_round_up(min_n_threads, m_n_output_dims),
                            m_n_output_dims};
                        const uint32_t n_threads = threads[2] * threads[1];
                        const sycl::range<3> blocks = {
                            1, 1,
                            div_round_up(input.n() * m_n_output_dims,
                                         n_threads)};

                        /*
                        DPCT1049:36: The work-group size passed to the SYCL
                        kernel may exceed the limit. To get the device limit,
                        query info::device::max_work_group_size. Adjust the
                        work-group size if needed.
                        */
            stream->submit([&](sycl::handler &cgh) {
                auto input_n_ct0 = input.n();
                auto input_view_ct2 = input.view();
                auto output_pitched_ptr_ct3 = output->pitched_ptr();

                cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1)
                                     [[intel::reqd_sub_group_size(32)]] {
                                         kernel_one_blob<T>(
                                             input_n_ct0, num_bins_log2,
                                             input_view_ct2,
                                             output_pitched_ptr_ct3, item_ct1);
                                     });
            });

                        // Padding
			parallel_for_gpu_aos(stream, input.n(), m_n_to_pad, [n_output_dims=m_n_output_dims, out=output->pitched_ptr()] (size_t elem, size_t dim) {
				out(elem)[n_output_dims + dim] = (T)1.0f;
			});
		} else {
			const uint32_t min_n_threads = N_THREADS_LINEAR;
                        const sycl::range<3> threads = {
                            1, div_round_up(min_n_threads, m_n_dims_to_encode),
                            m_n_dims_to_encode};
                        const uint32_t n_threads = threads[2] * threads[1];
                        const sycl::range<3> blocks = {
                            1, 1,
                            div_round_up(input.n() * m_n_dims_to_encode,
                                         n_threads)};

                        /*
                        DPCT1049:37: The work-group size passed to the SYCL
                        kernel may exceed the limit. To get the device limit,
                        query info::device::max_work_group_size. Adjust the
                        work-group size if needed.
                        */
            stream->submit([&](sycl::handler &cgh) {
                auto input_n_ct0 = input.n();
                auto m_n_dims_to_encode_ct2 = m_n_dims_to_encode;
                auto input_view_ct3 = input.view();
                auto output_data_ct4 = output->data();

                cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     kernel_one_blob_soa<T>(
                                         input_n_ct0, num_bins_log2,
                                         m_n_dims_to_encode_ct2, input_view_ct3,
                                         output_data_ct4, item_ct1);
                                 });
            });

                        // Padding
			parallel_for_gpu(stream, input.n() * m_n_to_pad, [out=output->data() + input.n() * m_n_dims_to_encode] (size_t i) {
				out[i] = (T)1.0f;
			});
		}

		return std::make_unique<Context>();
	}

    void backward_impl(dpct::queue_ptr stream, const Context &ctx,
                           const GPUMatrixDynamic<float> &input,
                           const GPUMatrixDynamic<T> &output,
                           const GPUMatrixDynamic<T> &dL_doutput,
                           GPUMatrixDynamic<float> *dL_dinput = nullptr,
                           bool use_inference_params = false,
                           GradientMode param_gradients_mode =
                               GradientMode::Overwrite) override {
        if (!dL_dinput) {
            return;
        }

		const uint32_t num_bins_log2 = (uint32_t)std::log2(m_n_bins);

		const uint32_t min_n_threads = N_THREADS_LINEAR;
                const sycl::range<3> threads = {
                    1, div_round_up(min_n_threads, m_n_dims_to_encode),
                    m_n_dims_to_encode};
                const uint32_t n_threads = threads[2] * threads[1];
                const sycl::range<3> blocks = {
                    1, 1,
                    div_round_up(input.n() * m_n_dims_to_encode, n_threads)};

                /*
                DPCT1049:38: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
        stream->submit([&](sycl::handler &cgh) {
            auto input_n_ct0 = input.n();
            auto m_n_dims_to_encode_ct1 = m_n_dims_to_encode;
            auto dL_doutput_view_ct3 = dL_doutput.view();
            auto input_view_ct4 = input.view();
            auto dL_dinput_view_ct5 = dL_dinput->view();

            cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                             [=](sycl::nd_item<3> item_ct1) {
                                 kernel_one_blob_backward<T>(
                                     input_n_ct0, m_n_dims_to_encode_ct1,
                                     num_bins_log2, dL_doutput_view_ct3,
                                     input_view_ct4, dL_dinput_view_ct5,
                                     item_ct1);
                             });
        });
        }

	uint32_t input_width() const override {
		return m_n_dims_to_encode;
	}

	uint32_t padded_output_width() const override {
		return m_n_output_dims + m_n_to_pad;
	}

	uint32_t output_width() const override {
		return padded_output_width();
	}

	uint32_t required_input_alignment() const override {
		return 1;
	}

	void set_padded_output_width(uint32_t padded_output_width) override {
		CHECK_THROW(padded_output_width >= m_n_output_dims);
		m_n_to_pad = padded_output_width - m_n_output_dims;
	}

	uint32_t required_output_alignment() const override {
		return 1;
	}

	MatrixLayout preferred_output_layout() const override {
		return AoS;
	}

	json hyperparams() const override {
		return {
			{"otype", "OneBlob"},
			{"n_bins", m_n_bins},
		};
	}

private:
	uint32_t m_n_bins;
	uint32_t m_n_dims_to_encode;

	// derived sizes
	uint32_t m_n_output_dims;
	uint32_t m_n_to_pad = 0;
};

}
