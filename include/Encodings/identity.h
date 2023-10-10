
#pragma once

#include <DeviceMem.h>
#include <common.h>
#include <common_device.h>
#include <encoding.h>
#include <gpu_matrix.h>
#include <stdint.h>

#include <dpct/dpct.hpp>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

#ifdef __SYCL_DEVICE_ONLY__

#define CONSTANT __attribute__((opencl_constant))

#else

#define CONSTANT

#endif
// template <typename T>
// void identity(
// 	const uint32_t num_outputs,
// 	const uint32_t num_to_encode,
// 	const uint32_t num_to_pad,
// 	const float scale,
// 	const T offset,
// 	sycl::accessor<float> data_in,
// 	sycl::accessor<T> data_out,
// 	const sycl::nd_item<3> &item_ct1)
// {
//         const uint32_t encoded_index =
//             item_ct1.get_local_id(2) +
//             item_ct1.get_group(2) * item_ct1.get_local_range(2);
//         if (encoded_index >= num_outputs) return;

// 	const uint32_t fan_out = num_to_encode + num_to_pad;
// 	const uint32_t i = encoded_index / fan_out;
// 	const uint32_t j = encoded_index - i * fan_out;

// 	if (j >= num_to_encode) {
// 		data_out[j] = 1;
// 	} else {
// 		data_out[j] = data_in[j] * scale + offset;
// 	}
// }

template <typename T>
void identity_backward(const uint32_t num_outputs,
                       const uint32_t n_dims_to_encode, const float scale,
                       MatrixView<const T> dL_dy, MatrixView<float> dL_dx,
                       const sycl::nd_item<3>& item_ct1) {
  const uint32_t output_index =
      item_ct1.get_local_id(2) +
      item_ct1.get_group(2) * item_ct1.get_local_range(2);
  if (output_index >= num_outputs) return;

  const uint32_t i = output_index / n_dims_to_encode;
  const uint32_t j = output_index - i * n_dims_to_encode;

  // The identity encoding can simply pass through the derivative.
  dL_dx(j, i) = (T)((float)dL_dy(j, i) * scale);
}

template <typename T>
class IdentityEncoding : public Encoding<T> {
 public:
  IdentityEncoding(uint32_t n_dims_to_encode, float scale = 1.0f,
                   float offset = 0.0f)
      : m_n_dims_to_encode{n_dims_to_encode}, m_scale{scale}, m_offset{offset} {
    m_n_output_dims = m_n_dims_to_encode;
    // std::cout << "Making Identity encoding with n_dims_to_encode: "
    //           << n_dims_to_encode << ", m_scale: " << m_scale
    //           << ", m_offset: " << m_offset << std::endl;
  }

  void forward_pass(const DeviceMem<bf16>& input, float* forward, float* A,
                    float* B, float* C, DeviceMem<float>& output)
      override{};  // inference_mixed_precision_impl -> forward

  // Perform backward pass through the encoding
  // tncc: inference_mixed_precision_impl() -> forward()
  virtual void backward_pass(
      const DeviceMem<bf16>& input, DeviceMem<bf16>& grads, float* out_inter,
      float* delta_temp, DeviceMem<bf16> loss, float* A, float* B, float* C,
      float* A_backward_last_layer, float* B_backward_last_layer,
      float* C_backward_last_layer, float* D_backward_last_layer,
      float* E_backward_last_layer, float* F_backward_last_layer,
      float* A_dgemm, float* B_dgemm, float* C_dgemm,
      float* forward) override{};

  std::unique_ptr<Context> forward_impl(
      dpct::queue_ptr stream, const GPUMatrixDynamic<float>& input,
      GPUMatrixDynamic<T>* output = nullptr, bool use_inference_params = false,
      bool prepare_input_gradients = false) override {
    if (!output || padded_output_width() == 0) {
      return std::make_unique<Context>();
    }
    // assert(input.n() == m_n_dims_to_encode);
    uint32_t n_elements = input.n() * padded_output_width();
    if (n_elements <= 0) {
      return std::make_unique<Context>();
    }

    // std::cout << "padded_output_width: " << padded_output_width()
    //           << ", input n: " << input.n() << ", input m: " << input.m()
    //           << ", stride: " << input.stride() << ", n_elements " <<
    //           n_elements
    //           << std::endl;
    {
      // Wrap our data variable in a buffer
      buffer<float, 1> inputBuf{input.data(), range<1>{n_elements}};
      buffer<T, 1> outputBuf{output->data(), range<1>{n_elements}};

      auto loc_m_n_dims_to_encode = m_n_dims_to_encode;
      auto loc_m_n_to_pad = m_n_to_pad;
      auto loc_m_scale = m_scale;
      auto loc_m_offset = m_offset;

      // manually, because we dont have MatrixView on device
      auto loc_m_stride = padded_output_width();
      auto unpadded_stride = input.stride();
      //   auto loc_m_stride = input.stride();
      // TODO: Check with NVCC, we cant forward MatrixView as is
      // MatrixView<T> view() const {
      // return {data(), layout() == CM ? 1u : stride(), layout() == CM ?
      // stride() : 1u};
      // }
      //   std::cout << "loc_m_stride: " << loc_m_stride
      //             << ", loc_m_n_to_pad: " << loc_m_n_to_pad
      //             << ", loc_m_n_dims_to_encode: " << loc_m_n_dims_to_encode
      //             << std::endl;

      // Create a command group to issue commands to the queue
      stream->submit([&](handler& cgh) {
        accessor input_acc{inputBuf, cgh, read_only};
        accessor output_acc{outputBuf, cgh, write_only};

        // Enqueue a parallel_for task with 1024 work-items
        cgh.parallel_for(n_elements, [=](id<1> index) {
          // sycl::nd_range{n_blocks_linear(n_elements) *
          // 	sycl::range<3>(1, 1, N_THREADS_LINEAR),
          // 	sycl::range<3>(1, 1, N_THREADS_LINEAR)},
          // [=](sycl::nd_item<3> item_ct1) {
          const uint32_t encoded_index = index;
          // const uint32_t encoded_index =
          //     item_ct1.get_local_id(2) +
          //     item_ct1.get_group(2) *
          //     item_ct1.get_local_range(2);
          if (encoded_index >= n_elements) {
            // exit(0);
            return;
          }

          // total padded amount
          const uint32_t fan_out = loc_m_n_dims_to_encode + loc_m_n_to_pad;

          // columns which are batch size
          const uint32_t i = encoded_index / fan_out;

          // rows which are the padded output dim
          const uint32_t j = encoded_index - i * fan_out;

          // MatrixView(i,j) => i * stride_i + j * stride_j
          const uint32_t idx = i * loc_m_stride + j;
          const uint32_t unpadded_idx = i * unpadded_stride + j;
          //   const uint32_t idx = j * loc_m_stride + i;

          //   static const CONSTANT char FMT[] =
          //       "Enc idx: %d, i: %d, j: %d, idx: %d\n";
          //   sycl::ext::oneapi::experimental::printf(FMT, encoded_index, i, j,
          //                                           idx);
          if (j >= loc_m_n_dims_to_encode) {
            output_acc[idx] = 1;
            // output_acc[idx] = 0;
          } else {
            output_acc[idx] =
                input_acc[unpadded_idx] * loc_m_scale + loc_m_offset;
          }
        });  // End of the kernel function
      });    // End of our commands for this queue
    }  // End of scope, so we wait for work producing resultBuf to complete

    return std::make_unique<Context>();
  }

  void backward_impl(
      dpct::queue_ptr stream, const Context& ctx,
      const GPUMatrixDynamic<float>& input, const GPUMatrixDynamic<T>& output,
      const GPUMatrixDynamic<T>& dL_doutput,
      GPUMatrixDynamic<float>* dL_dinput = nullptr,
      bool use_inference_params = false,
      GradientMode param_gradients_mode = GradientMode::Overwrite) override {
    if (!dL_dinput) {
      return;
    }

    uint32_t n_elements = input.n() * m_n_dims_to_encode;
    if (n_elements <= 0) {
      return;
    }

    {
      // Wrap our data variable in a buffer
      buffer<float, 1> buf_dL_dx{dL_dinput->data(), range<1>{n_elements}};
      buffer<T, 1> buf_dL_dy{dL_doutput.data(), range<1>{n_elements}};

      auto loc_m_n_dims_to_encode = m_n_dims_to_encode;
      auto loc_m_n_to_pad = m_n_to_pad;
      auto loc_m_scale = m_scale;
      auto loc_m_offset = m_offset;

      auto loc_m_stride =
          input
              .stride();  // manually, because we dont have MatrixView on device
      // TODO: Check with NVCC, we cant forward MatrixView as is
      // MatrixView<T> view() const {
      // return {data(), layout() == CM ? 1u : stride(), layout() == CM ?
      // stride() : 1u};
      // }

      // Create a command group to issue commands to the queue
      stream->submit([&](handler& cgh) {
        // Request write access to the buffer without initialization
        accessor dL_dx{buf_dL_dx, cgh, write_only};
        accessor dL_dy{buf_dL_dy, cgh, read_only};

        // Enqueue a parallel_for task with 1024 work-items
        cgh.parallel_for(  // n_elements, [=](nd_item<1> item_ct1) {
            sycl::nd_range{n_blocks_linear(n_elements) *
                               sycl::range<3>(1, 1, N_THREADS_LINEAR),
                           sycl::range<3>(1, 1, N_THREADS_LINEAR)},
            [=](sycl::nd_item<3> item_ct1) {
              const uint32_t output_index =
                  item_ct1.get_local_id(2) +
                  item_ct1.get_group(2) * item_ct1.get_local_range(2);
              if (output_index >= n_elements) return;

              const uint32_t i = output_index / loc_m_n_dims_to_encode;
              const uint32_t j = output_index - i * loc_m_n_dims_to_encode;

              // MatrixView(i,j) => i * stride_i + j * stride_j
              const uint32_t idx = j * loc_m_stride + i * loc_m_stride;

              // The identity encoding can simply pass through the
              // derivative.
              dL_dx[idx] = (T)((float)dL_dy[idx] * loc_m_scale);
            });  // End of the kernel function
      });        // End of our commands for this queue
    }  // End of scope, so we wait for work producing resultBuf to complete
  }

  uint32_t input_width() const override { return m_n_dims_to_encode; }

  uint32_t padded_output_width() const override {
    return m_n_output_dims + m_n_to_pad;
  }

  uint32_t output_width() const override { return padded_output_width(); }

  uint32_t required_input_alignment() const override { return 1; }

  void set_padded_output_width(uint32_t padded_output_width) override {
    CHECK_THROW(padded_output_width >= m_n_output_dims);
    m_n_to_pad = padded_output_width - m_n_output_dims;
  }

  uint32_t required_output_alignment() const override { return 1; }

  MatrixLayout preferred_output_layout() const override { return AoS; }

  // json hyperparams() const override {
  // 	return {
  // 		{"otype", "Identity"},
  // 		{"scale", m_scale},
  // 		{"offset", m_offset},
  // 	};
  // }

 private:
  uint32_t m_n_dims_to_encode;

  float m_scale;
  float m_offset;

  // derived sizes
  uint32_t m_n_output_dims;
  uint32_t m_n_to_pad = 0;
};