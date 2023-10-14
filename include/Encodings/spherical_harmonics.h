#pragma once

// #include <DeviceMem.h>
// #include <common.h>
// #include <common_device.h>
// #include <encoding.h>
// #include <stdint.h>

// #include <dpct/dpct.hpp>
// #include <numeric>
// #include <stdexcept>
// #include <string>
// #include <sycl/sycl.hpp>
// #include <vector>

// template <typename T>
// void kernel_sh(const uint32_t num_elements, const uint32_t degree,
//                const uint32_t num_to_pad, MatrixView<const float> data_in,
//                MatrixView<T> data_out, id<1> encoded_idx) {
// void kernel_sh(const uint32_t num_elements, const uint32_t degree,
//                const uint32_t num_to_pad) {
//   if (encoded_idx >= num_elements) return;

//   data_out.advance_cols(encoded_idx);

// #pragma unroll
//   for (uint32_t j = 0; j < num_to_pad; ++j) {
//     data_out(j) = (T)1.0f;
//   }

//   data_out.advance_rows(num_to_pad);

//   sh_enc<T, MatrixView<T>>(degree, data_in(0, encoded_idx) * 2.f - 1.f,
//                            data_in(1, encoded_idx) * 2.f - 1.f,
//                            data_in(2, encoded_idx) * 2.f - 1.f, data_out);
// }

// template <typename T>
// void kernel_sh_backward(const uint32_t num_elements, const uint32_t degree,
//                         const uint32_t num_to_pad, MatrixView<const T> dL_dy,
//                         MatrixView<const float> data_in,
//                         MatrixView<float> dL_dx,
//                         const sycl::nd_item<3> &item_ct1) {
//   const uint32_t i = item_ct1.get_local_id(2) +
//                      item_ct1.get_group(2) * item_ct1.get_local_range(2);
//   if (i >= num_elements) return;

//   dL_dy.advance(num_to_pad, i);

//   tnn::vec3 d = sh_enc_grad<T, MatrixView<const T>>(
//       degree, data_in(0, i) * 2.f - 1.f, data_in(1, i) * 2.f - 1.f,
//       data_in(2, i) * 2.f - 1.f, dL_dy);

//   // Multiplication by 2 due to the conversion
//   // from [0,1]^3 to directions in [-1,1]^3.
//   // See implementation in `kernel_sh`.
//   dL_dx.set_col(i, 2.0f * d);
// }

template <typename T>
class SphericalHarmonicsEncoding : public Encoding<T> {
 public:
  SphericalHarmonicsEncoding(uint32_t degree, uint32_t n_dims_to_encode)
      : m_degree{degree}, m_n_dims_to_encode{n_dims_to_encode} {
    m_n_output_dims = degree * degree;

    if (n_dims_to_encode != 3) {
      throw std::runtime_error{
          "Can only encode 3D directions in spherical harmonics."};
    }

    if (m_degree <= 0) {
      throw std::runtime_error{
          "Spherical harmonics must have positive degree."};
    }

    if (m_degree > 8) {
      throw std::runtime_error{
          "Spherical harmonics are only implemented up to degree 8."};
    }
  }

  std::unique_ptr<Context> forward_impl(
      dpct::queue_ptr stream, const GPUMatrixDynamic<float> &input,
      GPUMatrixDynamic<T> *output = nullptr, bool use_inference_params = false,
      bool prepare_input_gradients = false) override {
    uint32_t n_elements = input.n() * m_n_dims_to_encode;

    if (!output || padded_output_width() == 0) {
      return std::make_unique<Context>();
    }

    {
      // Wrap our data variable in a buffer
      buffer<float, 1> inputBuf{input.data(), range<1>{n_elements}};
      buffer<T, 1> outputBuf{output->data(), range<1>{n_elements}};
      auto loc_m_stride = input.stride();
      auto local_m_degree = m_degree;
      auto local_m_n_to_pad = m_n_to_pad;
      // Create a command group to issue commands to the queue
      stream->submit([&](handler &cgh) {
        accessor input_acc{inputBuf, cgh, read_only};
        accessor output_acc{outputBuf, cgh, write_only};

        // Enqueue a parallel_for task with 1024 work-items
        cgh.parallel_for(n_elements, [=](id<1> encoded_idx) {
        // kernel_sh(n_elements, local_m_degree, local_m_n_to_pad,
        // input.view(),
        //           output->view(), index);

        //   if (encoded_idx >= n_elements) {
        //     exit(0);
        //   };

        // output.advance_cols(encoded_idx);

#pragma unroll
          for (uint32_t j = 0; j < local_m_n_to_pad; ++j) {
            output_acc[encoded_idx * loc_m_stride + j] = (T)1.0f;
          }

          // data_out.advance_rows(m_n_to_pad);
          sh_enc<T>(local_m_degree,
                    input_acc[0 * loc_m_stride + encoded_idx] * 2.f - 1.f,
                    input_acc[1 * loc_m_stride + encoded_idx] * 2.f - 1.f,
                    input_acc[2 * loc_m_stride + encoded_idx] * 2.f - 1.f,
                    output_acc, local_m_n_to_pad);

          // kernel_sh(n_elements, local_m_degree, local_m_n_to_pad);
        });  // End of the kernel function
      });    // End of our commands for this queue
    }  // End of scope, so we wait for work producing resultBuf to complete

    return std::make_unique<Context>();
  }

  void backward_impl(
      dpct::queue_ptr stream, const Context &ctx,
      const GPUMatrixDynamic<float> &input, const GPUMatrixDynamic<T> &output,
      const GPUMatrixDynamic<T> &dL_doutput,
      GPUMatrixDynamic<float> *dL_dinput = nullptr,
      bool use_inference_params = false,
      GradientMode param_gradients_mode = GradientMode::Overwrite) override {
    //   if (!dL_dinput) {
    //     return;
    //   }

    //   linear_kernel(kernel_sh_backward<T>, 0, stream, input.n(), m_degree,
    //                 m_n_to_pad, dL_doutput.view(), input.view(),
    //                 dL_dinput->view());
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

  MatrixLayout preferred_output_layout() const override { return SoA; }

 private:
  uint32_t m_degree;
  uint32_t m_n_dims_to_encode;

  // derived sizes
  uint32_t m_n_output_dims;
  uint32_t m_n_to_pad = 0;
};
