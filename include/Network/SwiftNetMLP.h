#ifndef SWIFTNET_H
#define SWIFTNET_H

#include <CL/sycl.hpp>
#include <iostream>
#include <json/json.hpp>
#include <vector>

#include "DeviceMem.h"
#include "Network.h"
#include "activation.h"
#include "common.h"
#include "mkl.h"
#include "mkl_omp_offload.h"
#include "oneapi/mkl.hpp"
#include "sgd.h"
#include "trainer.h"
#ifdef __SYCL_DEVICE_ONLY__

#define CONSTANT __attribute__((opencl_constant))

#else

#define CONSTANT

#endif

/**
 * Multiplies matrices using DGEMM for gradient calculation in the SwiftNet
 * model.
 *
 * @param q                 SYCL queue for command submission.
 * @param grads_device      Pointer to device memory for gradients.
 * @param loss_gradients    Pointer to loss gradients for backpropagation.
 * @param fwd               Pointer to forward pass intermediate outputs.
 * @param A                 Pointer to matrix A (calculated activations).
 * @param B                 Pointer to matrix B (loss gradients).
 * @param C                 Pointer to matrix C (result of DGEMM).
 * @param k                 Index of the hidden matrix multiplication.
 * @param m_n_hidden_matrices Number of hidden matrix multiplications.
 * @param batch_size        Batch size of the data.
 * @tparam WIDTH            Width of the matrices.
 * @tparam ACTIVATION       Type of activation for hidden layers.
 */
template <int WIDTH, Activation ACTIVATION>
void dgemm_multiply(queue q, bf16* grads_device, float* loss_gradients,
                    float* fwd, float* A, float* B, float* C, int k,
                    int m_n_hidden_matrices, int batch_size, int m_inputs_width,
                    int& flops) {
  const int n_hidden_matrices = m_n_hidden_matrices;
  int layer_in_width;
  int offset_f1;
  int offset_g;
  int offset_c;
  if (k == (n_hidden_matrices - 1)) {
    // this is the 1st layer (input to 1st layer)
    // need this as input_width != net_width
    layer_in_width = m_inputs_width;
    offset_f1 = 0;
    offset_g = 0;
    offset_c = 0;
  } else {
    //  any layer between input and output (input to 1st layer and penultimate
    //  to last layer are handled separately)
    layer_in_width = WIDTH;
    offset_f1 = (n_hidden_matrices - k - 1) * WIDTH * batch_size;
    offset_g =
        (m_inputs_width + (n_hidden_matrices - k - 2) * WIDTH) * batch_size;
    offset_c =
        (m_inputs_width * WIDTH + (n_hidden_matrices - k - 2) * WIDTH * WIDTH);
  }
  // Calculate matrix A using the given activation function
  q.parallel_for<>(range<1>(layer_in_width * batch_size), [=](id<1> idx) {
     int i = idx / batch_size;
     int j = idx % batch_size;
     A[i * batch_size + j] = (float)elt_activation_ret<float>(
         ACTIVATION, fwd[i + j * layer_in_width + offset_g]);
     // int b_first;
     // int b_second;
     // int b_zeroes;
     // static const CONSTANT char FMT[] =
     //     "K: %d, offset_g: %d, A[%d] from %d: %d.%d\n";
     // get_float_as_integers_own(A[idx], b_first, b_second, b_zeroes);
     // if (A[i * batch_size + j] == 0) {
     //   sycl::ext::oneapi::experimental::printf(
     //       FMT, k, int(offset_g), int(idx),
     //       int(i + j * layer_in_width + offset_g), b_first, b_second);
     // }
   }).wait();

  // Assign matrix B using loss gradients
  q.parallel_for<>(range<1>(WIDTH * batch_size), [=](id<1> idx) {
     B[idx] = (float)loss_gradients[idx + offset_f1];
     // int b_first;
     // int b_second;
     // int b_zeroes;
     // static const CONSTANT char FMT[] = "K: %d, B[%d]: %d.%d \n";
     // get_float_as_integers_own(loss_gradients[idx + offset_f1], b_first,
     //                           b_second, b_zeroes);
     // if (B[idx] == 0) {
     //   sycl::ext::oneapi::experimental::printf(FMT, k, int(idx + offset_f1),
     //                                           b_first, b_second);
     // }
   }).wait();

  // Perform GEMM operation
  oneapi::mkl::blas::row_major::gemm(q, oneapi::mkl::transpose::nontrans,
                                     oneapi::mkl::transpose::nontrans,
                                     layer_in_width, WIDTH, batch_size, 1, A,
                                     batch_size, B, WIDTH, 0, C, WIDTH);

  // Update gradients_device with the computed values
  q.parallel_for<>(range<1>(layer_in_width * WIDTH), [=](id<1> idx) {
     grads_device[offset_c + idx] += C[idx];
     // int b_first;
     // int b_second;
     // int b_zeroes;
     // static const CONSTANT char FMT[] =
     //     "K: %d, offset_c: %d, C last[%d]: %d.%d\n";
     // get_float_as_integers_own(grads_device[offset_c + idx], b_first,
     // b_second,
     //                           b_zeroes);
     // if (C[idx] == 0) {
     //   sycl::ext::oneapi::experimental::printf(
     //       FMT, k, int(offset_c), int(offset_c + idx), b_first, b_second);
     // }
   }).wait();

  // Calculating flops
  // A Matrix
  flops =
      (layer_in_width * batch_size) + 2 * layer_in_width * WIDTH * batch_size;
  //(2 FLOPs per multiplication and addition).
}

void get_float_as_integers_own(float value, int& integer_val,
                               int& fractional_val);
using bf16 = sycl::ext::oneapi::bfloat16;

template <int WIDTH>
class SwiftNetMLP : public Network {
 public:
  SwiftNetMLP(queue q, int input_width, int output_width, int n_hidden_layers,
              Activation activation, Activation output_activation,
              int batch_size);
  ~SwiftNetMLP();
  void forward_pass(const DeviceMem<bf16>& input, float* forward, float* A,
                    float* B, float* C, DeviceMem<float>& output) override;

  void inference(const DeviceMem<bf16>& input, float* forward, float* A,
                 float* B, float* C, DeviceMem<float>& output) override;

  void backward_pass(const DeviceMem<bf16>& input, DeviceMem<bf16>& grads,
                     float* out_inter, DeviceMem<bf16> loss, float* A, float* B,
                     float* C, float* A_backward_last_layer,
                     float* B_backward_last_layer, float* C_backward_last_layer,
                     float* D_backward_last_layer, float* E_backward_last_layer,
                     float* F_backward_last_layer, float* A_dgemm,
                     float* B_dgemm, float* C_dgemm, float* forward) override;

  void dgemm_last_layer_backward(DeviceMem<bf16>& grads, float* forward,
                                 DeviceMem<bf16>& loss, int batch_size,
                                 float* A, float* B, float* C, float* D,
                                 float* E, float* F);
  void set_params(std::vector<bf16> params) override;
  void set_params(float* params) override;

  void save_to_file(std::string filename);
  void load_from_file(std::string filename);
  void initialize_params(int use_easy = 0) override;
  void free_mem(queue q) override;

  DeviceMem<bf16>* get_grads_matrices() override;

  DeviceMem<bf16>* get_weights_matrices() override;

  DeviceMem<bf16>* get_weightsT_matrices() override;
  std::vector<bf16> get_weights_matrices_as_vector() override;
  std::vector<bf16> get_weightsT_matrices_as_vector() override;

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
};

#endif
