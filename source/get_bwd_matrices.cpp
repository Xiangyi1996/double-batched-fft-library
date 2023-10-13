/*#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

#include "L2.h"
#include "SwiftNetMLP.h"
#include "activation.h"
#include "common.h"
#include "mkl.h"
#include "mkl_omp_offload.h"
#include "oneapi/mkl.hpp"
#include "sgd.h"
#include "trainer.h"

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
using bf16 = sycl::ext::oneapi::bfloat16;

#define INPUT_WIDTH 64
#define OUTPUT_WIDTH 64
#define HIDDEN_LAYERS 4
template <typename T>
void saveVectorToCSV(const std::vector<T>& data, const std::string& filename) {
  std::ofstream file(filename);

  if (!file.is_open()) {
    std::cerr << "Failed to open the file for writing: " << filename
              << std::endl;
    return;
  }

  for (const T& value : data) {
    file << value << ',';
  }

  // Remove the trailing comma
  if (!data.empty()) {
    file.seekp(-1, std::ios_base::end);
  }

  file << std::endl;
  file.close();
}

void get_matrices() {
  // SWIFTNET
  const int batch_size = 128;
  const int output_width = OUTPUT_WIDTH;
  const int WIDTH = 64;
  const int net_width = 64;

  const float scale = 1e-3f;

  queue q = queue();

  DeviceMem<bf16> inputs = DeviceMem<bf16>(INPUT_WIDTH * batch_size, q);
  DeviceMem<float> output = DeviceMem<float>(batch_size * output_width, q);
  DeviceMem<float> target = DeviceMem<float>(batch_size * output_width, q);
  DeviceMem<bf16> grads = DeviceMem<bf16>(batch_size * output_width, q);
  DeviceMem<float> losses = DeviceMem<float>(batch_size * output_width, q);

  SwiftNetMLP<64> network =
      SwiftNetMLP<64>(q, INPUT_WIDTH, output_width, HIDDEN_LAYERS,
                      Activation::ReLU, Activation::None, batch_size);
  //   Activation::None, Activation::None, batch_size);
  network.initialize_params(1);

  inputs.initialize_constant(0.1f, q);
  output.initialize_constant(0.0f, q);
  target.initialize_constant(1.0f, q);
  grads.initialize_arange(q);
  losses.initialize_constant(0.0f, q);

  auto p = network.m_forward;
  network.get_queue()
      .parallel_for<>(range<1>(inputs.size()),
                      [=](id<1> idx) { p[idx] = inputs.data()[idx]; })
      .wait();

  network.forward_pass(inputs, network.m_forward, network.m_A_forward,
                       network.m_B_forward, network.m_C_forward, output);

  network.backward_pass(
      inputs, grads, network.m_out_inter, network.m_deltas,
      network.m_A_backward, network.m_B_backward, network.m_C_backward,
      network.m_A_backward_last_layer, network.m_B_backward_last_layer,
      network.m_C_backward_last_layer, network.m_D_backward_last_layer,
      network.m_E_backward_last_layer, network.m_F_backward_last_layer,
      network.m_A_dgemm, network.m_B_dgemm, network.m_C_dgemm,
      network.m_forward);

  // Copying
  // Calculate sizes
  size_t m_forward_size =
      batch_size * (INPUT_WIDTH + OUTPUT_WIDTH + WIDTH * HIDDEN_LAYERS);
  size_t inputs_size = INPUT_WIDTH * batch_size;
  size_t grads_size = batch_size * output_width;
  size_t out_inter_size = batch_size * WIDTH * (HIDDEN_LAYERS - 1);
  size_t deltas_size = WIDTH * batch_size;
  size_t A_backward_size = WIDTH * batch_size;
  size_t B_backward_size = batch_size * OUTPUT_WIDTH;
  size_t C_backward_size = WIDTH * OUTPUT_WIDTH;
  size_t A_backward_last_layer_size = batch_size * OUTPUT_WIDTH;
  size_t B_backward_last_layer_size = OUTPUT_WIDTH * WIDTH;
  size_t C_backward_last_layer_size = WIDTH * batch_size;
  size_t D_backward_last_layer_size = WIDTH * batch_size;
  size_t E_backward_last_layer_size = batch_size * WIDTH;
  size_t F_backward_last_layer_size = WIDTH * WIDTH;

  // Allocate host memory
  std::vector<float> m_forward_vec(m_forward_size);
  std::vector<bf16> inputs_vec(inputs_size);
  std::vector<bf16> grads_vec(grads_size);
  std::vector<bf16> deltas_vec(deltas_size);
  std::vector<float> out_inter_vec(out_inter_size);
  std::vector<float> A_backward_vec(A_backward_size);
  std::vector<float> B_backward_vec(B_backward_size);
  std::vector<float> C_backward_vec(C_backward_size);
  std::vector<float> A_backward_last_layer_vec(A_backward_last_layer_size);
  std::vector<float> B_backward_last_layer_vec(B_backward_last_layer_size);
  std::vector<float> C_backward_last_layer_vec(C_backward_last_layer_size);
  std::vector<float> D_backward_last_layer_vec(D_backward_last_layer_size);
  std::vector<float> E_backward_last_layer_vec(E_backward_last_layer_size);
  std::vector<float> F_backward_last_layer_vec(F_backward_last_layer_size);

  // Copy data from device to host
  q.memcpy(m_forward_vec.data(), network.m_forward,
           m_forward_size * sizeof(float))
      .wait();
  inputs.copy_to_host(inputs_vec, q);
  grads.copy_to_host(grads_vec, q);
  network.m_deltas.copy_to_host(deltas_vec, q);

  q.memcpy(out_inter_vec.data(), network.m_out_inter,
           out_inter_size * sizeof(float))
      .wait();

  q.memcpy(A_backward_vec.data(), network.m_A_backward,
           A_backward_size * sizeof(float))
      .wait();
  q.memcpy(B_backward_vec.data(), network.m_B_backward,
           B_backward_size * sizeof(float))
      .wait();
  q.memcpy(C_backward_vec.data(), network.m_C_backward,
           C_backward_size * sizeof(float))
      .wait();
  q.memcpy(A_backward_last_layer_vec.data(), network.m_A_backward_last_layer,
           A_backward_last_layer_size * sizeof(float))
      .wait();
  q.memcpy(B_backward_last_layer_vec.data(), network.m_B_backward_last_layer,
           B_backward_last_layer_size * sizeof(float))
      .wait();
  q.memcpy(C_backward_last_layer_vec.data(), network.m_C_backward_last_layer,
           C_backward_last_layer_size * sizeof(float))
      .wait();
  q.memcpy(D_backward_last_layer_vec.data(), network.m_D_backward_last_layer,
           D_backward_last_layer_size * sizeof(float))
      .wait();
  q.memcpy(E_backward_last_layer_vec.data(), network.m_E_backward_last_layer,
           E_backward_last_layer_size * sizeof(float))
      .wait();
  q.memcpy(F_backward_last_layer_vec.data(), network.m_F_backward_last_layer,
           F_backward_last_layer_size * sizeof(float))
      .wait();

  saveVectorToCSV(m_forward_vec, "bwd_matrices/m_forward.csv");
  saveVectorToCSV(inputs_vec, "bwd_matrices/inputs.csv");
  saveVectorToCSV(grads_vec, "bwd_matrices/grads.csv");
  saveVectorToCSV(out_inter_vec, "bwd_matrices/out_inter.csv");
  saveVectorToCSV(deltas_vec, "bwd_matrices/deltas.csv");
  saveVectorToCSV(A_backward_vec, "bwd_matrices/A_backward.csv");
  saveVectorToCSV(B_backward_vec, "bwd_matrices/B_backward.csv");
  saveVectorToCSV(C_backward_vec, "bwd_matrices/C_backward.csv");
  saveVectorToCSV(A_backward_last_layer_vec,
                  "bwd_matrices/A_backward_last_layer.csv");
  saveVectorToCSV(B_backward_last_layer_vec,
                  "bwd_matrices/B_backward_last_layer.csv");
  saveVectorToCSV(C_backward_last_layer_vec,
                  "bwd_matrices/C_backward_last_layer.csv");
  saveVectorToCSV(D_backward_last_layer_vec,
                  "bwd_matrices/D_backward_last_layer.csv");
  saveVectorToCSV(E_backward_last_layer_vec,
                  "bwd_matrices/E_backward_last_layer.csv");
  saveVectorToCSV(F_backward_last_layer_vec,
                  "bwd_matrices/F_backward_last_layer.csv");

  inputs.free_mem(q);
  output.free_mem(q);
  target.free_mem(q);
  grads.free_mem(q);
  losses.free_mem(q);
  network.free_mem(q);
}
int main() {
  get_matrices();
  return 0;
}
*/