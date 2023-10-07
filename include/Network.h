#pragma once

#include "DeviceMem.h"

using bf16 = sycl::ext::oneapi::bfloat16;

// Base class for neural network
class Network {
 public:
  // Perform forward pass through the network
  virtual void forward_pass(const DeviceMem<bf16>& input, float* forward,
                            float* A, float* B, float* C,
                            DeviceMem<float>& output) = 0;

  // Perform inference through the network
  virtual void inference(const DeviceMem<bf16>& input, float* forward, float* A,
                         float* B, float* C, DeviceMem<float>& output) = 0;

  // Perform backward pass through the network
  virtual void backward_pass(
      const DeviceMem<bf16>& input, DeviceMem<bf16>& grads, float* out_inter,
      DeviceMem<bf16> loss, float* A, float* B, float* C,
      float* A_backward_last_layer, float* B_backward_last_layer,
      float* C_backward_last_layer, float* D_backward_last_layer,
      float* E_backward_last_layer, float* F_backward_last_layer,
      float* A_dgemm, float* B_dgemm, float* C_dgemm, float* forward) = 0;

  // Initialize network parameters
  virtual void initialize_params(int use_constant) = 0;
  // Free memory allocated by the network
  virtual void free_mem(queue q) = 0;

  // Get the SYCL queue associated with the network
  queue get_queue() { return m_q; }
  virtual void set_params(std::vector<bf16>& params) = 0;
  virtual void set_params(float* params) = 0;

  virtual DeviceMem<bf16>* get_grads_matrices() = 0;
  virtual DeviceMem<bf16>* get_weights_matrices() = 0;
  virtual DeviceMem<bf16>* get_weightsT_matrices() = 0;

  virtual std::vector<bf16> get_weights_matrices_as_vector() = 0;
  virtual std::vector<bf16> get_weightsT_matrices_as_vector() = 0;
  // Data members
  float* m_forward;
  int m_shmem_size;
  size_t m_alignment;

  float* m_A_forward;
  float* m_B_forward;
  float* m_C_forward;

  float* m_out_inter;
  //   float* m_deltas_temp;
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
};
