#pragma once

#include "DeviceMem.h"

using bf16 = sycl::ext::oneapi::bfloat16;

// Base class for neural network
class Network {
  public:
    // Perform forward pass through the network
    virtual std::vector<sycl::event> forward_pass(const DeviceMem<bf16> &input, float *forward, const size_t batch_size,
                                                  const std::vector<sycl::event> &deps) = 0;

    // Perform inference through the network
    virtual std::vector<sycl::event> inference(const DeviceMem<bf16> &input, float *forward, const size_t batch_size,
                                               const std::vector<sycl::event> &deps) = 0;

    // Perform backward pass through the network
    virtual std::vector<sycl::event> backward_pass(const DeviceMem<bf16> &grads, float *const out_inter,
                                                   float const *const forward, const size_t batch_size,
                                                   const std::vector<sycl::event> &deps) = 0;

    virtual std::vector<sycl::event> training(const DeviceMem<bf16> &input, const DeviceMem<bf16> &target,
                                              float *const intermediate_output_forward,
                                              float *const intermediate_output_backward, const size_t batch_size,
                                              const std::vector<sycl::event> &deps) = 0;

    // Initialize network parameters
    virtual void initialize_params(int use_constant) = 0;
    // Free memory allocated by the network
    virtual void free_mem(queue q) = 0;

    virtual bf16 const *const GetOutput(float const *const forward, const size_t batch_size) const { return nullptr; }

    // Get the SYCL queue associated with the network
    queue get_queue() { return m_q; }
    virtual void set_params(std::vector<bf16> &params) = 0;
    virtual void set_params(float *params) = 0;

    virtual DeviceMem<bf16> *get_grads_matrices() = 0;
    virtual DeviceMem<bf16> *get_weights_matrices() = 0;
    virtual DeviceMem<bf16> *get_weightsT_matrices() = 0;

    virtual std::vector<bf16> get_weights_matrices_as_vector() = 0;
    virtual std::vector<bf16> get_weightsT_matrices_as_vector() = 0;
    // Data members
    int m_shmem_size;

    queue m_q;

    DeviceMem<bf16> m_grads_matrices;
    DeviceMem<bf16> m_weights_matrices;
    DeviceMem<bf16> m_weightsT_matrices;

    virtual int get_n_hidden_layers() const = 0;
    virtual int get_n_hidden_matrices() const = 0;
    virtual int get_inputs_width() const = 0;
    virtual int get_net_width() const = 0;
    virtual int get_output_width() const = 0;
    virtual int get_padded_output_width() const = 0;
};
