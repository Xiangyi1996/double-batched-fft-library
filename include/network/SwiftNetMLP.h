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
#include "trainer.h"
#ifdef __SYCL_DEVICE_ONLY__

#define CONSTANT __attribute__((opencl_constant))

#else

#define CONSTANT

#endif

void get_float_as_integers_own(float value, int &integer_val, int &fractional_val);
using bf16 = sycl::ext::oneapi::bfloat16;

template <int WIDTH> class SwiftNetMLP : public Network {
  public:
    SwiftNetMLP(queue q, int input_width, int output_width, int n_hidden_layers, Activation activation,
                Activation output_activation);
    ~SwiftNetMLP();
    std::vector<sycl::event> forward_pass(const DeviceMem<bf16> &input, float *forward, const size_t batch_size,
                                          const std::vector<sycl::event> &deps) override;

    std::vector<sycl::event> inference(const DeviceMem<bf16> &input, float *forward, const size_t batch_size,
                                       const std::vector<sycl::event> &deps) override;

    std::vector<sycl::event> backward_pass(const DeviceMem<bf16> &grads, float *const out_inter,
                                           float const *const forward, const size_t batch_size,
                                           const std::vector<sycl::event> &deps) override;

    std::vector<sycl::event> training(const DeviceMem<bf16> &input, const DeviceMem<bf16> &target,
                                      float *const intermediate_output_forward,
                                      float *const intermediate_output_backward, const size_t batch_size,
                                      const std::vector<sycl::event> &deps) override;

    bf16 const *const GetOutput(float const *const forward, const size_t batch_size) const override {
        return reinterpret_cast<bf16 const *const>(forward) + m_inputs_width_padded * batch_size +
               WIDTH * batch_size * m_n_hidden_layers;
    }

    void set_params(std::vector<bf16> &params) override;
    void set_params(float *params) override;

    void save_to_file(std::string filename);
    void load_from_file(std::string filename);
    void initialize_params(int use_easy = 0) override;
    void free_mem(queue q) override;

    DeviceMem<bf16> *get_grads_matrices() override;

    DeviceMem<bf16> *get_weights_matrices() override;

    DeviceMem<bf16> *get_weightsT_matrices() override;
    std::vector<bf16> get_weights_matrices_as_vector() override;
    std::vector<bf16> get_weightsT_matrices_as_vector() override;

    int get_n_hidden_layers() const override { return m_n_hidden_layers; }

    int get_n_hidden_matrices() const override { return m_n_hidden_matrices; }

    int get_inputs_width() const override { return m_inputs_width; }

    int get_net_width() const override { return m_net_width; }

    int get_output_width() const override { return m_output_width; }

    int get_padded_output_width() const override { return m_output_width_padded; }

    void zero_pad_weight_matrix();

  private:
    Activation m_activation;
    Activation m_output_activation;
    int m_n_hidden_layers;
    int m_n_hidden_matrices;
    int m_inputs_width;
    int m_net_width;
    int m_output_width;
    int m_inputs_width_padded;
    int m_output_width_padded;

    DeviceMem<bf16> m_weights_matrices_inferences;

    int m_total_n_params;
    void check_parameters();
};

#endif
