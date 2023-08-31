#ifndef SWIFTNET_H
#define SWIFTNET_H

#include <iostream>
#include <vector>
#include <CL/sycl.hpp>
#include "activation.h"
#include "Network.h"
#include "DeviceMem.h"

#include "sgd.h"
#include "trainer.h"
#include "mkl.h"
#include "common.h"
#include "oneapi/mkl.hpp"

using bf16 = sycl::ext::oneapi::bfloat16;

template <int WIDTH>
class SwiftNetMLP : public Network {
public:
    SwiftNetMLP(queue q, int input_width, int output_width, int n_hidden_layers, Activation activation, Activation output_activation, int batch_size);
    ~SwiftNetMLP();
    void forward_pass(const DeviceMem<bf16>& input, float* forward, float* A, float* B, float* C, DeviceMem<float>& output) override;

    void inference(const DeviceMem<bf16>& input, float* forward, float* A, float* B, float* C, DeviceMem<float>& output) override;

    void backward_pass(
        const DeviceMem<bf16>& input,
        DeviceMem<bf16>& grads,
        float* out_inter,
        float* delta_temp, 
        DeviceMem<bf16> loss,
        float* A,
        float* B,
        float* C,
        float* A_backward_last_layer,
        float* B_backward_last_layer,
        float* C_backward_last_layer,
        float* D_backward_last_layer,
        float* E_backward_last_layer,
        float* F_backward_last_layer,
        float* A_dgemm,
        float* B_dgemm,
        float* C_dgemm,
        float* forward
    ) override;

    void dgemm_last_layer_backward(DeviceMem<bf16>& grads,
        float* forward,
        DeviceMem<bf16>& loss,
        int batch_size,
        float* A,
        float* B,
        float* C,
        float* D,
        float* E,
        float* F);
    //void set_params(float* params, float* inference_params, float* gradients);
    void save_to_file(std::string filename);
    void load_from_file(std::string filename);
    void initialize_params()  override;
    void free_mem(queue q) override;




    DeviceMem<bf16>* get_grads_matrices();

    DeviceMem<bf16>* get_weights_matrices();

    DeviceMem<bf16>* get_weightsT_matrices();


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
