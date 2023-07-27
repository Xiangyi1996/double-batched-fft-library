#pragma once

#include "DeviceMem.h"

using bf16 = sycl::ext::oneapi::bfloat16;

class Network {
public:

    virtual void forward_pass(const DeviceMem<bf16>& input, float* forward, bf16* act_mem, float* act_mem_temp, float* A, float* B, float* C, DeviceMem<float>& output) = 0;

    virtual void backward_pass(
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
    ) = 0;

    virtual void initialize_params() = 0;

    queue get_queue() {
        return m_q;
    }


    queue m_q;
    DeviceMem<bf16> m_grads_matrices;
    DeviceMem<bf16> m_weights_matrices;
    DeviceMem<bf16> m_weightsT_matrices;


};
