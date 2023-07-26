#pragma once

#include "DeviceMem.h"

using bf16 = sycl::ext::oneapi::bfloat16;

class Network {
public:

    virtual void forward_pass(const DeviceMem<bf16>& input, float* forward, DeviceMem<float>& output) = 0;

    virtual void backward_pass(
        const DeviceMem<bf16>& input, DeviceMem<bf16>& grads, float* forward
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
