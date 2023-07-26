#pragma once

#include "activation.h"
#include"Network.h"
#include "DeviceMem.h"
#include <json/json.hpp>

using bf16 = sycl::ext::oneapi::bfloat16;
using json = nlohmann::json;

template <int WIDTH>
class SwiftNetMLP : public Network {
public:
    SwiftNetMLP(queue q, int input_width, int output_width, int n_hidden_layers, Activation activation, Activation output_activation);

    void forward_pass(const DeviceMem<bf16>& input, float* forward,  DeviceMem<float>& output) override;

    void backward_pass(
        const DeviceMem<bf16>& input, DeviceMem<bf16>& grads, float* forward
    ) override;

    void dgemm_last_layer_backward(DeviceMem<bf16>& grads, float* forward, DeviceMem<bf16>& loss, int batch_size);
    //void set_params(float* params, float* inference_params, float* gradients);
    void save_to_file(std::string filename);
    void load_from_file(std::string filename);
    void initialize_params()  override;

    ~SwiftNetMLP() {
        m_weights_matrices_inferences.free_mem(m_q);
    }


    DeviceMem<bf16>* grads_matrices() {
        return &m_grads_matrices;
    }

    DeviceMem<bf16>* weights_matrices() {
        return &m_weights_matrices;
    } 

    DeviceMem<bf16>* weightsT_matrices() {
        return &m_weightsT_matrices;
    }

private:
    int m_n_hidden_layers;
    int m_n_hidden_matrices;
    int m_inputs_width;
    int m_net_width;
    int m_output_width;
    int m_padded_output_width;

    Activation m_activation;
    Activation m_output_activation;


    DeviceMem<bf16> m_weights_matrices_inferences;

    int m_total_n_params;
};

template<int WIDTH>
SwiftNetMLP<WIDTH>* create_network( queue q, const json& network);
