/**
 * @file tnn_api.cpp
 * @author Kai Yuan
 * @brief Definition of tnn_api functions.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "tnn_api.h"
#include "identity.h"

namespace tnn {
// template <typename T>
// EncodingModule<T>::EncodingModule(int input_width, const json &encoding_config, std::string device_name)
//     : Module<T>(device_name), m_input_width(input_width) {
//     encoding = create_encoding<T>(encoding_config);
//     sycl_queue = sycl::queue();
// }

// template <typename T> void EncodingModule<T>::initialize_params(float *params_full_precision, int use_easy) {
//     encoding->initialize_params();
// }

// template <typename T>
// torch::Tensor EncodingModule<T>::forward_pass(torch::Tensor input_tensor, torch::Tensor params, int use_inference) {
//     //   assert(input_tensor.sizes() == 2 &&p
//     //          "Tensor length for Encoding forward is not equal to 2!");
//     //   std::cout << "Input tensor sizes: " << input_tensor.sizes() << std::endl;
//     // int batch_size = input_tensor.sizes()[1];
//     // DeviceMatrix<float> input_matrix = DeviceMatrix<float>(input_tensor.data_ptr<float>(), m_input_width,
//     // batch_size);

//     // torch::Tensor output_tensor = torch::empty({encoding->output_width(), batch_size},
//     //                                            torch::TensorOptions().dtype(torch::kFloat32).device(m_device));
//     // DeviceMatrix<float> output_matrix = DeviceMatrix<float>(output_tensor.data_ptr<float>(), m_input_width,
//     // batch_size);

//     // std::unique_ptr<Context> model_ctx = encoding->forward_impl(&sycl_queue, input_matrix, &output_matrix);

//     // return output_tensor;
// }

// template <typename T>
// torch::Tensor EncodingModule<T>::backward_pass(torch::Tensor input_tensor, torch::Tensor grad_output,
//                                                torch::Tensor params) {}

// template <typename T> void EncodingModule<T>::free_memory() {}

template <typename T, int WIDTH>
NetworkWithEncodingModule<T, WIDTH>::NetworkWithEncodingModule(int width, int input_width, int output_width,
                                                               int n_hidden_layers, Activation activation,
                                                               Activation output_activation,
                                                               const json &encoding_config, std::string device_name)
    : Module<T>(device_name), m_input_width(input_width), m_output_width(output_width), m_width(width),
      m_n_hidden_layers(n_hidden_layers) {

    create_network_with_encoding<T, WIDTH>(sycl_queue, input_width, output_width, n_hidden_layers, activation,
                                           output_activation, encoding_config);
}

template <typename T, int WIDTH>
torch::Tensor NetworkWithEncodingModule<T, WIDTH>::forward_pass(torch::Tensor input_tensor, torch::Tensor params,
                                                                int use_inference) {
    // set_params(params);
    // int batch_size = input_tensor.sizes()[1];

    // DeviceMatrix<float> input_matrix = DeviceMatrix<float>(input_tensor.data_ptr<float>(), m_input_width,
    // batch_size); DeviceMatrix<T> output_encoding(batch_size, m_width, sycl_queue);

    // if (use_inference) {
    //     network->inference(input_matrix, output_encoding, output_network, {});
    // }

    // torch::Tensor output_tensor = get_converted_tensor_from_dev_mem(output_network);

    // return output_tensor;
}

template <typename T, int WIDTH>
torch::Tensor NetworkWithEncodingModule<T, WIDTH>::backward_pass(torch::Tensor input_tensor, torch::Tensor grad_output,
                                                                 torch::Tensor params) {
    // int batch_size = input_tensor.sizes()[1];

    // DeviceMem<bf16> input_backward = DeviceMem<bf16>(batch_size * m_input_width, sycl_queue);
    // DeviceMem<bf16> grads = DeviceMem<bf16>(batch_size * m_output_width, sycl_queue);

    // convert_tensor_to_dev_mem(grad_output, &grads);
    // convert_tensor_to_dev_mem(input_tensor, &input_backward);

    // set_params(params);

    // DeviceMem<bf16> *grads_matrices = network->backward_pass(input_backward, grads, forward, batch_size);

    // torch::Tensor grad_loss = get_converted_tensor_from_dev_mem(*grads_matrices);

    // return grad_loss;
}

template <typename T, int WIDTH> void NetworkWithEncodingModule<T, WIDTH>::set_params(torch::Tensor &params) {
    if (m_device_name == "cpu") {
        std::vector<bf16> params_bf16 = get_vector_from_tensor<bf16>(params);
        network->set_params(params_bf16);
    } else if (m_device_name == "xpu") {
        float *tensor_data = params.data_ptr<float>();
        network->set_params(tensor_data);
    } else {
        std::cout << "No behaviour for device " << m_device_name << ". Exiting code." << std::endl;
        exit(1);
    }
}

template <typename T, int WIDTH>
void NetworkWithEncodingModule<T, WIDTH>::initialize_params(float *params_full_precision, int use_easy) {
    network->initialize_params(use_easy);
    std::vector<bf16> params_full_precision_list(network->get_network()->get_weights_matrices()->size());
    network->get_network()->get_weights_matrices()->copy_to_host(params_full_precision_list, sycl_queue);

    for (int i = 0; i < params_full_precision_list.size(); i++) {
        params_full_precision[i] = float(params_full_precision_list[i]);
    }
}

template <typename T, int WIDTH> void NetworkWithEncodingModule<T, WIDTH>::free_memory() { network->free_memory(); }

template <typename T, int WIDTH> int NetworkWithEncodingModule<T, WIDTH>::n_params() {
    return network->get_network()->get_weights_matrices()->size();
}

} // namespace tnn
