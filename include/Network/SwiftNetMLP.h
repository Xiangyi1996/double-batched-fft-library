#pragma once

#include "activation.h"

using bf16 = sycl::ext::oneapi::bfloat16;

template <typename T, int WIDTH>
class SwiftNetMLP {
public:
	SwiftNetMLP(int input_width, int output_width, int n_hidden_layers, Activation activation, Activation output_activation);

	std::vector<float> forward_pass(const std::vector<bf16>& input, std::vector<T>& output);

	void backward_pass(
		const std::vector<bf16>& input, std::vector<bf16>& grads, std::vector<T>& forward
	);
	void dgemm_last_layer_backward(std::vector<bf16>& grads, std::vector<T> forward, int batch_size);

	void set_params(T* params, T* inference_params, T* gradients);

	void initialize_params();

	std::vector<bf16> m_weights_matrices;
	std::vector<bf16> m_weightsT_matrices;
	std::vector<bf16> m_grads_matrices;

private:
	int m_n_hidden_layers;
	int m_n_hidden_matrices;
	int m_inputs_width;
	int m_net_width;
	int m_output_width;
	int m_padded_output_width;

	Activation m_activation;
	Activation m_output_activation;
	std::vector<bf16> m_weights_matrices_inferences;


	int m_total_n_params;

};
