#pragma once

#include "activation.h"

template <typename T, int WIDTH>
class SwiftNetMLP {
public:
	SwiftNetMLP(int input_width, int output_width, int n_hidden_layers, Activation output_activation);
	
	std::unique_ptr<Context> forward_pass(vector<half>& input, vector<half>& output);

	void backward_impl(
		const vector<half>& input,
		const vector<half>& output,
		const vector<half>& dL_output,
		);

		void set_params(T* params, T* inference_params, T* gradients);

		void initialize_params();

private:
	int m_n_hidden_layers;
	int m_n_hidden_matmuls;
	int m_inputs_width;
	int m_net_width;
	int m_output_width;
	int m_padded_output_width;

	Activation m_activation;
	Activation m_output_activation;

	std::vector<std::vector<half>> m_weights_matrices;
	std::vector<std::vector<half>> m_weights_matrices_inferences;
	std::vector<std::vector<half>> m_grads_matrices;
	

	int m_total_n_params;

};