#pragma once

#include "activation.h"
#include "DeviceMem.h"

using bf16 = sycl::ext::oneapi::bfloat16;

template <int WIDTH>
class SwiftNetMLP {
public:
	SwiftNetMLP(queue q, int input_width, int output_width, int n_hidden_layers, Activation activation, Activation output_activation);

	DeviceMem<bf16> forward_pass(const DeviceMem<bf16>& input, DeviceMem<float>& output);

	void backward_pass(
		const DeviceMem<bf16>& input, DeviceMem<bf16>& grads, DeviceMem<bf16>& forward
	);
	void dgemm_last_layer_backward(DeviceMem<bf16>& grads, DeviceMem<bf16>& forward, DeviceMem<bf16>& loss, int batch_size);
	//void set_params(float* params, float* inference_params, float* gradients);

	void initialize_params();

	DeviceMem<bf16> m_grads_matrices;
	DeviceMem<bf16> m_weights_matrices;
	DeviceMem<bf16> m_weightsT_matrices;
	queue m_q;

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
