#include <iostream>
#include <vector>
#include <CL/sycl.hpp>
#include <ext/oneapi/matrix/matrix.hpp>
#include "SwiftNetMLP.h"
#include "activation.h"
#include "mkl.h"

template <typename T, int WIDTH>
SwiftNetMLP<T, WIDTH>::SwiftNetMLP(
	int input_width,
	int output_width,
	int n_hidden_layers,
	Activation activation,
	Activation output_activation
) :
	m_inputs_width{ input_width },
	m_net_width{ WIDTH },
	m_output_width{ output_width },
	m_n_hidden_layers{ n_hidden_layers },
	m_activation{ activation },
	m_output_activation{ output_activation }
{
	m_n_hidden_matrices = m_n_hidden_layers;

	m_weights_matrices.resize(m_net_width * m_inputs_width + (m_net_width * m_net_width) * m_n_hidden_matrices);
	m_weights_matrices_inferences.resize(m_net_width * m_inputs_width + (m_net_width * m_net_width) * m_n_hidden_matrices);
	m_grads_matrices.resize(m_net_width * m_inputs_width + (m_net_width * m_net_width) * m_n_hidden_matrices);


}

template <typename T, int WIDTH>
void SwiftNetMLP<T, WIDTH>::initialize_params() {
	for (int i = 0; i < m_net_width * m_inputs_width; i++) {
		m_weights_matrices[i] = bf16(1.0f);
		m_weights_matrices_inferences[i] = bf16(1.0f);
		m_grads_matrices[i] = bf16(1.0f);
	}
	for (int i = 0; i < m_n_hidden_matrices; i++) {
		for (int j = 0; j < m_net_width * m_net_width; j++) {

			m_weights_matrices[i * m_net_width * m_net_width + m_net_width * m_inputs_width + j] = bf16(1.0f);
			m_weights_matrices_inferences[i * m_net_width * m_net_width + m_net_width * m_inputs_width + j] = bf16(1.0f);
			m_grads_matrices[i * m_net_width * m_net_width + m_net_width * m_inputs_width + j] = bf16(1.0f);
		}

	}
}

template <typename T, int WIDTH>
std::vector<float> SwiftNetMLP<T, WIDTH>::forward_pass(const std::vector<bf16>& input, std::vector<T>& output) {

	int output_stride = WIDTH;
	int batch_size = input.size() / WIDTH;
	std::vector<float> forward(128 * WIDTH * (m_n_hidden_matrices + 1), 0.0f);
	std::cout << m_weights_matrices[0] << std::endl;

	switch (m_activation) {
	case Activation::None:        mlp_swift_forward<WIDTH, T, Activation::None>(m_output_activation, m_weights_matrices, input, forward, output, output_stride, m_n_hidden_matrices, 128, m_inputs_width, m_output_width); break;
	case Activation::Exponential: mlp_swift_forward<WIDTH, T, Activation::Exponential>(m_output_activation, m_weights_matrices, input, forward, output, output_stride, m_n_hidden_matrices, batch_size, m_inputs_width, m_output_width); break;
	case Activation::Sigmoid:     mlp_swift_forward<WIDTH, T, Activation::Sigmoid>(m_output_activation, m_weights_matrices, input, forward, output, output_stride, m_n_hidden_matrices, batch_size, m_inputs_width, m_output_width); break;
	case Activation::ReLU:        mlp_swift_forward<WIDTH, T, Activation::ReLU>(m_output_activation, m_weights_matrices, input, forward, output, output_stride, m_n_hidden_matrices, batch_size, m_inputs_width, m_output_width); break;
	case Activation::LeakyReLU:   mlp_swift_forward<WIDTH, T, Activation::LeakyReLU>(m_output_activation, m_weights_matrices, input, forward, output, output_stride, m_n_hidden_matrices, batch_size, m_inputs_width, m_output_width); break;
	case Activation::Squareplus:  mlp_swift_forward<WIDTH, T, Activation::Squareplus>(m_output_activation, m_weights_matrices, input, forward, output, output_stride, m_n_hidden_matrices, batch_size, m_inputs_width, m_output_width); break;
	case Activation::Softplus:    mlp_swift_forward<WIDTH, T, Activation::Softplus>(m_output_activation, m_weights_matrices, input, forward, output, output_stride, m_n_hidden_matrices, batch_size, m_inputs_width, m_output_width); break;
	case Activation::Tanh:        mlp_swift_forward<WIDTH, T, Activation::Tanh>(m_output_activation, m_weights_matrices, input, forward, output, output_stride, m_n_hidden_matrices, batch_size, m_inputs_width, m_output_width); break;
	default: throw std::runtime_error{"Unsupported activation."};
	}

	return forward;
}



////FIXME a voir comment est implemnte la backward pass
//template <typename T, int WIDTH>
//void SwiftNetMLP<T, WIDTH>::backward_pass(
//	stream outs,
//	const Context& ctx,
//	const std::vector<bf16> &input,
//	const std::vector<T> &output,
//	std::vector<bf16> &dL_doutput,
//	std::vector<bf16> *dL_dinput,
//	EGradientMode param_gradients_mode
//) {
//
//	int batch_size = dL_doutput.size();
//
//	std::vector<float> backward_tmp(batch_size * WIDTH, 0);
//
//	GPUMatrixDynamic<T> backward_output_tmp;
//	if (m_output_activation != Activation::None) {
//		backward_output_tmp = { m_padded_output_width, batch_size, outs, dL_doutput.layout() };
//		activation_backward_output_gpu(stream, dL_doutput.n_elements(), m_output_activation, output.data(), dL_doutput.data(), backward_output_tmp.data());
//	}
//
//	// Backprop
//	// - weight_gradient.T = activation * output_gradient.T
//	// - input_gradient = weights.T * output_gradient
//	// - RELU: pre_activation_gradinet = post_activation_gradient if val > 0 else 0
//
//	const float param_gradient_beta = param_gradients_mode == EGradientMode::Accumulate ? 1.0f : 0.0f;
//
//	std::vector<SyncedMultiStream> multi_streams;
//
//	const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
//
//	int split_k_factor = batch_size / std::min((uint32_t)(1 << 12), batch_size);
//
//	const GPUMatrixDynamic<T>& tmp_dL_doutput = m_output_activation == Activation::None ? dL_doutput : backward_output_tmp;
//
//	uint32_t tmp_idx = m_n_hidden_matmuls;
//	uint32_t backward_tmp_idx = 0;
//
//	// Output layer
//	if (param_gradients_mode != EGradientMode::Ignore) {
//		multi_streams.emplace_back(stream, 2);
//		fc_multiply_split_k<LastLayerK>(multi_streams.back().get(1), tmp_dL_doutput, forward.hidden.at(tmp_idx).transposed(), output_gradient_matrix(), split_k_factor, param_gradient_beta);
//	}
//
//	// If the output width is larger than 16 dims, we use cutlass to backpropagate through the last layer
//	// rather than fusing it with our kernel.
//	if (m_output_width > 16) {
//		fc_multiply<FullLayer>(stream, output_weight_matrix(use_inference_params).transposed(), tmp_dL_doutput, forward.hidden.at(tmp_idx), backward_tmp.at(backward_tmp_idx), m_activation, true);
//	}
//
//	// Only let the fully fused kernel compute gradients w.r.t. the input, if the input layer has the same size & layout as the other layers
//	auto dL_dinput_fused = input.m() == forward.hidden.at(0).m() && input.layout() == CM ? dL_dinput : nullptr;
//
//	// ASSUMPTION: weight matrices & forward_tmp matrices are contiguous in memory
//	switch (m_activation) {
//	case Activation::None:        mlp_swiftnet_backward<WIDTH, T, Activation::None>(outs, input_weight_matrix(use_inference_params), weight_matrix_at(use_inference_params, 0), tmp_dL_doutput, backward_tmp.at(backward_tmp_idx), forward.hidden.at(0), dL_dinput_fused, m_n_hidden_matmuls); break;
//	case Activation::Exponential: mlp_swiftnet_backward<WIDTH, T, Activation::Exponential>(outs, input_weight_matrix(use_inference_params), weight_matrix_at(use_inference_params, 0), tmp_dL_doutput, backward_tmp.at(backward_tmp_idx), forward.hidden.at(0), dL_dinput_fused, m_n_hidden_matmuls); break;
//	case Activation::Sigmoid:     mlp_swiftnet_backward<WIDTH, T, Activation::Sigmoid>(outs, input_weight_matrix(use_inference_params), weight_matrix_at(use_inference_params, 0), tmp_dL_doutput, backward_tmp.at(backward_tmp_idx), forward.hidden.at(0), dL_dinput_fused, m_n_hidden_matmuls); break;
//	case Activation::ReLU:        mlp_swiftnet_backward<WIDTH, T, Activation::ReLU>(outs, input_weight_matrix(use_inference_params), weight_matrix_at(use_inference_params, 0), tmp_dL_doutput, backward_tmp.at(backward_tmp_idx), forward.hidden.at(0), dL_dinput_fused, m_n_hidden_matmuls); break;
//	case Activation::LeakyReLU:   mlp_swiftnet_backward<WIDTH, T, Activation::LeakyReLU>(outs, input_weight_matrix(use_inference_params), weight_matrix_at(use_inference_params, 0), tmp_dL_doutput, backward_tmp.at(backward_tmp_idx), forward.hidden.at(0), dL_dinput_fused, m_n_hidden_matmuls); break;
//	case Activation::Squareplus:  mlp_swiftnet_backward<WIDTH, T, Activation::Squareplus>(outs, input_weight_matrix(use_inference_params), weight_matrix_at(use_inference_params, 0), tmp_dL_doutput, backward_tmp.at(backward_tmp_idx), forward.hidden.at(0), dL_dinput_fused, m_n_hidden_matmuls); break;
//	case Activation::Softplus:    mlp_swiftnet_backward<WIDTH, T, Activation::Softplus>(outs, input_weight_matrix(use_inference_params), weight_matrix_at(use_inference_params, 0), tmp_dL_doutput, backward_tmp.at(backward_tmp_idx), forward.hidden.at(0), dL_dinput_fused, m_n_hidden_matmuls); break;
//	case Activation::Tanh:        mlp_swiftnet_backward<WIDTH, T, Activation::Tanh>(outs, input_weight_matrix(use_inference_params), weight_matrix_at(use_inference_params, 0), tmp_dL_doutput, backward_tmp.at(backward_tmp_idx), forward.hidden.at(0), dL_dinput_fused, m_n_hidden_matmuls); break;
//	default: throw std::runtime_error{"Unsupported activation."};
//	}
//
//	tmp_idx -= 1;
//	++backward_tmp_idx;
//
//	// layers
//	for (uint32_t i = 0; i < m_n_hidden_matmuls; ++i) {
//		uint32_t matrix_idx = m_n_hidden_matmuls - i - 1;
//
//		if (param_gradients_mode != EGradientMode::Ignore) {
//			multi_streams.emplace_back(stream, 2);
//			fc_multiply_split_k<FullLayerK>(multi_streams.back().get(1), backward_tmp.at(backward_tmp_idx - 1), forward.hidden.at(tmp_idx).transposed(), gradient_matrix_at(matrix_idx), split_k_factor, param_gradient_beta);
//		}
//
//		tmp_idx -= 1;
//		++backward_tmp_idx;
//	}
//
//	if (param_gradients_mode != EGradientMode::Ignore) {
//		multi_streams.emplace_back(stream, 2);
//		fc_multiply_split_k<FullLayerK>(multi_streams.back().get(1), backward_tmp.at(backward_tmp_idx - 1), input.transposed(), input_gradient_matrix(), split_k_factor, param_gradient_beta);
//	}
//
//
//}
