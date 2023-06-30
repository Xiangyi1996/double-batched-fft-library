#include <iostream>
#include <vector>
#include <CL/sycl.hpp>
#include <ext/oneapi/experimental/bfloat16.hpp>
#include <ext/oneapi/matrix/matrix.hpp>
#include "activation.h"
#include "mkl.h"

template <typename T, int WIDTH>
SwiftNetMLP<T, WIDTH>::SwiftNetMLP(
	uint32_t input_width,
	uint32_t output_width,
	uint32_t n_hidden_layers,
	Activation activation,
	Activation output_activation
) :
	m_input_width{ input_width },
	m_net_width{ WIDTH },
	m_output_width{ output_width },
	m_n_hidden_layers{ n_hidden_layers },
	m_activation{ activation },
	m_output_activation{ output_activation }
{
	m_n_hidden_matrices = m_n_hidden_layers - 1;

	m_padded_output_width = next_multiple(m_output_width, REQUIRED_ALIGNMENT());

	m_weight_matrices.emplace_back(nullptr, m_network_width, m_input_width);
	m_weight_matrices_inference.emplace_back(nullptr, m_network_width, m_input_width);
	m_grad_matrices.emplace_back(nullptr, m_network_width, m_input_width);

	for (uint32_t i = 0; i < m_n_hidden_matmuls; ++i) {
		m_weight_matrices.emplace_back(nullptr, m_network_width, m_network_width);
		m_weight_matrices_inference.emplace_back(nullptr, m_network_width, m_network_width);
		m_gradient_matrices.emplace_back(nullptr, m_network_width, m_network_width);
	}

}


template <typename T, int WIDTH>
std::unique_ptr<Context> FullyFusedMLP<T, WIDTH>::forward_pass(stream outs, const const std::vector<bf16> &input, std::vector<T> *output) {

	int batch_size = input.size();
	std::vector<float> forward(batch_size * WIDTH, 0);

	switch (m_activation) {
		case Activation::None:        mlp_fused_forward<WIDTH, T, Activation::None>( m_output_activation, m_weight_matrices, input, forward, output, m_n_hidden_matmuls); break;
		case Activation::Exponential: mlp_fused_forward<WIDTH, T, Activation::Exponential( m_output_activation, m_weight_matrices, input, forward, output, m_n_hidden_matmuls); break;
		case Activation::Sigmoid:     mlp_fused_forward<WIDTH, T, Activation::Sigmoid>( m_output_activation, m_weight_matrices, input, forward, output, m_n_hidden_matmuls); break;
		case Activation::ReLU:        mlp_fused_forward<WIDTH, T, Activation::ReLU>( m_output_activation, m_weight_matrices, input, forward, output, m_n_hidden_matmuls); break;
		case Activation::LeakyReLU:   mlp_fused_forward<WIDTH, T, Activation::LeakyReLU>( m_output_activation, m_weight_matrices, input, forward, output, m_n_hidden_matmuls); break;
		case Activation::Squareplus:  mlp_fused_forward<WIDTH, T, Activation::Squareplus>( m_output_activation, m_weight_matrices, input, forward, output, m_n_hidden_matmuls); break;
		case Activation::Softplus:    mlp_fused_forward<WIDTH, T, Activation::Softplus>( m_output_activation, m_weight_matrices, input, forward, output, m_n_hidden_matmuls); break;
		case Activation::Tanh:        mlp_fused_forward<WIDTH, T, Activation::Tanh>( m_output_activation, m_weight_matrices, input, forward, output, m_n_hidden_matmuls); break;
		default: throw std::runtime_error{"Unsupported activation."};
		}

	return forward;
	}
