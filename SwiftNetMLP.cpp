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
	m_net_width{  WIDTH },
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

