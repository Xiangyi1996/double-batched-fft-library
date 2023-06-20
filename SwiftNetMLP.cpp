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
	}

	m_weight_matrices.emplace_back(nullptr, m_padded_output_width, m_network_width);
	m_weight_matrices_inference.emplace_back(nullptr, m_padded_output_width, m_network_width);
	m_gradient_matrices.emplace_back(nullptr, m_padded_output_width, m_network_width);

	m_n_params = 0;
	for (const auto& m : m_weight_matrices) {
		m_n_params += m.n_elements();
	}
}

template <typename T, int WIDTH>
std::unique_ptr SwiftNetMLP<T, WIDTH>::forward_pass() {
	
	switch (m_activation) {
	case Activation::None:        mlp_swift_forward<WIDTH, T, Activation::None, true>(stream, m_output_activation, input_weight_matrix(use_inference_params), input, inference_tmp, &output, m_n_hidden_matmuls); break;

	m_n_hidden_matmuls = n_hidden_layers-1;

	m_padded_output_width = next_multiple(m_output_width, REQUIRED_ALIGNMENT());

	// Create matrices related to weights
	m_weight_matrices.emplace_back(nullptr, m_network_width, m_input_width);
	m_weight_matrices_inference.emplace_back(nullptr, m_network_width, m_input_width);
	m_gradient_matrices.emplace_back(nullptr, m_network_width, m_input_width);

	for (uint32_t i = 0; i < m_n_hidden_matmuls; ++i) {
		m_weight_matrices.emplace_back(nullptr, m_network_width, m_network_width);
		m_weight_matrices_inference.emplace_back(nullptr, m_network_width, m_network_width);
		m_gradient_matrices.emplace_back(nullptr, m_network_width, m_network_width);
	}

	m_weight_matrices.emplace_back(nullptr, m_padded_output_width, m_network_width);
	m_weight_matrices_inference.emplace_back(nullptr, m_padded_output_width, m_network_width);
	m_gradient_matrices.emplace_back(nullptr, m_padded_output_width, m_network_width);

	// Determine total number of memory entries and set it
	m_total_n_params = 0;
	for (const auto& m : m_weight_matrices) {
		m_total_n_params += m.n_elements();
	}
}

template <typename T, int WIDTH>
void FullyFusedMLP<T, WIDTH>::inference_mixed_precision_impl(cudaStream_t stream, const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<T>& output, bool use_inference_params) {
	// Make sure our temporary buffers have the correct size for the given batch size
	uint32_t batch_size = input.n();

	GPUMatrix<T> inference_tmp = m_output_width > 16 ? GPUMatrix<T>{m_network_width, batch_size, stream} : GPUMatrix<T>{nullptr, m_network_width, batch_size};


	}
//FIXME
if (m_output_width > 16) {
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		m, n, k, 1, , k, B, n, beta, C, n);
	}
}