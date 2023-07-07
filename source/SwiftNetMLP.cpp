#include <iostream>
#include <vector>
#include <CL/sycl.hpp>
#include <ext/oneapi/matrix/matrix.hpp>
#include "SwiftNetMLP.h"
#include "activation.h"
#include "mkl.h"

template <int WIDTH>
SwiftNetMLP<WIDTH>::SwiftNetMLP(
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
	m_n_hidden_matrices = m_n_hidden_layers - 1;
	m_weightsT_matrices.resize(m_net_width * m_inputs_width + (m_net_width * m_net_width) * m_n_hidden_matrices + m_net_width * m_output_width);
	m_weights_matrices.resize(m_net_width * m_inputs_width + (m_net_width * m_net_width) * m_n_hidden_matrices + m_net_width * m_output_width);
	m_weights_matrices_inferences.resize(m_net_width * m_inputs_width + (m_net_width * m_net_width) * m_n_hidden_matrices + m_net_width * m_output_width);
	m_grads_matrices.resize(m_net_width * m_inputs_width + (m_net_width * m_net_width) * m_n_hidden_matrices + m_net_width * m_output_width);



}

template <int WIDTH>
void SwiftNetMLP<WIDTH>::initialize_params() {
	for (int i = 0; i < m_net_width * m_inputs_width; i++) {
		m_weights_matrices[i] = bf16(1.0f);
		m_weights_matrices_inferences[i] = bf16(1.0f);
		m_weightsT_matrices[i] = bf16(1.0f);

	}
	for (int i = 0; i < m_n_hidden_matrices; i++) {
		for (int j = 0; j < m_net_width * m_net_width; j++) {

			m_weights_matrices[i * m_net_width * m_net_width + m_net_width * m_inputs_width + j] = bf16(1.0f);
			m_weights_matrices_inferences[i * m_net_width * m_net_width + m_net_width * m_inputs_width + j] = bf16(1.0f);
			m_weightsT_matrices[i * m_net_width * m_net_width + m_net_width * m_inputs_width + j] = bf16(1.0f);
		}

	}
	for (int i = 0; i < m_net_width * m_output_width; i++) {
		m_weights_matrices[m_net_width * m_inputs_width + (m_net_width * m_net_width) * m_n_hidden_matrices + i] = bf16(1.0f);
		m_weights_matrices_inferences[m_net_width * m_inputs_width + (m_net_width * m_net_width) * m_n_hidden_matrices + i] = bf16(1.0f);
		m_weightsT_matrices[m_net_width * m_inputs_width + (m_net_width * m_net_width) * m_n_hidden_matrices + i] = bf16(1.0f);
	}


}

template <int WIDTH>
std::vector<bf16> SwiftNetMLP<WIDTH>::forward_pass(const std::vector<bf16>& input, std::vector<float>& output) {

	int output_stride = WIDTH;
	int batch_size = input.size() / m_inputs_width;
	std::vector<float> forward_f(128 * WIDTH * (m_n_hidden_matrices + 2), 0.0f);
	std::vector<bf16> forward(128 * WIDTH * (m_n_hidden_matrices + 2), 0.0f);


	switch (m_activation) {
	case Activation::None:        mlp_swift_forward<WIDTH, Activation::None>(m_output_activation, m_weights_matrices, input, forward_f, output, output_stride, m_n_hidden_layers, batch_size, m_inputs_width, m_output_width); break;
	case Activation::Exponential: mlp_swift_forward<WIDTH, Activation::Exponential>(m_output_activation, m_weights_matrices, input, forward_f, output, output_stride, m_n_hidden_layers, batch_size, m_inputs_width, m_output_width); break;
	case Activation::Sigmoid:     mlp_swift_forward<WIDTH, Activation::Sigmoid>(m_output_activation, m_weights_matrices, input, forward_f, output, output_stride, m_n_hidden_layers, batch_size, m_inputs_width, m_output_width); break;
	case Activation::ReLU:        mlp_swift_forward<WIDTH, Activation::ReLU>(m_output_activation, m_weights_matrices, input, forward_f, output, output_stride, m_n_hidden_layers, batch_size, m_inputs_width, m_output_width); break;
	case Activation::LeakyReLU:   mlp_swift_forward<WIDTH, Activation::LeakyReLU>(m_output_activation, m_weights_matrices, input, forward_f, output, output_stride, m_n_hidden_layers, batch_size, m_inputs_width, m_output_width); break;
	case Activation::Squareplus:  mlp_swift_forward<WIDTH, Activation::Squareplus>(m_output_activation, m_weights_matrices, input, forward_f, output, output_stride, m_n_hidden_layers, batch_size, m_inputs_width, m_output_width); break;
	case Activation::Softplus:    mlp_swift_forward<WIDTH, Activation::Softplus>(m_output_activation, m_weights_matrices, input, forward_f, output, output_stride, m_n_hidden_layers, batch_size, m_inputs_width, m_output_width); break;
	case Activation::Tanh:        mlp_swift_forward<WIDTH, Activation::Tanh>(m_output_activation, m_weights_matrices, input, forward_f, output, output_stride, m_n_hidden_layers, batch_size, m_inputs_width, m_output_width); break;
	default: throw std::runtime_error{"Unsupported activation."};
	}

	for (int i = 0; i < forward.size(); i++) {
		forward[i] = bf16(forward_f[i]);
	}

	return forward;
}

template <int WIDTH>
void SwiftNetMLP<WIDTH>::dgemm_last_layer_backward(std::vector<bf16>& grads, std::vector<bf16>& forward, std::vector<bf16>& act_fwd, std::vector<bf16>& loss, int batch_size) {
	double* A;
	double* B;
	double* C;
	A = (double*)mkl_malloc(grads.size() * sizeof(double), 64);
	//B = (MKL_BF16*)mkl_malloc(grads.size() * sizeof(MKL_BF16), 64);
	B = (double*)mkl_malloc(WIDTH * WIDTH * sizeof(double), 64);
	C = (double*)mkl_malloc(WIDTH * batch_size * sizeof(double), 64);
	for (int i = 0; i < grads.size(); i++) {
		A[i] = (double)loss[i];
	}
	for (int i = 0; i < WIDTH * WIDTH; i++) {
		B[i] = (double)m_weightsT_matrices[m_n_hidden_matrices * m_net_width * m_net_width + m_net_width * m_inputs_width + i];
	}
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		batch_size, m_net_width, m_output_width, 1, A, m_output_width, B, m_net_width, 0, C, m_net_width);

	bf16 x = 0;
	for (int i = 0; i < m_net_width * batch_size; i++) {
		elt_activation_bwd<double, float, bf16>(m_activation, C[i], forward[(m_n_hidden_matrices - 1) * batch_size * m_net_width + i], x);
		loss[i] = x;

		for (int j = 0; j < m_net_width; j++) {
			m_grads_matrices[m_inputs_width * m_net_width + (m_n_hidden_matrices - 1) * m_net_width * m_net_width + i % m_net_width + j * m_net_width] += x * act_fwd[m_inputs_width * batch_size + (m_n_hidden_matrices - 1) * m_net_width * batch_size + j + (i / m_net_width) * m_net_width];
		}

	}

	mkl_free(A);
	mkl_free(B);
	mkl_free(C);

}

template <int WIDTH>
void SwiftNetMLP<WIDTH>::backward_pass(const std::vector<bf16>& input, std::vector<bf16>& grads, std::vector<bf16>& forward, std::vector<bf16>& act_fwd) {
	int batch_size = input.size() / m_inputs_width;

	bf16 x;
	std::vector<bf16> loss(m_net_width * batch_size);
	for (int i = 0; i < batch_size * m_output_width; i++) {
		// On calcule les loss gradients du dernier layer

		elt_activation_bwd<bf16, float, bf16>(
			m_output_activation,
			grads[i],
			forward[(m_n_hidden_matrices + 1) * batch_size * m_net_width + i],
			x);
		loss[i] = x;
		for (int j = 0; j < m_net_width; j++) {
			m_grads_matrices[m_n_hidden_matrices * m_net_width * m_net_width + m_inputs_width * m_net_width + i % m_output_width + j * m_output_width] += x * act_fwd[m_inputs_width * batch_size + m_n_hidden_matrices * m_net_width * batch_size + j + (i / m_output_width) * m_net_width];
		}
	}

	/// Backpropagation through last layer
	dgemm_last_layer_backward(grads, forward, act_fwd, loss, batch_size);
	switch (m_activation) {
	case Activation::None:        mlp_swiftnet_backward<WIDTH, Activation::None>(act_fwd, m_weightsT_matrices, loss, m_grads_matrices, forward, batch_size, m_n_hidden_matrices); break;
	case Activation::ReLU:        mlp_swiftnet_backward<WIDTH, Activation::ReLU>(act_fwd, m_weightsT_matrices, loss, m_grads_matrices, forward, batch_size, m_n_hidden_matrices); break;
	case Activation::LeakyReLU:   mlp_swiftnet_backward<WIDTH, Activation::LeakyReLU>(act_fwd, m_weightsT_matrices, loss, m_grads_matrices, forward, batch_size, m_n_hidden_matrices); break;
	case Activation::Exponential: mlp_swiftnet_backward<WIDTH, Activation::Exponential>(act_fwd, m_weightsT_matrices, loss, m_grads_matrices, forward, batch_size, m_n_hidden_matrices); break;
	case Activation::Sigmoid:     mlp_swiftnet_backward<WIDTH, Activation::Sigmoid>(act_fwd, m_weightsT_matrices, loss, m_grads_matrices, forward, batch_size, m_n_hidden_matrices); break;
	case Activation::Tanh:        mlp_swiftnet_backward<WIDTH, Activation::Tanh>(act_fwd, m_weightsT_matrices, loss, m_grads_matrices, forward, batch_size, m_n_hidden_matrices); break;

	default: throw std::runtime_error{"Unsupported activation."};
	}
	for (int i = 0; i < m_grads_matrices.size(); i++) {
		m_grads_matrices[i] /= batch_size;
	}
}

