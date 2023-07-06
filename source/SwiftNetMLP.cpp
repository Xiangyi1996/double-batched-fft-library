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
	m_n_hidden_matrices = m_n_hidden_layers - 1;
	m_weightsT_matrices.resize(m_net_width * m_inputs_width + (m_net_width * m_net_width) * m_n_hidden_matrices + m_net_width * m_output_width);
	m_weights_matrices.resize(m_net_width * m_inputs_width + (m_net_width * m_net_width) * m_n_hidden_matrices + m_net_width * m_output_width);
	m_weights_matrices_inferences.resize(m_net_width * m_inputs_width + (m_net_width * m_net_width) * m_n_hidden_matrices + m_net_width * m_output_width);
	m_grads_matrices.resize(m_net_width * m_inputs_width + (m_net_width * m_net_width) * m_n_hidden_matrices + m_net_width * m_output_width);



}

template <typename T, int WIDTH>
void SwiftNetMLP<T, WIDTH>::initialize_params() {
	for (int i = 0; i < m_net_width * m_inputs_width; i++) {
		m_weights_matrices[i] = bf16(1.0f / 64);
		m_weights_matrices_inferences[i] = bf16(1.0f / 64);
		m_weightsT_matrices[i] = bf16(1.0f / 64);

	}
	for (int i = 0; i < m_n_hidden_matrices; i++) {
		for (int j = 0; j < m_net_width * m_net_width; j++) {

			m_weights_matrices[i * m_net_width * m_net_width + m_net_width * m_inputs_width + j] = bf16(1.0f / 64);
			m_weights_matrices_inferences[i * m_net_width * m_net_width + m_net_width * m_inputs_width + j] = bf16(1.0f / 64);
			m_weightsT_matrices[i * m_net_width * m_net_width + m_net_width * m_inputs_width + j] = bf16(1.0f / 64);
		}

	}
	for (int i = 0; i < m_net_width * m_output_width; i++) {
		m_weights_matrices[m_net_width * m_inputs_width + (m_net_width * m_net_width) * m_n_hidden_matrices + i] = bf16(1.0f / 64);
		m_weights_matrices_inferences[m_net_width * m_inputs_width + (m_net_width * m_net_width) * m_n_hidden_matrices + i] = bf16(1.0f / 64);
		m_weightsT_matrices[m_net_width * m_inputs_width + (m_net_width * m_net_width) * m_n_hidden_matrices + i] = bf16(1.0f / 64);
	}


}

template <typename T, int WIDTH>
std::vector<bf16> SwiftNetMLP<T, WIDTH>::forward_pass(const std::vector<bf16>& input, std::vector<T>& output) {

	int output_stride = WIDTH;
	int batch_size = input.size() / m_inputs_width;
	std::vector<float> forward_f(128 * WIDTH * (m_n_hidden_matrices + 2), 0.0f);
	std::vector<bf16> forward(128 * WIDTH * (m_n_hidden_matrices + 2), 0.0f);


	switch (m_activation) {
	case Activation::None:        mlp_swift_forward<WIDTH, T, Activation::None>(m_output_activation, m_weights_matrices, input, forward_f, output, output_stride, m_n_hidden_layers, batch_size, m_inputs_width, m_output_width); break;
	case Activation::Exponential: mlp_swift_forward<WIDTH, T, Activation::Exponential>(m_output_activation, m_weights_matrices, input, forward_f, output, output_stride, m_n_hidden_layers, batch_size, m_inputs_width, m_output_width); break;
	case Activation::Sigmoid:     mlp_swift_forward<WIDTH, T, Activation::Sigmoid>(m_output_activation, m_weights_matrices, input, forward_f, output, output_stride, m_n_hidden_layers, batch_size, m_inputs_width, m_output_width); break;
	case Activation::ReLU:        mlp_swift_forward<WIDTH, T, Activation::ReLU>(m_output_activation, m_weights_matrices, input, forward_f, output, output_stride, m_n_hidden_layers, batch_size, m_inputs_width, m_output_width); break;
	case Activation::LeakyReLU:   mlp_swift_forward<WIDTH, T, Activation::LeakyReLU>(m_output_activation, m_weights_matrices, input, forward_f, output, output_stride, m_n_hidden_layers, batch_size, m_inputs_width, m_output_width); break;
	case Activation::Squareplus:  mlp_swift_forward<WIDTH, T, Activation::Squareplus>(m_output_activation, m_weights_matrices, input, forward_f, output, output_stride, m_n_hidden_layers, batch_size, m_inputs_width, m_output_width); break;
	case Activation::Softplus:    mlp_swift_forward<WIDTH, T, Activation::Softplus>(m_output_activation, m_weights_matrices, input, forward_f, output, output_stride, m_n_hidden_layers, batch_size, m_inputs_width, m_output_width); break;
	case Activation::Tanh:        mlp_swift_forward<WIDTH, T, Activation::Tanh>(m_output_activation, m_weights_matrices, input, forward_f, output, output_stride, m_n_hidden_layers, batch_size, m_inputs_width, m_output_width); break;
	default: throw std::runtime_error{"Unsupported activation."};
	}

	for (int i = 0; i < forward.size(); i++) {
		forward[i] = bf16(forward_f[i]);
	}

	return forward;
}

template <typename T, int WIDTH>
void SwiftNetMLP<T, WIDTH>::dgemm_last_layer_backward(std::vector<bf16>& grads, std::vector<bf16>& forward, std::vector<bf16>& act_fwd, std::vector<bf16>& loss, int batch_size) {
	double* A;
	double* B;
	double* C;
	A = (double*)mkl_malloc(grads.size() * sizeof(double), 64);
	//B = (MKL_BF16*)mkl_malloc(grads.size() * sizeof(MKL_BF16), 64);
	B = (double*)mkl_malloc(m_output_width * m_net_width * sizeof(double), 64);
	C = (double*)mkl_malloc(m_net_width * batch_size * sizeof(double), 64);
	for (int i = 0; i < grads.size(); i++) {
		A[i] = (double)loss[i];
	}
	for (int i = 0; i < m_net_width * m_output_width; i++) {
		B[i] = (double)m_weightsT_matrices[m_n_hidden_matrices * m_net_width * m_net_width + m_net_width * m_inputs_width + i];
	}
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		batch_size, m_net_width, m_output_width, 1, A, m_output_width, B, m_net_width, 0, C, m_net_width);

	//std::cout <<"C : "<< C[0] << std::endl;
	bf16 x = 0;
	for (int i = 0; i < m_net_width * batch_size; i++) {
		elt_activation_bwd<double, T, bf16>(m_activation, C[i], forward[(m_n_hidden_matrices - 1) * batch_size * m_net_width + i], x);
		loss[i] = x;
		//std::cout << " act_fwd : " << act_fwd[m_inputs_width * batch_size + (m_n_hidden_matrices - 1) * m_net_width * batch_size + (i / m_net_width) * m_net_width] << std::endl;
		for (int j = 0; j < m_net_width; j++) {
			m_grads_matrices[m_inputs_width * m_net_width + (m_n_hidden_matrices - 1) * m_net_width * m_net_width + i % m_net_width + j * m_net_width] += x * act_fwd[m_inputs_width * batch_size + (m_n_hidden_matrices - 1) * m_net_width * batch_size + j + (i / m_net_width) * m_net_width];
		}

	}
	//std::cout << "m_grads_matrices : " << m_grads_matrices[m_inputs_width * m_net_width + (m_n_hidden_matrices - 1) * m_net_width * m_net_width] << std::endl;
	mkl_free(A);
	mkl_free(B);
	mkl_free(C);

}

template <typename T, int WIDTH>
void SwiftNetMLP<T, WIDTH>::backward_pass(const std::vector<bf16>& input, std::vector<bf16>& grads, std::vector<bf16>& forward, std::vector<bf16>& act_fwd) {
	int batch_size = input.size() / m_inputs_width;

	bf16 x;
	std::vector<bf16> loss(m_net_width * batch_size);
	for (int i = 0; i < batch_size * m_output_width; i++) {
		// On calcule les loss gradients du dernier layer

		elt_activation_bwd<bf16, T, bf16>(
			m_output_activation,
			grads[i],
			forward[(m_n_hidden_matrices + 1) * batch_size * m_net_width + i],
			x);
		loss[i] = x;
		for (int j = 0; j < m_net_width; j++) {
			m_grads_matrices[m_n_hidden_matrices * m_net_width * m_net_width + m_inputs_width * m_net_width + i % m_output_width + j * m_output_width] += x * act_fwd[m_inputs_width * batch_size + m_n_hidden_matrices * m_net_width * batch_size + j + (i / m_output_width) * m_net_width];
		}
		//if (i % m_output_width == 0) {
		//	std::cout << "x : " << x << std::endl;
		//	std::cout << "act forward : " << act_fwd[m_inputs_width * batch_size + m_n_hidden_matrices * m_net_width * batch_size + (i / m_output_width) * m_net_width] << std::endl;
		//	std::cout << " m_grads_matrices : " << m_grads_matrices[m_n_hidden_matrices * m_net_width * m_net_width + m_inputs_width * m_net_width + i % m_output_width] << std::endl;
		//	std::cout << "batch_size : " << batch_size << std::endl;
		//}

	}
	/*std::cout << "x : " << x << std::endl;
	std::cout << "act forward : " << act_fwd[m_inputs_width * batch_size + m_n_hidden_matrices * m_net_width * batch_size] << std::endl;
	std::cout << "act m_grads_matrices : " << m_grads_matrices[m_n_hidden_matrices * m_net_width * m_net_width + m_inputs_width * m_net_width ] << std::endl;
	std::cout << "batch_size : " << batch_size << std::endl;*/


	/// Backpropagation through last layer
	dgemm_last_layer_backward(grads, forward, act_fwd, loss, batch_size);
	switch (m_activation) {
	case Activation::None:        mlp_swiftnet_backward<WIDTH, T, Activation::None>(act_fwd, m_weightsT_matrices, loss, m_grads_matrices, forward, batch_size, m_n_hidden_matrices); break;
	case Activation::ReLU:        mlp_swiftnet_backward<WIDTH, T, Activation::ReLU>(act_fwd, m_weightsT_matrices, loss, m_grads_matrices, forward, batch_size, m_n_hidden_matrices); break;
	case Activation::LeakyReLU:   mlp_swiftnet_backward<WIDTH, T, Activation::LeakyReLU>(act_fwd, m_weightsT_matrices, loss, m_grads_matrices, forward, batch_size, m_n_hidden_matrices); break;
	case Activation::Exponential: mlp_swiftnet_backward<WIDTH, T, Activation::Exponential>(act_fwd, m_weightsT_matrices, loss, m_grads_matrices, forward, batch_size, m_n_hidden_matrices); break;
	case Activation::Sigmoid:     mlp_swiftnet_backward<WIDTH, T, Activation::Sigmoid>(act_fwd, m_weightsT_matrices, loss, m_grads_matrices, forward, batch_size, m_n_hidden_matrices); break;
	case Activation::Tanh:        mlp_swiftnet_backward<WIDTH, T, Activation::Tanh>(act_fwd, m_weightsT_matrices, loss, m_grads_matrices, forward, batch_size, m_n_hidden_matrices); break;

	default: throw std::runtime_error{"Unsupported activation."};
	}
	for (int i = 0; i < m_grads_matrices.size(); i++) {
		m_grads_matrices[i] /= batch_size;
	}
}
