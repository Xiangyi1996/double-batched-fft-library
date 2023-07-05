#pragma once

#include "SwiftNetMLP.h"
#include "L2.h"
//#include "optimizer.h"

template<typename T, int WIDTH>
class Trainer {
public:

	Trainer(int input_width, int output_width, int n_hidden_layers, Activation activation, Activation output_activation, L2Loss<T> loss, SGDOptimizer<T, WIDTH> optim) : m_model(input_width, output_width, n_hidden_layers, activation, output_activation) {
		m_loss = loss;
		m_optim = optim;
	}

	void training_step(const std::vector<bf16>& input, std::vector<T>& output, std::vector<float>& dL_output, std::vector<T>& target, std::vector<T>& grads, std::vector<T>& losses, const float scale) {
		auto forward = m_model.forward_pass(input, output);
		m_loss.evaluate(WIDTH, WIDTH, scale, output, target, grads, losses);
		//m_model.backward_pass(input, grads, forward);
		//std::cout << " weigth before optim : " << m_model.m_weights_matrices[0] << std::endl;
		m_optim.step(scale, m_model.m_weights_matrices, m_model.m_weightsT_matrices, grads);
		//std::cout << " weigth after optim : " << m_model.m_weights_matrices[0] << std::endl;

		/*for (int i = 0; i < 10; i++) {
			for (int j = 0; j < 10; j++) {
				std::cout << "m_weights_matrices : " << m_model.m_weights_matrices[64 * i + j] << std::endl;
				std::cout << "m_weightsT_matrices : " << m_model.m_weightsT_matrices[64 * j + i] << std::endl;
			}
		}*/

	}

	void initialize_params() {
		m_model.initialize_params();
	}

private:
	SwiftNetMLP<T, WIDTH> m_model;
	L2Loss<T> m_loss;
	SGDOptimizer<T, WIDTH> m_optim;
};