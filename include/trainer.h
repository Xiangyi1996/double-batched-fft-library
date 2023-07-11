#pragma once

#include "SwiftNetMLP.h"
#include "L2.h"
//#include "optimizer.h"

template<int WIDTH>
class Trainer {
public:

	Trainer(int input_width, int output_width, int n_hidden_layers, Activation activation, Activation output_activation, L2Loss loss, SGDOptimizer<WIDTH> optim) : m_model(input_width, output_width, n_hidden_layers, activation, output_activation) {
		m_loss = loss;
		m_optim = optim;
	}

	void training_step(const std::vector<bf16>& input, std::vector<float>& output, std::vector<float>& target, std::vector<bf16>& grads, std::vector<float>& losses, const float scale) {
		const int input_size = input.size();

		auto forward = m_model.forward_pass(input, output);

		m_loss.evaluate(WIDTH, WIDTH, scale, output, target, grads, losses);
		
		m_model.backward_pass(input, grads, forward);
		
		m_optim.step(scale, m_model.m_weights_matrices, m_model.m_weightsT_matrices, grads);

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 10; j++) {
				std::cout << "forward : " << i << " : " << forward[256 * 64 * i + j] << std::endl;
			}
		}

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 10; j++) {
				std::cout << "grads : "<< i<< " : " << m_model.m_grads_matrices[64 * 64 * i + j] << std::endl;
			}
		}
	}

	void initialize_params() {
		m_model.initialize_params();
	}

private:

	SwiftNetMLP<WIDTH> m_model;
	L2Loss m_loss;
	SGDOptimizer<WIDTH> m_optim;
};