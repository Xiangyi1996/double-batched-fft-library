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

	void training_step(const std::vector<bf16>& input, std::vector<T>& output, std::vector<float>& dL_output, std::vector<T>& target, std::vector<bf16>& grads, std::vector<T>& losses, const float scale) {
		auto forward = m_model.forward_pass(input, output);
		m_loss.evaluate(WIDTH, WIDTH, scale, output, target, grads, losses);
		std::vector<bf16> act_forward(128 * WIDTH * (3 + 2), 0.0f);
		const int input_size = input.size();

		for (int i = 0; i < input_size; i++) {
			act_forward[i] = input[i];
		}
		for (int i = 0; i < forward.size(); i++) {
			act_forward[i + input_size] = forward[i];
		}
		m_model.backward_pass(input, grads, forward, act_forward);
		std::cout << " weigth before optim : " << m_model.m_weights_matrices[0] << std::endl;
		m_optim.step(scale, m_model.m_weights_matrices, m_model.m_weightsT_matrices, grads);
		std::cout << " weigth after optim : " << m_model.m_weights_matrices[0] << std::endl;

		for (int i = 0; i < 10; i++) {
			std::cout << "grads : " << i << " : " << grads[i ] << std::endl;
		}
			for (int j = 0; j < 10; j++) {
				std::cout << "forward : " << 3 << " : " << act_forward[128 * 64 * 1 + j] << std::endl;
			}
			for (int j = 64*64; j <64*64+ 10; j++) {
				std::cout << "forward moitie : " << 3 << " : " << act_forward[128 * 64 * 1+ j] << std::endl;
			}
	

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 10; j++) {
				std::cout << "m_grads_matrices : " << i << " : " << m_model.m_grads_matrices[64 * 64 * i + j] << std::endl;
			}
		}

	}

	void initialize_params() {
		m_model.initialize_params();
	}

private:
	SwiftNetMLP<T, WIDTH> m_model;
	L2Loss<T> m_loss;
	SGDOptimizer<T, WIDTH> m_optim;
};