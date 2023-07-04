#pragma once

#include "SwiftNetMLP.h"
#include "L2.h"
//#include "optimizer.h"

template<typename T, int WIDTH>
class Trainer {
public:

	Trainer(int input_width, int output_width, int n_hidden_layers, Activation activation, Activation output_activation, L2Loss<T> loss ) : m_model(input_width, output_width, n_hidden_layers, activation, output_activation) {
		m_loss = loss;
		//m_optim = optim;
	}

	void training_step(const std::vector<bf16>& input, std::vector<T>& output, std::vector<float>& dL_output, std::vector<T>& target, std::vector<T>& grads, std::vector<T>& losses, const float scale) {
		m_model.initialize_params();
		auto forward = m_model.forward_pass(input, output);
		std::cout << target[0] << std::endl;
		m_loss.evaluate(WIDTH, WIDTH, scale, output, target, grads, losses);
		//m_model.backward_pass(grads, forward);
		//m_optim.step(scale, m_model.m_weights_matrices, grads);

	}

private:
	SwiftNetMLP<T, WIDTH> m_model;
	L2Loss<T> m_loss;
	//Optimizer m_optim;
};