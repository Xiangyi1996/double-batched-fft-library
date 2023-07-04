#pragma once

#include "SwiftNetMLP.h"
#include "loss.h"
#include "optimizer.h"

template< typename T, int WIDTH>
class Trainer {
public:

	Trainer(int input_width, int output_width, int n_hidden_layers, Activation activation, Activation output_activation, Loss loss, Optimizer optim ) : m_model(input_width, output_width, n_hidden_layer, activation, output_activation) {
		m_loss = loss;
		m_optim = optim;
	}

	training_step(const std::vector<bf16>& input, std::vector<T>& output, std::vector<bf16>& dL_output, const std::vector<T> target,std::vector<T> loss, std::vector<bf16> grads, float scale) {
		auto forward = m_model.forward_pass(input, output);
		loss.evaluate(WIDTH, WIDTH, scale, output, target, grads, loss);
		m_model.backward_pass(input, output, dL_output);
		m_optim.step(scale, m_model.m_weights_matrices, grads);

	}

private:
	SwiftNetMLP<T, WIDTH> m_model;
	Loss m_loss;
	Optimizer m_optim;
};