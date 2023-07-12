#pragma once

#include "SwiftNetMLP.h"
#include "L2.h"
#include "DeviceMem.h"
//#include "optimizer.h"

template<int WIDTH>
class Trainer {
public:

	Trainer(SwiftNetMLP<WIDTH>& network, Loss& loss, Optimizer& optim) {
		m_network = &network;
		m_loss = &loss;
		m_optim = &optim;
	}

	void training_step(DeviceMem<bf16>& input, DeviceMem<float>& output, DeviceMem<float>& target, DeviceMem<bf16>& grads, DeviceMem<float>& losses, const float scale) {
		const int input_size = input.size();

		auto forward = m_network->forward_pass(input, output);

		m_loss->evaluate(m_network->get_queue(), WIDTH, WIDTH, scale, output, target, grads, losses);

		m_network->backward_pass(input, grads, forward);

		m_optim->step(m_network->get_queue(), scale, m_network->m_weights_matrices, m_network->m_weightsT_matrices, grads);

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 10; j++) {
				std::cout << "forward : " << i << " : " << forward.data()[256 * 64 * i + j] << std::endl;
			}
		}

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 10; j++) {
				std::cout << "grads : " << i << " : " << m_network->m_grads_matrices.data()[64 * 64 * i + j] << std::endl;
			}
		}
		forward.free_mem(m_network->get_queue());
	}

	void initialize_params() {
		m_network->initialize_params();
	}

private:

	SwiftNetMLP<WIDTH>* m_network;
	Loss* m_loss;
	Optimizer* m_optim;
};