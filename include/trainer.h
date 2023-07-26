#pragma once

#include "SwiftNetMLP.h"
#include "L2.h"
#include "DeviceMem.h"
#include "optimizer.h"
#include "Network.h"

class Trainer {
public:

	Trainer(Network& network, Loss& loss, Optimizer& optim) {
		m_network = &network;
		m_loss = &loss;
		m_optim = &optim;
	}

	void training_step(DeviceMem<bf16>& input, float* forward,  DeviceMem<float>& output, DeviceMem<float>& target, DeviceMem<bf16>& grads, DeviceMem<float>& losses, const float scale, const int WIDTH) {
		//const int input_size = input.size();
		//const int batch_size = std::pow(2, 19);
		m_network->forward_pass(input, forward, output);

		m_loss->evaluate(m_network->get_queue(), WIDTH, WIDTH, scale, output, target, grads, losses);

		m_network->backward_pass(input, grads, forward);

		m_optim->step(m_network->get_queue(), scale, m_network->m_weights_matrices, m_network->m_weightsT_matrices, m_network->m_grads_matrices, WIDTH);

		/*for (int i = 0; i < 3; i++) {
			for (int j = 64; j < 74 ; j++) {
				std::cout << "forward : " << i << " : " << forward.data()[64 * batch_size * i + 64*j] << std::endl;
			}
		}
		for (int j = 0; j < 10; j++) {
			std::cout << "forward : " << 3 << " : " << forward.data()[64 * batch_size * 3 + 128 * j] << std::endl;
		}
		
		
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 10; j++) {
				std::cout << "grads : " << i << " : " << m_network->m_grads_matrices.data()[64 * 64 * i + j] << std::endl;
			}
		}
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 10; j++) {
				std::cout << "weight : " << i << " : " << m_network->m_weights_matrices.data()[64 * 64 * i + 64*j] << std::endl;
			}
		}*/
		std::vector<float> data = std::vector<float>(std::pow(2, 17) * (WIDTH + 64 + WIDTH * 4));
		m_network->get_queue().memcpy(data.data(), forward, std::pow(2, 17) * (WIDTH + 64 + WIDTH * 4) * sizeof(float));
		m_network->get_queue().wait();

		for (int i = 0; i < 3; i++) {
			for (int j = 64; j < 74 ; j++) {
				std::cout << "forward : " << i << " : " << data[64 * std::pow(2, 17) * i + 64*j] << std::endl;
			}
		}
		for (int j = 0; j < 10; j++) {
			std::cout << "forward : " << 3 << " : " << data[64 * std::pow(2, 17) * 3 + 128 * j] << std::endl;
		}

	}

	void initialize_params() {
		m_network->initialize_params();
	}

private:

	Network* m_network;
	Loss* m_loss;
	Optimizer* m_optim;
};
