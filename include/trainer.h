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

	void training_step(DeviceMem<bf16>& input,
		float* forward,
		bf16* act_mem,
		float* act_mem_temp,
		float* A_forward,
		float* B_forward,
		float* C_forward, 
		float* out_inter, 
		float* delta_temp,
		DeviceMem<bf16> loss,
		float* A_backward,
		float* B_backward,
		float* C_backward,
		float* A_backward_last_layer,
		float* B_backward_last_layer,
		float* C_backward_last_layer, 
		float* D_backward_last_layer, 
		float* E_backward_last_layer,
		float* F_backward_last_layer,
		float* A_dgemm,
		float* B_dgemm,
		float* C_dgemm,
		DeviceMem<float>& output, 
		DeviceMem<float>& target,
		DeviceMem<bf16>& grads,
		DeviceMem<float>& losses,
		const float scale,
		const int WIDTH) {
		//const int input_size = input.size();
		//const int batch_size = std::pow(2, 19);


		m_network->get_queue().parallel_for<>(range<1>(input.size()), [=](id<1> idx) {
			forward[idx] = (float)input.data()[idx];
			});

		m_network->forward_pass(input, forward, act_mem, act_mem_temp, A_forward, B_forward, C_forward, output);

		m_loss->evaluate(m_network->get_queue(), WIDTH, WIDTH, scale, output, target, grads, losses);

		m_network->backward_pass(input, 
			grads, 
			out_inter,
			delta_temp, 
			loss,
			A_backward,
			B_backward,
			C_backward,
			A_backward_last_layer,
			B_backward_last_layer,
			C_backward_last_layer,
			D_backward_last_layer,
			E_backward_last_layer, 
			F_backward_last_layer,
			A_dgemm,
			B_dgemm,
			C_dgemm,
			forward);

		m_optim->step(m_network->get_queue(), scale, m_network->m_weights_matrices, m_network->m_weightsT_matrices, m_network->m_grads_matrices, WIDTH);

	}

	void initialize_params() {
		m_network->initialize_params();
	}

private:

	Network* m_network;
	Loss* m_loss;
	Optimizer* m_optim;
};
