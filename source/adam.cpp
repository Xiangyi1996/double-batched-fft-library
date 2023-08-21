#pragma once
#include "adam.h"
#include <vector>

void adam_step(id<1> idx,
	const int n_elements,
	const float relative_weight_decay,
	const float absolute_weight_decay,
	const float weight_clipping_magnitude,
	const float loss_scale,
	float learning_rate,
	const float non_matrix_learning_rate_factor,
	const float beta1,
	const float beta2,
	const float epsilon,
	const float lower_lr_bound,
	const float upper_lr_bound,
	const float l2_reg,
	bf16* weights,
	const bf16* gradients,
	float* first_moments,
	float* second_moments,
	int WIDTH
) {

	const bf16 weight = weights[idx];
	bf16 gradient = gradients[idx] / loss_scale;

	gradient += l2_reg * weight;

	const float gradient_sq = gradient * gradient;

	float first_moment = first_moments[idx] = beta1 * first_moments[idx] + (1 - beta1) * gradient;
	const float second_moment = second_moments[idx] = beta2 * second_moments[idx] + (1 - beta2) * gradient_sq;

	const float effective_learning_rate = fminf(fmaxf(learning_rate / (sqrtf(second_moment) + epsilon), lower_lr_bound), upper_lr_bound);

	float new_weight = effective_learning_rate * first_moment;

	if (weight_clipping_magnitude != 0.0f) {
		new_weight = clamp(new_weight, -weight_clipping_magnitude, weight_clipping_magnitude);
	}

	weights[idx] = (bf16)new_weight;
}

void adam_stepT(id<1> idx,
	const int n_elements,
	const float relative_weight_decay,
	const float absolute_weight_decay,
	const float weight_clipping_magnitude,
	const float loss_scale,
	float learning_rate,
	const float non_matrix_learning_rate_factor,
	const float beta1,
	const float beta2,
	const float epsilon,
	const float lower_lr_bound,
	const float upper_lr_bound,
	const float l2_reg,
	bf16* weightsT,
	const bf16* gradients,
	float* first_moments,
	float* second_moments,
	int WIDTH
) {
	const int i = idx / WIDTH;
	const int j = idx % WIDTH;

	const int T_idx = WIDTH * j + i;

	const bf16 weight = weightsT[T_idx];
	bf16 gradient = gradients[T_idx] / loss_scale;

	gradient += l2_reg * weight;

	const float gradient_sq = gradient * gradient;

	float first_moment = first_moments[T_idx] = beta1 * first_moments[T_idx] + (1 - beta1) * gradient;
	const float second_moment = second_moments[T_idx] = beta2 * second_moments[T_idx] + (1 - beta2) * gradient_sq;

	const float effective_learning_rate = fminf(fmaxf(learning_rate / (sqrtf(second_moment) + epsilon), lower_lr_bound), upper_lr_bound);

	float new_weight = effective_learning_rate * first_moment;

	if (weight_clipping_magnitude != 0.0f) {
		new_weight = clamp(new_weight, -weight_clipping_magnitude, weight_clipping_magnitude);
	}

	weightsT[T_idx] = (bf16)new_weight;
}


void AdamOptimizer::step(queue q, float loss_scale, DeviceMem<bf16>& weights, DeviceMem<bf16>& weightsT, DeviceMem<bf16>& gradients, int WIDTH) {

	const int n_elements = weights.size();
	float learning_rate = m_learning_rate;
	float l2_reg = m_l2_reg;
	const float relative_weight_decay = 0.01f;
	const float absolute_weight_decay = 0.01f;
	const float weight_clipping_magnitude = 0.01f;
	const float non_matrix_learning_rate_factor = 0.01f;
	const float beta1 = 0.9f;
	const float beta2 = 0.99f;
	const float epsilon = 0.01f;
	const float lower_lr_bound = 0.0001f;
	const float upper_lr_bound = 0.1f;

	auto first_moment = m_first_moments.data();
	auto second_moment = m_second_moments.data();


	q.parallel_for<>(range<1>(n_elements), [=](id<1> idx) {
		adam_step(idx,
		n_elements,
		relative_weight_decay,
		absolute_weight_decay,
		weight_clipping_magnitude,
		loss_scale,
		learning_rate,
		non_matrix_learning_rate_factor,
		beta1,
		beta2,
		epsilon,
		lower_lr_bound,
		upper_lr_bound,
		l2_reg,
		weights.data(),
		gradients.data(),
		first_moment,
		second_moment,
		WIDTH);
		}).wait();

		q.parallel_for<>(range<1>(n_elements), [=](id<1> idx) {
			adam_stepT(idx,
			n_elements,
			relative_weight_decay,
			absolute_weight_decay,
			weight_clipping_magnitude,
			loss_scale,
			learning_rate,
			non_matrix_learning_rate_factor,
			beta1,
			beta2,
			epsilon,
			lower_lr_bound,
			upper_lr_bound,
			l2_reg,
			weightsT.data(),
			gradients.data(),
			first_moment,
			second_moment,
			WIDTH);
			}).wait();
}

void AdamOptimizer::set_learning_rate(const float learning_rate) {
	m_learning_rate = learning_rate;
}
