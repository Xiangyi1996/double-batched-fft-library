#pragma once
#include "optimizer.h"
#include <vector>

void adam_step(<id>1 idx,
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
	bf16*  weights,
	const bf16*  gradients,
	float*  first_moments,
	float*  second_moments
) {

	const bf16 weight = weights[idx];
	bf16 gradient = gradients[idx] / loss_scale;

	gradient += l2_reg * weight;

	const float gradient_sq = gradient * gradient;

	float first_moment = first_moments[idx] = beta1 * first_moments[idx] + (1 - beta1) * gradient;
	const float second_moment = second_moments[idx] = beta2 * second_moments[idx] + (1 - beta2) * gradient_sq;

	const float effective_learning_rate = fminf(fmaxf(learning_rate / (sqrtf(second_moment) + epsilon), lower_lr_bound), upper_lr_bound);

	const float decayed_weight = weight_decay(relative_weight_decay * learning_rate, absolute_weight_decay * learning_rate, weight_fp);
	float new_weight = decayed_weight - effective_learning_rate * first_moment;

	if (weight_clipping_magnitude != 0.0f) {
		new_weight = clamp(new_weight, -weight_clipping_magnitude, weight_clipping_magnitude);
	}

	weights[idx] = (bf16)new_weight;
}

void adam_stepT(< id>1 idx,
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
	float* second_moments
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

	const float decayed_weight = weight_decay(relative_weight_decay * learning_rate, absolute_weight_decay * learning_rate, weight_fp);
	float new_weight = decayed_weight - effective_learning_rate * first_moment;

	if (weight_clipping_magnitude != 0.0f) {
		new_weight = clamp(new_weight, -weight_clipping_magnitude, weight_clipping_magnitude);
	}

	weightsT[T_idx] = (bf16)new_weight;
}


template <int WIDTH>
class AdamOptimizer : public Optimizer {
public:

	void step(queue q, const int n_elements,
		const float relative_weight_decay,
		const float absolute_weight_decay,
		const float weight_clipping_magnitude,
		const float loss_scale,
		const float non_matrix_learning_rate_factor,
		const float beta1,
		const float beta2,
		const float epsilon,
		const float lower_lr_bound,
		const float upper_lr_bound,
		DeviceMem<bf16>& weights,
		DeviceMem<bf16>& weightsT, 
		DeviceMem<bf16>& gradients) const  override {

		const int n_elements = weights.size();
		float learning_rate = m_learning_rate;
		float l2_reg = m_l2_reg;


		q.parallel_for<>(range<1>(n_elements), [=](id<1> idx) {
			adam_step(idx, 
			n_elements,
			relative_weight_decay,
			absolute_weight_decay,
			weight_clipping_magnitude,
			loss_scale,
			m_learning_rate,
			non_matrix_learning_rate_factor,
			beta1,
			beta2,
			epsilon,
			lower_lr_bound,
			upper_lr_bound,
			m_l2_reg,
			weights.data(),
			gradients.data(),
			m_first_moments.data(),
			m_second_moments.data());
			}).wait();

		q.parallel_for<>(range<1>(n_elements), [=](id<1> idx) {
			adam_stepT<WIDTH>((idx,
			n_elements,
			relative_weight_decay,
			absolute_weight_decay,
			weight_clipping_magnitude,
			loss_scale,
			m_learning_rate,
			non_matrix_learning_rate_factor,
			beta1,
			beta2,
			epsilon,
			lower_lr_bound,
			upper_lr_bound,
			m_l2_reg,
			weightsT.data(),
			gradients.data(),
			m_first_moments.data(),
			m_second_moments.data());
			}).wait();
	}

	void set_learning_rate(const float learning_rate) {
		m_learning_rate = learning_rate;
	}

private:

	DeviceMem<float> m_first_moments;
	DeviceMem<float> m_second_moments;

	float m_learning_rate = 1e-3f;
	float m_l2_reg = 1e-8f;
};
