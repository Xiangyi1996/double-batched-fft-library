#pragma once
#include "optimizer.h"
#include <vector>

void sgd_step(id<1> idx,
	const int n_elements,
	const float loss_scale,
	const float learning_rate,
	const float l2_reg,
	bf16* weights,
	bf16* gradients
) {
	const bf16 weight = weights[idx];
	float gradient = gradients[idx] / loss_scale;

	gradient += l2_reg * weight;

	const bf16 new_weight = weight - learning_rate * gradient;

	weights[idx] = new_weight;
}

template<int WIDTH>
void sgd_stepT(id<1> idx,
	const int n_elements,
	const float loss_scale,
	const float learning_rate,
	const float l2_reg,
	bf16* weightsT,
	bf16* gradients
) {
	const int i = idx / WIDTH;
	const int j = idx % WIDTH;

	const int T_idx = WIDTH * j + i;

	const bf16 weightT = weightsT[idx];
	float gradient = gradients[T_idx] / loss_scale;

	gradient += l2_reg * weightT;

	const bf16 new_weightT = weightT - learning_rate * gradient;

	weightsT[idx] = new_weightT;
}

template <int WIDTH>
class SGDOptimizer : public Optimizer {
public:

	void step(queue q, float loss_scale, DeviceMem<bf16>& weights, DeviceMem<bf16>& weightsT, DeviceMem<bf16>& gradients) const  override {

		const int n_elements = weights.size();
		float learning_rate = m_learning_rate;
		float l2_reg = m_l2_reg;


		q.parallel_for<>(range<1>(n_elements), [=](id<1> idx) {
			sgd_step(idx, n_elements, loss_scale, learning_rate, l2_reg, weights.data(), gradients.data());
			}).wait();

			q.parallel_for<>(range<1>(n_elements), [=](id<1> idx) {
				sgd_stepT<WIDTH>(idx, n_elements, loss_scale, learning_rate, l2_reg, weightsT.data(), gradients.data());
				}).wait();


	}

	void set_learning_rate(const float learning_rate) {
		m_learning_rate = learning_rate;
	}

private:

	float m_learning_rate = 1e-3f;
	float m_l2_reg = 1e-8f;
};
