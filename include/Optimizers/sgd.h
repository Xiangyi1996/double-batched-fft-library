#pragma once
#include "optimizer.h"
#include <vector>

template<typename T>
void sgd_step(id<1> idx,
	const int n_elements,
	const float loss_scale,
	const float learning_rate,
	const float l2_reg,
	bf16* weights,
	T* gradients
) {

	const bf16 weight = weights[idx];
	float gradient = gradients[idx] / loss_scale;

	gradient += l2_reg * weight;

	const bf16 new_weight = weight - learning_rate * gradient;

	weights[idx] = new_weight;

}

template<typename T,int WIDTH>
void sgd_stepT(id<1> idx,
	const int n_elements,
	const float loss_scale,
	const float learning_rate,
	const float l2_reg,
	bf16* weightsT,
	T* gradients
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



template <typename T, int WIDTH>
class SGDOptimizer : public Optimizer<T, WIDTH> {
public:

	//SGDOptimizer(float learning_rate, float l2_reg) {
	//	m_learning_rate = learning_rate;
	//	m_l2_reg = l2_reg;
	//}

	//SGDOptimizer() {
	//	m_learning_rate = 1e-3f;
	//	m_l2_reg = 1e-8f;
	//}

	void step(float loss_scale, std::vector<bf16>& weights, std::vector<bf16>& weightsT, std::vector<T>& gradients) const  override {
		queue q;

		const int n_elements = weights.size();
		float learning_rate = m_learning_rate;
		float l2_reg = m_l2_reg;

		bf16* weights_device = malloc_shared<bf16>(weights.size(), q);
		bf16* weightsT_device = malloc_shared<bf16>(weightsT.size(), q);
		T* gradients_device = malloc_shared<T>(gradients.size(), q);

		q.memcpy(weights_device, weights.data(), weights.size() * sizeof(bf16));
		q.memcpy(weightsT_device, weightsT.data(), weightsT.size() * sizeof(bf16));
		q.memcpy(gradients_device, gradients.data(), gradients.size() * sizeof(T));
		q.wait();

		//std::cout << "grad " << gradients_device[0] << std::endl;
		//std::cout << "weights_device " << weights_device[0] << std::endl;
		//std::cout << "weightsT_device " << weightsT_device[0] << std::endl;

		q.parallel_for<>(range<1>(n_elements), [=](id<1> idx) {
			sgd_step(idx, n_elements, loss_scale, learning_rate, l2_reg, weights_device, gradients_device);
			}).wait();

		q.parallel_for<>(range<1>(n_elements), [=](id<1> idx) {
			sgd_stepT<T, WIDTH>(idx, n_elements, loss_scale, learning_rate, l2_reg, weightsT_device, gradients_device);
			}).wait();

		q.memcpy(weights.data(), weights_device, weights.size() * sizeof(bf16));
		q.memcpy(weightsT.data(), weightsT_device, weightsT.size() * sizeof(bf16));
		q.wait();
		//std::cout << "weights " << weights[0] << std::endl;


	}
private:


	float m_learning_rate = 1.0f;
	float m_l2_reg = 1e-8f;
};


