#pragma once
#include <optimizer.h>
#include <vector>

template<typename T>
void sgd_step(nd_item<1> item,
	const uint32_t n_elements,
	const float loss_scale,
	const float learning_rate,
	const float l2_reg,
	float* __restrict__ weights_full_precision,
	T* __restrict__ weights,
	const T* __restrict__ gradients
) {
	int id = item.get_global_id();
	if (id >= n_elements) return;
	const float weight_fp = weights_full_precision[i];
	float gradient = (float)gradients[i] / loss_scale;

	gradient += l2_reg * weight_fp;

	const float new_weight = weight_fp - learning_rate * gradient;

	weights_full_precision[i] = new_weight;
	weights[i] = (T)new_weight;
}



template <typename T>
class SGDOptimizer : public Optimizer<T> {
public:
	SGDOptimizer(const json& params) {
		update_hyperparams(params);
	}
	void step(queue& q, float loss_scale, float* weights_full_precision, T* weights, const T* gradients) override {
		++m_current_step;
		q.submit([&](handler& h) {
			m_nested->step(q, loss_scale, weights_full_precision, weights, gradients);
			h.parallel_for(nd_range<1>(n_elements, n_elements), [=]nd_item it [[intel::reqd_sub_group_size(16)]] {
				average_step(item,m_n_weights,loss_scale,m_learning_rate,m_l2_reg,weights_full_precision,weights,gradients);
				});
			});
		
	}
	void update_hyperparams(const json& params) override {
		if (params.contains("learning_rate")) {
			m_learning_rate = params["learning_rate"];
		}

		if (params.contains("l2_reg")) {
			m_l2_reg = params["l2_reg"];
		}
	}
	int step() {
		return current_step();
	}
private:
	uint32_t m_n_weights;
	uint32_t m_current_step = 0;

	// Hyperparameters
	float m_learning_rate = 1e-3f;
	float m_l2_reg = 1e-8f;
}


