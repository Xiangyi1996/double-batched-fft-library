#pragma once
#include <optimizer.h>
#include <vector>

template<typename T>
void lookahead_step(nd_item<1> item,
	const uint32_t n_elements,
	const float alpha,
	float* __restrict__ weights_full_precision,
	T* __restrict__ weights,
	const T* __restrict__ gradients
) {
	int id = item.get_global_id();
	if (id >= n_elements) return;
	
	float new_weight = ((float)weights_lookahead[i] * (1.0f - alpha) + weights_full_precisions[i] * alpha);
	weights_full_precision[i] = new_weight;
	weights_lookahead[i] = weights[i] = (T)new_weight;
}



template <typename T>
class LookaheadOptimizer : public Optimizer<T> {
public:
	LookaheadOptimizer(const json& params) {
		update_hyperparams(params);
	}
	void step(queue& q, float loss_scale, float* weights_full_precision, T* weights, const T* gradients) override {
		int current_step = m_nested->step();
		if (current_step % m_n_steps == 0) {
			q.submit([&](handler& h) {
				m_nested->step(q, loss_scale, weights_full_precision, weights, gradients);
				h.parallel_for(nd_range<1>(n_elements, n_elements), [=]nd_item it [[intel::reqd_sub_group_size(16)]] {
					lookahead_step(item,m_n_weights,loss_scale,m_learning_rate,m_l2_reg,weights_full_precision,weights,gradients);
					});
				});
		}
		m_nested->step(q, loss_scale, weights_full_precision, weights, gradients);

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
	float m_alpha = 0.5f;
	uint32_t m_n_steps = 16;
	
	
}


