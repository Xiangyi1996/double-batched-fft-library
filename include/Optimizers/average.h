#pragma once
#include<optimizer.h>
#include<vector>
template<typename T>
void average_step(
	nd_item<1> item,
	const int n_elements,
	const int n_samples,
	const T* weights,
	T* weights_average,
	T* weights_current_sample
) {
	int i = item.get_global_id();
	if (i >= n_elements) return;

	T weight = weights[i];
	weights_average[i] = ((float)weights_average + ((float)weight - (float)weights_current_sample[i]) / n_samples);
	weights_current_sample[i] = weight;
}

template <typename T>
class AverageOptimizer : public Optimizer<T> {
public:
	AverageOptimizer(const json& params) {
		m_nested.reset(create_optimizer<T>(params.value("nested", json::object())));
		update_hyperparams(params);
	}
	
	void step(queue& q ,float loss_scale, float* weights_full_precision, T* weights, const T* gradients) override {
		q.submit([&](handler& h) {
			h.parallel_for(nd_range<1>(n_elements, n_elements), [=]nd_item it [[intel::reqd_sub_group_size(16)]] {
				sgd_step(item,n_elements,weights,m_weights_average.data(),current_sample());
				});
			});
	}
private:
	uint32_t current_sample_idx() const {
		return step() % m_n_samples;
	}

	T* current_sample() const {
		return m_weights_samples.data() + current_sample_idx() * m_n_weights;
	}
	
	void update_hyperparams(const json& params) override {
		if (params.contains("n_samples")) {
			m_n_samples = params["n_samples"];
			if (m_n_weights > 0 || !m_layer_sizes.empty()) {
				allocate(m_n_weights, m_layer_sizes);
			}
		}

		if (params.contains("nested")) {
			m_nested->update_hyperparams(params["nested"]);
		}
	}
}
