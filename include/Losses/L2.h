#pragma once

#include "loss.h"

template<typename T>
void L2_loss(id<1> idx,
	const int n_elements,
	const int dims,
	const int stride,
	const float scale,
	T* preds,
	T* targets,
	T* grads,
	T* values) {
	

	const int intra_idx = idx % stride;
	const int inter_idx = idx / stride;

	const int N_total_elements = n_elements * dims / stride;

	const int target_idx = inter_idx * dims + intra_idx;

	const float difference = (preds[idx] - targets[target_idx]);

	values[idx] = difference * difference / N_total_elements;

	grads[idx] = (scale * 2 * (preds[idx] - targets[target_idx]) / N_total_elements);
	
}

template< typename T>
class L2Loss : public Loss<T> {
public:
	void evaluate(
		const int dims,
		const int stride,
		const float scale,
		std::vector<T>& preds,
		std::vector<T>& targets,
		std::vector<T>& grads,
		std::vector<T>& values
	) const override {

		queue q;

		int n_elements = preds.size();

		T* preds_device = malloc_shared<T>(preds.size(), q);
		T* targets_device = malloc_shared<T>(targets.size(), q);
		T* grads_device = malloc_shared<T>(grads.size(), q);
		T* values_device = malloc_shared<T>(values.size(), q);

		q.memcpy(preds_device, preds.data(), preds.size() * sizeof(T));
		q.memcpy(targets_device, targets.data(), targets.size() * sizeof(T));
		q.wait();

		q.parallel_for<>(range<1>(n_elements),[=](id<1> idx){
			L2_loss<T>(idx,
			n_elements,
			dims,
			stride,
			scale,
			preds_device,
			targets_device,
			grads_device,
			values_device);
		}).wait();

		q.memcpy(grads.data(), grads_device, grads.size() * sizeof(T));
		q.memcpy(values.data(), values_device, values.size() * sizeof(T));
		q.wait();

	}
};
