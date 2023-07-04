#pragma once
#pragma once

#include "loss.h"

template<typename T>
void Relative_L2_loss(nd_item<1> item,
	const int n_elements,
	const int dims,
	const int stride,
	const float scale,
	const T* preds,
	const float* targets,
	T* grads,
	float* values) {

	// get the index of the element considered by the kernel
	const int idx = item.get_global_linear_id();

	const int intra_idx = idx % stride;
	const int inter_idx = idx / stride;

	const int N_total_elements = n_elements * dims / stride;

	const int target_idx = inter_idx * dims + intra_odx;

	const float pred = (float)preds[i];
	const float target = targets[target_idx];
	const float difference = pred - target;
	const float var = pred * pred + 0.01f;

	values[i] = difference * difference / var / N_total_elements;

	grads[i] = (T)(loss_scale * 2 * difference / var / N_total_elements);

}

template< typename T>
class RelativeL2Loss : public Loss<T> {
public:
	void evaluate(
		const int dims,
		const int stride,
		const float scale,
		std::vector<T> preds,
		std::vector<float> targets,
		std::vector<T> grads,
		std::vector<float> values
	) const override {

		queue q;
		int n_elements = preds.size();

		T* preds_device = malloc_shared<T>(preds.size(), q);
		T* targets_device = malloc_shared<T>(targets.size(), q);
		T* grads_device = malloc_shared<T>(grads.size(), q);
		T* values_device = malloc_shared<T>(values.size(), q);

		q.memcpy(preds_device, preds.data(), preds.size() * sizeof(T));
		q.memcpy(targets_device, targets.data(), targets.size() * sizeof(T));
		q.memcpy(grads_device, grads.data(), grads.size() * sizeof(T));
		q.memcpy(values_device, values.data(), values.size() * sizeof(T));


		q.submit([&](handler& h) {
			h.parallel_for<>(nd_range<1>(128, 128), [=](nd_item<1> it) [[intel::reqd_sub_group_size(SG_SIZE)]] {
				L2_loss<T>(it,
					n_elements,
					dims,
					stride,
					scale,
					preds_device,
					targets_device,
					grads_device,
					values_device);
				});
			});
	}
};				