#pragma once

#include "loss.h"

void Relative_L1_loss(id<1> idx,
	const int n_elements,
	const int dims,
	const int stride,
	const float scale,
	float* preds,
	float* targets,
	bf16* grads,
	float* values) {

	const int intra_idx = idx % stride;
	const int inter_idx = idx / stride;

	const int N_total_elements = n_elements * dims / stride;

	const int target_idx = inter_idx * dims + intra_idx;

	const float difference = (preds[idx] - targets[target_idx]);
	const float norm = fabsf(preds[idx]) + 0.01f;

	values[idx] = fabsf(difference) / norm / N_total_elements;

	grads[idx] = bf16(scale * copysignf(1.0f, difference) / norm / N_total_elements);
}

class RelativeL1Loss : public Loss {
public:
	void evaluate(
		const int dims,
		const int stride,
		const float scale,
		std::vector<float>& preds,
		std::vector<float>& targets,
		std::vector<bf16>& grads,
		std::vector<float>& values
	) const override {
		queue q;

		int n_elements = preds.size();

		float* preds_device = malloc_shared<float>(preds.size(), q);
		float* targets_device = malloc_shared<float>(targets.size(), q);
		bf16* grads_device = malloc_shared<bf16>(grads.size(), q);
		float* values_device = malloc_shared<float>(values.size(), q);

		q.memcpy(preds_device, preds.data(), preds.size() * sizeof(float));
		q.memcpy(targets_device, targets.data(), targets.size() * sizeof(float));
		q.wait();

		q.parallel_for<>(range<1>(n_elements), [=](id<1> idx) {
			Relative_L1_loss(idx,
			n_elements,
			dims,
			stride,
			scale,
			preds_device,
			targets_device,
			grads_device,
			values_device);
			}).wait();

			q.memcpy(grads.data(), grads_device, grads.size() * sizeof(bf16));
			q.memcpy(values.data(), values_device, values.size() * sizeof(float));
			q.wait();
	}
};
