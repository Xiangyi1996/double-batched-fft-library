#pragma once

#include "loss.h"

void cross_entropy_loss(id<1> idx,
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

	const float weight = -targets[target_idx] / N_total_elements;
	const float pred = (float)preds[idx];

	values[idx] = weight * logf(pred);

	grads[idx] = bf16(scale * weight / pred);
}

class CrossEntropyLoss : public Loss {
public:
	void evaluate(
		const int dims,
		const int stride,
		const float scale,
		DeviceMem<float>& preds,
		DeviceMem<float>& targets,
		DeviceMem<bf16>& grads,
		DeviceMem<float>& values
	) const override {
		queue q;

		int n_elements = preds.size();

		q.parallel_for<>(range<1>(n_elements), [=](id<1> idx) {
			cross_entropy_loss(idx,
			n_elements,
			dims,
			stride,
			scale,
			preds.data(),
			targets.data(),
			grads.data(),
			values.data());
			}).wait();
	}
};
