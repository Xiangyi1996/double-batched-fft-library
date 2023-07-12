#pragma once

#include "loss.h"

void L2_loss(id<1> idx,
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

	values[idx] = difference * difference / N_total_elements;

	grads[idx] = bf16(scale * 2 * (preds[idx] - targets[target_idx]) );
}

class L2Loss : public Loss {
public:
	void evaluate(
		queue q,
		const int dims,
		const int stride,
		const float scale,
		DeviceMem<float>& preds,
		DeviceMem<float>& targets,
		DeviceMem<bf16>& grads,
		DeviceMem<float>& values
	) const override {

		int n_elements = preds.size();

		q.parallel_for<>(range<1>(n_elements), [=](id<1> idx) {
			L2_loss(idx,
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
