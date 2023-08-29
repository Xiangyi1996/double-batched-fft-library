#include "L1.h"

void L1_loss(
	id<1> idx,
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

	values[idx] = fabsf(difference) / N_total_elements;

	grads[idx] = bf16(scale * copysignf(1.0f, difference) / N_total_elements);
}

void L1Loss::evaluate(
		queue q,
		const int dims,
		const int stride,
		const float scale,
		DeviceMem<float>& preds,
		DeviceMem<float>& targets,
		DeviceMem<bf16>& grads,
		DeviceMem<float>& values
	) {

		int n_elements = preds.size();

		q.parallel_for<>(range<1>(n_elements), [=](id<1> idx) {
			L1_loss(idx,
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
