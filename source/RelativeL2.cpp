#include "RelativeL2.h"

void Relative_L2_loss(id<1> idx,
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

	const float pred = preds[idx];
	const float difference = (pred - targets[target_idx]);
	const float var = pred * pred + 0.01f;

	values[idx] = difference * difference / var / N_total_elements;

	grads[idx] = bf16(scale * 2 * difference / var / N_total_elements);
}

void RelativeL2Loss::evaluate(
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
		Relative_L2_loss(idx,
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
