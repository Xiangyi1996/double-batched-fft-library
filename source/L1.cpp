// Include necessary headers
#include "L1.h"

/**
 * Calculate L1 loss and gradients for a single element.
 *
 * @param idx Index of the element to process.
 * @param n_elements Total number of elements.
 * @param dims Number of dimensions.
 * @param stride Stride value for indexing.
 * @param scale Scaling factor for gradients.
 * @param preds Pointer to predicted values.
 * @param targets Pointer to target values.
 * @param grads Pointer to gradient values (bf16 type).
 * @param values Pointer to store loss values.
 */
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

	// Calculate intra and inter indices
	const int intra_idx = idx % stride;
	const int inter_idx = idx / stride;

	// Calculate target index
	const int N_total_elements = n_elements * dims / stride;
	const int target_idx = inter_idx * dims + intra_idx;

	// Calculate difference between predicted and target values
	const float difference = (preds[idx] - targets[target_idx]);

	// Calculate loss value and store in values array
	values[idx] = fabsf(difference) / N_total_elements;

	// Calculate and store gradient value
	grads[idx] = bf16(scale * copysignf(1.0f, difference) / N_total_elements);
}


/**
 * Evaluate L1 loss and gradients.
 *
 * @param q SYCL queue for parallel computation.
 * @param dims Number of dimensions.
 * @param stride Stride value for indexing.
 * @param scale Scaling factor for gradients.
 * @param preds Predicted values (DeviceMem<float>).
 * @param targets Target values (DeviceMem<float>).
 * @param grads Gradient values (DeviceMem<bf16>).
 * @param values Array to store loss values (DeviceMem<float>).
 */
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
	// Get the total number of elements
	int n_elements = preds.size();
		// Perform parallel computation using SYCL
	q.parallel_for<>(range<1>(n_elements), [=](id<1> idx) {
		// Call the L1_loss function to calculate loss and gradients
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
