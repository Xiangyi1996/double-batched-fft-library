// Include necessary header
#include "cross_entropy.h"

/**
 * Calculate cross-entropy loss and gradients for a single element.
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
void cross_entropy_loss(id<1> idx, const int n_elements, const int dims,
                        const int stride, const float scale, float* preds,
                        float* targets, bf16* grads, float* values) {
  const int intra_idx = idx % stride;
  const int inter_idx = idx / stride;

  const int N_total_elements = n_elements * dims / stride;

  const int target_idx = inter_idx * dims + intra_idx;

  const float weight = -targets[target_idx] / N_total_elements;
  const float pred = (float)preds[idx];

  values[idx] = weight * logf(pred);

  grads[idx] = bf16(scale * weight / pred);
}

/**
 * Evaluate cross-entropy loss and gradients.
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
void CrossEntropyLoss::evaluate(queue q, const int dims, const int stride,
                                const float scale, DeviceMem<float>& preds,
                                DeviceMem<float>& targets,
                                DeviceMem<bf16>& grads,
                                DeviceMem<float>& values) {
  // Get the total number of elements
  int n_elements = preds.size();
  // Perform parallel computation using SYCL
  q.parallel_for<>(range<1>(n_elements), [=](id<1> idx) {
     // Call the cross_entropy_loss function to calculate loss and gradients
     cross_entropy_loss(idx, n_elements, dims, stride, scale, preds.data(),
                        targets.data(), grads.data(), values.data());
   }).wait();
}
