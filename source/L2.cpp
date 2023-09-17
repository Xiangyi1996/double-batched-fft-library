#include "L2.h"

/**
 * Calculates L2 loss and gradients for each element in the input.
 *
 * This function computes the L2 loss and gradients between predicted values
 * and target values for each element in the input. It calculates the squared
 * difference between predicted and target values, and computes gradients using
 * the bf16 type.
 *
 * @param idx          The current index being processed.
 * @param n_elements   The number of elements in the batch.
 * @param dims         The total number of dimensions per element.
 * @param stride       The step size between consecutive elements.
 * @param scale        A scaling factor for normalization (unused).
 * @param preds        An array of predicted values.
 * @param targets      An array of target values.
 * @param grads        An array to store gradients.
 * @param values       An array to store squared differences.
 */
void L2_loss(id<1> idx, const int n_elements, const int dims, const int stride,
             const float scale, float* preds, float* targets, bf16* grads,
             float* values) {
  // Calculate intra and inter indices
  const int intra_idx = idx % stride;  // Index within each element
  const int inter_idx = idx / stride;  // Index of the element

  // Calculate total number of elements
  const int N_total_elements = n_elements * dims / stride;

  // Calculate target index
  const int target_idx = inter_idx * dims + intra_idx;

  // Compute the squared difference between preds and targets
  const float difference = (preds[idx] - targets[target_idx]);
  values[idx] = difference * difference;

  // Compute gradient using bf16 type
  grads[idx] = bf16(2 * ((bf16)preds[idx] - (bf16)targets[target_idx]));
}

/**
 * Evaluates L2 loss and gradients in parallel using OpenCL.
 *
 * This method calculates L2 loss and gradients for each element in parallel
 * using OpenCL. It uses the provided L2_loss function to perform the
 * calculations for each element.
 *
 * @param q          The OpenCL queue for parallel execution.
 * @param dims       The total number of dimensions per element.
 * @param stride     The step size between consecutive elements.
 * @param scale      A scaling factor for normalization (unused).
 * @param preds      The predicted values for each element.
 * @param targets    The target values for each element.
 * @param grads      An array to store gradients.
 * @param values     An array to store squared differences.
 */
void L2Loss::evaluate(queue q, const int dims, const int stride,
                      const float scale, DeviceMem<float>& preds,
                      DeviceMem<float>& targets, DeviceMem<bf16>& grads,
                      DeviceMem<float>& values) {
  // Get the total number of elements
  int n_elements = preds.size();

  // Parallel computation using OpenCL
  q.parallel_for<>(range<1>(n_elements), [=](id<1> idx) {
     L2_loss(idx, n_elements, dims, stride, scale, preds.data(), targets.data(),
             grads.data(), values.data());
   }).wait();
}
