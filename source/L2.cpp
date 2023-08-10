#include "L2.h"

// Function to compute L2 loss for an element
void L2_loss(
    id<1> idx,
    const int n_elements,
    const int dims,
    const int stride,
    const float scale,
    float* preds,
    float* targets,
    bf16* grads,
    float* values
) {
    // Calculate intra and inter indices
    const int intra_idx = idx % stride;
    const int inter_idx = idx / stride;

    // Calculate total number of elements
    const int N_total_elements = n_elements * dims / stride;

    // Calculate target index
    const int target_idx = inter_idx * dims + intra_idx;

    // Compute the squared difference between preds and targets
    const float difference = (preds[idx] - targets[target_idx]);
    values[idx] = difference * difference;

    // Compute gradient using bf16 type
    grads[idx] = bf16(((bf16)preds[idx] - (bf16)targets[target_idx]));
}

// Evaluate L2 loss using OpenCL queue
void L2Loss::evaluate(
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

    // Parallel computation using OpenCL
    q.parallel_for<>(range<1>(n_elements), [=](id<1> idx) {
        L2_loss(idx,
            n_elements,
            dims,
            stride,
            scale,
            preds.data(),
            targets.data(),
            grads.data(),
            values.data()
        );
    }).wait();
}
