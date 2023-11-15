#pragma once

#include "loss.h"

// We use it like this since then we have the l2_loss function
// available to be used in other kernels
template <typename T>
inline static void l2_loss(const float inv_n_elements, const float loss_scale, const T prediction, const float target,
                           float &value, T &gradient) {

    const float difference = static_cast<float>(prediction) - target;
    value = difference * difference * inv_n_elements;
    gradient = static_cast<T>(loss_scale * 2 * difference * inv_n_elements);
}

template <typename T> class L2Loss : public Loss<T> {
  protected:
    void Kernel(const sycl::queue &q, const size_t n_elements, const float loss_scale,
                T const *const __restrict__ predictions, float const *const __restrict__ targets,
                float *const __restrict__ values, T *const __restrict__ gradients) override {
        const float inv_n_elements = 1.0f / n_elements;
        q.parallel_for<>(range<1>(n_elements), [=](id<1> idx) {
            const int i = idx.get(0);
            l2_loss<T>(inv_n_elements, loss_scale, predictions[i], targets[i], values[i], gradients[i]);
        });
    }
};