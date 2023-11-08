#pragma once

#include "loss.h"

template <typename T> class RelativeL2Loss : public Loss<T> {
  protected:
    void Kernel(const sycl::queue &q, const size_t n_elements, const float loss_scale,
                T const *const __restrict__ predictions, float const *const __restrict__ targets,
                float *const __restrict__ values, T *const __restrict__ gradients) override {
        throw std::invalid_argument("Relative l2 loss not yet implemented.");
    }
};
