// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "DeviceMem.h"

template <typename T> class Loss {
  public:
    void evaluate(const sycl::queue &q, const float loss_scale, const DeviceMem<T> &predictions,
                  const DeviceMem<float> &targets, DeviceMem<float> &values, DeviceMem<T> &gradients) {
        SanityCheck(loss_scale, predictions, targets, values, gradients);
        Kernel(q, predictions.size(), loss_scale, predictions.data(), targets.data(), values.data(), gradients.data());
    }

  protected:
    void SanityCheck(const float loss_scale, const DeviceMem<T> &predictions, const DeviceMem<float> &targets,
                     DeviceMem<float> &values, DeviceMem<T> &gradients) {
        // Check if input dimensions match and if loss_scale is not 0
        const int n_elements = predictions.size();
        assert(values.size() == n_elements);
        assert(gradients.size() == n_elements);
        assert(loss_scale != 0.0f);
        assert(targets.size() == n_elements);
    }

  protected:
    virtual void Kernel(const sycl::queue &q, const size_t n_elements, const float loss_scale,
                        T const *const __restrict__ predictions, float const *const __restrict__ targets,
                        float *const __restrict__ values, T *const __restrict__ gradients) = 0;
};
