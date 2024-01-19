/**
 * @file common_kernel.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Base class for all the kernel implementations.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <sycl/sycl.hpp>
#include <vector>

#include "DeviceMatrix.h"

// abstract kernels class. We derive from this template classes for the various
// different kernels. They are then registered in the SwiftNetMLP?
template <typename T> class Kernels {

  public:
    virtual std::vector<sycl::event> forward_impl(sycl::queue &q, const DeviceMatricesView<T> &weights,
                                                  const DeviceMatrixView<T> &input,
                                                  DeviceMatricesView<T> intermediate_output, const int n_hidden_layers,
                                                  const std::vector<sycl::event> &deps) = 0;

    virtual std::vector<sycl::event> backward_impl(sycl::queue &q, const DeviceMatricesView<T> &weights,
                                                   const DeviceMatrixView<T> &input, DeviceMatricesView<T> output,
                                                   DeviceMatricesView<T> intermediate_backward,
                                                   const DeviceMatricesView<T> &intermediate_forward,
                                                   const int n_hidden_layers, const std::vector<sycl::event> &deps) = 0;

    virtual std::vector<sycl::event> inference_impl(sycl::queue &q, const DeviceMatricesView<T> &weights,
                                                    const DeviceMatrixView<T> &input,
                                                    DeviceMatricesView<T> intermediate_output,
                                                    const int n_hidden_layers,
                                                    const std::vector<sycl::event> &deps) = 0;

    virtual ~Kernels() {}
};