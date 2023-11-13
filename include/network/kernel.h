// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

// This file lists all the inference, forward, backward and fused (forward+backw)
// functions we have.
//
// In general, there should always be one 'general' implementation which
// ignores performance and then specialized implementations which are optimized
// for their use case.

// The netweork forward_impl, inference_impl, backward_impl functions will then
// decide at runtime which one to choose. May do an abstraction around this?
// The netweok *_impl functions may also have template specializations to
// make the choice quicker.

#pragma once

#include <sycl/sycl.hpp>
#include <vector>

#include "common.h"

namespace tinydpcppnn {
namespace kernels {

using bf16 = sycl::ext::oneapi::bfloat16;
using namespace sycl::ext::oneapi::experimental::matrix;

////////////////////////////GENERAL FUNCTIONS WHICH CAN DO EVERYTHING///////////

// Todo: May want to remove some of the template parameters of these functions and
// make them inputs.

// This is the general forward map which also doubles as inference. We use template
// specialization for all the versions
template <typename T, typename Tc, int INPUT_WIDTH, int OUTPUT_WIDTH, Activation activation,
          Activation output_activation, bool INFERENCE, size_t TN>
std::vector<sycl::event> mlp_swift_forward_4(sycl::queue &q, T const *const __restrict__ weights_ptr,
                                             T const *const __restrict__ inputs_ptr,
                                             T *const __restrict__ intermediate_output, const int n_hidden_layers,
                                             const int M, const std::vector<sycl::event> &deps);
} // namespace kernels
} // namespace tinydpcppnn
