#pragma once

// abstract kernels class. We derive from this template classes for the various
// different kernels. They are then registered in the SwiftNetMLP?
template <typename T> class Kernels {

  public:
    virtual std::vector<sycl::event> forward_impl_general(sycl::queue &q, T const *const __restrict__ weights_ptr,
                                                          T const *const __restrict__ inputs_ptr,
                                                          T *const __restrict__ intermediate_output,
                                                          const int n_hidden_layers, const int M,
                                                          const std::vector<sycl::event> &deps) = 0;

    virtual std::vector<sycl::event>
    backward_impl_general(sycl::queue &q, T const *const __restrict__ weights_ptr,
                          T const *const __restrict__ inputs_ptr, T *const __restrict__ output_ptr,
                          T *const __restrict__ intermediate_output, T const *const __restrict__ forward,
                          const int n_hidden_layers, const int M, const std::vector<sycl::event> &deps) = 0;
}