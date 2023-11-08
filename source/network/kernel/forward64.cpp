
#include "kernel.h"
#include "kernel_helper.h"

// general forward which can do forward and inference, small batchsizes,
// large batchsizes, and all input and output widths and all types
// WIDTH=64
template <typename T, int INPUT_WIDTH, int OUTPUT_WIDTH, Activation activation, bool INFERENCE, bool SMALL>
std::vector<sycl::event> mlp_swift_forward_64(sycl::queue &q, T const *const __restrict__ weights_ptr,
                                              T const *const __restrict__ inputs_ptr,
                                              T *const __restrict__ intermediate_output, const int n_hidden_layers,
                                              const int batch_size, const std::vector<sycl::event> &deps) {
    throw std::invalid_argument("General mlp_swfit_forward not yet implemented");
}

// Specialized forward
template <>
std::vector<sycl::event> mlp_swift_forward_64<bf16, 64, 64, Activation::ReLU, false, false>(
    sycl::queue &q, T const *const __restrict__ weights_ptr, T const *const __restrict__ inputs_ptr,
    T *const __restrict__ intermediate_output, const int n_hidden_layers, const int batch_size,
    const std::vector<sycl::event> &deps) {
    throw std::invalid_argument("General mlp_swfit_forward not yet implemented");
}

// Specialized forward for small batch sizes
template <>
std::vector<sycl::event> mlp_swift_forward_64<bf16, 64, 64, Activation::ReLU, false, true>(
    sycl::queue &q, T const *const __restrict__ weights_ptr, T const *const __restrict__ inputs_ptr,
    T *const __restrict__ intermediate_output, const int n_hidden_layers, const int batch_size,
    const std::vector<sycl::event> &deps) {
    throw std::invalid_argument("General mlp_swfit_forward not yet implemented");
}