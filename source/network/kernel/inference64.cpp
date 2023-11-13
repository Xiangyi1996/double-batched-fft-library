
#include "kernel.h"
#include "kernel_helper.h"

// // Specialized inference
// template <>
// std::vector<sycl::event> mlp_swift_forward_64<bf16, 64, 64, Activation::ReLU, true, false>(
//     sycl::queue &q, T const *const __restrict__ weights_ptr, T const *const __restrict__ inputs_ptr,
//     T *const __restrict__ intermediate_output, const int n_hidden_layers, const int batch_size,
//     const std::vector<sycl::event> &deps) {
//     throw std::invalid_argument("General mlp_swfit_forward not yet implemented");
// }

// // Specialized inference
// template <>
// std::vector<sycl::event> mlp_swift_forward_64<bf16, 64, 64, Activation::ReLU, true, true>(
//     sycl::queue &q, T const *const __restrict__ weights_ptr, T const *const __restrict__ inputs_ptr,
//     T *const __restrict__ intermediate_output, const int n_hidden_layers, const int batch_size,
//     const std::vector<sycl::event> &deps) {
//     throw std::invalid_argument("General mlp_swfit_forward not yet implemented");
// }
