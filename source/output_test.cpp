// #include <CL/sycl.hpp>
// #include <iostream>
// #include <vector>

// #include "L2.h"
// #include "SwiftNetMLP.h"
// #include "activation.h"
// #include "common.h"
// #include "mkl.h"
// #include "mkl_omp_offload.h"
// #include "oneapi/mkl.hpp"
// #include "sgd.h"
// #include "trainer.h"
// // #include "config.h"

// using namespace sycl;
// using namespace sycl::ext::oneapi::experimental::matrix;
// using bf16 = sycl::ext::oneapi::bfloat16;

// #define TM 8
// #define TK 16
// #define TN 8

// #define SG_SIZE 8
// #define WG_SIZE 8 * SG_SIZE
// #define BATCH_CHUNK 64

// void test_exactitude() {
//   // SWIFTNET
//   const int batch_size = 64;
//   const int input_width = 64;
//   const int output_width = 64;
//   const int WIDTH = 64;
//   const int layer_length = WIDTH * batch_size;
//   const size_t alignment = 4096;

//   queue q = queue();

//   DeviceMem<bf16> inputs = DeviceMem<bf16>(batch_size * WIDTH, q);
//   DeviceMem<float> output = DeviceMem<float>(batch_size * output_width, q);

//   float* forward =
//       malloc_device<float>(batch_size * (WIDTH + output_width + WIDTH * 2),
//       q);

//   auto A_forward =
//       sycl::aligned_alloc_device<float>(alignment, layer_length, q);
//   auto B_forward =
//       sycl::aligned_alloc_device<float>(alignment, output_width * 64, q);
//   auto C_forward = sycl::aligned_alloc_device<float>(
//       alignment, output_width * batch_size, q);

//   SwiftNetMLP<64> network =
//       SwiftNetMLP<64>(q, input_width, output_width, 2, Activation::None,
//                       Activation::None, batch_size);

//   inputs.initialize_test_input(q);
//   output.initialize_constant(0.0f, q);

//   network.initialize_params();
//   network.forward_pass(inputs, forward, A_forward, B_forward, C_forward,
//                        output);

//   std::vector<float> output_vec(batch_size * output_width);
//   output.copy_to_host(output_vec, q);

//   std::cout << "Output" << std::endl;
//   for (const float& element : output_vec) {
//     std::cout << element << " ";
//   }
//   std::cout << std::endl;
//   inputs.free_mem(q);
//   output.free_mem(q);
//   free(A_forward, q);
//   free(B_forward, q);
//   free(C_forward, q);
// }
// int main() {
//   test_exactitude();
//   return 0;
// }