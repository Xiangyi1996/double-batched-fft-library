// #include <CL/sycl.hpp>
// #include <iostream>
// #include <vector>

// #include "L2.h"
// #include "SwiftNetMLP.h"
// #include "activation.h"
// #include "mkl.h"
// #include "mkl_omp_offload.h"
// #include "network_with_encodings.h"
// #include "oneapi/mkl.hpp"

// using namespace sycl;
// using namespace sycl::ext::oneapi::experimental::matrix;
// using bf16 = sycl::ext::oneapi::bfloat16;

// #define INPUT_WIDTH 64
// #define OUTPUT_WIDTH 64
// #define HIDDEN_LAYERS 2

// void test_network_with_encoding() {
//   // SWIFTNET
//   const int batch_size = 64;
//   const int output_width = OUTPUT_WIDTH;
//   const int m_n_hidden_layers = HIDDEN_LAYERS;

//   GPUMatrix<float> input(INPUT_WIDTH, batch_size);
//   input.initialize_constant(0.01f);
//   DeviceMem<float>* output;

//   // Define the parameters for creating IdentityEncoding
//   std::unordered_map<std::string, std::string> encoding = {
//       {"n_dims_to_encode", "64"}, {"scale", "1.0"}, {"offset", "0.0"}};

//   NetworkWithEncoding network = NetworkWithEncoding(
//       INPUT_WIDTH, OUTPUT_WIDTH, m_n_hidden_layers, Activation::ReLU,
//       Activation::None, batch_size, "Identity", encoding);
//   network.initialize_params(1);
//   output = network.forward_pass(input, 0);

//   std::vector<float> out(batch_size * (OUTPUT_WIDTH));
//   output->copy_to_host(out, network.get_queue());

//   for (int j = 0; j < batch_size * OUTPUT_WIDTH; j++) {
//     std::cout << out[j] << ", ";
//   }
//   std::cout << std::endl;

//   network.free_memory();
// }

// int main() {
//   test_network_with_encoding();
//   return 0;
// }