// #include <CL/sycl.hpp>
// #include <iostream>
// #include <vector>

// #include "L2.h"
// #include "SwiftNetMLP.h"
// #include "activation.h"
// #include "mkl.h"
// #include "mkl_omp_offload.h"
// // #include "network_with_encodings.h"
// #include "oneapi/mkl.hpp"

// using namespace sycl;
// using namespace sycl::ext::oneapi::experimental::matrix;
// using bf16 = sycl::ext::oneapi::bfloat16;

// #define INPUT_WIDTH 64
// #define OUTPUT_WIDTH 64
// #define HIDDEN_LAYERS 2

// void test_encoding() {
//   // SWIFTNET
//   const int batch_size = 8;
//   const int output_width = OUTPUT_WIDTH;
//   const int m_n_hidden_layers = HIDDEN_LAYERS;

//   GPUMatrix<float> input(INPUT_WIDTH, batch_size);
//   GPUMatrix<bf16> output_float(INPUT_WIDTH, batch_size);

//   input.initialize_constant(0.01);
//   output_float.initialize_constant(1.00);
//   //   std::cout << "In" << std::endl;
//   //   input.print();
//   std::cout << "Out" << std::endl;
//   output_float.print();

//   //   //   // Define the parameters for creating IdentityEncoding
//   //   std::unordered_map<std::string, std::string> encoding = {
//   //       {"n_dims_to_encode", "64"}, {"scale", "1.0"}, {"offset", "0.0"}};
//   //   Encoding<bf16>* network =
//   //       create_encoding<bf16>(INPUT_WIDTH, "Identity", encoding);
//   //   Encoding<bf16>* network = new IdentityEncoding<bf16>(INPUT_WIDTH);

//   //   queue q;
//   //   std::unique_ptr<Context> model_ctx =
//   //       network->forward_impl(&q, input, &output_float);
//   //   std::cout << "Out2" << std::endl;

//   //   output_float.print();
// }
// int main() {
//   test_encoding();
//   return 0;
// }