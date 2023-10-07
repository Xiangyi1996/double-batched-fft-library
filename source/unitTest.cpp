// #include <iostream>
// #include <sycl/sycl.hpp>
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

// void test_encoding() {
//   // REFERENCE

//   L2Loss loss;
//   SGDOptimizer optim = SGDOptimizer(64, 2, 1e-3f, 1e-8f);

//   // SWIFTNET
//   const int batch_size = 64;
//   const int output_width = 2;
//   const int input_width = 64;

//   queue q = queue();

//   //   DeviceMem<float> input_f = DeviceMem<float>(batch_size * input_width,
//   //   q);
//   //   DeviceMem<float> output_f = DeviceMem<float>(batch_size *
//   //   output_width,
//   //   q); DeviceMem<float> target_f = DeviceMem<float>(batch_size *
//   //   output_width, q);

//   // Encoding<float>* encoding = create_grid_encoding_templated_2<float>(2,
//   // nullptr);
//   Encoding<float>* encoding = new IdentityEncoding<float>(input_width);
//   EncodingTrainer<float, float, float> train(*encoding, loss, optim);

//   train.initialize_params();  // encoding has none

//   GPUMatrix<float> input_float(input_width, batch_size);
//   GPUMatrix<float> output_float(input_width, batch_size);
//   GPUMatrix<float> target_float(input_width, batch_size);

//   input_float.initialize_constant(0.01f);
//   output_float.initialize_constant(0.00f);
//   target_float.initialize_constant(1.0f);
//   std::cout << "Performing one forward step" << std::endl;

//   std::cout << "Input before fwd pass" << std::endl;
//   input_float.print();
//   std::cout << "Output before fwd pass" << std::endl;
//   output_float.print();
//   std::cout << "Target before fwd pass" << std::endl;
//   target_float.print();

//   std::unique_ptr<Context> model_ctx =
//       encoding->forward_impl(&q, input_float, &output_float);

//   std::cout << "Input after fwd pass" << std::endl;
//   input_float.print();
//   std::cout << "Output after fwd pass" << std::endl;
//   output_float.print();
//   std::cout << "Target after fwd pass" << std::endl;
//   target_float.print();

//   //   std::cout << "Training step" << std::endl;

//   // dummy matrices

//   //   DeviceMem<float> dli_f = DeviceMem<float>(batch_size * input_width,
//   //   q);
//   //   DeviceMem<float> dlo_f = DeviceMem<float>(batch_size * output_width,
//   //   q);
//   //   GPUMatrix<float> dl_input(dli_f.data(), batch_size * input_width,
//   //                             batch_size * output_width);
//   //   GPUMatrix<float> dl_output(dlo_f.data(), batch_size * input_width,
//   //                              batch_size * output_width);
//   //   auto ctx = train.training_step(input_float, target_float);

//   //   std::cout << "Input after fwd pass" << std::endl;
//   //   input_float.print();
//   //   std::cout << "Output after fwd pass" << std::endl;
//   //   output_float.print();
//   //   std::cout << "Target after fwd pass" << std::endl;
//   //   target_float.print();

//   //   std::cout << "Training step done" << std::endl;

//   //   std::vector<float> output_train(output_float.cols() *
//   //   output_float.rows()); if (output_train.empty() || !ctx->output.data())
//   //   {
//   //     std::cerr << "Error: No training data! Empty..." << std::endl;
//   //     exit(0);
//   //   }

//   //   std::cout << "Performed encoded training step" << std::endl;
// }

// int main() {
//   test_encoding();
//   return 0;
// }