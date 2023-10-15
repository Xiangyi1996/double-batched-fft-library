/* #include <CL/sycl.hpp>
#include <iostream>
#include <vector>

#include "L2.h"
#include "SwiftNetMLP.h"
#include "activation.h"
#include "mkl.h"
#include "mkl_omp_offload.h"
// #include "network_with_encodings.h"
#include "Encodings/spherical_harmonics.h"
#include "encoding_factory.h"
#include "oneapi/mkl.hpp"
using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
using bf16 = sycl::ext::oneapi::bfloat16;

#define INPUT_WIDTH 3
#define OUTPUT_WIDTH 20
#define DEGREE 4

void test_encoding() {
  // SWIFTNET
  const int batch_size = 2;
  queue q;
  DeviceMem<float> input_float(INPUT_WIDTH * batch_size, q);
  input_float.initialize_arange(q);
  GPUMatrix<float> input(input_float.data(), INPUT_WIDTH, batch_size);

  GPUMatrix<bf16> output_float(OUTPUT_WIDTH, batch_size);

  output_float.initialize_constant(0.00f);

  std::cout << "In" << std::endl;
  input.print();
  std::cout << "Out" << std::endl;
  output_float.print();

  // Define the parameters for creating IdentityEncoding
  // std::unordered_map<std::string, std::string> encoding = {
  //     {"n_dims_to_encode", std::to_string(INPUT_WIDTH)},
  //     {"scale", "1.0"},
  //     {"offset", "0.0"}};
  //     Encoding<bf16>* network =
  //         create_encoding<bf16>(INPUT_WIDTH, "Identity", encoding);
  //   Encoding<bf16>* network =
  //       new SphericalHarmonicsEncoding<bf16>(DEGREE, INPUT_WIDTH);
  std::unordered_map<std::string, std::string> encoding = {
      {"n_dims_to_encode", std::to_string(INPUT_WIDTH)},
      {"degree", std::to_string(DEGREE)}};
  Encoding<bf16>* network =
      create_encoding<bf16>(INPUT_WIDTH, "SphericalHarmonics", encoding);
  network->set_padded_output_width(OUTPUT_WIDTH);

  std::unique_ptr<Context> model_ctx =
      network->forward_impl(&q, input, &output_float);
  std::cout << "out width: " << network->output_width() << std::endl;
  std::cout << "Out2" << std::endl;
  output_float.print();
}
int main() {
  test_encoding();
  return 0;
}
 */