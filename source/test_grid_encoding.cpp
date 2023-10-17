/* #include <CL/sycl.hpp>
#include <iostream>
#include <vector>

#include "L2.h"
#include "SwiftNetMLP.h"
#include "activation.h"
#include "mkl.h"
#include "mkl_omp_offload.h"
// #include "network_with_encodings.h"
// #include "Encodings/spherical_harmonics.h"
#include "Encodings/grid.h"
#include "encoding_factory.h"
#include "oneapi/mkl.hpp"
using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
using bf16 = sycl::ext::oneapi::bfloat16;

#define INPUT_WIDTH 3
#define OUTPUT_WIDTH 3
#define DEGREE 4

void test_encoding() {
  // SWIFTNET
  const int batch_size = 6;
  queue q;
  DeviceMem<float> input_float(INPUT_WIDTH * batch_size, q);
  //   input_float.initialize_arange(q);
  input_float.initialize_constant(1.0f, q);
  GPUMatrix<float> input(input_float.data(), INPUT_WIDTH, batch_size);

  GPUMatrix<float> output_float(32, batch_size);

  output_float.initialize_constant(0.001f);

  std::cout << "In" << std::endl;
  input.print();
  std::cout << "Out" << std::endl;
  output_float.print();

  // Define the parameters for creating IdentityEncoding
  std::unordered_map<std::string, std::string> encoding = {
      {"n_dims_to_encode", std::to_string(INPUT_WIDTH)},
      {"otype", "Grid"},
      {"type", "Hash"},
      {"n_levels", "16"},
      {"n_features_per_level", "2"},
      {"log2_hashmap_size", "19"},
      {"base_resolution", "16"},
      {"per_level_scale", "2.0"},
  };

  //   json encoding_json = {
  //       {"n_dims_to_encode", std::to_string(INPUT_WIDTH)},
  //       {"otype", "Grid"},
  //       {"type", "Hash"},
  //       {"n_levels", 16},
  //       {"n_features_per_level", 2},
  //       {"log2_hashmap_size", 19},
  //       {"base_resolution", 16},
  //       {"per_level_scale", 2.0},
  //   };

  //   GridEncoding<float>* network =
  //       create_grid_encoding<float>(INPUT_WIDTH, encoding_json);
  Encoding<float>* network = create_encoding<float>("Grid", encoding);

  DeviceMem<float> params_full_precision(network->n_params(), q);
  params_full_precision.initialize_arange(q);
  //   network->initialize_params(params_full_precision.data());
  network->set_params(params_full_precision.data(),
                      params_full_precision.data(), nullptr);
  //   std::unordered_map<std::string, std::string> encoding = {
  //       {"n_dims_to_encode", std::to_string(INPUT_WIDTH)},
  //       {"degree", std::to_string(DEGREE)}};
  //   Encoding<bf16>* network =
  //       create_encoding<bf16>(INPUT_WIDTH, "SphericalHarmonics",
  // encoding);
  //   network->set_padded_output_width(OUTPUT_WIDTH);
  //   std::cout << "About to fwd" << std::endl;
  std::unique_ptr<Context> model_ctx =
      network->forward_impl(&q, input, &output_float);
  //   std::cout << "out width: " << network->output_width() << std::endl;
  std::cout << "Out2" << std::endl;
  output_float.print();
}
int main() {
  test_encoding();
  return 0;
}
 */