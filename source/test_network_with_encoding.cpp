#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

#include "L2.h"
#include "SwiftNetMLP.h"
#include "activation.h"
#include "mkl.h"
#include "mkl_omp_offload.h"
#include "network_with_encodings.h"
#include "oneapi/mkl.hpp"

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
using bf16 = sycl::ext::oneapi::bfloat16;

#define INPUT_WIDTH 3
#define OUTPUT_WIDTH 20
#define HIDDEN_LAYERS 4

void test_network_with_encoding() {
  // SWIFTNET
  const int batch_size = 128;
  const int output_width = OUTPUT_WIDTH;
  const int m_n_hidden_layers = HIDDEN_LAYERS;

  GPUMatrix<float> input(INPUT_WIDTH, batch_size);
  //   GPUMatrix<float> input(batch_size, INPUT_WIDTH);
  input.initialize_constant(0.1f);
  DeviceMem<float> output;

  //   Define the parameters for creating IdentityEncoding
  std::unordered_map<std::string, std::string> encoding = {
      {"n_dims_to_encode", std::to_string(INPUT_WIDTH)},
      {"scale", "1.0"},
      {"offset", "0.0"}};
  std::string encoding_name = "Identity";

  //   std::unordered_map<std::string, std::string> encoding = {
  //       {"n_dims_to_encode", std::to_string(INPUT_WIDTH)},
  //       {"degree", std::to_string(4)}};
  //   std::string encoding_name = "SphericalHarmonics";
  NetworkWithEncoding network = NetworkWithEncoding(
      INPUT_WIDTH, OUTPUT_WIDTH, m_n_hidden_layers, Activation::ReLU,
      Activation::None, encoding_name, encoding);
  network.initialize_params(1);
  output = network.forward_pass(input, 0);

  std::vector<float> out(batch_size * (OUTPUT_WIDTH));
  output.copy_to_host(out, network.get_queue());

  for (int j = 0; j < batch_size * OUTPUT_WIDTH; j++) {
    std::cout << j << ": " << out[j] << std::endl;
  }

  network.free_memory();
}

int main() {
  test_network_with_encoding();
  return 0;
}