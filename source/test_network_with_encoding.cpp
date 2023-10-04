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

#define INPUT_WIDTH 2
#define OUTPUT_WIDTH 64
#define HIDDEN_LAYERS 2

void test_network_with_encoding() {
  // SWIFTNET
  const int batch_size = 64;
  const int output_width = OUTPUT_WIDTH;
  const int m_n_hidden_layers = HIDDEN_LAYERS;

  GPUMatrix<float> input(INPUT_WIDTH, batch_size);
  input.initialize_constant(0.01f);
  DeviceMem<float> *output;

  NetworkWithEncoding network = NetworkWithEncoding(
      INPUT_WIDTH, output_width, m_n_hidden_layers, Activation::None,
      Activation::None, batch_size, 1.0, 0.0);
  network.initialize_params(1);
  output = network.forward_pass(input, 0);

  std::vector<float> out(batch_size * (OUTPUT_WIDTH));
  output->copy_to_host(out, network.get_queue());

  for (int j = 0; j < OUTPUT_WIDTH; j++) {
    std::cout << out[j] << ", ";
  }
  std::cout << std::endl;

  network.free_memory();
}
int main() {
  test_network_with_encoding();
  return 0;
}