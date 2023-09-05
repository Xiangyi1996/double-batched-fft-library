#include "SwiftNetMLP.h"
#include "common.h"
#include "mkl.h"
#include "mkl_omp_offload.h"
#include "oneapi/mkl.hpp"
#include "trainer.h"

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

using bf16 = sycl::ext::oneapi::bfloat16;

void test1() {
  const int batch_size = 8192;
  const int output_width = 64;
  const int WIDTH = 64;
  const int n_hidden_layers = 4;
  const int intermediate_output_size = batch_size * WIDTH * n_hidden_layers;
  const int layer_length = WIDTH * batch_size;
  const int n_hidden_matrices = n_hidden_layers - 1;
  const int net_width = 64;
  const int inputs_width = 64;

  const float scale = 1e-3f;
  /*  device dev = device(gpu_selector_v);

    std::vector<device> subdev = {};

    subdev = dev.create_sub_devices<sycl::info::partition_property::
        partition_by_affinity_domain>(sycl::info::partition_affinity_domain::numa);

    queue q = queue(subdev[0]);
*/
  queue q = queue();

  DeviceMem<bf16> inputs = DeviceMem<bf16>(batch_size * WIDTH, q);
  DeviceMem<float> output = DeviceMem<float>(batch_size * output_width, q);
  DeviceMem<float> target = DeviceMem<float>(batch_size * output_width, q);
  DeviceMem<bf16> grads = DeviceMem<bf16>(batch_size * output_width, q);
  DeviceMem<float> losses = DeviceMem<float>(batch_size * output_width, q);

  float* forward = malloc_device<float>(
      batch_size * (WIDTH + output_width + WIDTH * n_hidden_matrices), q);
  int shmem_size = batch_size * WIDTH * 5;
  const size_t alignment = 4096;

  auto A_forward =
      sycl::aligned_alloc_device<float>(alignment, layer_length, q);
  auto B_forward =
      sycl::aligned_alloc_device<float>(alignment, output_width * 64, q);
  auto C_forward = sycl::aligned_alloc_device<float>(
      alignment, output_width * batch_size, q);

  float* out_inter =
      malloc_device<float>(batch_size * WIDTH * (n_hidden_matrices + 1), q);
  auto deltas_temp = sycl::aligned_alloc_device<float>(
      alignment, output_width * batch_size, q);
  DeviceMem<bf16> deltas(output_width * batch_size, q);

  auto A_backward =
      sycl::aligned_alloc_device<float>(alignment, WIDTH * batch_size, q);
  auto B_backward = sycl::aligned_alloc_device<float>(
      alignment, batch_size * output_width, q);
  auto C_backward =
      sycl::aligned_alloc_device<float>(alignment, WIDTH * output_width, q);

  auto A_backward_last_layer =
      sycl::aligned_alloc_device<float>(alignment, grads.size(), q);
  auto B_backward_last_layer =
      sycl::aligned_alloc_device<float>(alignment, output_width * WIDTH, q);
  auto C_backward_last_layer =
      sycl::aligned_alloc_device<float>(alignment, WIDTH * batch_size, q);
  auto D_backward_last_layer =
      sycl::aligned_alloc_device<float>(alignment, WIDTH * batch_size, q);
  auto E_backward_last_layer =
      sycl::aligned_alloc_device<float>(alignment, batch_size * WIDTH, q);
  auto F_backward_last_layer =
      sycl::aligned_alloc_device<float>(alignment, WIDTH * WIDTH, q);

  auto A_dgemm =
      sycl::aligned_alloc_device<float>(alignment, batch_size * WIDTH, q);
  auto B_dgemm =
      sycl::aligned_alloc_device<float>(alignment, batch_size * WIDTH, q);
  auto C_dgemm = sycl::aligned_alloc_device<float>(alignment, WIDTH * WIDTH, q);

  inputs.initialize_constant(bf16(2.0f), q);
  output.initialize_constant(0.0f, q);
  target.initialize_constant(8.0f, q);
  grads.initialize_constant(bf16(0.0f), q);
  losses.initialize_constant(0.0f, q);

  // auto model = create_from_config(q, config);

  L2Loss loss;
  SGDOptimizer optim = SGDOptimizer(64, n_hidden_layers, 1e-3f, 1e-8f);
  SwiftNetMLP<64> network =
      SwiftNetMLP<64>(q, 64, 64, n_hidden_layers, Activation::None,
                      Activation::None, batch_size);
  Trainer train(network, loss, optim);

  train.initialize_params();

  q.parallel_for<>(range<1>(inputs.size()),
                   [=](id<1> idx) { forward[idx] = inputs.data()[idx]; });

  for (int i = 0; i < 1000; i++) {
    std::cout << i << std::endl;
    train.training_step(
        inputs, forward, A_forward, B_forward, C_forward, out_inter,
        deltas_temp, deltas, A_backward, B_backward, C_backward,
        A_backward_last_layer, B_backward_last_layer, C_backward_last_layer,
        D_backward_last_layer, E_backward_last_layer, F_backward_last_layer,
        A_dgemm, B_dgemm, C_dgemm, output, target, grads, losses, scale, 64);
  }

  inputs.free_mem(q);
  output.free_mem(q);
  target.free_mem(q);
  grads.free_mem(q);
  losses.free_mem(q);
  free(out_inter, q);
  free(deltas_temp, q);
  free(A_forward, q);
  free(B_forward, q);
  free(C_forward, q);
  deltas.free_mem(q);
  free(A_backward, q);
  free(B_backward, q);
  free(C_backward, q);
  free(A_backward_last_layer, q);
  free(B_backward_last_layer, q);
  free(C_backward_last_layer, q);
  free(D_backward_last_layer, q);
  free(E_backward_last_layer, q);
  free(F_backward_last_layer, q);
  free(A_dgemm, q);
  free(B_dgemm, q);
  free(C_dgemm, q);
}

int main() {
  test1();
  return 0;
}
