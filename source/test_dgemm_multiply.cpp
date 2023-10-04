#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "SwiftNetMLP.h"
#include "activation.h"

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
using bf16 = sycl::ext::oneapi::bfloat16;

#define WIDTH 64
#define SHMEM_SIZE 2048  // 1024 on dg2, 2048 for pvc

// Function to initialize a buffer with random uniform values
template <typename T>
void initialize_buffer_with_random_uniform(queue& q, T* buffer_name,
                                           size_t size, float min_val,
                                           float max_val) {
  std::default_random_engine rng;
  std::uniform_real_distribution<float> dist(min_val, max_val);

  // Get a buffer access for the device
  auto buffer_device = buffer<T, 1>(buffer_name, range<1>(size))
                           .get_access<access::mode::write>(q);

  // Fill the buffer with random uniform values
  for (size_t i = 0; i < size; ++i) {
    buffer_device[i] = static_cast<T>(dist(rng));
  }
}

void test_dgemm(int batch_size, int input_width, int output_width,
                int m_n_hidden_layers, int& duration_us, int& flops_per_s,
                int& memory_bandwidth) {
  queue q = queue();
  int m_n_hidden_matrices = m_n_hidden_layers - 1;
  DeviceMem<bf16> grads_matrices = DeviceMem<bf16>(
      WIDTH * input_width + (WIDTH * WIDTH) * m_n_hidden_matrices +
          WIDTH * output_width,
      q);

  grads_matrices.initialize_uniform(q, 0.1);

  float* out_inter =
      malloc_device<float>(batch_size * WIDTH * m_n_hidden_matrices, q);

  float* forward = malloc_device<float>(
      batch_size * (input_width + output_width + WIDTH * m_n_hidden_layers), q);
  float* A_dgemm =
      sycl::aligned_alloc_device<float>(SHMEM_SIZE, batch_size * WIDTH, q);
  float* B_dgemm =
      sycl::aligned_alloc_device<float>(SHMEM_SIZE, batch_size * WIDTH, q);
  float* C_dgemm = sycl::aligned_alloc_device<float>(
      SHMEM_SIZE, WIDTH * WIDTH,
      q);  // WIDTH * WIDTH is the maximum, for the first layer, it's
           // technically input_width * WIDTH
  int flops = 0;
  auto start = std::chrono::high_resolution_clock::now();

  for (int k = 0; k < m_n_hidden_matrices; k++) {
    dgemm_multiply<64, Activation::ReLU>(
        q, grads_matrices.data(), out_inter, forward, A_dgemm, B_dgemm, C_dgemm,
        k, m_n_hidden_matrices, batch_size, input_width, flops);
  }
  flops *= m_n_hidden_matrices;

  // Stop the clock
  auto end = std::chrono::high_resolution_clock::now();

  // Calculate the duration
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  duration_us = static_cast<int>(duration.count());
  flops_per_s = flops * 1e6 / duration_us;
  memory_bandwidth = 0;

  // Print the message
  std::cout << "For batch_size = " << batch_size
            << ", input_width = " << input_width
            << ", output_width = " << output_width
            << ", m_n_hidden_layers = " << m_n_hidden_layers
            << ", we have duration = " << duration_us
            << " microseconds, flops per second = " << flops_per_s
            << " FLOPS/s, and memory bandwidth = " << memory_bandwidth
            << " bytes per second." << std::endl;
  // free
  grads_matrices.free_mem(q);
  free(out_inter, q);
  free(forward, q);
  free(A_dgemm, q);
  free(B_dgemm, q);
  free(C_dgemm, q);
}

void benchmark_dgemm_time() {
  //   std::vector<uint32_t> batch_sizes = {1 << 21, 1 << 20, 1 << 19, 1 << 18,
  //                                        1 << 17, 1 << 16, 1 << 15, 1 << 14};
  //   std::vector<uint32_t> hidden_layers = {2, 3, 4};
  //   std::vector<uint32_t> input_widths = {1, 8, 16, 32, 64};
  //   std::vector<uint32_t> output_widths = {1, 8, 16, 32, 64};
  std::vector<uint32_t> batch_sizes = {1 << 6};
  std::vector<uint32_t> hidden_layers = {4};
  std::vector<uint32_t> input_widths = {64};
  std::vector<uint32_t> output_widths = {64};

  nlohmann::json bench_result;
  std::string method = "SwiftNet";

  bench_result[method] = nlohmann::json::array();

  int duration_us;
  int flops_per_s;
  int memory_bandwidth;

  for (uint32_t batch_size : batch_sizes) {
    for (uint32_t input_width : input_widths) {
      for (uint32_t hidden_layer : hidden_layers) {
        for (uint32_t output_width : output_widths) {
          test_dgemm(batch_size, input_width, output_width, hidden_layer,
                     duration_us, flops_per_s, memory_bandwidth);

          bench_result[method].push_back({
              {"batch_size", batch_size},
              {"input_width", input_width},
              {"output_width", output_width},
              {"hidden_layer", hidden_layer},
              {"duration_us", duration_us},
              {"flops_per_s", flops_per_s},
              {"memory_bandwidth", memory_bandwidth},
          });
        }
      }
    }
  }

  std::string json_string = bench_result.dump(4);
  std::ofstream out{"bench_dgemm_results.json"};
  out << json_string;
}

int main() {
  benchmark_dgemm_time();
  return 0;
}