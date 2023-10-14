/* #include <CL/sycl.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "SwiftNetMLP.h"
#include "activation.h"
#include "common.h"
#include "mkl.h"
#include "mkl_omp_offload.h"
#include "oneapi/mkl.hpp"

// Define constants for the sizes
const int INPUT_WIDTH = 64;
const int OUTPUT_WIDTH = 64;
const int HIDDEN_LAYERS = 4;

template <typename T>
bool areVectorsWithinTolerance(const std::vector<T>& value,
                               const std::vector<T>& target, float tolerance) {
  assert(value.size() == target.size());  // Ensure vectors have the same length
  int total_values_checked = 0;
  bool allWithinTolerance = true;

  for (size_t i = 0; i < value.size(); ++i) {
    float diff = std::abs(value[i] - target[i]);
    // std::cout << "Checking idx: " << i << std::endl;
    total_values_checked++;
    if (diff > tolerance) {
      allWithinTolerance = false;
      std::cout << "Element at index " << i
                << " is not within tolerance. Value: " << value[i]
                << ", Target: " << target[i] << ". Diff: " << diff << std::endl;
    }
  }

  if (allWithinTolerance) {
    std::cout << "All elements are within tolerance. Total values checked: "
              << total_values_checked << std::endl;
  } else {
    std::cout
        << "Not all elements are within tolerance.. Total values checked: "
        << total_values_checked << std::endl;
  }

  return allWithinTolerance;
}

template <typename T>
std::vector<T> loadVectorFromCSV(const std::string& filename) {
  std::vector<T> data;
  std::ifstream file(filename);

  if (!file.is_open()) {
    std::cerr << "Failed to open the file for reading: " << filename
              << std::endl;
    return data;
  }

  std::string line;
  while (std::getline(file, line)) {
    std::istringstream ss(line);
    std::string token;
    while (std::getline(ss, token, ',')) {
      data.push_back(static_cast<T>(std::stof(token)));
    }
  }

  return data;
}

int main() {
  // SWIFTNET
  const int batch_size = 128;
  const int output_width = OUTPUT_WIDTH;
  const int WIDTH = 64;

  queue q = queue();

  DeviceMem<bf16> inputs = DeviceMem<bf16>(INPUT_WIDTH * batch_size, q);
  DeviceMem<float> output = DeviceMem<float>(batch_size * output_width, q);
  DeviceMem<float> target = DeviceMem<float>(batch_size * output_width, q);
  DeviceMem<bf16> grads = DeviceMem<bf16>(batch_size * output_width, q);
  DeviceMem<float> losses = DeviceMem<float>(batch_size * output_width, q);

  SwiftNetMLP<64> network =
      SwiftNetMLP<64>(q, INPUT_WIDTH, output_width, HIDDEN_LAYERS,
                      Activation::ReLU, Activation::None, batch_size);
  //   Activation::None, Activation::None, batch_size);
  network.initialize_params(1);

  inputs.initialize_constant(0.1f, q);
  output.initialize_constant(0.0f, q);
  target.initialize_constant(1.0f, q);
  grads.initialize_arange(q);
  losses.initialize_constant(0.0f, q);

  auto p = network.m_forward;
  network.get_queue()
      .parallel_for<>(range<1>(inputs.size()),
                      [=](id<1> idx) { p[idx] = inputs.data()[idx]; })
      .wait();

  network.forward_pass(inputs, network.m_forward, network.m_A_forward,
                       network.m_B_forward, network.m_C_forward, output);

  network.backward_pass(
      inputs, grads, network.m_out_inter, network.m_deltas,
      network.m_A_backward, network.m_B_backward, network.m_C_backward,
      network.m_A_backward_last_layer, network.m_B_backward_last_layer,
      network.m_C_backward_last_layer, network.m_D_backward_last_layer,
      network.m_E_backward_last_layer, network.m_F_backward_last_layer,
      network.m_A_dgemm, network.m_B_dgemm, network.m_C_dgemm,
      network.m_forward);

  // Copying
  // Calculate sizes
  size_t m_forward_size =
      batch_size * (INPUT_WIDTH + OUTPUT_WIDTH + WIDTH * HIDDEN_LAYERS);
  size_t inputs_size = INPUT_WIDTH * batch_size;
  size_t grads_size = batch_size * output_width;
  size_t out_inter_size = batch_size * WIDTH * (HIDDEN_LAYERS - 1);
  size_t deltas_size = WIDTH * batch_size;
  size_t A_backward_size = WIDTH * batch_size;
  size_t B_backward_size = batch_size * OUTPUT_WIDTH;
  size_t C_backward_size = WIDTH * OUTPUT_WIDTH;
  size_t A_backward_last_layer_size = batch_size * OUTPUT_WIDTH;
  size_t B_backward_last_layer_size = OUTPUT_WIDTH * WIDTH;
  size_t C_backward_last_layer_size = WIDTH * batch_size;
  size_t D_backward_last_layer_size = WIDTH * batch_size;
  size_t E_backward_last_layer_size = batch_size * WIDTH;
  size_t F_backward_last_layer_size = WIDTH * WIDTH;

  // Allocate host memory
  std::vector<float> m_forward_vec(m_forward_size);
  std::vector<bf16> inputs_vec(inputs_size);
  std::vector<bf16> grads_vec(grads_size);
  std::vector<bf16> deltas_vec(deltas_size);
  std::vector<float> out_inter_vec(out_inter_size);
  std::vector<float> A_backward_vec(A_backward_size);
  std::vector<float> B_backward_vec(B_backward_size);
  std::vector<float> C_backward_vec(C_backward_size);
  std::vector<float> A_backward_last_layer_vec(A_backward_last_layer_size);
  std::vector<float> B_backward_last_layer_vec(B_backward_last_layer_size);
  std::vector<float> C_backward_last_layer_vec(C_backward_last_layer_size);
  std::vector<float> D_backward_last_layer_vec(D_backward_last_layer_size);
  std::vector<float> E_backward_last_layer_vec(E_backward_last_layer_size);
  std::vector<float> F_backward_last_layer_vec(F_backward_last_layer_size);

  // Copy data from device to host
  q.memcpy(m_forward_vec.data(), network.m_forward,
           m_forward_size * sizeof(float))
      .wait();
  inputs.copy_to_host(inputs_vec, q);
  grads.copy_to_host(grads_vec, q);
  network.m_deltas.copy_to_host(deltas_vec, q);

  q.memcpy(out_inter_vec.data(), network.m_out_inter,
           out_inter_size * sizeof(float))
      .wait();

  q.memcpy(A_backward_vec.data(), network.m_A_backward,
           A_backward_size * sizeof(float))
      .wait();
  q.memcpy(B_backward_vec.data(), network.m_B_backward,
           B_backward_size * sizeof(float))
      .wait();
  q.memcpy(C_backward_vec.data(), network.m_C_backward,
           C_backward_size * sizeof(float))
      .wait();
  q.memcpy(A_backward_last_layer_vec.data(), network.m_A_backward_last_layer,
           A_backward_last_layer_size * sizeof(float))
      .wait();
  q.memcpy(B_backward_last_layer_vec.data(), network.m_B_backward_last_layer,
           B_backward_last_layer_size * sizeof(float))
      .wait();
  q.memcpy(C_backward_last_layer_vec.data(), network.m_C_backward_last_layer,
           C_backward_last_layer_size * sizeof(float))
      .wait();
  q.memcpy(D_backward_last_layer_vec.data(), network.m_D_backward_last_layer,
           D_backward_last_layer_size * sizeof(float))
      .wait();
  q.memcpy(E_backward_last_layer_vec.data(), network.m_E_backward_last_layer,
           E_backward_last_layer_size * sizeof(float))
      .wait();
  q.memcpy(F_backward_last_layer_vec.data(), network.m_F_backward_last_layer,
           F_backward_last_layer_size * sizeof(float))
      .wait();

  // Load the CSV files into vectors
  std::vector<float> m_forward_vec_ref =
      loadVectorFromCSV<float>("bwd_matrices/m_forward.csv");
  std::vector<bf16> inputs_vec_ref =
      loadVectorFromCSV<bf16>("bwd_matrices/inputs.csv");
  std::vector<bf16> grads_vec_ref =
      loadVectorFromCSV<bf16>("bwd_matrices/grads.csv");
  std::vector<float> out_inter_vec_ref =
      loadVectorFromCSV<float>("bwd_matrices/out_inter.csv");
  std::vector<bf16> deltas_vec_ref =
      loadVectorFromCSV<bf16>("bwd_matrices/deltas.csv");
  std::vector<float> A_backward_vec_ref =
      loadVectorFromCSV<float>("bwd_matrices/A_backward.csv");
  std::vector<float> B_backward_vec_ref =
      loadVectorFromCSV<float>("bwd_matrices/B_backward.csv");
  std::vector<float> C_backward_vec_ref =
      loadVectorFromCSV<float>("bwd_matrices/C_backward.csv");
  std::vector<float> A_backward_last_layer_vec_ref =
      loadVectorFromCSV<float>("bwd_matrices/A_backward_last_layer.csv");
  std::vector<float> B_backward_last_layer_vec_ref =
      loadVectorFromCSV<float>("bwd_matrices/B_backward_last_layer.csv");
  std::vector<float> C_backward_last_layer_vec_ref =
      loadVectorFromCSV<float>("bwd_matrices/C_backward_last_layer.csv");
  std::vector<float> D_backward_last_layer_vec_ref =
      loadVectorFromCSV<float>("bwd_matrices/D_backward_last_layer.csv");
  std::vector<float> E_backward_last_layer_vec_ref =
      loadVectorFromCSV<float>("bwd_matrices/E_backward_last_layer.csv");
  std::vector<float> F_backward_last_layer_vec_ref =
      loadVectorFromCSV<float>("bwd_matrices/F_backward_last_layer.csv");

  float tolerance = 1e-3;
  // Compare each vector with its corresponding reference

  // Compare each vector with its corresponding reference and print a message
  std::cout << "Checking m_forward..." << std::endl;
  areVectorsWithinTolerance(m_forward_vec, m_forward_vec_ref, tolerance);

  std::cout << "Checking inputs..." << std::endl;
  areVectorsWithinTolerance(inputs_vec, inputs_vec_ref, tolerance);

  std::cout << "Checking grads..." << std::endl;
  areVectorsWithinTolerance(grads_vec, grads_vec_ref, tolerance);

  std::cout << "Checking out_inter..." << std::endl;
  areVectorsWithinTolerance(out_inter_vec, out_inter_vec_ref, tolerance);

  std::cout << "Checking deltas..." << std::endl;
  areVectorsWithinTolerance(deltas_vec, deltas_vec_ref, tolerance);

  std::cout << "Checking A_backward..." << std::endl;
  areVectorsWithinTolerance(A_backward_vec, A_backward_vec_ref, tolerance);

  std::cout << "Checking B_backward..." << std::endl;
  areVectorsWithinTolerance(B_backward_vec, B_backward_vec_ref, tolerance);

  std::cout << "Checking C_backward..." << std::endl;
  areVectorsWithinTolerance(C_backward_vec, C_backward_vec_ref, tolerance);

  std::cout << "Checking A_backward_last_layer..." << std::endl;
  areVectorsWithinTolerance(A_backward_last_layer_vec,
                            A_backward_last_layer_vec_ref, tolerance);

  std::cout << "Checking B_backward_last_layer..." << std::endl;
  areVectorsWithinTolerance(B_backward_last_layer_vec,
                            B_backward_last_layer_vec_ref, tolerance);

  std::cout << "Checking C_backward_last_layer..." << std::endl;
  areVectorsWithinTolerance(C_backward_last_layer_vec,
                            C_backward_last_layer_vec_ref, tolerance);

  std::cout << "Checking D_backward_last_layer..." << std::endl;
  areVectorsWithinTolerance(D_backward_last_layer_vec,
                            D_backward_last_layer_vec_ref, tolerance);

  std::cout << "Checking E_backward_last_layer..." << std::endl;
  areVectorsWithinTolerance(E_backward_last_layer_vec,
                            E_backward_last_layer_vec_ref, tolerance);

  std::cout << "Checking F_backward_last_layer..." << std::endl;
  areVectorsWithinTolerance(F_backward_last_layer_vec,
                            F_backward_last_layer_vec_ref, tolerance);

  return 0;
}
 */