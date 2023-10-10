// #include <CL/sycl.hpp>
// #include <chrono>
// #include <cmath>
// #include <iostream>
// #include <thread>
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

// using namespace sycl;
// using namespace sycl::ext::oneapi::experimental::matrix;
// using bf16 = sycl::ext::oneapi::bfloat16;

// #define INPUT_WIDTH 64
// #define OUTPUT_WIDTH 64
// #define HIDDEN_LAYERS 4
// bool areVectorsWithinTolerance(const std::vector<float>& value,
//                                const std::vector<float>& target,
//                                float tolerance) {
//   //   assert(a.size() == b.size());  // Ensure vectors have the same length

//   bool allWithinTolerance = true;

//   for (size_t i = 0; i < target.size(); ++i) {
//     float diff = std::abs(value[i] - target[i]);

//     if (diff > tolerance) {
//       allWithinTolerance = false;
//       std::cout << "Element at index " << i
//                 << " is not within tolerance. Value: " << value[i]
//                 << ", Target: " << target[i] << ". Diff: " << diff <<
//                 std::endl;
//     }
//   }

//   if (allWithinTolerance) {
//     std::cout << "All elements are within tolerance." << std::endl;
//   } else {
//     std::cout << "Not all elements are within tolerance." << std::endl;
//   }

//   return allWithinTolerance;
// }

// // Function to read target vectors from a file with a specified delimiter
// std::vector<std::vector<float>> readTargetVectorsFromFile(
//     const std::string& filename, char delimiter) {
//   std::vector<std::vector<float>> targetVectors;

//   std::ifstream inputFile(filename);
//   if (!inputFile.is_open()) {
//     std::cerr << "Error opening " << filename << std::endl;
//     return targetVectors;  // Return an empty vector in case of an error
//   }

//   std::string line;

//   while (std::getline(inputFile, line)) {
//     std::vector<float> vectorFromCSV;
//     std::istringstream lineStream(line);
//     std::string valueStr;

//     while (std::getline(lineStream, valueStr, delimiter)) {
//       float value = std::stod(valueStr);  // Convert the string to a double
//       vectorFromCSV.push_back(value);
//     }

//     targetVectors.push_back(vectorFromCSV);
//   }

//   inputFile.close();

//   return targetVectors;
// }
// void benchmark_time() {
//   // SWIFTNET

//   // Fourth step: train the model by sampling the above image and optimizing
//   // relative squared error using Adam.
//   // Experimental not sure how many batch sizes are possible
//   //   std::vector<uint32_t> batch_sizes = {
//   //       1 << 29, 1 << 28, 1 << 27, 1 << 26, 1 << 25, 1 << 24, 1 << 23, 1
//   <<
//   //       22, 1 << 21, 1 << 20, 1 << 19, 1 << 18, 1 << 17, 1 << 16, 1 << 15,
//   1
//   //       << 14};
//   std::vector<uint32_t> batch_sizes = {1 << 21, 1 << 20, 1 << 19, 1 << 18,
//                                        1 << 17, 1 << 16, 1 << 15, 1 << 14};
//   std::string method = "SwiftNet";
//   nlohmann::json bench_result;
//   bench_result[method] = nlohmann::json::array();

//   const int output_width = OUTPUT_WIDTH;
//   const int WIDTH = 64;
//   const int m_n_hidden_layers = HIDDEN_LAYERS;

//   const float scale = 1.0f;

//   L2Loss loss;
//   SGDOptimizer optim =
//       SGDOptimizer(OUTPUT_WIDTH, m_n_hidden_layers, 1e-3f, 1e-8f);

//   for (uint32_t batch_size : batch_sizes) {
//     std::cout << "Batch size 2^" << std::log2(batch_size) << std::endl;
//     queue q = queue();

//     DeviceMem<bf16> inputs = DeviceMem<bf16>(INPUT_WIDTH * batch_size, q);
//     DeviceMem<float> output = DeviceMem<float>(batch_size * output_width, q);
//     DeviceMem<float> target = DeviceMem<float>(batch_size * output_width, q);
//     DeviceMem<bf16> grads = DeviceMem<bf16>(batch_size * output_width, q);
//     DeviceMem<float> losses = DeviceMem<float>(batch_size * output_width, q);

//     SwiftNetMLP<64> network =
//         SwiftNetMLP<64>(q, INPUT_WIDTH, output_width, m_n_hidden_layers,
//                         Activation::ReLU, Activation::None, batch_size);
//     // Activation::None, Activation::None, batch_size);

//     Trainer train(network, loss, optim);

//     train.initialize_params(1);

//     inputs.initialize_constant(0.001f, q);
//     output.initialize_constant(0.0f, q);
//     target.initialize_constant(0.1f, q);
//     grads.initialize_constant(bf16(0.0f), q);
//     losses.initialize_constant(0.0f, q);

//     // Various constants for the network and optimization
//     uint32_t n_iterations = std::max(1000 * (1 << 18) / batch_size, 250u);
//     uint32_t n_iterations_warmup = n_iterations / 2;

//     std::chrono::steady_clock::time_point begin =
//         std::chrono::steady_clock::now();

//     float tmp_loss = 0;
//     uint32_t tmp_loss_counter = 0;

//     uint32_t print_interval = n_iterations / 10;
//     const uint32_t STEPS_INCREMENT = 5;

//     double mean_training_throughput = 0;
//     size_t mean_counter = 0;

//     std::cout << "Iterations: " << n_iterations
//               << ", steps increment: " << STEPS_INCREMENT << std::endl;

//     for (uint32_t i = 0; i < n_iterations; i += STEPS_INCREMENT) {
//       bool print_loss = i % print_interval == 0;

//       for (uint32_t j = 0; j < STEPS_INCREMENT; ++j) {
//         train.training_step(inputs, output, target, grads, losses, scale,
//         WIDTH,
//                             0, 0);

//         if (j == STEPS_INCREMENT - 1) {
//           tmp_loss += 0;
//           ++tmp_loss_counter;
//         }
//       }

//       // Debug outputs
//       if (print_loss) {
//         std::chrono::steady_clock::time_point end =
//             std::chrono::steady_clock::now();
//         auto microseconds =
//             std::chrono::duration_cast<std::chrono::microseconds>(end -
//             begin)
//                 .count();
//         double throughput =
//             print_interval * batch_size / ((double)microseconds / 1000000.0);
//         std::cout << "Iteration#" << i << ": "
//                   << "loss=" << tmp_loss / (float)tmp_loss_counter
//                   << " time=" << microseconds << "[µs] thp=" << throughput
//                   << "/s" << std::endl;

//         begin = end;
//         tmp_loss = 0;
//         tmp_loss_counter = 0;

//         if (i >= n_iterations_warmup) {
//           mean_training_throughput += throughput;
//           ++mean_counter;
//         }
//       }
//     }

//     mean_training_throughput /= (double)mean_counter;

//     std::cout << "Finished training benchmark. Mean throughput is "
//               << mean_training_throughput
//               << "/s. Waiting 10 seconds for GPU to cool down." << std::endl;

//     // Sanity check: we run with aranged weights and 4 layers and 0.001 as
//     // input. Values generated from test_compare_torch_dpcpp.py and saved in
//     // python/dpcpp.csv (bf16 vals). python/torch.csv is float vals

//     float tolerance = 0.001;

//     std::vector<float> out(batch_size * (OUTPUT_WIDTH));
//     output.copy_to_host(out, q);

//     std::vector<float> fwd(
//         batch_size * (INPUT_WIDTH + OUTPUT_WIDTH + WIDTH *
//         m_n_hidden_layers));
//     q.memcpy(fwd.data(), network.m_forward, fwd.size() * sizeof(float));
//     q.wait();

//     std::vector<bf16> grads_vec(network.m_grads_matrices.size());
//     network.m_grads_matrices.copy_to_host(grads_vec, q);

//     // for (int i_g = 0; i_g < grads_vec.size(); i_g++) {
//     //   std::cout << i_g << ": " << grads_vec[i_g] << std::endl;
//     // }

//     std::vector<std::vector<float>> targetVectors =
//         readTargetVectorsFromFile("python/torch.csv", ',');

//     std::vector<std::vector<float>> targetGrads =
//         readTargetVectorsFromFile("python/torch_grads.csv", ',');

//     // get every layer from fwd:
//     for (int i = 0; i < m_n_hidden_layers + 2; i++) {
//       int layer_size;
//       int grads_layer_size;

//       int start_idx;
//       int end_idx;

//       int start_idx_grads;
//       int end_idx_grads;

//       if (i == 0) {
//         layer_size = INPUT_WIDTH;
//         start_idx = 0;
//         end_idx = INPUT_WIDTH * batch_size;

//         grads_layer_size = WIDTH * INPUT_WIDTH;
//         start_idx_grads = 0;
//         end_idx_grads = WIDTH * INPUT_WIDTH;
//       } else if (i == m_n_hidden_layers + 1) {
//         layer_size = OUTPUT_WIDTH;
//         start_idx = (INPUT_WIDTH + WIDTH * m_n_hidden_layers) * batch_size;
//         end_idx = (INPUT_WIDTH + OUTPUT_WIDTH + WIDTH * m_n_hidden_layers) *
//                   batch_size;

//         grads_layer_size = WIDTH * OUTPUT_WIDTH;
//         start_idx_grads = WIDTH * INPUT_WIDTH + (WIDTH * WIDTH) * (i);
//         end_idx_grads =
//             WIDTH * INPUT_WIDTH + (WIDTH * WIDTH) * (i) + WIDTH *
//             OUTPUT_WIDTH;
//         std::cout << "For output layer (out): ";
//         tolerance *= 2;  // increase tolerance for last layer (accumulated
//         error
//                          // through layers between bf16 and float)
//         areVectorsWithinTolerance(out, targetVectors[i], tolerance);
//       } else {
//         layer_size = WIDTH;
//         start_idx = (INPUT_WIDTH + WIDTH * (i - 1)) * batch_size;
//         end_idx = (INPUT_WIDTH + WIDTH * (i)) * batch_size;

//         grads_layer_size = WIDTH * WIDTH;
//         start_idx_grads = WIDTH * INPUT_WIDTH + (WIDTH * WIDTH) * (i - 1);
//         end_idx_grads = WIDTH * INPUT_WIDTH + (WIDTH * WIDTH) * (i);
//       }

//       std::vector<float> layer(batch_size * layer_size);
//       std::copy(fwd.begin() + start_idx, fwd.begin() + end_idx,
//       layer.begin());

//       std::vector<bf16> layer_grads(grads_layer_size);
//       std::copy(grads_vec.begin() + start_idx_grads,
//                 grads_vec.begin() + end_idx_grads, layer_grads.begin());

//       std::cout << "Layer " << i << ", start_idx: " << start_idx
//                 << ", end_idx: " << end_idx << ". ";

//       areVectorsWithinTolerance(layer, targetVectors[i], tolerance);

//       //   std::cout << "Grads Layer " << i << ", start_idx: " <<
//       //   start_idx_grads
//       //             << ", end_idx: " << end_idx_grads;

//       // areVectorsWithinTolerance(layer_grads, targetGrads[i], tolerance);
//     }

//     std::this_thread::sleep_for(std::chrono::seconds{10});
//     begin = std::chrono::steady_clock::now();

//     // Inference benchmark
//     double mean_inference_throughput = 0;
//     mean_counter = 0;

//     print_interval *= 5;
//     n_iterations *= 5;
//     n_iterations_warmup *= 5;
//     for (uint32_t i = 0; i < n_iterations; ++i) {
//       bool print_loss = i % print_interval == 0;

//       // Inference step
//       train.training_step(inputs, output, target, grads, losses, scale,
//       WIDTH,
//                           1, 1);
//       // Debug outputs
//       if (print_loss) {
//         std::chrono::steady_clock::time_point end =
//             std::chrono::steady_clock::now();
//         auto microseconds =
//             std::chrono::duration_cast<std::chrono::microseconds>(end -
//             begin)
//                 .count();
//         double throughput =
//             print_interval * batch_size / ((double)microseconds / 1000000.0);
//         std::cout << "Iteration#" << i << ": "
//                   << "time=" << microseconds << "[µs] thp=" << throughput
//                   << "/s" << std::endl;

//         begin = end;

//         if (i >= n_iterations_warmup) {
//           mean_inference_throughput += throughput;
//           ++mean_counter;
//         }
//       }
//     }

//     mean_inference_throughput /= (double)mean_counter;

//     std::cout << "Finished inference benchmark. Mean throughput is "
//               << mean_inference_throughput
//               << "/s. Waiting 10 seconds for GPU to cool down." << std::endl;

//     output.copy_to_host(out, q);
//     areVectorsWithinTolerance(out, targetVectors.back(), tolerance * 2);
//     // accumulated tolerance for last layer due to
//     // accumulated error between bf16 and float

//     // std::cout << "Output[0] (all are the same): " << out[0]
//     //           << ", ref val: " << target_value
//     //           << ". Within tolerance: " << abs(out[0] - target_value)
//     //           << tolerance << std::endl;

//     inputs.free_mem(q);
//     output.free_mem(q);
//     target.free_mem(q);
//     grads.free_mem(q);
//     losses.free_mem(q);

//     std::this_thread::sleep_for(std::chrono::seconds{10});

//     bench_result[method].push_back({
//         {"batch_size", batch_size},
//         {"training_throughput", mean_training_throughput},
//         {"inference_throughput", mean_inference_throughput},
//     });
//   }

//   std::string json_string = bench_result.dump(4);
//   std::ofstream out{"bench_result_ours.json"};
//   out << json_string;
// }
// // int main() {
// //   benchmark_time();
// //   return 0;
// // }