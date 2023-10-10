// #include <CL/sycl.hpp>
// #include <chrono>
// #include <iostream>
// #include <json/json.hpp>
// #include <random>
// #include <vector>

// #include "DeviceMem.h"
// #include "activation.h"
// #include "mkl.h"
// #include "mkl_omp_offload.h"
// #include "oneapi/mkl.hpp"
// using namespace sycl;
// using namespace sycl::ext::oneapi::experimental::matrix;
// using bf16 = sycl::ext::oneapi::bfloat16;

// #define WIDTH 64
// #define ITERATIONS 100

// // // for dg2
// // #define TM 8
// // #define TK 16
// // #define TN 8
// // #define SKEW 0

// // #define SG_SIZE 8
// // #define WG_SIZE 8 * SG_SIZE

// // #define BATCH_CHUNK 16
// // #define SHMEM_SIZE 1024

// // uncomment below for pvc
// #define TM 8
// #define TK 16
// #define TN 16
// #define SKEW 8

// #define SG_SIZE 16
// #define WG_SIZE 4 * SG_SIZE

// #define BATCH_CHUNK 32
// #define SHMEM_SIZE 2048

// template <int NET_WIDTH, int N_ITERS, bool BACKWARD = false>
// void matmul_act_layer(
//     nd_item<1> item, Activation activation,
//     multi_ptr<bf16, access::address_space::local_space, (access::decorated)2> a,
//     multi_ptr<float, access::address_space::local_space, (access::decorated)2>
//         at,
//     bf16* weights_layer, float* out_inter, float* forward_act = nullptr,
//     int print = 0) {
//   // Get sub-group and local IDs

//   auto sg = item.get_sub_group();
//   int id = item.get_local_id() % SG_SIZE;
//   int sgId = sg.get_group_id();

//   // Device pointers to memory
//   device_ptr<bf16> w(weights_layer);
//   device_ptr<float> o(out_inter);
//   device_ptr<float> f(forward_act);

//   // Define matrices and load weights
//   joint_matrix<sub_group, bf16, use::a, TM, TK, layout::row_major> act_matrix;
//   joint_matrix<sub_group, bf16, use::b, TK, TN,
//                sycl::ext::intel::experimental::matrix::layout::packed>
//       weight_matrix0;
//   joint_matrix<sub_group, bf16, use::b, TK, TN,
//                sycl::ext::intel::experimental::matrix::layout::packed>
//       weight_matrix1;
//   joint_matrix<sub_group, bf16, use::b, TK, TN,
//                sycl::ext::intel::experimental::matrix::layout::packed>
//       weight_matrix2;
//   joint_matrix<sub_group, bf16, use::b, TK, TN,
//                sycl::ext::intel::experimental::matrix::layout::packed>
//       weight_matrix3;
//   joint_matrix<sub_group, float, use::accumulator, TM, TN> result_matrix;

//   joint_matrix_load(sg, weight_matrix0,
//                     w + TN * 2 * sgId + TK / 2 * 0 * NET_WIDTH * 2,
//                     NET_WIDTH * 2);
//   joint_matrix_load(sg, weight_matrix1,
//                     w + TN * 2 * sgId + TK / 2 * 1 * NET_WIDTH * 2,
//                     NET_WIDTH * 2);
//   joint_matrix_load(sg, weight_matrix2,
//                     w + TN * 2 * sgId + TK / 2 * 2 * NET_WIDTH * 2,
//                     NET_WIDTH * 2);
//   joint_matrix_load(sg, weight_matrix3,
//                     w + TN * 2 * sgId + TK / 2 * 3 * NET_WIDTH * 2,
//                     NET_WIDTH * 2);
// #pragma unroll
//   for (int l = 0; l < N_ITERS; l++) {
//     joint_matrix_fill(sg, result_matrix, 0.0f);
//     // Load activation matrix and perform matrix multiplication and accumulation
//     joint_matrix_load(sg, act_matrix, a + TK * 0 + TM * l * (NET_WIDTH + SKEW),
//                       NET_WIDTH + SKEW);

//     result_matrix =
//         joint_matrix_mad(sg, act_matrix, weight_matrix0, result_matrix);
//     joint_matrix_load(sg, act_matrix, a + TK * 1 + TM * l * (NET_WIDTH + SKEW),
//                       NET_WIDTH + SKEW);
//     result_matrix =
//         joint_matrix_mad(sg, act_matrix, weight_matrix1, result_matrix);
//     joint_matrix_load(sg, act_matrix, a + TK * 2 + TM * l * (NET_WIDTH + SKEW),
//                       NET_WIDTH + SKEW);
//     result_matrix =
//         joint_matrix_mad(sg, act_matrix, weight_matrix2, result_matrix);
//     joint_matrix_load(sg, act_matrix, a + TK * 3 + TM * l * (NET_WIDTH + SKEW),
//                       NET_WIDTH + SKEW);
//     result_matrix =
//         joint_matrix_mad(sg, act_matrix, weight_matrix3, result_matrix);

//     joint_matrix_store(sg, result_matrix,
//                        at + TN * sgId + TM * l * (NET_WIDTH + SKEW),
//                        NET_WIDTH + SKEW, layout::row_major);
//   }

// #pragma unroll
//   for (int i = 0; i < N_ITERS; i++) {
//     if (BACKWARD) {
//       int stride = (NET_WIDTH + SKEW);
//       int offset = TN * sgId * (NET_WIDTH + SKEW) + TM * i + id;
//       // Apply backward activation matrix if required
//       matrix_activation_backward<float, float, bf16, SG_SIZE>(
//           activation, at, f, a, TN * sgId + (NET_WIDTH + SKEW) * TM * i + id,
//           (NET_WIDTH + SKEW));
//     } else {
//       //   Apply forward activation matrix
//       matrix_activation<float, bf16, SG_SIZE>(
//           activation, at, a, TN * sgId + (NET_WIDTH + SKEW) * TM * i + id,
//           (NET_WIDTH + SKEW));
//     }
//   }

//   if (out_inter) {
// #pragma unroll
//     for (int i = 0; i < N_ITERS; i++) {
//       for (int k = 0; k < TM; k++) {
//         // Copy results to the output intermediate matrix
//         if (BACKWARD) {
//           out_inter[TN * sgId + NET_WIDTH * TM * i + k * NET_WIDTH + id] =
//               a[TN * sgId + (NET_WIDTH + SKEW) * TM * i +
//                 k * (NET_WIDTH + SKEW) + id];

//         } else {
//           out_inter[TN * sgId + NET_WIDTH * TM * i + k * NET_WIDTH + id] =
//               a[TN * sgId + (NET_WIDTH + SKEW) * TM * i +
//                 k * (NET_WIDTH + SKEW) + id];
//         }
//       }
//     }
//   }
// }
// // Copy paste from SwiftNetMLP, but with T1, T2 instead of bf16 and float
// template <int NET_WIDTH, Activation ACTIVATION, typename T1, typename T2>
// void dgemm_multiply(queue q, T1* grads_device, T2* loss_gradients, T2* fwd,
//                     T2* A, T2* B, T2* C, int k, int m_n_hidden_matrices,
//                     int batch_size, int m_inputs_width, int& flops) {
//   const int n_hidden_matrices = m_n_hidden_matrices;
//   int layer_in_width;
//   int offset_f1;
//   int offset_g;
//   int offset_c;
//   if (k == (n_hidden_matrices - 1)) {
//     // this is the 1st layer (input to 1st layer)
//     // need this as input_width != net_width
//     layer_in_width = m_inputs_width;
//     offset_f1 = 0;
//     offset_g = 0;
//     offset_c = 0;
//   } else {
//     //  any layer between input and output (input to 1st layer and penultimate
//     //  to last layer are handled separately)
//     layer_in_width = NET_WIDTH;
//     offset_f1 = (n_hidden_matrices - k - 1) * NET_WIDTH * batch_size;
//     offset_g =
//         (m_inputs_width + (n_hidden_matrices - k - 2) * NET_WIDTH) * batch_size;
//     offset_c = (m_inputs_width * NET_WIDTH +
//                 (n_hidden_matrices - k - 2) * NET_WIDTH * NET_WIDTH);
//   }
//   // Calculate matrix A using the given activation function
//   q.parallel_for<>(range<1>(layer_in_width * batch_size), [=](id<1> idx) {
//      int i = idx / batch_size;
//      int j = idx % batch_size;
//      A[i * batch_size + j] = (T2)elt_activation_ret<T2>(
//          ACTIVATION, fwd[i + j * layer_in_width + offset_g]);
//    }).wait();

//   // Assign matrix B using loss gradients
//   q.parallel_for<>(range<1>(NET_WIDTH * batch_size), [=](id<1> idx) {
//      B[idx] = (T2)loss_gradients[idx + offset_f1];
//    }).wait();

//   // Perform GEMM operation
//   oneapi::mkl::blas::row_major::gemm(
//       q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
//       layer_in_width, NET_WIDTH, batch_size, 1, A, batch_size, B, NET_WIDTH, 0,
//       C, NET_WIDTH);

//   // Update gradients_device with the computed values
//   q.parallel_for<>(range<1>(layer_in_width * NET_WIDTH), [=](id<1> idx) {
//      grads_device[offset_c + idx] += C[idx];
//    }).wait();

//   // Calculating flops
//   // A Matrix
//   flops = (layer_in_width * batch_size) +
//           2 * layer_in_width * NET_WIDTH * batch_size;
//   //(2 FLOPs per multiplication and addition).
// }

// void test_xmx(int batch_size, int input_width, int output_width,
//               int m_n_hidden_layers, float* forward, int& duration_us,
//               int& flops_per_s, int& memory_bandwidth) {
//   const int N_ITERS = BATCH_CHUNK / TM;
//   int m_n_hidden_matrices = m_n_hidden_layers - 1;
//   queue q;
//   DeviceMem<bf16> weights_matrices;
//   weights_matrices.allocate(WIDTH * input_width +
//                                 (WIDTH * WIDTH) * m_n_hidden_matrices +
//                                 WIDTH * output_width,
//                             q);
//   weights_matrices.initialize_arange(q, input_width, WIDTH, output_width,
//                                      m_n_hidden_matrices);

//   q.submit([&](handler& cgh) {
//      local_accessor<bf16> act_mem = local_accessor<bf16>(
//          range<1>(SHMEM_SIZE + BATCH_CHUNK * SKEW) * WIDTH / 64, cgh);
//      local_accessor<float> act_mem_temp = local_accessor<float>(
//          range<1>(SHMEM_SIZE + BATCH_CHUNK * SKEW) * WIDTH / 64, cgh);

//      cgh.parallel_for(
//          nd_range<1>(batch_size * WG_SIZE / BATCH_CHUNK, WG_SIZE),
//          [=](nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
//            auto a = act_mem.get_pointer();
//            auto at = act_mem_temp.get_pointer();
//            auto wg = item.get_group();
//            const int wg_idx = wg.get_group_id();
//            const int elem_idx = BATCH_CHUNK * wg_idx;
//            for (int k = 0; k < m_n_hidden_matrices; k++) {
//              matmul_act_layer<WIDTH, N_ITERS, false>(
//                  item, Activation::ReLU, a, at,
//                  weights_matrices.data() + input_width * WIDTH +
//                      k * WIDTH * WIDTH,
//                  (forward + elem_idx * WIDTH + ((k + 1) * WIDTH) * batch_size),
//                  nullptr, 0);
//            }
//          });
//    }).wait();
// }
// // Function to initialize a buffer with random uniform values
// template <typename T>
// void initialize_buffer_with_random_uniform(queue& q, T* buffer_name,
//                                            size_t size, float min_val,
//                                            float max_val) {
//   std::default_random_engine rng;
//   std::uniform_real_distribution<float> dist(min_val, max_val);

//   // Get a buffer access for the device
//   auto buffer_device = buffer<T, 1>(buffer_name, range<1>(size))
//                            .get_access<access::mode::write>(q);

//   // Fill the buffer with random uniform values
//   for (size_t i = 0; i < size; ++i) {
//     buffer_device[i] = static_cast<T>(dist(rng));
//   }
// }

// template <typename T1, typename T2>
// void test_dgemm(int batch_size, int input_width, int output_width,
//                 int m_n_hidden_layers, int& duration_us, int& flops_per_s,
//                 int& memory_bandwidth, int use_xmx) {
//   queue q = queue();
//   int m_n_hidden_matrices = m_n_hidden_layers - 1;
//   DeviceMem<T1> grads_matrices = DeviceMem<T1>(
//       WIDTH * input_width + (WIDTH * WIDTH) * m_n_hidden_matrices +
//           WIDTH * output_width,
//       q);

//   grads_matrices.initialize_uniform(q, 0.1);

//   T2* out_inter =
//       malloc_device<T2>(batch_size * WIDTH * m_n_hidden_matrices, q);

//   T2* forward = malloc_device<T2>(
//       batch_size * (input_width + output_width + WIDTH * m_n_hidden_layers), q);
//   T2* A_dgemm =
//       sycl::aligned_alloc_device<T2>(SHMEM_SIZE, batch_size * WIDTH, q);
//   T2* B_dgemm =
//       sycl::aligned_alloc_device<T2>(SHMEM_SIZE, batch_size * WIDTH, q);
//   T2* C_dgemm = sycl::aligned_alloc_device<T2>(
//       SHMEM_SIZE, WIDTH * WIDTH,
//       q);  // WIDTH * WIDTH is the maximum, for the first layer, it's
//            // technically input_width * WIDTH
//   int flops = 0;
//   auto start = std::chrono::high_resolution_clock::now();

//   for (int it = 0; it < ITERATIONS; it++) {
//     if (use_xmx) {
//       test_xmx(batch_size, input_width, output_width, m_n_hidden_layers,
//                forward, duration_us, flops_per_s, memory_bandwidth);
//     } else {
//       for (int k = 0; k < m_n_hidden_matrices; k++) {
//         dgemm_multiply<64, Activation::ReLU, T1, T2>(
//             q, grads_matrices.data(), out_inter, forward, A_dgemm, B_dgemm,
//             C_dgemm, k, m_n_hidden_matrices, batch_size, input_width, flops);
//       }
//     }
//   }

//   flops *= m_n_hidden_matrices;

//   // Stop the clock
//   auto end = std::chrono::high_resolution_clock::now();

//   // Calculate the duration
//   auto duration =
//       std::chrono::duration_cast<std::chrono::microseconds>(end - start);

//   duration_us = static_cast<int>(duration.count());
//   flops_per_s = flops * 1e6 / duration_us;
//   memory_bandwidth = 0;

//   // Print the message
//   std::cout << "Use XMX: " << use_xmx
//             << ", grads datatype = " << typeid(T1).name()
//             << ", array datatype = " << typeid(T2).name()
//             << ", batch_size = " << batch_size
//             << ", input_width = " << input_width
//             << ", output_width = " << output_width
//             << ", m_n_hidden_layers = " << m_n_hidden_layers
//             << ", we have duration = " << duration_us
//             << " microseconds, flops per second = " << flops_per_s
//             << " FLOPS/s, and memory bandwidth = " << memory_bandwidth
//             << " bytes per second." << std::endl;
//   // free
//   grads_matrices.free_mem(q);
//   free(out_inter, q);
//   free(forward, q);
//   free(A_dgemm, q);
//   free(B_dgemm, q);
//   free(C_dgemm, q);
// }

// template <typename T1, typename T2>
// void benchmark_dgemm_time(int use_xmx) {
//   //   std::vector<uint32_t> batch_sizes = {1 << 21, 1 << 20, 1 << 19, 1 <<
//   //   18,
//   //                                        1 << 17, 1 << 16, 1 << 15, 1 <<
//   //                                        14};
//   //   std::vector<uint32_t> hidden_layers = {2, 3, 4};
//   //   std::vector<uint32_t> input_widths = {1, 8, 16, 32, 64};
//   //   std::vector<uint32_t> output_widths = {1, 8, 16, 32, 64};
//   std::vector<uint32_t> batch_sizes = {1 << 6};
//   std::vector<uint32_t> hidden_layers = {4};
//   std::vector<uint32_t> input_widths = {64};
//   std::vector<uint32_t> output_widths = {64};

//   nlohmann::json bench_result;
//   std::string method = "SwiftNet";

//   bench_result[method] = nlohmann::json::array();

//   int duration_us;
//   int flops_per_s;
//   int memory_bandwidth;

//   for (uint32_t batch_size : batch_sizes) {
//     for (uint32_t input_width : input_widths) {
//       for (uint32_t hidden_layer : hidden_layers) {
//         for (uint32_t output_width : output_widths) {
//           // Loop over the types and call the function

//           test_dgemm<T1, T2>(batch_size, input_width, output_width,
//                              hidden_layer, duration_us, flops_per_s,
//                              memory_bandwidth, use_xmx);

//           bench_result[method].push_back(
//               {{"batch_size", batch_size},
//                {"input_width", input_width},
//                {"output_width", output_width},
//                {"hidden_layer", hidden_layer},
//                {"duration_us", duration_us},
//                {"flops_per_s", flops_per_s},
//                {"memory_bandwidth", memory_bandwidth},
//                {"datatype_grads", typeid(T1).name()},
//                {"datatype_arrays", typeid(T2).name()}});
//         }
//       }
//     }
//   }

//   std::string json_string = bench_result.dump(4);
//   std::ofstream out{"bench_dgemm_results.json"};
//   out << json_string;
// }

// int main() {
//   benchmark_dgemm_time<float, float>(0);
//   benchmark_dgemm_time<bf16, float>(0);
//   benchmark_dgemm_time<bf16, float>(1);  // xmx
//   // No GEMM with bf16 yet
//   //   benchmark_dgemm_time<float, bf16>();
//   //   benchmark_dgemm_time<bf16, bf16>();
//   return 0;
// }