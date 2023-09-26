#define TM 8
#define TK 16
#define TN 8
#define SKEW 4

#define SG_SIZE 8
#define WG_SIZE 8*SG_SIZE

#define BATCH_CHUNK 16
#define SHMEM_SIZE 1024

#include "SwiftNetMLP.h"

#include "common.h"
#include "mkl.h"
#include "mkl_omp_offload.h"
#include "oneapi/mkl.hpp"
#include "trainer.h"

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
using bf16 = sycl::ext::oneapi::bfloat16;

void get_float_as_integers_own(float value, int& integer_val,
                               int& fractional_val) {
  // careful with the code. Leading zeros not shown in fractional_val. This is
  // only to debug whether it's zero or non-zero. only for 4 decimals after
  // comma

  // Extract the integer part
  int integerPart = static_cast<int>(value);

  // Extract the fractional part as an integer
  int fractionalPart =
      //   static_cast<int>(std::fabs((value - static_cast<float>(integerPart))
      //   *
      static_cast<int>(((value - static_cast<float>(integerPart)) *
                        1000000));  // Adjust the multiplier as needed
  integer_val = integerPart;
  fractional_val = fractionalPart;
}
void get_float_as_integers_own(float value, int& integer_val,
                               int& fractional_val, int& leading_zeros) {
  // careful with the code. Leading zeros not shown in fractional_val. This is
  // only to debug whether it's zero or non-zero. only for 4 decimals after
  // comma

  // Extract the integer part
  int integerPart = static_cast<int>(value);

  // Extract the fractional part as an integer
  int fractionalPart =
      //   static_cast<int>(std::fabs((value - static_cast<float>(integerPart))
      //   *
      static_cast<int>(((value - static_cast<float>(integerPart)) *
                        1000000));  // Adjust the multiplier as needed
  integer_val = integerPart;
  fractional_val = fractionalPart;
  //   // Print the integer and fractional parts as integers
  //   std::cout << "Integer part: " << integerPart << std::endl;
  //   std::cout << "Fractional part: " << fractionalPart << std::endl;

  // Count leading zeros in the fractional part
  leading_zeros = 0;
}
template <typename T>
void printVector(const std::vector<T>& vec) {
  std::cout << "Vector contents: ";
  for (const T& element : vec) {
    std::cout << element << " ";
  }
  std::cout << std::endl;
}

/**
 * Execute the action made by a work-group to calculate the next layer.
 *
 * @param item          The SYCL nd_item representing the work item.
 * @param activation    The type of activation to be applied.
 * @param a             Pointer to activation memory.
 * @param a             Pointer to temporary activation memory.
 * @param weights_layer Pointer to weights for the layer.
 * @param out_inter     Pointer to output intermediate memory.
 * @param out           Pointer to final output memory.
 * @param forward_act   Optional pointer to forward activation memory.
 * @tparam WIDTH        Width of the layer.
 * @tparam N_ITERS      Number of iterations.
 * @tparam BACKWARD     Flag indicating if backward activation is applied.
 */
template <int WIDTH, int N_ITERS, bool BACKWARD = false>
void matmul_act_layer(
    nd_item<1> item, Activation activation,
    multi_ptr<bf16, access::address_space::local_space, (access::decorated)2> a,
    multi_ptr<float, access::address_space::local_space, (access::decorated)2>
        at,
    bf16* weights_layer, float* out_inter, float* forward_act = nullptr,
    int print = 0) {
  // Get sub-group and local IDs

  auto sg = item.get_sub_group();
  int id = item.get_local_id() % SG_SIZE;
  int sgId = sg.get_group_id();

  // Device pointers to memory
  device_ptr<bf16> w(weights_layer);
  device_ptr<float> o(out_inter);
  device_ptr<float> f(forward_act);

  // Define matrices and load weights
  joint_matrix<sub_group, bf16, use::a, TM, TK, layout::row_major> act_matrix;
  joint_matrix<sub_group, bf16, use::b, TK, TN,
               sycl::ext::intel::experimental::matrix::layout::packed>
      weight_matrix0;
  joint_matrix<sub_group, bf16, use::b, TK, TN,
               sycl::ext::intel::experimental::matrix::layout::packed>
      weight_matrix1;
  joint_matrix<sub_group, bf16, use::b, TK, TN,
               sycl::ext::intel::experimental::matrix::layout::packed>
      weight_matrix2;
  joint_matrix<sub_group, bf16, use::b, TK, TN,
               sycl::ext::intel::experimental::matrix::layout::packed>
      weight_matrix3;
  joint_matrix<sub_group, float, use::accumulator, TM, TN> result_matrix;

  joint_matrix_load(sg, weight_matrix0,
                    w + TN * 2 * sgId + TK / 2 * 0 * WIDTH * 2, WIDTH * 2);
  joint_matrix_load(sg, weight_matrix1,
                    w + TN * 2 * sgId + TK / 2 * 1 * WIDTH * 2, WIDTH * 2);
  joint_matrix_load(sg, weight_matrix2,
                    w + TN * 2 * sgId + TK / 2 * 2 * WIDTH * 2, WIDTH * 2);
  joint_matrix_load(sg, weight_matrix3,
                    w + TN * 2 * sgId + TK / 2 * 3 * WIDTH * 2, WIDTH * 2);
//   for (int m_idx = 0; m_idx < 4; m_idx++) {
//     for (int w_idx = TN * 2 * sgId + TK / 2 * m_idx * WIDTH * 2;
//          w_idx < TN * 2 * sgId + TK / 2 * m_idx * WIDTH * 2 + WIDTH * 2;
//          w_idx++) {
//       int b_first;
//       int b_second;
//       int b_zeroes;
//       get_float_as_integers_own(w[w_idx], b_first, b_second, b_zeroes);
//       int wg_id = item.get_group().get_group_id();
//       int sg_id = item.get_sub_group().get_group_id();
//       int local_id = item.get_local_id();
//       static const CONSTANT char FMT[] =
//           "W_idx: %d, m_idx: %d,  group id: %d, sub_group "
//           "id: %d, local id: "
//           "%d, overall id: %d, val: %d.%d \n ";
//       if (wg_id == 0 && id == 0 && print) {
//         sycl::ext::oneapi::experimental::printf(
//             FMT, w_idx, m_idx, int(wg_id), int(sg_id), int(local_id),
//             int(wg_id * WG_SIZE + local_id), b_first, b_second);
//       }
//     }
//   }
#pragma unroll
  for (int l = 0; l < N_ITERS; l++) {
    joint_matrix_fill(sg, result_matrix, 0.0f);

    // Load activation matrix and perform matrix multiplication and accumulation
    joint_matrix_load(sg, act_matrix, a + TK * 0 + TM * l * (WIDTH + SKEW),
                      WIDTH + SKEW);

    result_matrix =
        joint_matrix_mad(sg, act_matrix, weight_matrix0, result_matrix);
    joint_matrix_load(sg, act_matrix, a + TK * 1 + TM * l * (WIDTH + SKEW),
                      WIDTH + SKEW);
    result_matrix =
        joint_matrix_mad(sg, act_matrix, weight_matrix1, result_matrix);
    joint_matrix_load(sg, act_matrix, a + TK * 2 + TM * l * (WIDTH + SKEW),
                      WIDTH + SKEW);
    result_matrix =
        joint_matrix_mad(sg, act_matrix, weight_matrix2, result_matrix);
    joint_matrix_load(sg, act_matrix, a + TK * 3 + TM * l * (WIDTH + SKEW),
                      WIDTH + SKEW);
    result_matrix =
        joint_matrix_mad(sg, act_matrix, weight_matrix3, result_matrix);

    joint_matrix_store(sg, result_matrix,
                       at + TN * sgId + TM * l * (WIDTH + SKEW), WIDTH + SKEW,
                       layout::row_major);
  }

#pragma unroll
  for (int i = 0; i < N_ITERS; i++) {
    if (BACKWARD) {
      int stride = (WIDTH + SKEW);
      int offset = TN * sgId * (WIDTH + SKEW) + TM * i + id;
      // Apply backward activation matrix if required
      matrix_activation_backward<float, float, bf16, SG_SIZE>(
          activation, at, f, a, TN * sgId + (WIDTH + SKEW) * TM * i + id,
          (WIDTH + SKEW));
    } else {
      //   Apply forward activation matrix
      matrix_activation<float, bf16, SG_SIZE>(
          activation, at, a, TN * sgId + (WIDTH + SKEW) * TM * i + id,
          (WIDTH + SKEW));
    }
  }

  if (out_inter) {
#pragma unroll
    for (int i = 0; i < N_ITERS; i++) {
      for (int k = 0; k < TM; k++) {
        // Copy results to the output intermediate matrix
        if (BACKWARD) {
          out_inter[TN * sgId + WIDTH * TM * i + k * WIDTH + id] =
              a[TN * sgId + (WIDTH + SKEW) * TM * i + k * (WIDTH + SKEW) + id];
        } else {
          out_inter[TN * sgId + WIDTH * TM * i + k * WIDTH + id] =
              a[TN * sgId + (WIDTH + SKEW) * TM * i + k * (WIDTH + SKEW) + id];
        }
      }
    }
  }
}

/**
 * Loads input data into the activation memory using a static pattern for work
 * groups.
 *
 * @param item      The SYCL nd_item representing the work item.
 * @param a   Pointer to the activation memory.
 * @param input     Pointer to the input data.
 * @tparam WIDTH    Width of the data.
 * @tparam N_ITERS  Number of iterations.
 */
template <int WIDTH, int N_ITERS>
void workgroup_prefetch(
    nd_item<1> item,
    multi_ptr<bf16, access::address_space::local_space, (access::decorated)2> a,
    const bf16* input, int print = 0) {
  // Get local ID and sub-group information
  int id = item.get_local_id() % SG_SIZE;
  auto sg = item.get_sub_group();
  int sgId = sg.get_group_id();
  int wg_id = item.get_group().get_group_id();

  int sg_id = item.get_sub_group().get_group_id();
  int local_id = item.get_local_id();
#pragma unroll
  for (int i = 0; i < N_ITERS; i++) {
    for (int k = 0; k < TM; k++) {
      // Copy input data to activation memory
      a[TN * sgId + (WIDTH + SKEW) * TM * i + k * (WIDTH + SKEW) + id] =
          input[TN * sgId + WIDTH * TM * i + k * WIDTH + id];
    }
  }
}
/**
 * Loads input data into the activation memory using a static pattern for work
 * groups.
 *
 * @param item      The SYCL nd_item representing the work item.
 * @param a   Pointer to the activation memory.
 * @param input     Pointer to the input data.
 * @tparam WIDTH    Width of the data.
 * @tparam N_ITERS  Number of iterations.
 */
template <int WIDTH, int N_ITERS>
void workgroup_prefetch(
    nd_item<1> item,
    multi_ptr<bf16, access::address_space::local_space, (access::decorated)2> a,
    const float* forward, int print = 0) {
  // Get local ID and sub-group information
  int id = item.get_local_id() % SG_SIZE;
  auto sg = item.get_sub_group();
  int sgId = sg.get_group_id();
  int wg_id = item.get_group().get_group_id();

  int sg_id = item.get_sub_group().get_group_id();
  int local_id = item.get_local_id();
#pragma unroll
  for (int i = 0; i < N_ITERS; i++) {
    for (int k = 0; k < TM; k++) {
      // Copy input data to activation memory
      a[TN * sgId + (WIDTH + SKEW) * TM * i + k * (WIDTH + SKEW) + id] =
          forward[TN * sgId + WIDTH * TM * i + k * WIDTH + id];
    }
  }
}
/*
 * Writes data from shared memory to the output thread block using a static
 * pattern for work groups.
 *
 * @param item               The SYCL nd_item representing the work item.
 * @param a                  Pointer to the shared memory containing activation
 * data.
 * @param output_threadblock Pointer to the output thread block.
 * @tparam WIDTH             Width of the data.
 * @tparam N_ITERS           Number of iterations.
 */
template <int WIDTH, int N_ITERS>
void workgroup_write_output_static(
    nd_item<1> item,
    multi_ptr<bf16, access::address_space::local_space, (access::decorated)2> a,
    float* output_threadblock) {
  // Get local ID and sub-group information
  int id = item.get_local_id() % SG_SIZE;
  auto sg = item.get_sub_group();
  int sgId = sg.get_group_id();

#pragma unroll
  for (int i = 0; i < N_ITERS; i++) {
    for (int k = 0; k < TM; k++) {
      output_threadblock[TN * sgId + WIDTH * TM * i + k * WIDTH + id] =
          a[TN * sgId + (WIDTH + SKEW) * TM * i + k * (WIDTH + SKEW) + id];
    }
  }
}

/**
 * Performs forward dynamic input layer computation within a work group.
 *
 * @param item                  The SYCL nd_item representing the work item.
 * @param activation            The type of activation to be applied.
 * @param a                     Pointer to the shared memory containing
 * activation data.
 * @param at                    Pointer to the sfohared memory containing
 * temporary activation data.
 * @param input                 Pointer to the input data.
 * @param weights_layer         Pointer to weights for the layer.
 * @param out_intermediate_layer Pointer to output intermediate memory for the
 * layer.
 * @param input_width           Width of the input data.
 * @tparam WIDTH                Width of the layer.
 * @tparam N_ITERS              Number of iterations.
 */
template <int WIDTH, int N_ITERS>
void workgroup_matmul_act_dynamic(
    nd_item<1> item, Activation activation,
    multi_ptr<bf16, access::address_space::local_space, (access::decorated)2> a,
    multi_ptr<float, access::address_space::local_space, (access::decorated)2>
        at,
    bf16* input, bf16* weights_layer, float* out_intermediate_layer,
    const int input_width, const int batch_size) {
  auto sg = item.get_sub_group();
  int id = item.get_local_id() % SG_SIZE;
  int sgId = sg.get_group_id();

  // Device pointers to memory
  device_ptr<bf16> in(input);
  device_ptr<bf16> w(weights_layer);
  device_ptr<float> o(out_intermediate_layer);

  // Define matrices and load weights
  joint_matrix<sub_group, bf16, use::a, TM, TK, layout::row_major> act_matrix;
  joint_matrix<sub_group, bf16, use::b, TK, TN,
               sycl::ext::intel::experimental::matrix::layout::packed>
      weight_matrix;

  joint_matrix<sub_group, float, use::accumulator, TM, TN> result_matrix;

  const int n_operations = input_width / TK;

  for (int l = 0; l < N_ITERS; l++) {
    joint_matrix_fill(sg, result_matrix, 0.0f);
    for (int i = 0; i < n_operations; i++) {
      joint_matrix_load(sg, act_matrix, in + 16 * i * batch_size + 16 * l,
                        input_width);
      joint_matrix_load(sg, weight_matrix,
                        w + TN * 2 * sgId + TK / 2 * i * input_width * 2,
                        input_width * 2);

      result_matrix =
          joint_matrix_mad(sg, act_matrix, weight_matrix, result_matrix);

      joint_matrix_store(sg, result_matrix, at + TN * sgId + TM * l * WIDTH,
                         WIDTH, layout::row_major);
    }

    matrix_activation<float, bf16, SG_SIZE>(
        activation, at, a, TN * sgId + TM * l * WIDTH + id, WIDTH);
  }
  for (int i = 0; i < N_ITERS; i++) {
    for (int k = 0; k < TM; k++) {
      o[TN * sgId + WIDTH * TM * i + k * WIDTH + id] =
          (bf16)at[TN * sgId + WIDTH * TM * i + k * WIDTH + id];
    }
  }
}
/**
 * Performs forward computation for the last layer within a work group.
 *
 * @param item              The SYCL nd_item representing the work item.
 * @param activation        The type of activation to be applied.
 * @param a                 Pointer to activation memory.
 * @param weights_layer     Pointer to weights for the layer.
 * @param out               Pointer to the output memory.
 * @tparam WIDTH            Width of the layer.
 * @tparam N_ITERS          Number of iterations.
 */
template <int WIDTH, int N_ITERS>
void workgroup_last_layer(
    nd_item<1> item,
    multi_ptr<bf16, access::address_space::local_space, (access::decorated)2> a,
    bf16* weights_layer, float* out) {
  auto sg = item.get_sub_group();
  int sgId = sg.get_group_id();
  device_ptr<bf16> w(weights_layer);
  device_ptr<float> o(out);

  joint_matrix<sub_group, bf16, use::a, TM, TK, layout::row_major> act_matrix;

  joint_matrix<sub_group, bf16, use::b, TK, TN,
               sycl::ext::intel::experimental::matrix::layout::packed>
      weight_matrix0;
  joint_matrix<sub_group, bf16, use::b, TK, TN,
               sycl::ext::intel::experimental::matrix::layout::packed>
      weight_matrix1;
  joint_matrix<sub_group, bf16, use::b, TK, TN,
               sycl::ext::intel::experimental::matrix::layout::packed>
      weight_matrix2;
  joint_matrix<sub_group, bf16, use::b, TK, TN,
               sycl::ext::intel::experimental::matrix::layout::packed>
      weight_matrix3;
  joint_matrix<sub_group, float, use::accumulator, TM, TN> result_matrix;

  joint_matrix_load(sg, weight_matrix0,
                    w + TN * 2 * sgId + TK / 2 * 0 * WIDTH * 2, WIDTH * 2);
  joint_matrix_load(sg, weight_matrix1,
                    w + TN * 2 * sgId + TK / 2 * 1 * WIDTH * 2, WIDTH * 2);
  joint_matrix_load(sg, weight_matrix2,
                    w + TN * 2 * sgId + TK / 2 * 2 * WIDTH * 2, WIDTH * 2);
  joint_matrix_load(sg, weight_matrix3,
                    w + TN * 2 * sgId + TK / 2 * 3 * WIDTH * 2, WIDTH * 2);
#pragma unroll

  for (int l = 0; l < N_ITERS; l++) {
    joint_matrix_fill(sg, result_matrix, 0.0f);

    joint_matrix_load(sg, act_matrix, a + TK * 0 + TM * l * (WIDTH + SKEW),
                      (WIDTH + SKEW));
    result_matrix =
        joint_matrix_mad(sg, act_matrix, weight_matrix0, result_matrix);
    joint_matrix_load(sg, act_matrix, a + TK * 1 + TM * l * (WIDTH + SKEW),
                      (WIDTH + SKEW));
    result_matrix =
        joint_matrix_mad(sg, act_matrix, weight_matrix1, result_matrix);
    joint_matrix_load(sg, act_matrix, a + TK * 2 + TM * l * (WIDTH + SKEW),
                      (WIDTH + SKEW));
    result_matrix =
        joint_matrix_mad(sg, act_matrix, weight_matrix2, result_matrix);
    joint_matrix_load(sg, act_matrix, a + TK * 3 + TM * l * (WIDTH + SKEW),
                      (WIDTH + SKEW));
    result_matrix =
        joint_matrix_mad(sg, act_matrix, weight_matrix3, result_matrix);

    joint_matrix_store(sg, result_matrix,
                       o + TN * sgId + TM * l * (WIDTH + SKEW), (WIDTH + SKEW),
                       layout::row_major);
  }
}

/**
 * Kernel function for the forward pass of the Swift MLP model.
 *
 * @param item                  The SYCL nd_item representing the work item.
 * @param output_activation     The type of activation to be applied for output
 * layer.
 * @param input                 Pointer to input data.
 * @param weights_layer         Pointer to weights for the layer.
 * @param out_intermediate_layer Pointer to intermediate output memory.
 * @param act_mem               Pointer to activation memory.
 * @param act_mem_temp          Pointer to temporary activation memory.
 * @param out                   Pointer to output memory.
 * @param input_width           Width of the input data.
 * @param output_width          Width of the output data.
 * @param n_hidden_matmuls      Number of hidden matrix multiplications.
 * @param batch_size            Batch size of the data.
 * @tparam WIDTH                Width of the layers.
 * @tparam N_ITERS              Number of iterations.
 * @tparam activation           Type of activation for hidden layers.
 */
template <int WIDTH, int N_ITERS, Activation activation, bool INFERENCE = false>
void kernel_swift_mlp(nd_item<1> item, const Activation output_activation,
                      bf16* input, bf16* weights_layer,
                      float* out_intermediate_layer,
                      local_accessor<bf16> act_mem,
                      local_accessor<float> act_mem_temp, float* out,
                      const uint32_t input_width, const uint32_t output_width,
                      const uint32_t n_hidden_matmuls, int batch_size) {
  auto a = act_mem.get_pointer();
  auto at = act_mem_temp.get_pointer();

  // Handle first layer because it has different input

  auto wg = item.get_group();
  const int wg_idx = wg.get_group_id();
  const int elem_idx = BATCH_CHUNK * wg_idx;
  const int hidden_weight_lenght = WIDTH * WIDTH;
  //   for (int w_idx = 0; w_idx < 1024; w_idx++) {
  //     int b_first;
  //     int b_second;
  //     int b_zeroes;
  //     get_float_as_integers_own(at[w_idx], b_first, b_second, b_zeroes);
  //     int wg_id = item.get_group().get_group_id();
  //     int sg_id = item.get_sub_group().get_group_id();
  //     int local_id = item.get_local_id();
  //     static const CONSTANT char FMT[] =
  //         "at 0,  w_idx: %d,  group id: %d, sub_group "
  //         "id: %d, local id: "
  //         "%d, overall id: %d, val: %d.%d \n ";
  //     if ((wg_id == 0)) {
  //       sycl::ext::oneapi::experimental::printf(
  //           FMT, w_idx, int(wg_id), int(sg_id), int(local_id),
  //           int(wg_id * WG_SIZE + local_id), b_first, b_second);
  //     }
  //   }
  if (input_width == WIDTH) {
    workgroup_prefetch<WIDTH, N_ITERS>(item, a, input + elem_idx * WIDTH);
    matmul_act_layer<WIDTH, N_ITERS, false>(
        item, activation, a, at, weights_layer,
        !INFERENCE ? (out_intermediate_layer + elem_idx * WIDTH) : nullptr,
        nullptr, 0);
  } else if (input_width >= 16) {
    // if < 16, then handled in forward pass via gemm
    workgroup_matmul_act_dynamic<WIDTH, N_ITERS>(
        item, activation, a, at, input + elem_idx * WIDTH, weights_layer,
        !INFERENCE ? (out_intermediate_layer + elem_idx * WIDTH) : nullptr,
        input_width, batch_size);
  } else {
    // load fwd into act_mem
    workgroup_prefetch<WIDTH, N_ITERS>(
        item, a,
        out_intermediate_layer + input_width * batch_size + elem_idx * WIDTH);
  }

  //   for (int w_idx = 0; w_idx < 1024; w_idx++) {
  //     int b_first;
  //     int b_second;
  //     int b_zeroes;
  //     get_float_as_integers_own(at[w_idx], b_first, b_second, b_zeroes);
  //     int wg_id = item.get_group().get_group_id();
  //     int sg_id = item.get_sub_group().get_group_id();
  //     int local_id = item.get_local_id();
  //     static const CONSTANT char FMT[] =
  //         "at2,  w_idx: %d,  group id: %d, sub_group "
  //         "id: %d, local id: "
  //         "%d, overall id: %d, val: %d.%d \n ";
  //     if ((wg_id == 0)) {
  //       sycl::ext::oneapi::experimental::printf(
  //           FMT, w_idx, int(wg_id), int(sg_id), int(local_id),
  //           int(wg_id * WG_SIZE + local_id), b_first, b_second);
  //     }
  //   }
  //   Handle hidden layers all together
  //   std::cout << "n_hidden_matmuls: " << n_hidden_matmuls << std::endl;
  const int first_weight_length = input_width * WIDTH;

  for (int k = 0; k < n_hidden_matmuls; k++) {
    matmul_act_layer<WIDTH, N_ITERS, false>(
        item, activation, a, at,
        weights_layer + first_weight_length + k * hidden_weight_lenght,
        !INFERENCE ? (out_intermediate_layer + elem_idx * WIDTH +
                      (k * WIDTH + input_width) * batch_size)
                   : nullptr,
        nullptr, 0);
  }

  // Handle output layer
  if (output_width > 16) {
    if (INFERENCE) {
      workgroup_write_output_static<WIDTH, N_ITERS>(
          item, a,
          out_intermediate_layer + elem_idx * WIDTH +
              (n_hidden_matmuls * WIDTH + input_width) * batch_size);
    }
  } else if (out) {
    // static const CONSTANT char FMT[] =
    //     "first_weight_length: %d hidden_weight_lenght: %d, "
    //     "n_hidden_matmuls:%d, write to: %d\n";
    // if (wg_idx == 0) {
    //   sycl::ext::oneapi::experimental::printf(FMT, int(first_weight_length),
    //                                           int(hidden_weight_lenght),
    //                                           int(n_hidden_matmuls));
    // }
    workgroup_last_layer<WIDTH, N_ITERS>(
        item, a,
        weights_layer + first_weight_length +
            hidden_weight_lenght * n_hidden_matmuls,
        out + elem_idx * WIDTH);
  }
}

/**
 * Performs forward pass for the Swift MLP model.
 *
 * @param q                  SYCL queue for command submission.
 * @param output_activation The type of activation to be applied for output
 * layer.
 * @param weights            Device memory containing weights for the model.
 * @param inputs             Device memory containing input data.
 * @param intermediate_output Pointer to intermediate output memory.
 * @param act_mem            Pointer to activation memory.
 * @param act_mem_temp       Pointer to temporary activation memory.
 * @param output             Device memory for storing the output.
 * @param n_hidden_layers    Number of hidden layers.
 * @param input_width        Width of the input data.
 * @param output_width       Width of the output data.
 * @param batch_size         Batch size of the data.
 * @tparam WIDTH             Width of the layers.
 * @tparam activation        Type of activation for hidden layers.
 */
template <int WIDTH, Activation activation, bool INFERENCE>
void mlp_swift_forward(queue q, Activation output_activation,
                       const DeviceMem<bf16>& weights,
                       const DeviceMem<bf16>& inputs,
                       float* intermediate_output, DeviceMem<float>& output,
                       const int n_hidden_layers, const int input_width,
                       const int output_width, int batch_size) {
  const int N_ITERS = BATCH_CHUNK / TM;

  q.submit([&](handler& cgh) {
     local_accessor<bf16> act_mem = local_accessor<bf16>(
         range<1>(SHMEM_SIZE + BATCH_CHUNK * SKEW) * WIDTH / 64, cgh);
     local_accessor<float> act_mem_temp = local_accessor<float>(
         range<1>(SHMEM_SIZE + BATCH_CHUNK * SKEW) * WIDTH / 64, cgh);

     cgh.parallel_for(
         nd_range<1>(batch_size * WG_SIZE / BATCH_CHUNK, WG_SIZE),
         [=](nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
           kernel_swift_mlp<WIDTH, N_ITERS, activation, INFERENCE>(
               item, output_activation, inputs.data(), weights.data(),
               intermediate_output, act_mem, act_mem_temp, output.data(),
               input_width, output_width, n_hidden_layers - 1, batch_size);
         });
   }).wait();
}

/**
 * Kernel function for backpropagation in the SwiftNet model.
 *
 * @param item             The SYCL nd_item representing the work item.
 * @param deltas           Pointer to the losses deltas from where the
 * backpropagation starts.
 * @param a                Pointer to loss gradients for backpropagation.
 * @param at               Pointer to temporary loss gradients memory.
 * @param grads            Pointer to gradients for weight updates.
 * @param weights          Pointer to weights of the model.
 * @param forward          Pointer to forward pass intermediate outputs.
 * @param out_inter        Pointer to intermediate output memory.
 * @param n_hidden_matmuls Number of hidden matrix multiplications.
 * @param batch_size       Batch size of the data.
 * @tparam WIDTH           Width of the layers.
 * @tparam N_ITERS         Number of iterations.
 * @tparam ACTIVATION      Type of activation for hidden layers.
 */
template <int WIDTH, int N_ITERS, Activation ACTIVATION>
void kernel_swiftnet_backward(nd_item<1> item, bf16* deltas,
                              local_accessor<bf16> deltas_layers,
                              local_accessor<float> delta_temp, bf16* grads,
                              bf16* weights, float* forward, float* out_inter,
                              uint32_t n_hidden_matmuls, int batch_size) {
  auto a = deltas_layers.get_pointer();
  auto at = delta_temp.get_pointer();
  auto sg = item.get_sub_group();

  int groupId = item.get_group(0);
  int sgId = sg.get_group_id();
  const int layer_length = WIDTH * batch_size;

  workgroup_prefetch<WIDTH, N_ITERS>(item, a,
                                     deltas + groupId * BATCH_CHUNK * WIDTH, 0);
  // Iterate through hidden layers for backpropagation
  for (int k = 0; k < n_hidden_matmuls; k++) {
    matmul_act_layer<WIDTH, N_ITERS, true>(
        item, ACTIVATION, a, at,
        weights + WIDTH * WIDTH * (n_hidden_matmuls - k),
        out_inter + groupId * BATCH_CHUNK * WIDTH +
            (n_hidden_matmuls - k - 1) * layer_length,
        forward + WIDTH * batch_size +
            WIDTH * batch_size * (n_hidden_matmuls - k - 1) +
            groupId * BATCH_CHUNK * WIDTH,
        1);
  }
}

/**
 * Multiplies matrices using DGEMM for gradient calculation in the SwiftNet
 * model.
 *
 * @param q                 SYCL queue for command submission.
 * @param grads_device      Pointer to device memory for gradients.
 * @param loss_gradients    Pointer to loss gradients for backpropagation.
 * @param fwd               Pointer to forward pass intermediate outputs.
 * @param A                 Pointer to matrix A (calculated activations).
 * @param B                 Pointer to matrix B (loss gradients).
 * @param C                 Pointer to matrix C (result of DGEMM).
 * @param k                 Index of the hidden matrix multiplication.
 * @param m_n_hidden_matrices Number of hidden matrix multiplications.
 * @param batch_size        Batch size of the data.
 * @tparam WIDTH            Width of the matrices.
 * @tparam ACTIVATION       Type of activation for hidden layers.
 */
template <int WIDTH, Activation ACTIVATION>
void dgemm_multiply(queue q, bf16* grads_device, float* loss_gradients,
                    float* fwd, float* A, float* B, float* C, int k,
                    int m_n_hidden_matrices, int batch_size) {
  const int layer_lenght = WIDTH * batch_size;
  const int n_hidden_matrices = m_n_hidden_matrices;

  // Calculate matrix A using the given activation function
  q.parallel_for<>(range<1>(WIDTH * batch_size), [=](id<1> idx) {
    int i = idx / batch_size;
    int j = idx % batch_size;
    A[i * batch_size + j] = (float)elt_activation_ret<float>(
        ACTIVATION,
        fwd[i + j * WIDTH + (n_hidden_matrices - k - 1) * layer_lenght]);
    // int b_first;
    // int b_second;
    // int b_zeroes;
    // static const CONSTANT char FMT[] = "A[%d]: %d.%d (%d zeroes), \n";
    // get_float_as_integers_own(A[idx], b_first, b_second, b_zeroes);
    // sycl::ext::oneapi::experimental::printf(FMT, int(idx), b_first, b_second,
    //                                         b_zeroes);
  });

  // Assign matrix B using loss gradients
  q.parallel_for<>(range<1>(WIDTH * batch_size), [=](id<1> idx) {
    B[idx] =
        (float)loss_gradients[idx + (n_hidden_matrices - k - 1) * layer_lenght];
    // int b_first;
    // int b_second;
    // int b_zeroes;
    // static const CONSTANT char FMT[] = "B[%d]: %d.%d (%d zeroes), \n";
    // get_float_as_integers_own(B[idx], b_first, b_second, b_zeroes);
    // sycl::ext::oneapi::experimental::printf(FMT, int(idx), b_first, b_second,
    //                                         b_zeroes);
  });

  // Perform GEMM operation
  oneapi::mkl::blas::row_major::gemm(
      q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
      WIDTH, WIDTH, batch_size, 1, A, batch_size, B, WIDTH, 0, C, WIDTH);

  // Update gradients_device with the computed values
  q.parallel_for<>(range<1>(WIDTH * WIDTH), [=](id<1> idx) {
    // int b_first;
    // int b_second;
    // int b_zeroes;
    // static const CONSTANT char FMT[] = "C last[%d]: %d.%d (%d zeroes), \n";
    // get_float_as_integers_own(C[idx], b_first, b_second, b_zeroes);
    // sycl::ext::oneapi::experimental::printf(
    //     FMT, int((m_n_hidden_matrices - k - 1) * WIDTH * WIDTH + idx),
    //     b_first, b_second, b_zeroes);
    grads_device[(m_n_hidden_matrices - k - 1) * WIDTH * WIDTH + idx] += C[idx];
  });
}

/**
 * Backward pass for gradient calculation in the SwiftNet model.
 *
 * @param q                 SYCL queue for command submission.
 * @param weights_transposed Pointer to transposed and packed weights.
 * @param deltas            Pointer to delta values.
 * @param grads_matrices    Pointer to matrices for gradients.
 * @param out_inter         Pointer to intermediate outputs.
 * @param delta_temp        Pointer to temporary delta memory.
 * @param forward           Pointer to forward pass intermediate outputs.
 * @param A_dgemm           Pointer to matrix A for DGEMM.
 * @param B_dgemm           Pointer to matrix B for DGEMM.
 * @param C_dgemm           Pointer to matrix C for DGEMM.
 * @param n_hidden_matmuls Number of hidden matrix multiplications.
 * @param batch_size        Batch size of the data.
 * @tparam WIDTH            Width of the matrices.
 * @tparam ACTIVATION       Type of activation for hidden layers.
 */
template <int WIDTH, Activation ACTIVATION>
void mlp_swiftnet_backward(queue q, DeviceMem<bf16>& weights_transposed,
                           DeviceMem<bf16>& deltas,
                           DeviceMem<bf16>& grads_matrices, float* out_inter,
                           float* delta_temp_, float* forward, float* A_dgemm,
                           float* B_dgemm, float* C_dgemm,
                           const uint32_t n_hidden_matmuls, int batch_size) {
  // here, weights are already transposed and packed
  // in deltas, the last layer has already been calculated

  const int N_ITERS = BATCH_CHUNK / TM;

  // Execute the kernel for backward pass
  q.submit([&](handler& h) {
     local_accessor<bf16> deltas_layers = local_accessor<bf16>(
         range<1>(SHMEM_SIZE + BATCH_CHUNK * SKEW) * WIDTH / 64, h);
     local_accessor<float> delta_temp = local_accessor<float>(
         range<1>(SHMEM_SIZE + BATCH_CHUNK * SKEW) * WIDTH / 64, h);

     // Execute DGEMM multiply for each hidden layer
     h.parallel_for(nd_range<1>(batch_size * WG_SIZE / BATCH_CHUNK, WG_SIZE),
                    [=](nd_item<1> item)
                        [[intel::reqd_sub_group_size(SG_SIZE)]] {
                          kernel_swiftnet_backward<WIDTH, N_ITERS, ACTIVATION>(
                              item, deltas.data(), deltas_layers, delta_temp,
                              grads_matrices.data(), weights_transposed.data(),
                              forward, out_inter, n_hidden_matmuls, batch_size);
                        });
   }).wait();

  for (int k = 0; k < n_hidden_matmuls; k++) {
    dgemm_multiply<WIDTH, ACTIVATION>(q, grads_matrices.data(), out_inter,
                                      forward, A_dgemm, B_dgemm, C_dgemm, k,
                                      n_hidden_matmuls, batch_size);
  }
}

/**
 * Constructor for the SwiftNetMLP class.
 *
 * @param q                  SYCL queue for command submission.
 * @param input_width        Width of the input data.
 * @param output_width       Width of the output data.
 * @param n_hidden_layers    Number of hidden layers.
 * @param activation         Activation function for hidden layers.
 * @param output_activation  Activation function for the output layer.
 * @param batch_size         Batch size of the data.
 * @tparam WIDTH             Width of the matrices.
 */
template <int WIDTH>
SwiftNetMLP<WIDTH>::SwiftNetMLP(queue q, int input_width, int output_width,
                                int n_hidden_layers, Activation activation,
                                Activation output_activation, int batch_size)
    : m_inputs_width{input_width},
      m_net_width{WIDTH},
      m_output_width{output_width},
      m_n_hidden_layers{n_hidden_layers},
      m_activation{activation},
      m_output_activation{output_activation},
      m_batch_size{batch_size} {
  // Store provided parameters
  m_q = q;
  m_n_hidden_matrices = m_n_hidden_layers - 1;

  // Allocate memory for various matrices
  m_weightsT_matrices.allocate(
      m_net_width * m_inputs_width +
          (m_net_width * m_net_width) * m_n_hidden_matrices +
          m_net_width * m_output_width,
      m_q);
  m_weights_matrices.allocate(
      m_net_width * m_inputs_width +
          (m_net_width * m_net_width) * m_n_hidden_matrices +
          m_net_width * m_output_width,
      m_q);
  m_weights_matrices_inferences.allocate(
      m_net_width * m_inputs_width +
          (m_net_width * m_net_width) * m_n_hidden_matrices +
          m_net_width * m_output_width,
      m_q);
  m_grads_matrices.allocate(
      m_net_width * m_inputs_width +
          (m_net_width * m_net_width) * m_n_hidden_matrices +
          m_net_width * m_output_width,
      m_q);

  // Initialize constants and allocations
  m_alignment = SHMEM_SIZE;

  // Allocate and initialize various memory buffers
  m_forward =
      malloc_device<float>(m_batch_size * (m_inputs_width + m_output_width +
                                           WIDTH * m_n_hidden_layers),
                           q);

  m_A_forward =
      sycl::aligned_alloc_device<float>(m_alignment, m_inputs_width * WIDTH, q);
  m_B_forward =
      sycl::aligned_alloc_device<float>(m_alignment, m_output_width * WIDTH, q);
  m_C_forward = sycl::aligned_alloc_device<float>(
      m_alignment, m_output_width * m_batch_size, q);

  m_out_inter = malloc_device<float>(
      m_batch_size * (m_output_width + WIDTH * m_n_hidden_matrices), q);
  //   malloc_device<float>(m_batch_size * WIDTH * (m_n_hidden_layers), q);
  m_deltas_temp = sycl::aligned_alloc_device<float>(
      m_alignment, m_output_width * m_batch_size, q);
  m_deltas.allocate(m_output_width * m_batch_size, q);

  m_A_backward =
      sycl::aligned_alloc_device<float>(m_alignment, WIDTH * m_batch_size, q);
  m_B_backward = sycl::aligned_alloc_device<float>(
      m_alignment, m_batch_size * m_output_width, q);
  m_C_backward =
      sycl::aligned_alloc_device<float>(m_alignment, WIDTH * m_output_width, q);

  m_A_backward_last_layer = sycl::aligned_alloc_device<float>(
      m_alignment, m_batch_size * m_output_width, q);
  m_B_backward_last_layer =
      sycl::aligned_alloc_device<float>(m_alignment, m_output_width * WIDTH, q);
  m_C_backward_last_layer =
      sycl::aligned_alloc_device<float>(m_alignment, WIDTH * m_batch_size, q);
  m_D_backward_last_layer =
      sycl::aligned_alloc_device<float>(m_alignment, WIDTH * m_batch_size, q);
  m_E_backward_last_layer =
      sycl::aligned_alloc_device<float>(m_alignment, m_batch_size * WIDTH, q);
  m_F_backward_last_layer =
      sycl::aligned_alloc_device<float>(m_alignment, WIDTH * WIDTH, q);

  m_A_dgemm =
      sycl::aligned_alloc_device<float>(m_alignment, m_batch_size * WIDTH, q);
  m_B_dgemm =
      sycl::aligned_alloc_device<float>(m_alignment, m_batch_size * WIDTH, q);
  m_C_dgemm = sycl::aligned_alloc_device<float>(m_alignment, WIDTH * WIDTH, q);
}

template <int WIDTH>
SwiftNetMLP<WIDTH>::~SwiftNetMLP() {}
/**
 * Get a pointer to the gradients matrices.
 *
 * @return A pointer to the gradients matrices.
 */
template <int WIDTH>
DeviceMem<bf16>* SwiftNetMLP<WIDTH>::get_grads_matrices() {
  return &m_grads_matrices;
}

/**
 * Get a pointer to the weights matrices.
 *
 * @return A pointer to the weights matrices.
 */
template <int WIDTH>
DeviceMem<bf16>* SwiftNetMLP<WIDTH>::get_weights_matrices() {
  return &m_weights_matrices;
}

/**
 * Get a pointer to the transposed weights matrices.
 *
 * @return A pointer to the transposed weights matrices.
 */
template <int WIDTH>
DeviceMem<bf16>* SwiftNetMLP<WIDTH>::get_weightsT_matrices() {
  return &m_weightsT_matrices;
}

/**
 * Initialize parameters for the neural network.
 * This function initializes the weights matrices with uniform random values.
 */
template <int WIDTH>
void SwiftNetMLP<WIDTH>::initialize_params() {
  // Initialize weights matrices with uniform random values, you can choose a
  // different initialization ( look in DeviceMem.cpp )
  //   m_weights_matrices.initialize_uniform(
  //   0.01, m_weightsT_matrices, m_inputs_width, m_net_width, m_output_width,
  //   m_n_hidden_matrices, m_q);
  //   m_weights_matrices.initialize_uniform(
  //       0.01, m_weightsT_matrices, m_inputs_width, m_net_width,
  //       m_output_width, m_n_hidden_matrices, m_q);
  m_weights_matrices.intitialize_he_normal(m_inputs_width, m_q);

  //   m_weights_matrices.initialize_constant(0.01, m_q);

  //   m_weights_matrices.initialize_arange(m_q, m_inputs_width, m_net_width,
  //                                        m_output_width,
  //                                        m_n_hidden_matrices);

  m_weights_matrices.make_transposed(m_weightsT_matrices, m_inputs_width,
                                     m_net_width, m_output_width,
                                     m_n_hidden_matrices, m_q);
};
// };

/**
 * Save the neural network parameters to a file.
 *
 * @param filename The name of the file to save the parameters to.
 */
template <int WIDTH>
void SwiftNetMLP<WIDTH>::save_to_file(std::string filename) {
  // Open the file for writing
  std::ofstream file;
  file.open(filename);

  // Write parameters to the file
  file << m_inputs_width << "\n";
  file << m_net_width << "\n";
  file << m_output_width << "\n";
  file << m_n_hidden_layers << "\n";
  file << m_n_hidden_matrices << "\n";

  // Write each value of the weights matrices to the file
  for (int i = 0; i < m_weights_matrices.size(); i++) {
    file << m_weights_matrices.data()[i] << "\n";
  }

  // Close the file
  file.close();
  return;
}

/**
 * Load neural network parameters from a file.
 *
 * @param filename The name of the file to load parameters from.
 */
template <int WIDTH>
void SwiftNetMLP<WIDTH>::load_from_file(std::string filename) {
  // Open the file for reading
  std::ifstream file;
  file.open(filename);
  std::string line;

  // Read parameters from the file
  file >> m_inputs_width;
  file >> m_net_width;
  file >> m_output_width;
  file >> m_n_hidden_layers;
  file >> m_n_hidden_matrices;

  // Read each value from the file and set it as a bf16 value in weights
  // matrices
  for (int i = 0; i < m_weights_matrices.size(); i++) {
    float x;
    file >> x;
    m_weights_matrices.data()[i] = bf16(x);
  }

  // Close the file
  file.close();

  // Make the weights matrices transposed using the transposed weights matrices
  m_weights_matrices.make_transposed(m_weightsT_matrices, m_inputs_width,
                                     m_net_width, m_output_width,
                                     m_n_hidden_matrices, m_q);
  return;
}

/**
 * Free memory allocated on the device for various arrays.
 *
 * @param q The SYCL queue used for device operations.
 */
template <int WIDTH>
void SwiftNetMLP<WIDTH>::free_mem(queue q) {
  // Free memory for arrays allocated using sycl::aligned_alloc_device
  free(m_out_inter, q);
  free(m_deltas_temp, q);
  free(m_A_forward, q);
  free(m_B_forward, q);
  free(m_C_forward, q);

  // Free memory for DeviceMem<bf16> arrays using their free_mem member function
  m_deltas.free_mem(q);

  free(m_A_backward, q);
  free(m_B_backward, q);
  free(m_C_backward, q);
  free(m_A_backward_last_layer, q);
  free(m_B_backward_last_layer, q);
  free(m_C_backward_last_layer, q);
  free(m_D_backward_last_layer, q);
  free(m_E_backward_last_layer, q);
  free(m_F_backward_last_layer, q);
  free(m_A_dgemm, q);
  free(m_B_dgemm, q);
  free(m_C_dgemm, q);
}

/**
 * Perform a forward pass of the SwiftNetMLP model.
 *
 * @param input The input data on the device.
 * @param forward Pointer to the forward intermediate array.
 * @param act_mem Pointer to activation memory.
 * @param act_mem_temp Pointer to temporary activation memory.
 * @param A Temporary array A for matrix multiplication.
 * @param B Temporary array B for matrix multiplication.
 * @param C Temporary array C for matrix multiplication.
 * @param output The output data on the device.
 */
template <int WIDTH>
void SwiftNetMLP<WIDTH>::forward_pass(const DeviceMem<bf16>& input,
                                      float* forward, float* A, float* B,
                                      float* C, DeviceMem<float>& output) {
  // Constants and dimensions

  //   std::vector<bf16> weightsT(m_weightsT_matrices.size());
  //   //   std::cout << " grads T before " << std::endl;
  //   m_q.memcpy(weightsT.data(), m_weightsT_matrices.data(),
  //              m_weightsT_matrices.size() * sizeof(bf16))
  //       .wait();
  //   for (int i = 0; i < weightsT.size(); i++) {
  //     std::cout << "fwd Weight at " << i << ": " << weightsT[i] << std::endl;
  //   }
  //   std::cout << "== == == == == == == == == == == == == == == == == == == ==
  //   == "
  //                "== == == == == == == == "
  //             << std::endl;
  auto output_activation = m_output_activation;
  auto activation = m_activation;
  const int batch_size = m_batch_size;
  const int n_hidden_matrices = m_n_hidden_matrices;
  const int net_width = m_net_width;
  const int inputs_width = m_inputs_width;
  const int output_width = m_output_width;

  // Static assertion and assertion checks
  static_assert(WIDTH % 16 == 0, "Width must be a multiple of 16.");
  assert(m_batch_size % 64 == 0);

  // Get a pointer to the weights matrices data
  auto p = m_weights_matrices.data();
  if (inputs_width < 16) {
    m_q.parallel_for<>(range<1>(input.size()),
                       [=](id<1> idx) { forward[idx] = input.data()[idx]; });

    m_q.parallel_for<>(range<1>(inputs_width * WIDTH), [=](id<1> idx) {
      A[idx] = (float)p[toPackedLayoutCoord(idx, inputs_width, WIDTH)];
    });

    oneapi::mkl::blas::row_major::gemm(
        m_q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
        m_batch_size, WIDTH, inputs_width, 1, forward, inputs_width, A, WIDTH,
        0, forward + input.size(), WIDTH);
    m_q.parallel_for<>(range<1>(WIDTH * m_batch_size),
                       [=](id<1> idx) {
                         forward[idx + (inputs_width * batch_size)] =
                             elt_activation_ret<float>(
                                 activation,
                                 forward[idx + (inputs_width * batch_size)]);
                       })
        .wait();
  }
  // Perform forward pass based on activation function
  switch (m_activation) {
    case Activation::None:
      mlp_swift_forward<WIDTH, Activation::None, false>(
          m_q, m_output_activation, m_weights_matrices, input,
          forward + input.size(), output, m_n_hidden_layers, m_inputs_width,
          m_output_width, m_batch_size);
      break;
    case Activation::Exponential:
      mlp_swift_forward<WIDTH, Activation::Exponential, false>(
          m_q, m_output_activation, m_weights_matrices, input,
          forward + input.size(), output, m_n_hidden_layers, m_inputs_width,
          m_output_width, m_batch_size);
      break;
    case Activation::Sigmoid:
      mlp_swift_forward<WIDTH, Activation::Sigmoid, false>(
          m_q, m_output_activation, m_weights_matrices, input,
          forward + input.size(), output, m_n_hidden_layers, m_inputs_width,
          m_output_width, m_batch_size);
      break;
    case Activation::ReLU:
      mlp_swift_forward<WIDTH, Activation::ReLU, false>(
          m_q, m_output_activation, m_weights_matrices, input,
          forward + input.size(), output, m_n_hidden_layers, m_inputs_width,
          m_output_width, m_batch_size);
      break;
    case Activation::LeakyReLU:
      mlp_swift_forward<WIDTH, Activation::LeakyReLU, false>(
          m_q, m_output_activation, m_weights_matrices, input,
          forward + input.size(), output, m_n_hidden_layers, m_inputs_width,
          m_output_width, m_batch_size);
      break;
    case Activation::Squareplus:
      mlp_swift_forward<WIDTH, Activation::Squareplus, false>(
          m_q, m_output_activation, m_weights_matrices, input,
          forward + input.size(), output, m_n_hidden_layers, m_inputs_width,
          m_output_width, m_batch_size);
      break;
    case Activation::Softplus:
      mlp_swift_forward<WIDTH, Activation::Softplus, false>(
          m_q, m_output_activation, m_weights_matrices, input,
          forward + input.size(), output, m_n_hidden_layers, m_inputs_width,
          m_output_width, m_batch_size);
      break;
    case Activation::Tanh:
      mlp_swift_forward<WIDTH, Activation::Tanh, false>(
          m_q, m_output_activation, m_weights_matrices, input,
          forward + input.size(), output, m_n_hidden_layers, m_inputs_width,
          m_output_width, m_batch_size);
      break;
    default:
      return;
  }

  // Handle the case when output_width is greater than 16

  // TODO: Not doing check again. According to Darius, it's faster to use
  // workgroup_last_layer, but somehow results are flawed
  //   if (m_output_width > 16) {
  m_q.parallel_for<>(range<1>(m_output_width * m_net_width), [=](id<1> idx) {
    B[idx] =
        (float)p[toPackedLayoutCoord(idx, net_width, output_width) +
                 net_width * (inputs_width + n_hidden_matrices * net_width)];
  });

  oneapi::mkl::blas::row_major::gemm(
      m_q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
      m_batch_size, m_output_width, WIDTH, 1,
      forward + (n_hidden_matrices * WIDTH + m_inputs_width) * m_batch_size,
      WIDTH, B, m_output_width, 0, C, m_output_width);

  const int intermediate_output_size =
      m_batch_size * (WIDTH * m_n_hidden_layers);

  m_q.parallel_for<>(range<1>(m_output_width * m_batch_size),
                     [=](id<1> idx) {
                       output.data()[idx] =
                           elt_activation_ret<float>(output_activation, C[idx]);
                       forward[intermediate_output_size + input.size() + idx] =
                           elt_activation_ret<float>(output_activation, C[idx]);
                     })
      .wait();
  //   } else {
  //     m_q.parallel_for<>(range<1>(m_output_width * m_batch_size),
  //                        [=](id<1> idx) {
  //                          output.data()[idx] = elt_activation_ret<float>(
  //                              output_activation, output.data()[idx]);
  //                        })
  //         .wait();
  //   }

  //   output.copy_to_host(out, m_q);
  //   for (int i = 0; i < out.size(); i++) {
  //     std::cout << i << ": " << out[i] << std::endl;
  //   }
}

template <int WIDTH>
void SwiftNetMLP<WIDTH>::inference(const DeviceMem<bf16>& input, float* forward,
                                   float* A, float* B, float* C,
                                   DeviceMem<float>& output) {
  const int input_size = input.size();
  const int n_hidden_matrices = m_n_hidden_matrices;
  const int net_width = m_net_width;
  const int inputs_width = m_inputs_width;
  const int output_width = m_output_width;

  auto activation = m_activation;
  const int batch_size = m_batch_size;

  static_assert(WIDTH % 16 == 0, "Width must be a multiply of 16.");
  assert(m_batch_size % 64 == 0);
  auto p = m_weights_matrices.data();

  if (inputs_width < 16) {
    m_q.parallel_for<>(range<1>(input.size()),
                       [=](id<1> idx) { forward[idx] = input.data()[idx]; });

    m_q.parallel_for<>(range<1>(inputs_width * WIDTH), [=](id<1> idx) {
      A[idx] = (float)p[toPackedLayoutCoord(idx, inputs_width, WIDTH)];
    });

    oneapi::mkl::blas::row_major::gemm(
        m_q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
        m_batch_size, WIDTH, inputs_width, 1, forward, inputs_width, A, WIDTH,
        0, forward + input.size(), WIDTH);
    m_q.parallel_for<>(range<1>(WIDTH * m_batch_size),
                       [=](id<1> idx) {
                         forward[idx + (inputs_width * batch_size)] =
                             elt_activation_ret<float>(
                                 activation,
                                 forward[idx + (inputs_width * batch_size)]);
                       })
        .wait();
  }
  switch (m_activation) {
    case Activation::None:
      mlp_swift_forward<WIDTH, Activation::None, true>(
          m_q, m_output_activation, m_weights_matrices, input, forward, output,
          m_n_hidden_layers, m_inputs_width, m_output_width, m_batch_size);
      break;
    case Activation::Exponential:
      mlp_swift_forward<WIDTH, Activation::Exponential, true>(
          m_q, m_output_activation, m_weights_matrices, input, forward, output,
          m_n_hidden_layers, m_inputs_width, m_output_width, m_batch_size);
      break;
    case Activation::Sigmoid:
      mlp_swift_forward<WIDTH, Activation::Sigmoid, true>(
          m_q, m_output_activation, m_weights_matrices, input, forward, output,
          m_n_hidden_layers, m_inputs_width, m_output_width, m_batch_size);
      break;
    case Activation::ReLU:
      mlp_swift_forward<WIDTH, Activation::ReLU, true>(
          m_q, m_output_activation, m_weights_matrices, input, forward, output,
          m_n_hidden_layers, m_inputs_width, m_output_width, m_batch_size);
      break;
    case Activation::LeakyReLU:
      mlp_swift_forward<WIDTH, Activation::LeakyReLU, true>(
          m_q, m_output_activation, m_weights_matrices, input, forward, output,
          m_n_hidden_layers, m_inputs_width, m_output_width, m_batch_size);
      break;
    case Activation::Squareplus:
      mlp_swift_forward<WIDTH, Activation::Squareplus, true>(
          m_q, m_output_activation, m_weights_matrices, input, forward, output,
          m_n_hidden_layers, m_inputs_width, m_output_width, m_batch_size);
      break;
    case Activation::Softplus:
      mlp_swift_forward<WIDTH, Activation::Softplus, true>(
          m_q, m_output_activation, m_weights_matrices, input, forward, output,
          m_n_hidden_layers, m_inputs_width, m_output_width, m_batch_size);
      break;
    case Activation::Tanh:
      mlp_swift_forward<WIDTH, Activation::Tanh, true>(
          m_q, m_output_activation, m_weights_matrices, input, forward, output,
          m_n_hidden_layers, m_inputs_width, m_output_width, m_batch_size);
      break;
    default:
      throw std::runtime_error{"Unsupported activation."};
  }

  //   if (m_output_width > 16) {
  m_q.parallel_for<>(range<1>(m_output_width * m_net_width), [=](id<1> idx) {
    B[idx] = p[toPackedLayoutCoord(idx, net_width, output_width) +
               net_width * (inputs_width + n_hidden_matrices * net_width)];
  });
  oneapi::mkl::blas::row_major::gemm(
      m_q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
      m_batch_size, m_output_width, WIDTH, 1,
      forward + (n_hidden_matrices * WIDTH + inputs_width) * m_batch_size,
      WIDTH, B, m_output_width, 0, C, m_output_width);
  auto output_activation = m_output_activation;

  m_q.parallel_for<>(range<1>(m_output_width * m_batch_size),
                     [=](id<1> idx) {
                       output.data()[idx] =
                           elt_activation_ret<float>(output_activation, C[idx]);
                     })
      .wait();
  //   }
}

/**
 * Perform matrix multiplications and activation backpropagation for the last
 * layer (beginning of the backward pass) .
 *
 * @param grads The gradients on the device.
 * @param forward Pointer to the forward intermediate array.
 * @param loss The loss gradients on the device.
 * @param batch_size The batch size.
 * @param A Temporary array A for matrix multiplication.
 * @param B Temporary array B for matrix multiplication.
 * @param C Temporary array C for matrix multiplication.
 * @param D Temporary array D for activation backpropagation.
 * @param E Temporary array E for activation backpropagation.
 * @param F Temporary array F for matrix multiplication.
 */
template <int WIDTH>
void SwiftNetMLP<WIDTH>::dgemm_last_layer_backward(DeviceMem<bf16>& grads,
                                                   float* forward,
                                                   DeviceMem<bf16>& loss,
                                                   int batch_size, float* A,
                                                   float* B, float* C, float* D,
                                                   float* E, float* F) {
  auto p_w = m_weightsT_matrices.data();

  auto p_g = m_grads_matrices.data();
  //   std::cout << "Total weights: " << m_weightsT_matrices.size() <<
  //   std::endl;
  const int offset_w = m_n_hidden_matrices * m_net_width * m_net_width +
                       m_net_width * m_inputs_width;
  const int offset_g = m_inputs_width * m_net_width +
                       (m_n_hidden_matrices - 1) * m_net_width * m_net_width;
  const int offset_f =
      (m_inputs_width + (m_n_hidden_matrices - 1) * batch_size) * m_net_width;
  //   std::cout << "offsets: " << offset_w << "," << offset_g << ", " <<
  //   offset_f
  //             << std::endl;
  const int output_width = m_output_width;
  const int net_width = m_net_width;

  int i = 0;
  int j = 0;
  auto activation = m_activation;

  m_q.parallel_for<>(range<1>(grads.size()),
                     [=](id<1> idx) {
                       A[idx] = (float)loss.data()[idx];

                       //    int b_first;
                       //    int b_second;
                       //    static const CONSTANT char FMT[] = "A[%d]
                       //    %d.%d,\n"; get_float_as_integers_own(A[idx],
                       //    b_first, b_second);
                       //    sycl::ext::oneapi::experimental::printf(
                       //        FMT, int(idx), b_first, b_second);
                     })
      .wait();
  //   m_q.parallel_for<>(range<1>(m_weightsT_matrices.size()),
  //                      [=](id<1> idx) {
  //                        int b_first;
  //                        int b_second;
  //                        static const CONSTANT char FMT[] = "pw[%d]
  //                        %d.%d,\n"; get_float_as_integers_own(p_w[idx],
  //                        b_first, b_second);
  //                        sycl::ext::oneapi::experimental::printf(
  //                            FMT, int(idx), b_first, b_second);
  //                      })
  //       .wait();
  m_q.parallel_for<>(range<1>(m_output_width * WIDTH),
                     [=](id<1> idx) {
                       B[idx] =
                           p_w[offset_w + toPackedLayoutCoord(idx, output_width,
                                                              net_width)];
                       //    int b_first;
                       //    int b_second;
                       //    static const CONSTANT char FMT[] = "B[%d]
                       //    %d.%d,\n"; get_float_as_integers_own(B[idx],
                       //    b_first, b_second);
                       //    sycl::ext::oneapi::experimental::printf(
                       //        FMT, int(idx), b_first, b_second);
                     })
      .wait();

  oneapi::mkl::blas::row_major::gemm(
      m_q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
      //   m_q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans,
      batch_size, m_net_width, m_output_width, 1, A, m_output_width, B,
      m_net_width, 0, C, m_net_width);

  //   m_q.parallel_for<>(range<1>(m_net_width * m_output_width),
  //                      [=](id<1> idx) {
  //                        int b_first;
  //                        int b_second;
  //                        static const CONSTANT char FMT[] = "C[%d]
  //                        %d.%d,\n"; get_float_as_integers_own(C[idx],
  //                        b_first, b_second);
  //                        sycl::ext::oneapi::experimental::printf(
  //                            FMT, int(idx), b_first, b_second);
  //                      })
  //       .wait();
  m_q.parallel_for<>(range<1>(WIDTH * batch_size),
                     [=](id<1> idx) {
                       int i = idx / batch_size;
                       int j = idx % batch_size;
                       D[i * batch_size + j] = elt_activation_ret<float>(
                           activation, forward[offset_f + j * net_width + i]);

                       //    int b_first;
                       //    int b_second;
                       //    static const CONSTANT char FMT[] = "D[%d]
                       //    %d.%d\n"; get_float_as_integers_own(D[i *
                       //    batch_size + j], b_first,
                       //                              b_second);
                       //    sycl::ext::oneapi::experimental::printf(
                       //        FMT, int(i * batch_size + j), b_first,
                       //        b_second);
                     })
      .wait();

  m_q.parallel_for<>(range<1>(m_net_width * batch_size),
                     [=](id<1> idx) {
                       elt_activation_bwd<float, float, float>(
                           activation, C[idx], forward[offset_f + idx], E[idx]);
                       loss.data()[idx] = (bf16)E[idx];
                       //    int b_first;
                       //    int b_second;
                       //    static const CONSTANT char FMT[] = "E[%d]
                       //    %d.%d\n"; get_float_as_integers_own(E[idx],
                       //    b_first, b_second);
                       //    sycl::ext::oneapi::experimental::printf(
                       //        FMT, int(idx), b_first, b_second);
                     })
      .wait();

  oneapi::mkl::blas::row_major::gemm(
      m_q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
      m_net_width, m_net_width, batch_size, 1, D, batch_size, E, m_net_width, 0,
      F, m_net_width);

  m_q.parallel_for<>(range<1>(m_net_width * m_net_width),
                     [=](id<1> idx) {
                       p_g[idx + offset_g] = (float)F[idx];
                       //    int b_first;
                       //    int b_second;
                       //    static const CONSTANT char FMT[] = "F[%d]
                       //    %d.%d,\n"; get_float_as_integers_own(F[idx],
                       //    b_first, b_second);
                       //    sycl::ext::oneapi::experimental::printf(
                       //        FMT, int(idx), b_first, b_second);
                     })
      .wait();
}

/**
 * Perform the backward pass of the neural network.
 *
 * @param input The input data on the device.
 * @param grads The gradients on the device.
 * @param out_inter Intermediate array for storing outputs.
 * @param delta_temp Temporary array for deltas.
 * @param loss Loss array on the device.
 * @param A Temporary array A for activation backpropagation.
 * @param B Temporary array B for activation backpropagation.
 * @param C Temporary array C for matrix multiplication.
 * @param A_backward_last_layer Temporary array A for last layer backward
 * pass.
 * @param B_backward_last_layer Temporary array B for last layer backward
 * pass.
 * @param C_backward_last_layer Temporary array C for last layer backward
 * pass.
 * @param D_backward_last_layer Temporary array D for last layer backward
 * pass.
 * @param E_backward_last_layer Temporary array E for last layer backward
 * pass.
 * @param F_backward_last_layer Temporary array F for last layer backward
 * pass.
 * @param A_dgemm Temporary array A for DGEMM.
 * @param B_dgemm Temporary array B for DGEMM.
 * @param C_dgemm Temporary array C for DGEMM.
 * @param forward Pointer to the forward intermediate array.
 */
template <int WIDTH>
void SwiftNetMLP<WIDTH>::backward_pass(
    const DeviceMem<bf16>& input, DeviceMem<bf16>& grads, float* out_inter,
    float* delta_temp, DeviceMem<bf16> loss, float* A, float* B, float* C,
    float* A_backward_last_layer, float* B_backward_last_layer,
    float* C_backward_last_layer, float* D_backward_last_layer,
    float* E_backward_last_layer, float* F_backward_last_layer, float* A_dgemm,
    float* B_dgemm, float* C_dgemm, float* forward) {
  int batch_size = m_batch_size;
  auto p = m_grads_matrices.data();
  int s = m_grads_matrices.size();
  auto activation = m_activation;
  auto output_activation = m_output_activation;
  const int offset_grad = m_n_hidden_matrices * m_net_width * m_net_width +
                          m_inputs_width * m_net_width;
  const int offset_f = m_inputs_width * batch_size +
                       m_n_hidden_matrices * m_net_width * batch_size;

  //   const size_t alignment = 1024;
  //   std::vector<bf16> grad_vec_out(m_grads_matrices.size());
  //   std::cout << " grads before " << std::endl;
  //   m_q.memcpy(grad_vec_out.data(), m_grads_matrices.data(),
  //              m_grads_matrices.size() * sizeof(bf16))
  //       .wait();
  //   for (int i = 0; i < grad_vec_out.size(); i++) {
  //     std::cout << "Grad at " << i << ": " << grad_vec_out[i] << std::endl;
  //   }
  //   std::cout << "== == == == == == == == == == == == == == == == == == == ==
  //   == "
  //                "== == == == == == == == "
  //             << std::endl;

  //   std::vector<bf16> weights(m_weights_matrices.size());
  //   std::cout << " grads T before " << std::endl;
  //   m_q.memcpy(weights.data(), m_weights_matrices.data(),
  //              m_weights_matrices.size() * sizeof(bf16))
  //       .wait();
  //   for (int i = 0; i < weights.size(); i++) {
  //     std::cout << "Weight at " << i << ": " << weights[i] << std::endl;
  //   }
  //   std::cout << "== == == == == == == == == == == == == == == == == == == ==
  //   == "
  //                "== == == == == == == == "
  //             << std::endl;
  //   std::vector<bf16> weightsT(m_weightsT_matrices.size());
  //   m_q.memcpy(weightsT.data(), m_weightsT_matrices.data(),
  //              m_weightsT_matrices.size() * sizeof(bf16))
  //       .wait();
  //   for (int i = 0; i < weightsT.size(); i++) {
  //     std::cout << "Weight T at " << i << ": " << weightsT[i] << std::endl;
  //   }
  //   std::cout << "== == == == == == == == == == == == == == == == == == == ==
  //   == "
  //                "== == == == == == == == "
  //             << std::endl;
  // Compute activation backpropagation using parallel_for
  m_q.parallel_for<>(range<1>(WIDTH * batch_size),
                     [=](id<1> idx) {
                       int i = idx / batch_size;
                       int j = idx % batch_size;
                       A[i * batch_size + j] = elt_activation_ret<float>(
                           activation, forward[offset_f + j * WIDTH + i]);
                       //    int b_first;
                       //    int b_second;
                       //    static const CONSTANT char FMT[] =
                       //        "Fwd[%d] at i: %d,j: %d: %d.%d,  ";
                       //    get_float_as_integers_own(A[i * batch_size + j],
                       //    b_first, b_second);
                       //    get_float_as_integers_own(forward[offset_f + j *
                       //    WIDTH + i], b_first,
                       //                              b_second);
                       //    sycl::ext::oneapi::experimental::printf(FMT,
                       //    int(i
                       //    * batch_size + j),
                       //                                            i, j,
                       //                                            b_first,
                       //                                            b_second);
                     })
      .wait();
  // Compute output activation backpropagation using parallel_for and copy to
  // loss array
  m_q.parallel_for<>(range<1>(batch_size * m_output_width),
                     [=](id<1> idx) {
                       elt_activation_bwd<bf16, float, float>(
                           output_activation, grads.data()[idx],
                           forward[offset_f + batch_size * WIDTH + idx],
                           B[idx]);
                       loss.data()[idx] = (bf16)B[idx];
                       //    int b_first;
                       //    int b_second;
                       //    int a_first;
                       //    int a_second;
                       //    static const CONSTANT char FMT[] = "loss[%d]
                       //    %d.%d,"; get_float_as_integers_own(B[idx],
                       //    b_first, b_second);
                       //    sycl::ext::oneapi::experimental::printf(
                       //        FMT, int(idx), b_first, b_second);
                     })
      .wait();
  // Perform matrix multiplication using MKL BLAS
  oneapi::mkl::blas::row_major::gemm(
      m_q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
      m_net_width, m_output_width, batch_size, 1, A, batch_size, B,
      m_output_width, 0, C, m_output_width);
  // Copy the result back to the gradients matrix
  m_q.parallel_for<>(range<1>(m_net_width * m_output_width),
                     [=](id<1> idx) {
                       p[idx + offset_grad] = (float)C[idx];
                       //    int b_first;
                       //    int b_second;
                       //    static const CONSTANT char FMT[] = "p[%d]
                       //    %d.%d,"; get_float_as_integers_own(C[idx],
                       //    b_first, b_second);
                       //    sycl::ext::oneapi::experimental::printf(
                       //        FMT, int(idx + offset_grad), b_first,
                       //        b_second);
                     })
      .wait();

  // Backpropagation through last layer using dgemm_last_layer_backward
  dgemm_last_layer_backward(grads, forward, loss, batch_size,
                            A_backward_last_layer, B_backward_last_layer,
                            C_backward_last_layer, D_backward_last_layer,
                            E_backward_last_layer, F_backward_last_layer);

  //   std::cout << " grads should be penultimate layer " << std::endl;
  //   m_q.memcpy(grad_vec_out.data(), m_grads_matrices.data(),
  //              m_grads_matrices.size() * sizeof(bf16))
  //       .wait();
  //   for (int i = 0; i < grad_vec_out.size(); i++) {
  //     std::cout << "Grad at " << i << ": " << grad_vec_out[i] << std::endl;
  //   }
  //   std::cout <<
  //   "=========================================================="
  //             << std::endl;
  // Choose appropriate mlp_swiftnet_backward based on activation
  switch (m_activation) {
    case Activation::None:
      mlp_swiftnet_backward<WIDTH, Activation::None>(
          m_q, m_weightsT_matrices, loss, m_grads_matrices, out_inter,
          delta_temp, forward, A_dgemm, B_dgemm, C_dgemm, m_n_hidden_matrices,
          m_batch_size);
      break;
    case Activation::ReLU:
      mlp_swiftnet_backward<WIDTH, Activation::ReLU>(
          m_q, m_weightsT_matrices, loss, m_grads_matrices, out_inter,
          delta_temp, forward, A_dgemm, B_dgemm, C_dgemm, m_n_hidden_matrices,
          m_batch_size);
      break;
    case Activation::LeakyReLU:
      mlp_swiftnet_backward<WIDTH, Activation::LeakyReLU>(
          m_q, m_weightsT_matrices, loss, m_grads_matrices, out_inter,
          delta_temp, forward, A_dgemm, B_dgemm, C_dgemm, m_n_hidden_matrices,
          m_batch_size);
      break;
    case Activation::Exponential:
      mlp_swiftnet_backward<WIDTH, Activation::Exponential>(
          m_q, m_weightsT_matrices, loss, m_grads_matrices, out_inter,
          delta_temp, forward, A_dgemm, B_dgemm, C_dgemm, m_n_hidden_matrices,
          m_batch_size);
      break;
    case Activation::Sigmoid:
      mlp_swiftnet_backward<WIDTH, Activation::Sigmoid>(
          m_q, m_weightsT_matrices, loss, m_grads_matrices, out_inter,
          delta_temp, forward, A_dgemm, B_dgemm, C_dgemm, m_n_hidden_matrices,
          m_batch_size);
      break;
    case Activation::Tanh:
      mlp_swiftnet_backward<WIDTH, Activation::Tanh>(
          m_q, m_weightsT_matrices, loss, m_grads_matrices, out_inter,
          delta_temp, forward, A_dgemm, B_dgemm, C_dgemm, m_n_hidden_matrices,
          m_batch_size);
      break;
    default:
      return;
  }
  //   std::cout << "For all hidden " << std::endl;
  //   m_q.memcpy(grad_vec_out.data(), m_grads_matrices.data(),
  //              m_grads_matrices.size() * sizeof(bf16))
  //       .wait();
  //   for (int i = 0; i < grad_vec_out.size(); i++) {
  //     std::cout << "Grad at " << i << ": " << grad_vec_out[i] << std::endl;
  //   }
  //   std::cout << "
  //   =========================================================="
  //             << std::endl;
  //   // Normalize gradients
  //   m_q.parallel_for<>(range<1>(s), [=](id<1> idx) { p[idx] /= batch_size;
  //   })
  //       .wait();
  //   std::vector<bf16> grad_vec_out(m_grads_matrices.size());

  //   std::cout << " grads final" << std::endl;
  //   m_q.memcpy(grad_vec_out.data(), m_grads_matrices.data(),
  //              m_grads_matrices.size() * sizeof(bf16))
  //       .wait();
  //   for (int i = 0; i < grad_vec_out.size(); i++) {
  //     std::cout << "Grad at " << i << ": " << grad_vec_out[i] << std::endl;
  //   }
}

template <int WIDTH>
void SwiftNetMLP<WIDTH>::set_params(float* params) {
  auto p = m_weights_matrices.data();
  int s = m_weights_matrices.size();

  m_q.parallel_for<>(range<1>(s),
                     [=](id<1> idx) { p[idx] = bf16(params[idx]); })
      .wait();
  m_weights_matrices.make_transposed(m_weightsT_matrices, m_inputs_width, WIDTH,
                                     m_output_width, m_n_hidden_matrices, m_q);
}

template <int WIDTH>
void SwiftNetMLP<WIDTH>::set_params(std::vector<bf16> params) {
  m_weights_matrices.copy_from_host(params, m_q);
  m_weights_matrices.make_transposed(m_weightsT_matrices, m_inputs_width, WIDTH,
                                     m_output_width, m_n_hidden_matrices, m_q);
}

template <int WIDTH>
std::vector<bf16> SwiftNetMLP<WIDTH>::get_weights_matrices_as_vector() {
  std::vector<bf16> list_float(m_weights_matrices.size());
  m_weights_matrices.copy_to_host(list_float, m_q);
  return list_float;
}

template <int WIDTH>
std::vector<bf16> SwiftNetMLP<WIDTH>::get_weightsT_matrices_as_vector() {
  std::vector<bf16> list_float(m_weightsT_matrices.size());
  m_weightsT_matrices.copy_to_host(list_float, m_q);
  return list_float;
}
template class SwiftNetMLP<64>;
template class SwiftNetMLP<128>;