#define TM 8
#define TK 16
#define TN 8

#define SG_SIZE 8
#define WG_SIZE 8*SG_SIZE

#define BATCH_CHUNK 16
#define BATCH_SIZE 4 * 2048
#define SHMEM_SIZE 1024

#include "SwiftNetMLP.h"
#include "trainer.h"
#include "mkl.h"
#include "mkl_omp_offload.h"
#include "common.h"
#include "oneapi/mkl.hpp"

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
using bf16 = sycl::ext::oneapi::bfloat16;



/**
 * Execute the action made by a work-group to calculate the next layer.
 *
 * @param item          The SYCL nd_item representing the work item.
 * @param activation    The type of activation to be applied.
 * @param act_mem       Pointer to activation memory.
 * @param act_mem_temp  Pointer to temporary activation memory.
 * @param weights_layer Pointer to weights for the layer.
 * @param out_inter     Pointer to output intermediate memory.
 * @param out           Pointer to final output memory.
 * @param forward_act   Optional pointer to forward activation memory.
 * @tparam WIDTH        Width of the layer.
 * @tparam N_ITERS      Number of iterations.
 * @tparam BACKWARD     Flag indicating if backward activation is applied.
 */
template <int WIDTH, int N_ITERS, bool BACKWARD = false>
void work_group_layer(nd_item<1> item, Activation activation, multi_ptr<bf16, access::address_space::local_space, (access::decorated)2> a, multi_ptr<float, access::address_space::local_space, (access::decorated)2> at, bf16* weights_layer, float* out_inter, float* forward_act = nullptr) {
   
    constexpr int SKEW = WIDTH % 16 == 0 ? 8 : 0;
    
    // Get sub-group and local IDs
    auto sg = item.get_sub_group();
    int id = item.get_local_id() % SG_SIZE;
    int sgId = sg.get_group_id();
    const int N_BLOCKS = WIDTH / TK;

    // Device pointers to memory
    device_ptr<bf16> w(weights_layer);
    device_ptr<float> o(out_inter);
    device_ptr<float> f(forward_act);

    // Define matrices and load weights
    joint_matrix<sub_group, bf16, use::a, TM, TK, layout::row_major> act_matrix;
    joint_matrix<sub_group, bf16, use::b, TK, TN, sycl::ext::intel::experimental::matrix::layout::packed> weight_matrix0;
    joint_matrix<sub_group, bf16, use::b, TK, TN, sycl::ext::intel::experimental::matrix::layout::packed> weight_matrix1;
    joint_matrix<sub_group, bf16, use::b, TK, TN, sycl::ext::intel::experimental::matrix::layout::packed> weight_matrix2;
    joint_matrix<sub_group, bf16, use::b, TK, TN, sycl::ext::intel::experimental::matrix::layout::packed> weight_matrix3;
    joint_matrix<sub_group, float, use::accumulator, TM, TN> result_matrix;

    joint_matrix_load(sg, weight_matrix0, w + TN * 2 * sgId + TK / 2 * 0 * WIDTH * 2, WIDTH * 2);
    joint_matrix_load(sg, weight_matrix1, w + TN * 2 * sgId + TK / 2 * 1 * WIDTH * 2, WIDTH * 2);
    joint_matrix_load(sg, weight_matrix2, w + TN * 2 * sgId + TK / 2 * 2 * WIDTH * 2, WIDTH * 2);
    joint_matrix_load(sg, weight_matrix3, w + TN * 2 * sgId + TK / 2 * 3 * WIDTH * 2, WIDTH * 2);

#pragma unroll
    for (int l = 0; l < N_ITERS; l++) {
        joint_matrix_fill(sg, result_matrix, 0.0f);

        // Load activation matrix and perform matrix multiplication and accumulation
        joint_matrix_load(sg, act_matrix, a + TK * 0 + TM * l * (WIDTH + SKEW), WIDTH + SKEW);
        result_matrix = joint_matrix_mad(sg, act_matrix, weight_matrix0, result_matrix);
        joint_matrix_load(sg, act_matrix, a + TK * 1 + TM * l * (WIDTH + SKEW), WIDTH + SKEW);
        result_matrix = joint_matrix_mad(sg, act_matrix, weight_matrix1, result_matrix);
        joint_matrix_load(sg, act_matrix, a + TK * 2 + TM * l * (WIDTH + SKEW), WIDTH + SKEW);
        result_matrix = joint_matrix_mad(sg, act_matrix, weight_matrix2, result_matrix);
        joint_matrix_load(sg, act_matrix, a + TK * 3 + TM * l * (WIDTH + SKEW), WIDTH + SKEW);
        result_matrix = joint_matrix_mad(sg, act_matrix, weight_matrix3, result_matrix);

        // Store the result matrix
        joint_matrix_store(sg, result_matrix, at + TN * sgId + TM * l * (WIDTH + SKEW), WIDTH + SKEW, layout::row_major);
    }

#pragma unroll
    for (int i = 0; i < N_ITERS; i++) {
        if (BACKWARD) {
            // Apply backward activation matrix if required
            matrix_activation_backward<float, float, bf16, SG_SIZE>(activation, at, f, a, TN * sgId * (WIDTH + SKEW) + TM * i + id, (WIDTH + SKEW));
        }
        else {
            // Apply forward activation matrix
           matrix_activation<float, bf16, SG_SIZE>(activation, at, a, TN * sgId + (WIDTH + SKEW) * TM * i + id, (WIDTH + SKEW));
        }
    }

    if (out_inter) {
#pragma unroll
        for (int i = 0; i < N_ITERS; i++) {
            for (int k = 0; k < TM; k++) {
               // Copy results to the output intermediate matrix
               out_inter[TN * sgId + WIDTH * TM * i + k * WIDTH + id] = at[TN * sgId + (WIDTH + SKEW) * TM * i + k * (WIDTH + SKEW) + id];
            }
        }
    }
}


/**
 * Loads input data into the activation memory using a static pattern for work groups.
 *
 * @param item      The SYCL nd_item representing the work item.
 * @param act_mem   Pointer to the activation memory.
 * @param input     Pointer to the input data.
 * @tparam WIDTH    Width of the data.
 * @tparam N_ITERS  Number of iterations.
 */
template <int WIDTH, int N_ITERS>
void workgroup_load_input_static(nd_item<1> item, multi_ptr<bf16, access::address_space::local_space, (access::decorated)2> a, const bf16* input) {
    constexpr int SKEW = WIDTH % 16 == 0 ? 8 : 0;
    
    // Get local ID and sub-group information
    int id = item.get_local_id() % SG_SIZE;
    auto sg = item.get_sub_group();
    int sgId = sg.get_group_id();

#pragma unroll
    for (int i = 0; i < N_ITERS; i++) {
        for (int k = 0; k < TM; k++) {
            // Copy input data to activation memory
            //a[TN * sgId + ( WIDTH + SKEW ) * TM * i + k * (WIDTH + SKEW) + id] = input[TN * sgId + WIDTH * TM * i + k * WIDTH + id];
        }
    }
}



/*
 * Writes data from shared memory to the output thread block using a static pattern for work groups.
 *
 * @param item              The SYCL nd_item representing the work item.
 * @param act_shmem         Pointer to the shared memory containing activation data.
 * @param output_threadblock Pointer to the output thread block.
 * @tparam WIDTH            Width of the data.
 * @tparam N_ITERS          Number of iterations.
 */
template <int WIDTH, int N_ITERS>
void workgroup_write_output_static(nd_item<1> item, multi_ptr<bf16, access::address_space::local_space, (access::decorated)2> a, float* output_threadblock) {
    constexpr int SKEW = WIDTH % 16 == 0 ? 8 : 0;
    
    // Get local ID and sub-group information
    int id = item.get_local_id() % SG_SIZE;
    auto sg = item.get_sub_group();
    int sgId = sg.get_group_id();

#pragma unroll
    for (int i = 0; i < N_ITERS; i++) {
        for (int k = 0; k < TM; k++) {
            // Copy data from shared memory to output thread block
            output_threadblock[TN * sgId * WIDTH + TM * i + k * WIDTH + id] = a[TN * sgId + (WIDTH + SKEW) * TM * i + k * WIDTH + id];
        }
    }
}


/**
 * Performs forward dynamic input layer computation within a work group.
 *
 * @param item                  The SYCL nd_item representing the work item.
 * @param activation            The type of activation to be applied.
 * @param act_shmem             Pointer to the shared memory containing activation data.
 * @param input                 Pointer to the input data.
 * @param weights_layer         Pointer to weights for the layer.
 * @param out_intermediate_layer Pointer to output intermediate memory for the layer.
 * @param input_width           Width of the input data.
 * @param batch_size            Batch size of the data.
 * @tparam WIDTH                Width of the layer.
 * @tparam N_ITERS              Number of iterations.
 */
template <int WIDTH, int N_ITERS>
void workgroup_input_layer_forward_dynamic(nd_item<1> item,
    Activation activation,
    multi_ptr<bf16, access::address_space::local_space, (access::decorated)2> a,
    multi_ptr<float, access::address_space::local_space, (access::decorated)2> at,
    const bf16* input,
    bf16* weights_layer,
    float* out_intermediate_layer,
    const int input_width
)
{
    auto sg = item.get_sub_group();
    int id = item.get_local_id() % SG_SIZE;
    int sgId = sg.get_group_id();
    const int N_BLOCKS = WIDTH / TK;

    // Device pointers to memory
    device_ptr<bf16> w(weights_layer);
    device_ptr<float> o(out_intermediate_layer);

    // Define matrices and load weights
    joint_matrix<sub_group, bf16, use::a, TM, TK, layout::row_major> act_matrix;
    joint_matrix<sub_group, bf16, use::b, TK, TN, sycl::ext::intel::experimental::matrix::layout::packed> weight_matrix;

    joint_matrix<sub_group, float, use::accumulator, TM, TN> result_matrix;

    const int n_operations = input_width / TK;

    for (int l = 0; l < N_ITERS; l++) {
        for (int j = 0; j < N_ITERS; j++) {
            for (int k = 0; k < TM; k++) {
                a[TN * sgId + WIDTH * TM * j + k * WIDTH + id] = input[TN * sgId + WIDTH * TM * j + k * WIDTH + id];
            }
        }

        joint_matrix_fill(sg, result_matrix, 0.0f);
        for (int i = 0; i < n_operations; i++) {
            joint_matrix_load(sg, act_matrix, a + TK * i, input_width);
            joint_matrix_load(sg, weight_matrix, w + TN / 2 * 2 * sgId * 8 * input_width + TK * i * 2, input_width * 2);

            result_matrix = joint_matrix_mad(sg, act_matrix, weight_matrix, result_matrix);

            joint_matrix_store(sg, result_matrix, at + TN * sgId + TM * l * WIDTH, WIDTH, layout::row_major);
        }

        matrix_activation<float, bf16, SG_SIZE>(activation, at, a, TN * sgId + TM * l * WIDTH + id, WIDTH);
    }
    for (int i = 0; i < N_ITERS; i++) {
        for (int k = 0; k < TM; k++) {
            o[TN * sgId + WIDTH * TM * i + k * WIDTH + id] = (bf16)at[TN * sgId + WIDTH * TM * i + k * WIDTH + id];
        }
    }
}

/**
 * Performs forward computation for the last layer within a work group.
 *
 * @param item              The SYCL nd_item representing the work item.
 * @param activation        The type of activation to be applied.
 * @param act_mem           Pointer to activation memory.
 * @param weights_layer     Pointer to weights for the layer.
 * @param out               Pointer to the output memory.
 * @param output_stride     The stride for the output memory.
 * @tparam WIDTH            Width of the layer.
 * @tparam N_ITERS          Number of iterations.
 */
template <int WIDTH, int N_ITERS>
void workgroup_last_layer_forward(nd_item<1> item,
    Activation activation,
    multi_ptr<bf16, access::address_space::local_space, (access::decorated)2> a,
    bf16* weights_layer,
    float* out,
    const int output_stride) {

    auto sg = item.get_sub_group();
    int sgId = sg.get_group_id();
    const int li = item.get_local_id(0);
    int N_BLOCKS = WIDTH / 16;
    device_ptr<bf16> w(weights_layer);
    device_ptr<float> o(out);

    joint_matrix<sub_group, bf16, use::a, TM, TK, layout::row_major> act_matrix;

    joint_matrix<sub_group, bf16, use::b, TK, TN, sycl::ext::intel::experimental::matrix::layout::packed> weight_matrix0;
    joint_matrix<sub_group, bf16, use::b, TK, TN, sycl::ext::intel::experimental::matrix::layout::packed> weight_matrix1;
    joint_matrix<sub_group, bf16, use::b, TK, TN, sycl::ext::intel::experimental::matrix::layout::packed> weight_matrix2;
    joint_matrix<sub_group, bf16, use::b, TK, TN, sycl::ext::intel::experimental::matrix::layout::packed> weight_matrix3;
    joint_matrix<sub_group, float, use::accumulator, TM, TN> result_matrix;



    joint_matrix_load(sg, weight_matrix0, w + TN * 2 * sgId + TK / 2 * 0 * WIDTH * 2, WIDTH * 2);
    joint_matrix_load(sg, weight_matrix1, w + TN * 2 * sgId + TK / 2 * 1 * WIDTH * 2, WIDTH * 2);
    joint_matrix_load(sg, weight_matrix2, w + TN * 2 * sgId + TK / 2 * 2 * WIDTH * 2, WIDTH * 2);
    joint_matrix_load(sg, weight_matrix3, w + TN * 2 * sgId + TK / 2 * 3 * WIDTH * 2, WIDTH * 2);

    for (int l = 0; l < N_ITERS; l++) {
        joint_matrix_fill(sg, result_matrix, 0.0f);

        joint_matrix_load(sg, act_matrix, a + TK * 0 + TM * l * WIDTH, WIDTH);
        result_matrix = joint_matrix_mad(sg, act_matrix, weight_matrix0, result_matrix);
        joint_matrix_load(sg, act_matrix, a + TK * 1 + TM * l * WIDTH, WIDTH);
        result_matrix = joint_matrix_mad(sg, act_matrix, weight_matrix1, result_matrix);
        joint_matrix_load(sg, act_matrix, a + TK * 2 + TM * l * WIDTH, WIDTH);
        result_matrix = joint_matrix_mad(sg, act_matrix, weight_matrix2, result_matrix);
        joint_matrix_load(sg, act_matrix, a + TK * 3 + TM * l * WIDTH, WIDTH);
        result_matrix = joint_matrix_mad(sg, act_matrix, weight_matrix3, result_matrix);

        joint_matrix_store(sg, result_matrix, o + TM * sgId + TN * l * WIDTH, WIDTH, layout::row_major);
    }
}



/**
 * Kernel function for the forward pass of the Swift MLP model.
 *
 * @param item                  The SYCL nd_item representing the work item.
 * @param output_activation     The type of activation to be applied for output layer.
 * @param input                 Pointer to input data.
 * @param weights_layer         Pointer to weights for the layer.
 * @param out_intermediate_layer Pointer to intermediate output memory.
 * @param act_mem               Pointer to activation memory.
 * @param act_mem_temp          Pointer to temporary activation memory.
 * @param out                   Pointer to output memory.
 * @param output_stride         The stride for the output memory.
 * @param input_width           Width of the input data.
 * @param output_width          Width of the output data.
 * @param n_hidden_matmuls      Number of hidden matrix multiplications.
 * @param batch_size            Batch size of the data.
 * @tparam WIDTH                Width of the layers.
 * @tparam N_ITERS              Number of iterations.
 * @tparam activation           Type of activation for hidden layers.
 */
template <int WIDTH, int N_ITERS, Activation activation, bool INFERENCE = false>
void kernel_swift_mlp(nd_item<1> item,
    const Activation output_activation,
    bf16* input,
    bf16* weights_layer,
    float* out_intermediate_layer,
    local_accessor<bf16> act_mem,
    local_accessor<float> act_mem_temp,
    float* out,
    const uint32_t output_stride,
    const uint32_t input_width,
    const uint32_t output_width,
    const uint32_t n_hidden_matmuls,
    int batch_size) {


    auto a = act_mem.get_pointer();
    auto at = act_mem_temp.get_pointer();

    // Handle first layer because it has different input

    auto wg = item.get_group();
    const int wg_idx = wg.get_group_id();
    const int elem_idx = BATCH_CHUNK * wg_idx;
    const int first_weight_length = input_width * WIDTH;
    const int hidden_weight_lenght = WIDTH * WIDTH;
    const int layer_lenght = WIDTH * batch_size;

    if (input_width == WIDTH) {
        workgroup_load_input_static<WIDTH, N_ITERS>(item, a, input + elem_idx * WIDTH);
        work_group_layer<WIDTH, N_ITERS, false>(item, activation, a, at, weights_layer, !INFERENCE ? (out_intermediate_layer + elem_idx * WIDTH) : nullptr);
    }
    else {
        /*workgroup_input_layer_forward_dynamic<WIDTH, N_ITERS>(item,
            activation,
            a,
            at,
            input + elem_idx * input_width,
            weights_layer,
            !INFERENCE ? (out_intermediate_layer + elem_idx * WIDTH) : nullptr,
            input_width);*/
    }

    // Handle hidden layers all together

    for (int k = 0; k < n_hidden_matmuls; k++) {
        work_group_layer<WIDTH, N_ITERS, false>(item,
            activation,
            a,
            at,
            weights_layer + first_weight_length + k * hidden_weight_lenght,
            !INFERENCE ? (out_intermediate_layer + elem_idx * WIDTH + (k + 1) * layer_lenght) : nullptr);
    }

    // Handle output layer
    if (output_width > 16) {
        if (INFERENCE) {
           workgroup_write_output_static<WIDTH, N_ITERS>(item, a, out_intermediate_layer + elem_idx * WIDTH + (n_hidden_matmuls + 1) * layer_lenght);
        }
    }
    else if (out) {
        /*workgroup_last_layer_forward<WIDTH, N_ITERS>(item,
            output_activation,
            a,
            weights_layer + first_weight_length + hidden_weight_lenght * n_hidden_matmuls,
            out + elem_idx * WIDTH,
            output_stride);*/
    }
}


/**
 * Performs forward pass for the Swift MLP model.
 *
 * @param q                  SYCL queue for command submission.
 * @param output_activation The type of activation to be applied for output layer.
 * @param weights            Device memory containing weights for the model.
 * @param inputs             Device memory containing input data.
 * @param intermediate_output Pointer to intermediate output memory.
 * @param act_mem            Pointer to activation memory.
 * @param act_mem_temp       Pointer to temporary activation memory.
 * @param output             Device memory for storing the output.
 * @param output_stride      The stride for the output memory.
 * @param n_hidden_layers    Number of hidden layers.
 * @param input_width        Width of the input data.
 * @param output_width       Width of the output data.
 * @param batch_size         Batch size of the data.
 * @tparam WIDTH             Width of the layers.
 * @tparam activation        Type of activation for hidden layers.
 */
template <int WIDTH, Activation activation, bool INFERENCE>
void mlp_swift_forward(queue q,
    Activation output_activation,
    const DeviceMem<bf16>& weights,
    const DeviceMem<bf16>& inputs,
    float* intermediate_output,
    DeviceMem<float>& output,
    const int output_stride,
    const int n_hidden_layers,
    const int input_width,
    const int output_width,
    int batch_size)
{
    constexpr int SKEW = WIDTH % 16 == 0 ? 8 : 0;

    const int N_BLOCKS = WIDTH / TK;
    const int N_ITERS = BATCH_CHUNK / TM;

    q.submit([&](handler& cgh)
        {
            local_accessor<bf16> act_mem = local_accessor<bf16>(range<1>(SHMEM_SIZE + BATCH_CHUNK * SKEW), cgh);
            local_accessor<float> act_mem_temp = local_accessor<float>(range<1>(SHMEM_SIZE + BATCH_CHUNK * SKEW), cgh);

            cgh.parallel_for(
                nd_range<1>(batch_size * WG_SIZE / BATCH_CHUNK, WG_SIZE),
                [=](nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]]
                {
                    kernel_swift_mlp<WIDTH, N_ITERS, activation, INFERENCE>(item,
                        output_activation,
                        inputs.data(),
                        weights.data(),
                        intermediate_output,
                        act_mem,
                        act_mem_temp,
                        output.data(),
                        output_stride,
                        input_width,
                        output_width,
                        n_hidden_layers - 1,
                        batch_size);

                });
        }).wait();
}


/**
 * Kernel function for backpropagation in the SwiftNet model.
 *
 * @param item             The SYCL nd_item representing the work item.
 * @param loss_gradients   Pointer to loss gradients for backpropagation.
 * @param loss_gradients_temp Pointer to temporary loss gradients memory.
 * @param grads            Pointer to gradients for weight updates.
 * @param weights          Pointer to weights of the model.
 * @param forward          Pointer to forward pass intermediate outputs.
 * @param out_inter        Pointer to intermediate output memory.
 * @param n_hidden_matmuls Number of hidden matrix multiplications.
 * @param batch_size         Batch size of the data.
 * @tparam WIDTH           Width of the layers.
 * @tparam N_ITERS         Number of iterations.
 * @tparam ACTIVATION      Type of activation for hidden layers.
 */
template <int WIDTH, int N_ITERS, Activation ACTIVATION>
void kernel_swiftnet_backward(
    nd_item<1> item,
    bf16* deltas,
    multi_ptr<bf16, access::address_space::local_space, (access::decorated)2> a,
    multi_ptr<float, access::address_space::local_space, (access::decorated)2> at,
    bf16* grads,
    bf16* weights,
    float* forward,
    float* out_inter,
    uint32_t n_hidden_matmuls,
    int batch_size
) {
    auto sg = item.get_sub_group();


    int groupId = item.get_group(0);
    int sgId = sg.get_group_id();
    const int layer_length = WIDTH * batch_size;

    workgroup_load_input_static<WIDTH, N_ITERS>(item, a, deltas + groupId * BATCH_CHUNK * WIDTH);

    // Iterate through hidden layers for backpropagation
    for (int k = 0; k < n_hidden_matmuls; k++) {
        work_group_layer<WIDTH, N_ITERS, true>(
            item,
            ACTIVATION,
            a,
            at,
            weights + WIDTH * WIDTH * (n_hidden_matmuls - k),
            out_inter + groupId * BATCH_CHUNK * WIDTH + (n_hidden_matmuls - k - 1) * layer_length,
            forward + WIDTH * batch_size + WIDTH * batch_size * (n_hidden_matmuls - k - 1) + groupId * BATCH_CHUNK * WIDTH
        );
    }
}

/**
 * Multiplies matrices using DGEMM for gradient calculation in the SwiftNet model.
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
void dgemm_multiply(queue q,
    bf16* grads_device,
    float* loss_gradients,
    float* fwd,
    float* A,
    float* B,
    float* C,
    int k,
    int m_n_hidden_matrices,
    int batch_size) {
    const int layer_lenght = WIDTH * batch_size;
    const int n_hidden_matrices = m_n_hidden_matrices;

    // Calculate matrix A using the given activation function
    q.parallel_for<>(range<1>(WIDTH * batch_size), [=](id<1> idx) {
        int i = idx / batch_size;
        int j = idx % batch_size;
        A[i * batch_size + j] = (float)elt_activation_ret<float>(ACTIVATION, fwd[i + j * WIDTH + (n_hidden_matrices - k - 1) * layer_lenght]);
        });

    // Assign matrix B using loss gradients
    q.parallel_for<>(range<1>(WIDTH * batch_size), [=](id<1> idx) {
        B[idx] = (float)loss_gradients[idx + (n_hidden_matrices - k - 1) * layer_lenght];
        });

    // Perform DGEMM operation
    oneapi::mkl::blas::row_major::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
        WIDTH, WIDTH, batch_size, 1, A, batch_size, B, WIDTH, 0, C, WIDTH);

    // Update gradients_device with the computed values
    q.parallel_for<>(range<1>(WIDTH * WIDTH), [=](id<1> idx) {
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
template<int WIDTH, Activation ACTIVATION>
void mlp_swiftnet_backward(
    queue q,
    DeviceMem<bf16>& weights_transposed,
    DeviceMem<bf16>& deltas,
    DeviceMem<bf16>& grads_matrices,
    float* out_inter,
    float* delta_temp_,
    float* forward,
    float* A_dgemm,
    float* B_dgemm,
    float* C_dgemm,
    const uint32_t n_hidden_matmuls,
    int batch_size
) {

    constexpr int SKEW = WIDTH % 16 == 0 ? 8 : 0;

    // here, weights are already transposed and packed
    // in deltas, the last layer has already been calculated

    const int layer_lenght = WIDTH * batch_size;
    const int N_ITERS = BATCH_CHUNK / TM;

    // Execute the kernel for backward pass
    q.submit([&](handler& h) {

        local_accessor<bf16> deltas_layers = local_accessor<bf16>(range<1>(SHMEM_SIZE + BATCH_CHUNK * SKEW), h);
        local_accessor<float> delta_temp = local_accessor<float>(range<1>(SHMEM_SIZE + BATCH_CHUNK * SKEW), h);
        auto a = deltas_layers.get_pointer();
        auto at = delta_temp.get_pointer();

        // Execute DGEMM multiply for each hidden layer
        h.parallel_for(nd_range<1>(batch_size * WG_SIZE / BATCH_CHUNK, WG_SIZE), [=](nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
            kernel_swiftnet_backward<WIDTH, N_ITERS, ACTIVATION>(item, deltas.data(), a, at, grads_matrices.data(), weights_transposed.data(), forward, out_inter, n_hidden_matmuls, batch_size);
            });
        }).wait();

        for (int k = 0; k < n_hidden_matmuls; k++) {
            dgemm_multiply<WIDTH, ACTIVATION>(q, grads_matrices.data(), out_inter, forward, A_dgemm, B_dgemm, C_dgemm, k, n_hidden_matmuls, batch_size);
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
SwiftNetMLP<WIDTH>::SwiftNetMLP(
    queue q,
    int input_width,
    int output_width,
    int n_hidden_layers,
    Activation activation,
    Activation output_activation,
    int batch_size
) :
    m_inputs_width{ input_width },
    m_net_width{ WIDTH },
    m_output_width{ output_width },
    m_n_hidden_layers{ n_hidden_layers },
    m_activation{ activation },
    m_output_activation{ output_activation },
    m_batch_size{ batch_size }
{
    // Store provided parameters
    m_q = q;
    m_n_hidden_matrices = m_n_hidden_layers - 1;

    // Allocate memory for various matrices
    m_weightsT_matrices.allocate(m_net_width * m_inputs_width + (m_net_width * m_net_width) * m_n_hidden_matrices + m_net_width * m_output_width, m_q);
    m_weights_matrices.allocate(m_net_width * m_inputs_width + (m_net_width * m_net_width) * m_n_hidden_matrices + m_net_width * m_output_width, m_q);
    m_weights_matrices_inferences.allocate(m_net_width * m_inputs_width + (m_net_width * m_net_width) * m_n_hidden_matrices + m_net_width * m_output_width, m_q);
    m_grads_matrices.allocate(m_net_width * m_inputs_width + (m_net_width * m_net_width) * m_n_hidden_matrices + m_net_width * m_output_width, m_q);

    // Initialize constants and allocations
    const int layer_length = WIDTH * m_batch_size;
    m_alignment = SHMEM_SIZE;

    // Allocate and initialize various memory buffers
    m_forward = malloc_device<float>(m_batch_size * (WIDTH + m_output_width + WIDTH * m_n_hidden_layers), q);

    m_shmem_size = m_batch_size * WIDTH * m_n_hidden_layers;
    m_act_mem = sycl::aligned_alloc_device<bf16>(m_alignment, m_shmem_size, q);
    m_act_mem_temp = sycl::aligned_alloc_device<float>(m_alignment, m_shmem_size, q);

    m_A_forward = sycl::aligned_alloc_device<float>(m_alignment, layer_length, q);
    m_B_forward = sycl::aligned_alloc_device<float>(m_alignment, m_output_width * WIDTH, q);
    m_C_forward = sycl::aligned_alloc_device<float>(m_alignment, m_output_width * m_batch_size, q);

    m_out_inter = malloc_device<float>(m_batch_size * WIDTH * (m_n_hidden_layers), q);
    m_deltas_temp = sycl::aligned_alloc_device<float>(m_alignment, m_output_width * m_batch_size, q);
    m_deltas.allocate(m_output_width * m_batch_size, q);

    m_A_backward = sycl::aligned_alloc_device<float>(m_alignment, WIDTH * m_batch_size, q);
    m_B_backward = sycl::aligned_alloc_device<float>(m_alignment, m_batch_size * m_output_width, q);
    m_C_backward = sycl::aligned_alloc_device<float>(m_alignment, WIDTH * m_output_width, q);

    m_A_backward_last_layer = sycl::aligned_alloc_device<float>(m_alignment, m_batch_size * m_output_width, q);
    m_B_backward_last_layer = sycl::aligned_alloc_device<float>(m_alignment, m_output_width * WIDTH, q);
    m_C_backward_last_layer = sycl::aligned_alloc_device<float>(m_alignment, WIDTH * m_batch_size, q);
    m_D_backward_last_layer = sycl::aligned_alloc_device<float>(m_alignment, WIDTH * m_batch_size, q);
    m_E_backward_last_layer = sycl::aligned_alloc_device<float>(m_alignment, m_batch_size * WIDTH, q);
    m_F_backward_last_layer = sycl::aligned_alloc_device<float>(m_alignment, WIDTH * WIDTH, q);

    m_A_dgemm = sycl::aligned_alloc_device<float>(m_alignment, m_batch_size * WIDTH, q);
    m_B_dgemm = sycl::aligned_alloc_device<float>(m_alignment, m_batch_size * WIDTH, q);
    m_C_dgemm = sycl::aligned_alloc_device<float>(m_alignment, WIDTH * WIDTH, q);
}

template<int WIDTH>
SwiftNetMLP<WIDTH>::~SwiftNetMLP() {

}
/**
 * Get a pointer to the gradients matrices.
 *
 * @return A pointer to the gradients matrices.
 */
template<int WIDTH>
DeviceMem<bf16>* SwiftNetMLP<WIDTH>::get_grads_matrices() {
    return &m_grads_matrices;
}

/**
 * Get a pointer to the weights matrices.
 *
 * @return A pointer to the weights matrices.
 */
template<int WIDTH>
DeviceMem<bf16>* SwiftNetMLP<WIDTH>::get_weights_matrices() {
    return &m_weights_matrices;
}

/**
 * Get a pointer to the transposed weights matrices.
 *
 * @return A pointer to the transposed weights matrices.
 */
template<int WIDTH>
DeviceMem<bf16>* SwiftNetMLP<WIDTH>::get_weightsT_matrices() {
    return &m_weightsT_matrices;
}

/**
 * Initialize parameters for the neural network.
 * This function initializes the weights matrices with uniform random values.
 */
template <int WIDTH>
void SwiftNetMLP<WIDTH>::initialize_params() {
    // You can uncomment the following lines if needed
    // m_weights_matrices.initialize_constant(1e-4f, m_q);
    // m_weightsT_matrices.initialize_constant(1e-4f, m_q);

    // Initialize weights matrices with uniform random values
    m_weights_matrices.initialize_uniform(0.01, m_weightsT_matrices, m_inputs_width, m_net_width, m_output_width, m_n_hidden_matrices, m_q);
};


/**
 * Save the neural network parameters to a file.
 *
 * @param filename The name of the file to save the parameters to.
 */
template<int WIDTH>
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
template<int WIDTH>
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

    // Read each value from the file and set it as a bf16 value in weights matrices
    for (int i = 0; i < m_weights_matrices.size(); i++) {
        float x;
        file >> x;
        m_weights_matrices.data()[i] = bf16(x);
    }

    // Close the file
    file.close();

    // Make the weights matrices transposed using the transposed weights matrices
    m_weights_matrices.make_transposed(m_weightsT_matrices, m_inputs_width, m_net_width, m_output_width, m_n_hidden_matrices, m_q);
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
    free(m_act_mem, q);
    free(m_act_mem_temp, q);
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
void SwiftNetMLP<WIDTH>::forward_pass(const DeviceMem<bf16>& input, float* forward, float* A, float* B, float* C, DeviceMem<float>& output) {
    // Constants and dimensions
    const int output_stride = WIDTH;
    const int intermediate_output_size = m_batch_size * WIDTH * m_n_hidden_layers;
    const int layer_length = WIDTH * m_batch_size;
    const int n_hidden_matrices = m_n_hidden_matrices;
    const int net_width = m_net_width;
    const int inputs_width = m_inputs_width;
    const int output_width = m_output_width;

    // Static assertion and assertion checks
    static_assert(WIDTH % 16 == 0, "Width must be a multiple of 16.");
    assert(m_batch_size % 64 == 0);

    // Get a pointer to the weights matrices data
    auto p = m_weights_matrices.data();

    // Perform forward pass based on activation function
    switch (m_activation) {
    case Activation::None:
        mlp_swift_forward<WIDTH, Activation::None, false>(m_q, m_output_activation, m_weights_matrices, input, forward + input.size(), output, output_stride, m_n_hidden_layers, m_inputs_width, m_output_width, m_batch_size);
        break;
    case Activation::Exponential:
        mlp_swift_forward<WIDTH, Activation::None, false>(m_q, m_output_activation, m_weights_matrices, input, forward + input.size(), output, output_stride, m_n_hidden_layers, m_inputs_width, m_output_width, m_batch_size);
        break;
    case Activation::Sigmoid:
        mlp_swift_forward<WIDTH, Activation::Sigmoid, false>(m_q, m_output_activation, m_weights_matrices, input, forward + input.size(), output, output_stride, m_n_hidden_layers, m_inputs_width, m_output_width, m_batch_size);
        break;
    case Activation::ReLU:
        mlp_swift_forward<WIDTH, Activation::ReLU, false>(m_q, m_output_activation, m_weights_matrices, input, forward + input.size(), output, output_stride, m_n_hidden_layers, m_inputs_width, m_output_width, m_batch_size);
        break;
    case Activation::LeakyReLU:
        mlp_swift_forward<WIDTH, Activation::LeakyReLU, false>(m_q, m_output_activation, m_weights_matrices, input, forward + input.size(), output, output_stride, m_n_hidden_layers, m_inputs_width, m_output_width, m_batch_size);
        break;
    case Activation::Squareplus:
        mlp_swift_forward<WIDTH, Activation::Squareplus, false>(m_q, m_output_activation, m_weights_matrices, input, forward + input.size(), output, output_stride, m_n_hidden_layers, m_inputs_width, m_output_width, m_batch_size);
        break;
    case Activation::Softplus:
        mlp_swift_forward<WIDTH, Activation::Softplus, false>(m_q, m_output_activation, m_weights_matrices, input, forward + input.size(), output, output_stride, m_n_hidden_layers, m_inputs_width, m_output_width, m_batch_size);
        break;
    case Activation::Tanh:
        mlp_swift_forward<WIDTH, Activation::Tanh, false>(m_q, m_output_activation, m_weights_matrices, input, forward + input.size(), output, output_stride, m_n_hidden_layers, m_inputs_width, m_output_width, m_batch_size);
        break;
    default: return;
    }

    // Handle the case when output_width is greater than 16
    if (m_output_width > 16) {
        m_q.parallel_for<>(range<1>(m_output_width * m_net_width), [=](id<1> idx) {
            B[idx] = (float)p[toPackedLayoutCoord(idx, net_width, output_width) + net_width * (inputs_width + n_hidden_matrices * net_width)];
            });

        oneapi::mkl::blas::row_major::gemm(m_q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
            m_batch_size, m_output_width, WIDTH, 1, forward + (n_hidden_matrices + 1) * layer_length, WIDTH, B, m_output_width, 0, C, m_output_width);

        m_q.parallel_for<>(range<1>(m_output_width * m_batch_size), [=](id<1> idx) {
            output.data()[idx] = C[idx];
            forward[intermediate_output_size + input.size() + idx] = C[idx];
            }).wait();
    }
}

template <int WIDTH>
void SwiftNetMLP<WIDTH>::inference(const DeviceMem<bf16>& input, float* forward, float* A, float* B, float* C, DeviceMem<float>& output) {

    const int output_stride = WIDTH;
    const int input_size = input.size();
    const int layer_length = WIDTH * m_batch_size;
    const int n_hidden_matrices = m_n_hidden_matrices;
    const int net_width = m_net_width;
    const int inputs_width = m_inputs_width;
    const int output_width = m_output_width;

    static_assert(WIDTH % 16 == 0, "Width must be a multiply of 16.");
    assert(m_batch_size % 64 == 0);
    auto p = m_weights_matrices.data();



    switch (m_activation) {
    case Activation::None:        mlp_swift_forward<WIDTH, Activation::None, true>(m_q, m_output_activation, m_weights_matrices, input, forward, output, output_stride, m_n_hidden_layers, m_inputs_width, m_output_width, m_batch_size); break;
    case Activation::Exponential: mlp_swift_forward<WIDTH, Activation::Exponential, true>(m_q, m_output_activation, m_weights_matrices, input, forward, output, output_stride, m_n_hidden_layers, m_inputs_width, m_output_width, m_batch_size); break;
    case Activation::Sigmoid:     mlp_swift_forward<WIDTH, Activation::Sigmoid, true>(m_q, m_output_activation, m_weights_matrices, input, forward, output, output_stride, m_n_hidden_layers, m_inputs_width, m_output_width, m_batch_size); break;
    case Activation::ReLU:        mlp_swift_forward<WIDTH, Activation::ReLU, true>(m_q, m_output_activation, m_weights_matrices, input, forward, output, output_stride, m_n_hidden_layers, m_inputs_width, m_output_width, m_batch_size); break;
    case Activation::LeakyReLU:   mlp_swift_forward<WIDTH, Activation::LeakyReLU, true>(m_q, m_output_activation, m_weights_matrices, input, forward, output, output_stride, m_n_hidden_layers, m_inputs_width, m_output_width, m_batch_size); break;
    case Activation::Squareplus:  mlp_swift_forward<WIDTH, Activation::Squareplus, true>(m_q, m_output_activation, m_weights_matrices, input, forward, output, output_stride, m_n_hidden_layers, m_inputs_width, m_output_width, m_batch_size); break;
    case Activation::Softplus:    mlp_swift_forward<WIDTH, Activation::Softplus, true>(m_q, m_output_activation, m_weights_matrices, input, forward, output, output_stride, m_n_hidden_layers, m_inputs_width, m_output_width, m_batch_size); break;
    case Activation::Tanh:        mlp_swift_forward<WIDTH, Activation::Tanh, true>(m_q, m_output_activation, m_weights_matrices, input, forward, output, output_stride, m_n_hidden_layers, m_inputs_width, m_output_width, m_batch_size); break;
    default: throw std::runtime_error{"Unsupported activation."};
    }

    if (m_output_width > 16) {
        m_q.parallel_for<>(range<1>(m_output_width * m_net_width), [=](id<1> idx) {
            B[idx] = p[toPackedLayoutCoord(idx, net_width, output_width) + net_width * (inputs_width + n_hidden_matrices * net_width)];
            });

        oneapi::mkl::blas::row_major::gemm(m_q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
            m_batch_size, m_output_width, WIDTH, 1, A, WIDTH, B, m_output_width, 0, output.data(), m_output_width);
    }
}

/**
 * Perform matrix multiplications and activation backpropagation for the last layer (beginning of the backward pass) .
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
    int batch_size,
    float* A,
    float* B,
    float* C,
    float* D,
    float* E,
    float* F) {

    auto p_w = m_weightsT_matrices.data();
    auto p_g = m_grads_matrices.data();
    const int offset_w = m_n_hidden_matrices * m_net_width * m_net_width + m_net_width * m_inputs_width;
    const int offset_g = m_inputs_width * m_net_width + (m_n_hidden_matrices - 1) * m_net_width * m_net_width;
    const int offset_f = (m_inputs_width + (m_n_hidden_matrices - 1) * batch_size) * m_net_width;
    const int output_width = m_output_width;
    const int net_width = m_net_width;

    int i = 0;
    int j = 0;
    auto activation = m_activation;

    m_q.parallel_for<>(range<1>(grads.size()), [=](id<1> idx) {
        A[idx] = (float)loss.data()[idx];
        }).wait();

        m_q.parallel_for<>(range<1>(m_output_width * WIDTH), [=](id<1> idx) {
            B[idx] = p_w[offset_w + toPackedLayoutCoord(idx, output_width, net_width)];
            }).wait();

            oneapi::mkl::blas::row_major::gemm(m_q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
                batch_size, m_net_width, m_output_width, 1, A, m_output_width, B, m_net_width, 0, C, m_net_width);

            m_q.parallel_for<>(range<1>(WIDTH * batch_size), [=](id<1> idx) {
                int i = idx / batch_size;
                int j = idx % batch_size;
                D[i * batch_size + j] = elt_activation_ret<float>(activation, forward[offset_f + j * net_width + i]);
                }).wait();

                m_q.parallel_for<>(range<1>(m_net_width * batch_size), [=](id<1> idx) {
                    elt_activation_bwd<float, float, float>(activation, C[idx], forward[offset_f + idx], E[idx]);
                    loss.data()[idx] = (bf16)E[idx];
                    }).wait();


                    oneapi::mkl::blas::row_major::gemm(m_q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
                        m_net_width, m_net_width, batch_size, 1, D, batch_size, E, m_net_width, 0, F, m_net_width);

                    m_q.parallel_for<>(range<1>(m_net_width * m_net_width), [=](id<1> idx) {
                        p_g[idx + offset_g] = (float)F[idx];
                        }).wait();

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
 * @param A_backward_last_layer Temporary array A for last layer backward pass.
 * @param B_backward_last_layer Temporary array B for last layer backward pass.
 * @param C_backward_last_layer Temporary array C for last layer backward pass.
 * @param D_backward_last_layer Temporary array D for last layer backward pass.
 * @param E_backward_last_layer Temporary array E for last layer backward pass.
 * @param F_backward_last_layer Temporary array F for last layer backward pass.
 * @param A_dgemm Temporary array A for DGEMM.
 * @param B_dgemm Temporary array B for DGEMM.
 * @param C_dgemm Temporary array C for DGEMM.
 * @param forward Pointer to the forward intermediate array.
 */
template <int WIDTH>
void SwiftNetMLP<WIDTH>::backward_pass(const DeviceMem<bf16>& input,
    DeviceMem<bf16>& grads,
    float* out_inter,
    float* delta_temp,
    DeviceMem<bf16> loss,
    float* A,
    float* B,
    float* C,
    float* A_backward_last_layer,
    float* B_backward_last_layer,
    float* C_backward_last_layer,
    float* D_backward_last_layer,
    float* E_backward_last_layer,
    float* F_backward_last_layer,
    float* A_dgemm,
    float* B_dgemm,
    float* C_dgemm,
    float* forward) {

    int batch_size = m_batch_size;
    auto p = m_grads_matrices.data();
    int s = m_grads_matrices.size();
    auto activation = m_activation;
    auto output_activation = m_output_activation;
    const int offset_grad = m_n_hidden_matrices * m_net_width * m_net_width + m_inputs_width * m_net_width;
    const int offset_f = m_inputs_width * batch_size + m_n_hidden_matrices * m_net_width * batch_size;

    const size_t alignment = 1024;

    // Compute activation backpropagation using parallel_for
    m_q.parallel_for<>(range<1>(WIDTH * batch_size), [=](id<1> idx) {
        int i = idx / batch_size;
        int j = idx % batch_size;
        A[i * batch_size + j] = elt_activation_ret<float>(activation, forward[offset_f + j * WIDTH + i]);
        }).wait();

        // Compute output activation backpropagation using parallel_for and copy to loss array
        m_q.parallel_for<>(range<1>(batch_size * m_output_width), [=](id<1> idx) {
            elt_activation_bwd<bf16, float, float>(output_activation, grads.data()[idx], forward[offset_f + batch_size * WIDTH + idx], B[idx]);
            loss.data()[idx] = (bf16)B[idx];
            }).wait();

            // Perform matrix multiplication using MKL BLAS
            oneapi::mkl::blas::row_major::gemm(m_q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
                m_net_width, m_output_width, batch_size, 1, A, batch_size, B, m_output_width, 0, C, m_output_width);

            // Copy the result back to the gradients matrix
            m_q.parallel_for<>(range<1>(m_net_width * m_output_width), [=](id<1> idx) {
                p[idx + offset_grad] = (float)C[idx];
                }).wait();

                // Backpropagation through last layer using dgemm_last_layer_backward
                dgemm_last_layer_backward(grads, forward, loss, batch_size, A_backward_last_layer, B_backward_last_layer, C_backward_last_layer, D_backward_last_layer, E_backward_last_layer, F_backward_last_layer);

                // Choose appropriate mlp_swiftnet_backward based on activation
                switch (m_activation) {
                case Activation::None: mlp_swiftnet_backward<WIDTH, Activation::None>(m_q, m_weightsT_matrices, loss, m_grads_matrices, out_inter, delta_temp, forward, A_dgemm, B_dgemm, C_dgemm, m_n_hidden_matrices, m_batch_size); break;
                case Activation::ReLU: mlp_swiftnet_backward<WIDTH, Activation::ReLU>(m_q, m_weightsT_matrices, loss, m_grads_matrices, out_inter, delta_temp, forward, A_dgemm, B_dgemm, C_dgemm, m_n_hidden_matrices, m_batch_size); break;
                case Activation::LeakyReLU: mlp_swiftnet_backward<WIDTH, Activation::LeakyReLU>(m_q, m_weightsT_matrices, loss, m_grads_matrices, out_inter, delta_temp, forward, A_dgemm, B_dgemm, C_dgemm, m_n_hidden_matrices, m_batch_size); break;
                case Activation::Exponential: mlp_swiftnet_backward<WIDTH, Activation::Exponential>(m_q, m_weightsT_matrices, loss, m_grads_matrices, out_inter, delta_temp, forward, A_dgemm, B_dgemm, C_dgemm, m_n_hidden_matrices, m_batch_size); break;
                case Activation::Sigmoid: mlp_swiftnet_backward<WIDTH, Activation::Sigmoid>(m_q, m_weightsT_matrices, loss, m_grads_matrices, out_inter, delta_temp, forward, A_dgemm, B_dgemm, C_dgemm, m_n_hidden_matrices, m_batch_size); break;
                case Activation::Tanh: mlp_swiftnet_backward<WIDTH, Activation::Tanh>(m_q, m_weightsT_matrices, loss, m_grads_matrices, out_inter, delta_temp, forward, A_dgemm, B_dgemm, C_dgemm, m_n_hidden_matrices, m_batch_size); break;
                default: return;
                }

                // Normalize gradients
                m_q.parallel_for<>(range<1>(s), [=](id<1> idx) {
                    p[idx] /= batch_size;
                    }).wait();
}

template class SwiftNetMLP<64>;
template class SwiftNetMLP<128>;
