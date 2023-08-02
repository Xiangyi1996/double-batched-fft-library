#include <iostream>
#include <vector>
#include <CL/sycl.hpp>
#include "activation.h"
#include "SwiftNetMLP.h"
#include "L2.h"
#include "sgd.h"
#include "trainer.h"
#include "mkl.h"
#include "mkl_omp_offload.h"
#include "common.h"
#include "oneapi/mkl.hpp"
//#include "config.h"

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
using bf16 = sycl::ext::oneapi::bfloat16;

#define TM 8
#define TK 16
#define TN 8

#define SG_SIZE 8
#define WG_SIZE 8*SG_SIZE
#define BATCH_CHUNK 64

template <int WIDTH, int N_ITERS, bool BACKWARD = false>
void work_group_layer(nd_item<1> item, Activation activation, bf16* act_mem, float* act_mem_temp, bf16* weights_layer, float* out_inter, bf16* out, float* forward_act = nullptr) {

    auto sg = item.get_sub_group();
    int id = item.get_local_id() % SG_SIZE;
    int sgId = sg.get_group_id();
    const int N_BLOCKS = WIDTH / TK;

    device_ptr<bf16> w(weights_layer);
    device_ptr<bf16> a(act_mem);
    device_ptr<float> at(act_mem_temp);
    device_ptr<float> o(out_inter);
    device_ptr<float> f(forward_act);

    item.barrier();

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

        joint_matrix_store(sg, result_matrix, at + TN * sgId + TM * l * WIDTH, WIDTH, layout::row_major);

    }

    item.barrier();

    for (int i = 0; i < N_ITERS; i++) {
        if (BACKWARD) {
            matrix_activation_backward<float, float, bf16, SG_SIZE>(activation, at + TN * sgId + TM * i * WIDTH, f + TN * sgId + i * TM * WIDTH + id, act_mem + TN * sgId + TM * i * WIDTH + id, WIDTH);
        }
        else {

            matrix_activation<float, bf16, SG_SIZE>(activation, at + TN * sgId + TM * i * WIDTH + id, act_mem + TN * sgId + TM * i * WIDTH + id, WIDTH);
        }

    }

    for (int i = 0; i < N_ITERS; i++) {
        for (int k = 0; k < TM; k++) {
            out_inter[TN * sgId + TM * i * WIDTH + k * WIDTH + id] = act_mem_temp[TN * sgId + TM * i * WIDTH + k * WIDTH + id];
        }
    }
}

template <int WIDTH, int N_ITERS>
void workgroup_load_input_static(nd_item<1> item, bf16* act_mem, const bf16* input) {
    int id = item.get_local_id() % SG_SIZE;
    auto sg = item.get_sub_group();
    int sgId = sg.get_group_id();

    for (int i = 0; i < N_ITERS; i++) {
        for (int k = 0; k < TM; k++) {
            act_mem[TN * sgId + TM * i * WIDTH + k * WIDTH + id] = input[TN * sgId + TM * i * WIDTH + k * WIDTH + id];
        }
    }
}


template <int WIDTH, int N_ITERS>
void workgroup_write_output_static(nd_item<1> item, bf16* act_shmem, float* output_threadblock) {
    int id = item.get_local_id() % SG_SIZE;
    auto sg = item.get_sub_group();
    int sgId = sg.get_group_id();

    for (int i = 0; i < N_ITERS; i++) {
        for (int k = 0; k < TM; k++) {
            output_threadblock[TN * sgId + TM * i * WIDTH + k * WIDTH + id] = act_shmem[TN * sgId + TM * i * WIDTH + k * WIDTH + id];
        }
    }
}

template <int WIDTH, int N_ITERS>
void workgroup_input_layer_forward_dynamic(nd_item<1> item,
    Activation activation,
    bf16* act_shmem,
    const bf16* input,
    bf16* weights_layer,
    float* out_intermediate_layer,
    const int input_width,
    const int batch_size)
{
    auto sg = item.get_sub_group();
    int sgId = sg.get_group_id();
    const int N_BLOCKS = WIDTH / TK;
    const int li = item.get_local_id(0);

    device_ptr<bf16> w(weights_layer);
    device_ptr<bf16> a(act_shmem);
    device_ptr<float> o(out_intermediate_layer);

    bf16* weights_shmem = act_shmem + 16 * input_width;

    const int n_element_load = N_BLOCKS * 32 * 8;
    const int wi_elem_idx = (li + sgId * 32) * 8;

    const int n_elements_weights = WIDTH * input_width;

    for (int i = wi_elem_idx; i < n_elements_weights; i += n_element_load) {
        weights_shmem[i] = weights_layer[i];
    }

    joint_matrix<sub_group, bf16, use::a, TM, TK, layout::row_major> act_matrix;
    joint_matrix<sub_group, bf16, use::b, TK, TN, sycl::ext::intel::experimental::matrix::layout::packed> weight_matrix;

    joint_matrix<sub_group, float, use::accumulator, TM, TN> result_matrix;

    const int n_operations = input_width / TK;

    for (int l = 0; l < N_ITERS; l++) {
        const int n_elems_input = TK * input_width;
        for (int i = wi_elem_idx; i < n_elems_input; i += n_element_load) {
            act_shmem[i] = input[l * n_element_load + i];
        }

        joint_matrix_fill(sg, result_matrix, 0.0f);
        for (int i = 0; i < n_operations; i++) {
            joint_matrix_load(sg, act_matrix, a + TK * i, input_width);
            joint_matrix_load(sg, weight_matrix, w + TN / 2 * 2 * sgId * 8 * input_width + TK * i * 2, input_width * 2);

            result_matrix = joint_matrix_mad(sg, act_matrix, weight_matrix, result_matrix);



            joint_matrix_store(sg, result_matrix, o + TN * sgId + TM * l * WIDTH, WIDTH, layout::row_major);
        }
    }
}


template <int WIDTH, int N_ITERS>
void workgroup_last_layer_forward(nd_item<1> item,
    Activation activation,
    bf16* act_mem,
    bf16* weights_layer,
    float* out,
    const int output_stride) {

    auto sg = item.get_sub_group();
    int sgId = sg.get_group_id();
    const int li = item.get_local_id(0);
    int N_BLOCKS = WIDTH / 16;
    device_ptr<bf16> w(weights_layer);
    device_ptr<bf16> a(act_mem);
    device_ptr<float> o(out);

    joint_matrix<sub_group, bf16, use::a, TM, TK, layout::row_major> act_matrix;

    joint_matrix<sub_group, bf16, use::b, TK, TN, sycl::ext::intel::experimental::matrix::layout::packed> weight_matrix0;
    joint_matrix<sub_group, bf16, use::b, TK, TN, sycl::ext::intel::experimental::matrix::layout::packed> weight_matrix1;
    joint_matrix<sub_group, bf16, use::b, TK, TN, sycl::ext::intel::experimental::matrix::layout::packed> weight_matrix2;
    joint_matrix<sub_group, bf16, use::b, TK, TN, sycl::ext::intel::experimental::matrix::layout::packed> weight_matrix3;
    joint_matrix<sub_group, float, use::accumulator, TM, TN> result_matrix;

    bf16* weights_shmem = act_mem + N_ITERS * 16 * WIDTH;

    const int weights_row = (8 * li) % WIDTH;
    const int weights_col = (8 * li + 8 * 32 * sgId) / WIDTH;

    weights_shmem[weights_row + weights_col * WIDTH] = weights_layer[weights_row + weights_col * WIDTH];

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


template <int WIDTH, int N_ITERS, Activation activation>
void kernel_swift_mlp(nd_item<1> item,
    const Activation output_activation,
    bf16* input,
    bf16* weights_layer,
    float* out_intermediate_layer,
    bf16* act_mem,
    float* act_mem_temp,
    float* out,
    const int batch_size,
    const uint32_t output_stride,
    const uint32_t input_width,
    const uint32_t output_width,
    const uint32_t n_hidden_matmuls,
    const layout input_layout,
    const layout output_layout) {

    // Handle first layer because it has different input

    auto wg = item.get_group();
    const int wg_idx = wg.get_group_id();
    const int elem_idx = BATCH_CHUNK * wg_idx;
    const int first_weight_length = input_width * WIDTH;
    const int hidden_weight_lenght = WIDTH * WIDTH;
    const int layer_lenght = WIDTH * batch_size;

    if (input_width == WIDTH) {

        workgroup_load_input_static<WIDTH, N_ITERS>(item, act_mem + elem_idx * WIDTH, input + elem_idx * WIDTH);
        work_group_layer<WIDTH, N_ITERS, false>(item, activation, act_mem + elem_idx * WIDTH, act_mem_temp + elem_idx * WIDTH, weights_layer, out_intermediate_layer + elem_idx * WIDTH, nullptr);
    }
    else {
        workgroup_input_layer_forward_dynamic<WIDTH, N_ITERS>(item,
            activation,
            act_mem,
            input + elem_idx * input_width,
            weights_layer,
            out_intermediate_layer + elem_idx * WIDTH,
            input_width,
            batch_size);

    }
    item.barrier();

    // Handle hidden layers all together

    for (int k = 0; k < n_hidden_matmuls; k++) {
        work_group_layer<WIDTH, N_ITERS, false>(item,
            activation,
            act_mem + elem_idx * WIDTH,
            act_mem_temp + elem_idx * WIDTH,
            weights_layer + first_weight_length + k * hidden_weight_lenght,
            out_intermediate_layer + elem_idx * WIDTH + (k + 1) * layer_lenght,
            nullptr);
        item.barrier();
    }

    // Handle output layer
    if (output_width > 16) {
        /*work_group_layer<WIDTH, N_ITERS, false>(item,
            activation,
            act_mem + elem_idx * WIDTH,
            act_mem_temp + elem_idx * WIDTH,
            weights_layer + first_weight_length + n_hidden_matmuls * hidden_weight_lenght,
            out_intermediate_layer + elem_idx * WIDTH + (n_hidden_matmuls + 1) * layer_lenght,
            nullptr);*/

            //workgroup_write_output_static<WIDTH, N_ITERS>(item, act_mem, out + elem_idx * WIDTH);
    }

    else if (out) {
        workgroup_last_layer_forward<WIDTH, N_ITERS>(item,
            output_activation,
            act_mem,
            weights_layer + first_weight_length + hidden_weight_lenght * n_hidden_matmuls,
            out + elem_idx * WIDTH + (n_hidden_matmuls + 1) * layer_lenght,
            output_stride);

    }
}

template <int WIDTH, Activation activation>
void mlp_swift_forward(queue q,
    Activation output_activation,
    const DeviceMem<bf16>& weights,
    const DeviceMem<bf16>& inputs,
    float* intermediate_output,
    bf16* act_mem,
    float* act_mem_temp,
    DeviceMem<float>& output,
    const int output_stride,
    const int n_hidden_layers,
    const int batch_size,
    const int input_width,
    const int output_width)
{
    const int N_BLOCKS = WIDTH / TK;
    const int N_ITERS = BATCH_CHUNK / TM;

    q.submit([&](handler& cgh)
        {
            cgh.parallel_for(
                nd_range<1>(batch_size * WG_SIZE / BATCH_CHUNK, WG_SIZE),
                [=](nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]]
                {
                    kernel_swift_mlp<WIDTH, N_ITERS, activation>(item,
                        output_activation,
                        inputs.data(),
                        weights.data(),
                        intermediate_output,
                        act_mem,
                        act_mem_temp,
                        output.data(),
                        batch_size,
                        output_stride,
                        input_width,
                        output_width,
                        n_hidden_layers - 1,
                        layout::col_major,
                        layout::col_major);

                });
        }).wait();
}

template <int WIDTH, int N_ITERS, Activation ACTIVATION>
void kernel_swiftnet_backward(
    nd_item<1> item,
    bf16* loss_gradients,
    float* loss_gradients_temp,
    bf16* grads,
    bf16* weights,
    float* forward,
    float* out_inter,
    int batch_number,
    uint32_t n_hidden_matmuls

) {
    auto sg = item.get_sub_group();

    int groupId = item.get_group(0);
    int sgId = sg.get_group_id();
    int idx = 8 * groupId * N_ITERS;
    const int layer_length = WIDTH * WIDTH * batch_number;

    for (int k = 0; k < n_hidden_matmuls; k++) {
        work_group_layer<WIDTH, N_ITERS, true>(
            item,
            ACTIVATION,
            loss_gradients + groupId * WIDTH * WIDTH,
            loss_gradients_temp + groupId * WIDTH * WIDTH,
            weights + WIDTH * WIDTH * (n_hidden_matmuls - k),
            out_inter + groupId * WIDTH * WIDTH + (n_hidden_matmuls - k - 1) * layer_length,
            loss_gradients + groupId * WIDTH * WIDTH,
            forward + WIDTH * WIDTH * batch_number + WIDTH * WIDTH * batch_number * (n_hidden_matmuls - k - 1) + groupId * WIDTH * WIDTH
        );
        item.barrier();
    }
}

template <int WIDTH, Activation ACTIVATION>
void dgemm_multiply(queue q,
    bf16* grads_device,
    float* loss_gradients,
    float* fwd,
    float* A,
    float* B,
    float* C,
    int k,
    int batch_size,
    int m_n_hidden_matrices) {
    const int layer_length = WIDTH * batch_size;
    const int n_hidden_matrices = m_n_hidden_matrices;
    int i = 0;
    int j = 0;

    q.parallel_for<>(range<1>(WIDTH * batch_size), [=](id<1> idx) {
        int i = idx / batch_size;
        int j = idx % batch_size;
        A[i * batch_size + j] = (float)elt_activation_ret<float>(ACTIVATION, fwd[i + j * WIDTH + (n_hidden_matrices - k - 1) * layer_length]);
        });

    q.parallel_for<>(range<1>(WIDTH * batch_size), [=](id<1> idx) {
        B[idx] = (float)loss_gradients[idx + (n_hidden_matrices - k - 1) * layer_length];
        });

    oneapi::mkl::blas::row_major::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
        WIDTH, WIDTH, batch_size, 1, A, batch_size, B, WIDTH, 0, C, WIDTH);

    q.parallel_for<>(range<1>(WIDTH * WIDTH), [=](id<1> idx) {
        grads_device[(m_n_hidden_matrices - k - 1) * WIDTH * WIDTH + idx] += C[idx];
        }).wait();
}

template<int WIDTH, Activation ACTIVATION>
void mlp_swiftnet_backward(
    queue q,
    DeviceMem<bf16>& weights_transposed,
    DeviceMem<bf16>& deltas,
    DeviceMem<bf16>& grads_matrices,
    float* out_inter,
    float* delta_temp,
    float* forward,
    float* A_dgemm,
    float* B_dgemm,
    float* C_dgemm,
    int batch_size,
    const uint32_t n_hidden_matmuls
) {
    // here, weights are already transposed and packed
    // in deltas, the last layer has already been calculated

    const int layer_lenght = WIDTH * batch_size;
    const int N_ITERS = BATCH_CHUNK / TM;
    int batch_number = batch_size / BATCH_CHUNK;
    try {
        q.submit([&](handler& h) {

            h.parallel_for(nd_range<1>(batch_size * WG_SIZE / BATCH_CHUNK, WG_SIZE), [=](nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                kernel_swiftnet_backward<WIDTH, N_ITERS, ACTIVATION>(item, deltas.data(), delta_temp, grads_matrices.data(), weights_transposed.data(), forward, out_inter, batch_number, n_hidden_matmuls);
                });
            }).wait();

            for (int k = 0; k < n_hidden_matmuls; k++) {
                dgemm_multiply<WIDTH, ACTIVATION>(q, grads_matrices.data(), out_inter, forward, A_dgemm, B_dgemm, C_dgemm, k, batch_size, n_hidden_matmuls);
            }
            q.wait();
    }

    catch (std::exception const& e)
    {
        std::cout << "An exception was caught when performing AMX/XMX matrix multiply.\n";
        std::terminate();
    }
}


template <int WIDTH>
SwiftNetMLP<WIDTH>::SwiftNetMLP(
    queue q,
    int input_width,
    int output_width,
    int n_hidden_layers,
    Activation activation,
    Activation output_activation
) :
    m_inputs_width{ input_width },
    m_net_width{ WIDTH },
    m_output_width{ output_width },
    m_n_hidden_layers{ n_hidden_layers },
    m_activation{ activation },
    m_output_activation{ output_activation }
{

    m_q = q;
    m_n_hidden_matrices = m_n_hidden_layers - 1;
    m_weightsT_matrices.allocate(m_net_width * m_inputs_width + (m_net_width * m_net_width) * m_n_hidden_matrices + m_net_width * m_output_width, m_q);
    m_weights_matrices.allocate(m_net_width * m_inputs_width + (m_net_width * m_net_width) * m_n_hidden_matrices + m_net_width * m_output_width, m_q);
    m_weights_matrices_inferences.allocate(m_net_width * m_inputs_width + (m_net_width * m_net_width) * m_n_hidden_matrices + m_net_width * m_output_width, m_q);
    m_grads_matrices.allocate(m_net_width * m_inputs_width + (m_net_width * m_net_width) * m_n_hidden_matrices + m_net_width * m_output_width, m_q);


    const int batch_size = std::pow(2, 17);
    const int layer_length = WIDTH * batch_size;
    m_alignment = 4096;

    m_forward = malloc_device<float>(batch_size * (WIDTH + m_output_width + WIDTH * m_n_hidden_layers), q);

    m_shmem_size = batch_size * WIDTH * m_n_hidden_layers;

    m_act_mem = sycl::aligned_alloc_device<bf16>(m_alignment, m_shmem_size, q);
    m_act_mem_temp = sycl::aligned_alloc_device<float>(m_alignment, m_shmem_size, q);

    m_A_forward = sycl::aligned_alloc_device<float>(m_alignment, layer_length, q);
    m_B_forward = sycl::aligned_alloc_device<float>(m_alignment, m_output_width * 64, q);
    m_C_forward = sycl::aligned_alloc_device<float>(m_alignment, m_output_width * batch_size, q);

    m_out_inter = malloc_device<float>(batch_size * WIDTH * (m_n_hidden_layers), q);
    m_deltas_temp = sycl::aligned_alloc_device<float>(m_alignment, m_output_width * batch_size, q);
    m_deltas.allocate(m_output_width * batch_size, q);

    m_A_backward = sycl::aligned_alloc_device<float>(m_alignment, WIDTH * batch_size, q);
    m_B_backward = sycl::aligned_alloc_device<float>(m_alignment, batch_size * m_output_width, q);
    m_C_backward = sycl::aligned_alloc_device<float>(m_alignment, WIDTH * m_output_width, q);

    m_A_backward_last_layer = sycl::aligned_alloc_device<float>(m_alignment, batch_size * m_output_width, q);
    m_B_backward_last_layer = sycl::aligned_alloc_device<float>(m_alignment, m_output_width * WIDTH, q);
    m_C_backward_last_layer = sycl::aligned_alloc_device<float>(m_alignment, WIDTH * batch_size, q);
    m_D_backward_last_layer = sycl::aligned_alloc_device<float>(m_alignment, WIDTH * batch_size, q);
    m_E_backward_last_layer = sycl::aligned_alloc_device<float>(m_alignment, batch_size * WIDTH, q);
    m_F_backward_last_layer = sycl::aligned_alloc_device<float>(m_alignment, WIDTH * WIDTH, q);

    m_A_dgemm = sycl::aligned_alloc_device<float>(m_alignment, batch_size * WIDTH, q);
    m_B_dgemm = sycl::aligned_alloc_device<float>(m_alignment, batch_size * WIDTH, q);
    m_C_dgemm = sycl::aligned_alloc_device<float>(m_alignment, WIDTH * WIDTH, q);

}

template <int WIDTH>
void SwiftNetMLP<WIDTH>::initialize_params() {
    //m_weights_matrices.initialize_constant(1e-4f, m_q);
    //m_weightsT_matrices.initialize_constant(1e-4f, m_q);
    m_weights_matrices.initialize_uniform(0.01, m_weightsT_matrices, m_inputs_width, m_net_width, m_output_width, m_n_hidden_matrices, m_q);
};

template<int WIDTH>
void SwiftNetMLP<WIDTH>::save_to_file(std::string filename) {
    std::ofstream file;
    file.open(filename);
    file << m_inputs_width << "\n";
    file << m_net_width << "\n";
    file << m_output_width << "\n";
    file << m_n_hidden_layers << "\n";
    file << m_n_hidden_matrices << "\n";
    for (int i = 0; i < m_weights_matrices.size(); i++) {
        file << m_weights_matrices.data()[i] << "\n";
    }
    file.close();
    return;
}

template<int WIDTH>
void SwiftNetMLP<WIDTH>::load_from_file(std::string filename) {
    std::ifstream file;
    std::string line;
    file.open(filename);
    file >> m_inputs_width;
    file >> m_net_width;
    file >> m_output_width;
    file >> m_n_hidden_layers;
    file >> m_n_hidden_matrices;
    for (int i = 0; i < m_weights_matrices.size(); i++) {
        float x;
        file >> x;
        m_weights_matrices.data()[i] = bf16(x);
    }
    file.close();
    m_weights_matrices.make_transposed(m_weightsT_matrices, m_inputs_width, m_net_width, m_output_width, m_n_hidden_matrices);
    return;
}

template <int WIDTH>
void SwiftNetMLP<WIDTH>::free_mem(queue q) {
    free(m_act_mem, q);
    free(m_act_mem_temp, q);
    free(m_out_inter, q);
    free(m_deltas_temp, q);
    free(m_A_forward, q);
    free(m_B_forward, q);
    free(m_C_forward, q);
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

template <int WIDTH>
void SwiftNetMLP<WIDTH>::forward_pass(const DeviceMem<bf16>& input, float* forward, bf16* act_mem, float* act_mem_temp, float* A, float* B, float* C, DeviceMem<float>& output) {

    const int output_stride = WIDTH;
    const int batch_size = input.size() / m_inputs_width;
    const int input_size = input.size();
    const int intermediate_output_size = batch_size * WIDTH * m_n_hidden_layers;
    const int layer_length = WIDTH * batch_size;
    const int n_hidden_matrices = m_n_hidden_matrices;
    const int net_width = m_net_width;
    const int inputs_width = m_inputs_width;
    const int output_width = m_output_width;

    static_assert(WIDTH % 16 == 0, "Width must be a multiply of 16.");
    assert(batch_size % 64 == 0);
    auto p = m_weights_matrices.data();



    switch (m_activation) {
    case Activation::None:        mlp_swift_forward<WIDTH, Activation::None>(m_q, m_output_activation, m_weights_matrices, input, forward + input.size(), act_mem, act_mem_temp, output, output_stride, m_n_hidden_layers, batch_size, m_inputs_width, m_output_width); break;
    case Activation::Exponential: mlp_swift_forward<WIDTH, Activation::Exponential>(m_q, m_output_activation, m_weights_matrices, input, forward + input.size(), act_mem, act_mem_temp, output, output_stride, m_n_hidden_layers, batch_size, m_inputs_width, m_output_width); break;
    case Activation::Sigmoid:     mlp_swift_forward<WIDTH, Activation::Sigmoid>(m_q, m_output_activation, m_weights_matrices, input, forward + input.size(), act_mem, act_mem_temp, output, output_stride, m_n_hidden_layers, batch_size, m_inputs_width, m_output_width); break;
    case Activation::ReLU:        mlp_swift_forward<WIDTH, Activation::ReLU>(m_q, m_output_activation, m_weights_matrices, input, forward + input.size(), act_mem, act_mem_temp, output, output_stride, m_n_hidden_layers, batch_size, m_inputs_width, m_output_width); break;
    case Activation::LeakyReLU:   mlp_swift_forward<WIDTH, Activation::LeakyReLU>(m_q, m_output_activation, m_weights_matrices, input, forward + input.size(), act_mem, act_mem_temp, output, output_stride, m_n_hidden_layers, batch_size, m_inputs_width, m_output_width); break;
    case Activation::Squareplus:  mlp_swift_forward<WIDTH, Activation::Squareplus>(m_q, m_output_activation, m_weights_matrices, input, forward + input.size(), act_mem, act_mem_temp, output, output_stride, m_n_hidden_layers, batch_size, m_inputs_width, m_output_width); break;
    case Activation::Softplus:    mlp_swift_forward<WIDTH, Activation::Softplus>(m_q, m_output_activation, m_weights_matrices, input, forward + input.size(), act_mem, act_mem_temp, output, output_stride, m_n_hidden_layers, batch_size, m_inputs_width, m_output_width); break;
    case Activation::Tanh:        mlp_swift_forward<WIDTH, Activation::Tanh>(m_q, m_output_activation, m_weights_matrices, input, forward + input.size(), act_mem, act_mem_temp, output, output_stride, m_n_hidden_layers, batch_size, m_inputs_width, m_output_width); break;
    default: throw std::runtime_error{"Unsupported activation."};
    }

    if (m_output_width > 16) {



        m_q.parallel_for<>(range<1>(layer_length), [=](id<1> idx) {
            A[idx] = (float)forward[idx + (n_hidden_matrices + 1) * layer_length];
            });

        m_q.parallel_for<>(range<1>(m_output_width * m_net_width), [=](id<1> idx) {
            B[idx] = (float)p[toPackedLayoutCoord(idx, net_width, output_width) + net_width * (inputs_width + n_hidden_matrices * net_width)];
            });

        oneapi::mkl::blas::row_major::gemm(m_q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
            batch_size, m_output_width, WIDTH, 1, A, WIDTH, B, m_output_width, 0, C, m_output_width);


        m_q.parallel_for<>(range<1>(m_output_width * batch_size), [=](id<1> idx) {
            output.data()[idx] = (float)C[idx];
            forward[intermediate_output_size + input.size() + idx] = (float)C[idx];
            }).wait();
    }

}

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
            B[idx] = (float)p_w[offset_w + toPackedLayoutCoord(idx, output_width, net_width)];
            }).wait();

            oneapi::mkl::blas::row_major::gemm(m_q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
                batch_size, m_net_width, m_output_width, 1, A, m_output_width, B, m_net_width, 0, C, m_net_width);

            m_q.parallel_for<>(range<1>(WIDTH * batch_size), [=](id<1> idx) {
                int i = idx / batch_size;
                int j = idx % batch_size;
                D[i * batch_size + j] = (float)elt_activation_ret<float>(activation, forward[offset_f + j * net_width + i]);
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

    int batch_size = input.size() / m_inputs_width;
    auto p = m_grads_matrices.data();
    int s = m_grads_matrices.size();
    auto activation = m_activation;
    auto output_activation = m_output_activation;
    const int offset_grad = m_n_hidden_matrices * m_net_width * m_net_width + m_inputs_width * m_net_width;
    const int offset_f = m_inputs_width * batch_size + m_n_hidden_matrices * m_net_width * batch_size;

    const size_t alignment = 4096;


    m_q.parallel_for<>(range<1>(WIDTH * batch_size), [=](id<1> idx) {
        int i = idx / batch_size;
        int j = idx % batch_size;
        A[i * batch_size + j] = (float)elt_activation_ret<float>(activation, forward[offset_f + j * WIDTH + i]);
        }).wait();

        m_q.parallel_for<>(range<1>(batch_size * m_output_width), [=](id<1> idx) {
            elt_activation_bwd<bf16, float, float>(output_activation, grads.data()[idx], forward[offset_f + batch_size * WIDTH + idx], B[idx]);
            loss.data()[idx] = (bf16)B[idx];
            }).wait();


            oneapi::mkl::blas::row_major::gemm(m_q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
                m_net_width, m_output_width, batch_size, 1, A, batch_size, B, m_output_width, 0, C, m_output_width);

            m_q.parallel_for<>(range<1>(m_net_width * m_output_width), [=](id<1> idx) {
                p[idx + offset_grad] = (float)C[idx];
                }).wait();

                /// Backpropagation through last layer
                dgemm_last_layer_backward(grads,
                    forward,
                    loss,
                    batch_size,
                    A_backward_last_layer,
                    B_backward_last_layer,
                    C_backward_last_layer,
                    D_backward_last_layer,
                    E_backward_last_layer,
                    F_backward_last_layer);
                switch (m_activation) {
                case Activation::None:        mlp_swiftnet_backward<WIDTH, Activation::None>(m_q, m_weightsT_matrices, loss, m_grads_matrices, out_inter, delta_temp, forward, A_dgemm, B_dgemm, C_dgemm, batch_size, m_n_hidden_matrices); break;
                case Activation::ReLU:        mlp_swiftnet_backward<WIDTH, Activation::ReLU>(m_q, m_weightsT_matrices, loss, m_grads_matrices, out_inter, delta_temp, forward, A_dgemm, B_dgemm, C_dgemm, batch_size, m_n_hidden_matrices); break;
                case Activation::LeakyReLU:   mlp_swiftnet_backward<WIDTH, Activation::LeakyReLU>(m_q, m_weightsT_matrices, loss, m_grads_matrices, out_inter, delta_temp, forward, A_dgemm, B_dgemm, C_dgemm, batch_size, m_n_hidden_matrices); break;
                case Activation::Exponential: mlp_swiftnet_backward<WIDTH, Activation::Exponential>(m_q, m_weightsT_matrices, loss, m_grads_matrices, out_inter, delta_temp, forward, A_dgemm, B_dgemm, C_dgemm, batch_size, m_n_hidden_matrices); break;
                case Activation::Sigmoid:     mlp_swiftnet_backward<WIDTH, Activation::Sigmoid>(m_q, m_weightsT_matrices, loss, m_grads_matrices, out_inter, delta_temp, forward, A_dgemm, B_dgemm, C_dgemm, batch_size, m_n_hidden_matrices); break;
                case Activation::Tanh:        mlp_swiftnet_backward<WIDTH, Activation::Tanh>(m_q, m_weightsT_matrices, loss, m_grads_matrices, out_inter, delta_temp, forward, A_dgemm, B_dgemm, C_dgemm, batch_size, m_n_hidden_matrices); break;

                default: throw std::runtime_error{"Unsupported activation."};
                }

                m_q.parallel_for<>(range<1>(s), [=](id<1> idx) {
                    p[idx] /= batch_size;
                    }).wait();
}



void test1() {

    const int batch_size = std::pow(2, 10);
    const int output_width = 128;
    const int WIDTH = 64;
    const int intermediate_output_size = batch_size * WIDTH * 2;
    const int layer_length = WIDTH * batch_size;
    const int n_hidden_matrices = 1;
    const int net_width = 64;
    const int inputs_width = 64;


    const float scale = 1e-3f;
    /*device dev = device(gpu_selector_v);

    std::vector<device> subdev = {};

    subdev = dev.create_sub_devices<sycl::info::partition_property::
        partition_by_affinity_domain>(sycl::info::partition_affinity_domain::numa);

    queue q = queue(subdev[0]);*/

    queue q = queue();

    DeviceMem<bf16> inputs = DeviceMem<bf16>(batch_size * WIDTH, q);
    DeviceMem<float> output = DeviceMem<float>(batch_size * output_width, q);
    DeviceMem<float> target = DeviceMem<float>(batch_size * output_width, q);
    DeviceMem<bf16> grads = DeviceMem<bf16>(batch_size * output_width, q);
    DeviceMem<float> losses = DeviceMem<float>(batch_size * output_width, q);


    float* forward = malloc_device<float>(batch_size * (WIDTH + output_width + WIDTH * 4), q);
    int shmem_size = batch_size * WIDTH * 2;
    const size_t alignment = 4096;

    auto act_mem = sycl::aligned_alloc_device<bf16>(alignment, shmem_size, q);
    auto act_mem_temp = sycl::aligned_alloc_device<float>(alignment, shmem_size, q);

    auto A_forward = sycl::aligned_alloc_device<float>(alignment, layer_length, q);
    auto B_forward = sycl::aligned_alloc_device<float>(alignment, output_width * 64, q);
    auto C_forward = sycl::aligned_alloc_device<float>(alignment, output_width * batch_size, q);

    float* out_inter = malloc_device<float>(batch_size * WIDTH * (n_hidden_matrices + 1), q);
    auto deltas_temp = sycl::aligned_alloc_device<float>(alignment, output_width * batch_size, q);
    DeviceMem<bf16> deltas(output_width * batch_size, q);

    auto A_backward = sycl::aligned_alloc_device<float>(alignment, WIDTH * batch_size, q);
    auto B_backward = sycl::aligned_alloc_device<float>(alignment, batch_size * output_width, q);
    auto C_backward = sycl::aligned_alloc_device<float>(alignment, WIDTH * output_width, q);

    auto A_backward_last_layer = sycl::aligned_alloc_device<float>(alignment, grads.size(), q);
    auto B_backward_last_layer = sycl::aligned_alloc_device<float>(alignment, output_width * WIDTH, q);
    auto C_backward_last_layer = sycl::aligned_alloc_device<float>(alignment, WIDTH * batch_size, q);
    auto D_backward_last_layer = sycl::aligned_alloc_device<float>(alignment, WIDTH * batch_size, q);
    auto E_backward_last_layer = sycl::aligned_alloc_device<float>(alignment, batch_size * WIDTH, q);
    auto F_backward_last_layer = sycl::aligned_alloc_device<float>(alignment, WIDTH * WIDTH, q);

    auto A_dgemm = sycl::aligned_alloc_device<float>(alignment, batch_size * WIDTH, q);
    auto B_dgemm = sycl::aligned_alloc_device<float>(alignment, batch_size * WIDTH, q);
    auto C_dgemm = sycl::aligned_alloc_device<float>(alignment, WIDTH * WIDTH, q);


    inputs.initialize_constant(bf16(2.0f), q);
    output.initialize_constant(0.0f, q);
    target.initialize_constant(8.0f, q);
    grads.initialize_constant(bf16(0.0f), q);
    losses.initialize_constant(0.0f, q);

    nlohmann::json config = {
    {"loss", {
            {"otype", "L2"}
    }},
    {"optimizer", {
            {"otype", "sgd"},
            {"output_width", 128},
            {"n_hidden_layers", 2},
            {"learning_rate", 1e-3},
            {"l2_reg", 1e-8f}
    }},
    {"encoding", {
            {"otype", "HashGrid"},
            {"n_levels", 16},
            {"n_features_per_level", 2},
            {"log2_hashmap_size", 19},
            {"base_resolution", 16},
            {"per_level_scale", 2.0},
    }},
    {"network", {
            {"otype", "SwiftNetMLP"},
            {"activation", "ReLU"},
            {"output_activation", "None"},
            {"n_neurons", 64},
            {"n_hidden_layers", 2},
    }},
    };

    //auto model = create_from_config(q, config);

    L2Loss loss;
    SGDOptimizer optim = SGDOptimizer(128, 2, 1e-3f, 1e-8f);
    SwiftNetMLP<64> network = SwiftNetMLP<64>(q, 64, 128, 2, Activation::None, Activation::None);
    Trainer train(network, loss, optim);

    train.initialize_params();

    q.parallel_for<>(range<1>(inputs.size()), [=](id<1> idx) {
        forward[idx] = (float)inputs.data()[idx];
        });

    for (int i = 0; i < 10; i++) {
        std::cout << i << std::endl;
        train.training_step(inputs,
            forward,
            act_mem,
            act_mem_temp,
            A_forward,
            B_forward,
            C_forward,
            out_inter,
            deltas_temp,
            deltas,
            A_backward,
            B_backward,
            C_backward,
            A_backward_last_layer,
            B_backward_last_layer,
            C_backward_last_layer,
            D_backward_last_layer,
            E_backward_last_layer,
            F_backward_last_layer,
            A_dgemm,
            B_dgemm,
            C_dgemm,
            output,
            target,
            grads,
            losses,
            scale,
            64);
    }

    inputs.free_mem(q);
    output.free_mem(q);
    target.free_mem(q);
    grads.free_mem(q);
    losses.free_mem(q);
    free(act_mem, q);
    free(act_mem_temp, q);
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


