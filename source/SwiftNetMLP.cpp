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
#include "config.h"

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
void work_group_layer(nd_item<1> item, Activation activation, bf16* act_mem, bf16* weights_layer, float* out_inter, bf16* out, stream outs, bf16* forward_act = nullptr) {

	auto sg = item.get_sub_group();
	int sgId = sg.get_group_id();
	const int N_BLOCKS = WIDTH / TK;

	device_ptr<bf16> w(weights_layer);
	device_ptr<bf16> a(act_mem);
	device_ptr<float> o(out_inter);
	device_ptr<bf16> f(forward_act);

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

		joint_matrix_store(sg, result_matrix, o + TN * sgId + TM * l * WIDTH, WIDTH, layout::row_major);


	}
	for (int i = 0; i < N_ITERS; i++) {
		for (int j = 0; j < TN; j++) {
			for (int k = 0; k < TM; k++) {
				act_mem[TN * sgId + TM * i * WIDTH + j + k * WIDTH] = out_inter[TN * sgId + TM * i * WIDTH + j + k * WIDTH];
			}
		}
	}
	for (int i = 0; i < N_ITERS; i++) {
		if (BACKWARD) {
			matrix_activation_backward<float, bf16, bf16, SG_SIZE>(item, activation, o + TN * sgId + TM * i * WIDTH, f + TN * sgId + i * TM * WIDTH, out + TN * sgId + TM * i * WIDTH, WIDTH);
		}

		else {
			matrix_activation<bf16>(item, activation, a + TN * sgId + TM * i * WIDTH, WIDTH, outs);
		}
	}
}

template <int WIDTH, int N_ITERS>
void workgroup_load_input_static(nd_item<1> item, bf16* act_shmem, const bf16* input) {
	int localId = item.get_local_id();
	auto sg = item.get_sub_group();
	int sgId = sg.get_group_id();

	for (int i = 0; i < N_ITERS; i++) {
		for (int j = 0; j < TN; j++) {
			for (int k = 0; k < TM; k++) {
				act_shmem[TN * sgId + TM * i * WIDTH + j + k * WIDTH] = input[TN * sgId + TM * i * WIDTH + j + k * WIDTH];
			}
		}
	}
}


template <int WIDTH, int N_ITERS>
void workgroup_write_output_static(nd_item<1> item, bf16* act_shmem, float* output_threadblock) {
	int localId = item.get_local_id();
	auto sg = item.get_sub_group();
	int sgId = sg.get_group_id();

	for (int i = 0; i < N_ITERS; i++) {
		for (int j = 0; j < TN; j++) {
			for (int k = 0; k < TM; k++) {
				output_threadblock[TN * sgId + TM * i * WIDTH + j + k * WIDTH] = act_shmem[TN * sgId + TM * i * WIDTH + j + k * WIDTH];
			}
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
	const int batch_size,
	stream outs)
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

			matrix_activation<float>(item, activation, o + TK * sgId + TM * l * WIDTH, WIDTH, outs);

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
	const int output_stride,
	stream outs) {

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
	bf16* act_shmem,
	float* out,
	const int batch_size,
	const uint32_t output_stride,
	const uint32_t input_width,
	const uint32_t output_width,
	const uint32_t n_hidden_matmuls,
	const layout input_layout,
	const layout output_layout,
	stream outs) {

	// Handle first layer because it has different input

	auto wg = item.get_group();
	const int wg_idx = wg.get_group_id();
	const int elem_idx = WIDTH * wg_idx;

	if (input_width == WIDTH) {

		workgroup_load_input_static<WIDTH, N_ITERS>(item, act_shmem + elem_idx * WIDTH, input + elem_idx * WIDTH);
		work_group_layer<WIDTH, N_ITERS, false>(item, activation, act_shmem + elem_idx * WIDTH, weights_layer, out_intermediate_layer + elem_idx * WIDTH, nullptr, outs);
	}
	else {
		workgroup_input_layer_forward_dynamic<WIDTH, N_ITERS>(item,
			activation,
			act_shmem,
			input + elem_idx * input_width,
			weights_layer,
			out_intermediate_layer + elem_idx * WIDTH,
			input_width,
			batch_size,
			outs);

	}

	// Handle hidden layers all together

	const int first_weight_length = input_width * WIDTH;
	const int hidden_weight_lenght = WIDTH * WIDTH;
	const int layer_lenght = WIDTH * batch_size;

	for (int k = 0; k < n_hidden_matmuls; k++) {
		work_group_layer<WIDTH, N_ITERS, false>(item, activation, act_shmem + elem_idx * WIDTH, weights_layer + first_weight_length + k * hidden_weight_lenght, out_intermediate_layer + elem_idx * WIDTH + (k + 1) * layer_lenght, nullptr, outs);
	}

	//// Handle output layer

	if (output_width > 16) {
		//work_group_layer<WIDTH, N_ITERS, false>(item, activation, act_shmem + elem_idx * WIDTH, weights_layer + first_weight_length + n_hidden_matmuls * hidden_weight_lenght, out_intermediate_layer + elem_idx * WIDTH + (n_hidden_matmuls + 1) * layer_lenght, nullptr, outs);

		//workgroup_write_output_static<WIDTH, N_ITERS>(item, act_shmem, out + elem_idx * WIDTH);
	}
	else if (out) {
		workgroup_last_layer_forward<WIDTH, N_ITERS>(item,
			output_activation,
			act_shmem,
			weights_layer + first_weight_length + hidden_weight_lenght * n_hidden_matmuls,
			out + elem_idx * WIDTH + (n_hidden_matmuls + 1) * layer_lenght,
			output_stride,
			outs);

	}
}

template <int WIDTH, Activation activation>
void mlp_swift_forward(queue q,
	Activation output_activation,
	const DeviceMem<bf16>& weights,
	const DeviceMem<bf16>& inputs,
	DeviceMem<float>& intermediate_output,
	DeviceMem<float>& output,
	const int output_stride,
	const int n_hidden_layers,
	const int batch_size,
	const int input_width,
	const int output_width)
{
	const int N_BLOCKS = WIDTH / TK;
	const int N_ITERS = WIDTH / TM;

	int shmem_size = batch_size * WIDTH;

	DeviceMem<bf16> act(shmem_size, q);

	q.submit([&](handler& cgh)
		{
			stream outs(1024, 256, cgh);
			cgh.parallel_for<>(
				nd_range<1>(batch_size * WG_SIZE / BATCH_CHUNK, WG_SIZE),
				[=](nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]]
				{
					kernel_swift_mlp<WIDTH, N_ITERS, activation>(item,
						output_activation,
						inputs.data(),
						weights.data(),
						intermediate_output.data(),
						act.data(),
						output.data(),
						batch_size,
						output_stride,
						input_width,
						output_width,
						n_hidden_layers - 1,
						layout::col_major,
						layout::col_major,
						outs);
				});
		}).wait();

}

template <int WIDTH, int N_ITERS, Activation ACTIVATION>
void kernel_swiftnet_backward(
	nd_item<1> item,
	bf16* loss_gradients,
	bf16* grads,
	bf16* weights,
	bf16* forward,
	float* out_inter,
	int batch_number,
	uint32_t n_hidden_matmuls,
	stream outs

) {
	auto sg = item.get_sub_group();

	int groupId = item.get_group(0);
	int sgId = sg.get_group_id();
	int idx = 8 * groupId * N_ITERS;
	const int layer_length = WIDTH * WIDTH * batch_number;


	//On suppose qu'on a déjà fait la backprop dans le dernier layer
	// Hidden Layers
	for (int k = 0; k < n_hidden_matmuls; k++) {
		work_group_layer<WIDTH, N_ITERS, true>(
			item,
			ACTIVATION,
			loss_gradients + groupId * WIDTH * WIDTH,
			weights + WIDTH * WIDTH * (n_hidden_matmuls - k),
			out_inter + groupId * WIDTH * WIDTH + (n_hidden_matmuls - k - 1) * layer_length,
			loss_gradients + groupId * WIDTH * WIDTH,
			outs,
			forward + WIDTH * WIDTH * batch_number + WIDTH * WIDTH * batch_number * (n_hidden_matmuls - k - 1) + groupId * WIDTH * WIDTH//Le premier WIDTH*WIDTH*batch_number correspond à l'input, à remplacer par batch_size * input_width
		);
		item.barrier();
	}
}

template <int WIDTH, Activation ACTIVATION>
void dgemm_multiply(bf16* grads_device, float* loss_gradients, bf16* fwd, int k, int batch_size, int m_n_hidden_matrices) {
	const int layer_length = WIDTH * batch_size;

	double* A;
	double* B;
	double* C;
	A = (double*)mkl_malloc(batch_size * WIDTH * sizeof(double), 64);

	B = (double*)mkl_malloc(batch_size * WIDTH * sizeof(double), 64);
	C = (double*)mkl_malloc(WIDTH * WIDTH * sizeof(double), 64);
	for (int i = 0; i < WIDTH; i++) {
		for (int j = 0; j < batch_size; j++) {
			A[i * batch_size + j] = (double)elt_activation_ret<bf16>(ACTIVATION, fwd[i + j * WIDTH + (m_n_hidden_matrices - k - 1) * layer_length]);
		}
	}
	for (int i = 0; i < WIDTH * batch_size; i++) {
		B[i] = (double)loss_gradients[i + (m_n_hidden_matrices - k - 1) * layer_length];
	}
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		WIDTH, WIDTH, batch_size, 1, A, batch_size, B, WIDTH, 0, C, WIDTH);
	//for (int i = 0; i < 64; i++) {
	//	std::cout << "B : " << B[64*i] << std::endl;
	//}
	//for (int i = 0; i < 64; i++) {
	//	std::cout << "loss_gradients : " << loss_gradients[ i] << std::endl;
	//}
	bf16 x = 0;
	for (int i = 0; i < WIDTH * WIDTH; i++) {
		grads_device[(m_n_hidden_matrices - k - 1) * WIDTH * WIDTH + i] += C[i];
	}

	mkl_free(A);
	mkl_free(B);
	mkl_free(C);
}

template<int WIDTH, Activation ACTIVATION>
void mlp_swiftnet_backward(
	queue q,
	DeviceMem<bf16>& weights_transposed,
	DeviceMem<bf16>& deltas,
	DeviceMem<bf16>& grads_matrices,
	DeviceMem<bf16>& forward,
	int batch_size,
	const uint32_t n_hidden_matmuls
) {
	// here, weights are already transposed and packed
	// in deltas, the last layer has already been calculated

	const int layer_lenght = WIDTH * batch_size;
	const int N_ITERS = 8;
	int batch_number = batch_size / 64;
	try {

		float* out_inter = malloc_shared<float>(batch_size * WIDTH * (n_hidden_matmuls + 2), q);

		q.wait();
		q.submit([&](handler& h) {
			//Transfer data to device memory
			stream outs(1024, 256, h);
			h.parallel_for(nd_range<1>(batch_size * WG_SIZE / BATCH_CHUNK, WG_SIZE), [=](nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
				kernel_swiftnet_backward<WIDTH, N_ITERS, ACTIVATION>(item, deltas.data(), grads_matrices.data(), weights_transposed.data(), forward.data(), out_inter, batch_number, n_hidden_matmuls, outs);
				});
			}).wait();

			for (int k = 0; k < n_hidden_matmuls; k++) {
				dgemm_multiply<WIDTH, ACTIVATION>(grads_matrices.data(), out_inter, forward.data(), k, batch_size, n_hidden_matmuls);
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
	m_q{ q },
	m_inputs_width{ input_width },
	m_net_width{ WIDTH },
	m_output_width{ output_width },
	m_n_hidden_layers{ n_hidden_layers },
	m_activation{ activation },
	m_output_activation{ output_activation }
{
	m_n_hidden_matrices = m_n_hidden_layers - 1;
	m_weightsT_matrices.allocate(m_net_width * m_inputs_width + (m_net_width * m_net_width) * m_n_hidden_matrices + m_net_width * m_output_width, m_q);
	m_weights_matrices.allocate(m_net_width * m_inputs_width + (m_net_width * m_net_width) * m_n_hidden_matrices + m_net_width * m_output_width, m_q);
	m_weights_matrices_inferences.allocate(m_net_width * m_inputs_width + (m_net_width * m_net_width) * m_n_hidden_matrices + m_net_width * m_output_width, m_q);
	m_grads_matrices.allocate(m_net_width * m_inputs_width + (m_net_width * m_net_width) * m_n_hidden_matrices + m_net_width * m_output_width, m_q);

}

template <int WIDTH>
void SwiftNetMLP<WIDTH>::initialize_params() {
	for (int i = 0; i < m_net_width * m_inputs_width; i++) {
		m_weights_matrices.data()[i] = bf16(1.0f / 64);
		m_weights_matrices_inferences.data()[i] = bf16(1.0f / 64);
		m_weightsT_matrices.data()[i] = bf16(1.0f / 64);
	}

	for (int i = 0; i < m_n_hidden_matrices; i++) {
		for (int j = 0; j < m_net_width * m_net_width; j++) {

			m_weights_matrices.data()[i * m_net_width * m_net_width + m_net_width * m_inputs_width + j] = bf16(1.0f / 64);
			m_weights_matrices_inferences.data()[i * m_net_width * m_net_width + m_net_width * m_inputs_width + j] = bf16(1.0f / 64);
			m_weightsT_matrices.data()[i * m_net_width * m_net_width + m_net_width * m_inputs_width + j] = bf16(1.0f / 64);
		}
	}

	for (int i = 0; i < m_net_width * m_output_width; i++) {
		m_weights_matrices.data()[m_net_width * m_inputs_width + (m_net_width * m_net_width) * m_n_hidden_matrices + i] = bf16(1.0f / 64);
		m_weights_matrices_inferences.data()[m_net_width * m_inputs_width + (m_net_width * m_net_width) * m_n_hidden_matrices + i] = bf16(1.0f / 64);
		m_weightsT_matrices.data()[m_net_width * m_inputs_width + (m_net_width * m_net_width) * m_n_hidden_matrices + i] = bf16(1.0f / 64);
	}
	//m_weights_matrices.initialize_uniform(0.1, m_weightsT_matrices, m_inputs_width, m_net_width, m_output_width, m_n_hidden_matrices);
}

template <int WIDTH>
DeviceMem<bf16> SwiftNetMLP<WIDTH>::forward_pass(const DeviceMem<bf16>& input, DeviceMem<float>& output) {

	const int output_stride = WIDTH;
	const int batch_size = input.size() / m_inputs_width;
	const int input_size = input.size();
	const int intermediate_output_size = batch_size * WIDTH * m_n_hidden_layers;
	DeviceMem<float> forward_f = DeviceMem<float>(intermediate_output_size, m_q);
	DeviceMem<bf16> forward = DeviceMem<bf16>(batch_size * (m_inputs_width + m_output_width + WIDTH * m_n_hidden_layers), m_q);

	static_assert(WIDTH % 16 == 0, "Width must be a multiply of 16.");
	assert(batch_size % 64 == 0);



	switch (m_activation) {
	case Activation::None:        mlp_swift_forward<WIDTH, Activation::None>(m_q, m_output_activation, m_weights_matrices, input, forward_f, output, output_stride, m_n_hidden_layers, batch_size, m_inputs_width, m_output_width); break;
	case Activation::Exponential: mlp_swift_forward<WIDTH, Activation::Exponential>(m_q, m_output_activation, m_weights_matrices, input, forward_f, output, output_stride, m_n_hidden_layers, batch_size, m_inputs_width, m_output_width); break;
	case Activation::Sigmoid:     mlp_swift_forward<WIDTH, Activation::Sigmoid>(m_q, m_output_activation, m_weights_matrices, input, forward_f, output, output_stride, m_n_hidden_layers, batch_size, m_inputs_width, m_output_width); break;
	case Activation::ReLU:        mlp_swift_forward<WIDTH, Activation::ReLU>(m_q, m_output_activation, m_weights_matrices, input, forward_f, output, output_stride, m_n_hidden_layers, batch_size, m_inputs_width, m_output_width); break;
	case Activation::LeakyReLU:   mlp_swift_forward<WIDTH, Activation::LeakyReLU>(m_q, m_output_activation, m_weights_matrices, input, forward_f, output, output_stride, m_n_hidden_layers, batch_size, m_inputs_width, m_output_width); break;
	case Activation::Squareplus:  mlp_swift_forward<WIDTH, Activation::Squareplus>(m_q, m_output_activation, m_weights_matrices, input, forward_f, output, output_stride, m_n_hidden_layers, batch_size, m_inputs_width, m_output_width); break;
	case Activation::Softplus:    mlp_swift_forward<WIDTH, Activation::Softplus>(m_q, m_output_activation, m_weights_matrices, input, forward_f, output, output_stride, m_n_hidden_layers, batch_size, m_inputs_width, m_output_width); break;
	case Activation::Tanh:        mlp_swift_forward<WIDTH, Activation::Tanh>(m_q, m_output_activation, m_weights_matrices, input, forward_f, output, output_stride, m_n_hidden_layers, batch_size, m_inputs_width, m_output_width); break;
	default: throw std::runtime_error{"Unsupported activation."};
	}

	if (m_output_width > 16) {
		const int layer_length = WIDTH * batch_size;

		double* A;
		double* B;
		double* C;
		A = (double*)mkl_malloc(layer_length * sizeof(double), 64);

		B = (double*)mkl_malloc(m_output_width * m_net_width * sizeof(double), 64);
		C = (double*)mkl_malloc(m_output_width * batch_size * sizeof(double), 64);
		for (int i = 0; i < layer_length; i++) {
			A[i] = (double)forward_f.data()[i + m_n_hidden_matrices * layer_length];
		}
		for (int i = 0; i < m_output_width * m_net_width; i++) {
			B[i] = (double)m_weights_matrices.data()[toPackedLayoutCoord(i, m_net_width, m_output_width) + m_net_width * (m_inputs_width + m_n_hidden_matrices * m_net_width)];
		}
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			batch_size, m_output_width, WIDTH, 1, A, WIDTH, B, m_output_width, 0, C, m_output_width);

		for (int i = 0; i < m_output_width * batch_size; i++) {
			output.data()[i] = (float)C[i];
		}

		mkl_free(A);
		mkl_free(B);
		mkl_free(C);
	}

	m_q.parallel_for<>(range<1>(forward.size()), [=](id<1> idx) {
		if (idx < input.size()) {
			forward.data()[idx] = (bf16)input.data()[idx];
		}
		else if (idx < forward_f.size() + input.size()) {
			forward.data()[idx] = (bf16)(forward_f.data()[idx - input.size()]);
		}
		else {
			forward.data()[idx] = (bf16)output.data()[idx - intermediate_output_size - input_size];
		}
		}).wait();


		/*for (int i = 0; i < input_size; i++) {
			forward.data()[i] = input.data()[i];
		}
		for (int i = 0; i < forward_f.size(); i++) {
			forward.data()[i + input_size] = bf16(forward_f.data()[i]);
		}
		for (int i = 0; i < output.size(); i++) {
			forward.data()[i + intermediate_output_size + input_size] = output.data()[i];
		}*/
		forward_f.free_mem(m_q);

		return forward;
}

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
void SwiftNetMLP<WIDTH>::dgemm_last_layer_backward(DeviceMem<bf16>& grads, DeviceMem<bf16>& forward, DeviceMem<bf16>& loss, int batch_size) {
	double* A;
	double* B;
	double* C;
	double* D;
	A = (double*)mkl_malloc(grads.size() * sizeof(double), 64);
	B = (double*)mkl_malloc(m_output_width * WIDTH * sizeof(double), 64);
	C = (double*)mkl_malloc(WIDTH * batch_size * sizeof(double), 64);
	for (int i = 0; i < grads.size(); i++) {
		A[i] = (double)loss.data()[i];
	}
	for (int i = 0; i < m_output_width * WIDTH; i++) {
		B[i] = (double)m_weightsT_matrices.data()[m_n_hidden_matrices * m_net_width * m_net_width + m_net_width * m_inputs_width + toPackedLayoutCoord(i, m_output_width, m_net_width)];
	}
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		batch_size, m_net_width, m_output_width, 1, A, m_output_width, B, m_net_width, 0, C, m_net_width);


	mkl_free(A);
	mkl_free(B);;

	A = (double*)mkl_malloc(m_net_width * batch_size * sizeof(double), 64);
	B = (double*)mkl_malloc(batch_size * m_net_width * sizeof(double), 64);
	D = (double*)mkl_malloc(m_net_width * m_net_width * sizeof(double), 64);
	for (int i = 0; i < m_net_width; i++) {
		for (int j = 0; j < batch_size; j++) {
			A[i * batch_size + j] = (double)elt_activation_ret<bf16>(m_activation, forward.data()[m_inputs_width + (m_n_hidden_matrices - 1) * batch_size * m_net_width + j * m_net_width + i]);
		}
	}
	for (int i = 0; i < m_net_width * batch_size; i++) {
		elt_activation_bwd<double, double, double>(m_activation, C[i], forward.data()[m_inputs_width + (m_n_hidden_matrices - 1) * batch_size * m_net_width + i], B[i]);
		loss.data()[i] = (bf16)B[i];
	}

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		m_net_width, m_net_width, batch_size, 1, A, batch_size, B, m_net_width, 0, D, m_net_width);

	for (int i = 0; i < m_net_width * m_net_width; i++) {
		m_grads_matrices.data()[m_inputs_width * m_net_width + (m_n_hidden_matrices - 1) * m_net_width * m_net_width + i] = D[i];
	}

	mkl_free(A);
	mkl_free(B);
	mkl_free(C);
	mkl_free(D);


}

template <int WIDTH>
void SwiftNetMLP<WIDTH>::backward_pass(const DeviceMem<bf16>& input, DeviceMem<bf16>& grads, DeviceMem<bf16>& forward) {
	int batch_size = input.size() / m_inputs_width;
	auto p = m_grads_matrices.data();
	int s = m_grads_matrices.size();
	int offset_grad = m_n_hidden_matrices * m_net_width * m_net_width + m_inputs_width * m_net_width;

	bf16 x;
	DeviceMem<bf16> loss(m_output_width * batch_size, m_q);

	double* A;
	double* B;
	double* C;
	A = (double*)mkl_malloc(m_net_width * batch_size * sizeof(double), 64);
	B = (double*)mkl_malloc(batch_size * m_output_width * sizeof(double), 64);
	C = (double*)mkl_malloc(m_net_width * m_output_width * sizeof(double), 64);
	for (int i = 0; i < m_net_width; i++) {
		for (int j = 0; j < batch_size; j++) {
			A[i * batch_size + j] = (double)elt_activation_ret<bf16>(m_activation, forward.data()[m_inputs_width * batch_size + m_n_hidden_matrices * m_net_width * batch_size + j * m_net_width + i]);
		}
	}
	for (int i = 0; i < batch_size * m_output_width; i++) {
		elt_activation_bwd<bf16, double, double>(
			m_output_activation,
			grads.data()[i],
			forward.data()[input.size() + (m_n_hidden_matrices + 1) * batch_size * m_net_width + i],
			B[i]);
		loss.data()[i] = (bf16)B[i];
	}
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		m_net_width, m_output_width, batch_size, 1, A, batch_size, B, m_output_width, 0, C, m_output_width);

	for (int i = 0; i < m_net_width * m_output_width; i++) {
		m_grads_matrices.data()[m_n_hidden_matrices * m_net_width * m_net_width + m_inputs_width * m_net_width + i] = C[i];
	}

	mkl_free(A);
	mkl_free(B);
	mkl_free(C);


	/// Backpropagation through last layer
	dgemm_last_layer_backward(grads, forward, loss, batch_size);
	switch (m_activation) {
	case Activation::None:        mlp_swiftnet_backward<WIDTH, Activation::None>(m_q, m_weightsT_matrices, loss, m_grads_matrices, forward, batch_size, m_n_hidden_matrices); break;
	case Activation::ReLU:        mlp_swiftnet_backward<WIDTH, Activation::ReLU>(m_q, m_weightsT_matrices, loss, m_grads_matrices, forward, batch_size, m_n_hidden_matrices); break;
	case Activation::LeakyReLU:   mlp_swiftnet_backward<WIDTH, Activation::LeakyReLU>(m_q, m_weightsT_matrices, loss, m_grads_matrices, forward, batch_size, m_n_hidden_matrices); break;
	case Activation::Exponential: mlp_swiftnet_backward<WIDTH, Activation::Exponential>(m_q, m_weightsT_matrices, loss, m_grads_matrices, forward, batch_size, m_n_hidden_matrices); break;
	case Activation::Sigmoid:     mlp_swiftnet_backward<WIDTH, Activation::Sigmoid>(m_q, m_weightsT_matrices, loss, m_grads_matrices, forward, batch_size, m_n_hidden_matrices); break;
	case Activation::Tanh:        mlp_swiftnet_backward<WIDTH, Activation::Tanh>(m_q, m_weightsT_matrices, loss, m_grads_matrices, forward, batch_size, m_n_hidden_matrices); break;

	default: throw std::runtime_error{"Unsupported activation."};
	}

	m_q.parallel_for<>(range<1>(s), [=](id<1> idx) {
		p[idx] /= batch_size;
		}).wait();
}

Activation string_to_activation(const std::string& activation_name) {
	if (isequalstring(activation_name, "None")) {
		return Activation::None;
	}
	else if (isequalstring(activation_name, "ReLU")) {
		return Activation::ReLU;
	}
	else if (isequalstring(activation_name, "Exponential")) {
		return Activation::Exponential;
	}
	else if (isequalstring(activation_name, "Sigmoid")) {
		return Activation::Sigmoid;
	}
	else if (isequalstring(activation_name, "Sine")) {
		return Activation::Sine;
	}
	else if (isequalstring(activation_name, "Tanh")) {
		return Activation::Tanh;
	}
	throw std::runtime_error{"Invalid activation name:}"};
}

template<int WIDTH>
SwiftNetMLP<WIDTH>* create_network(queue q, const json& network) {


	int n_neurons = network.value("n_neurons", 128u);
	switch (n_neurons) {
	case  16: return new SwiftNetMLP<16>{ q, network["n_input_dims"], network["n_output_dims"], network["n_hidden_layers"], string_to_activation(network.value("activation", "ReLU")),string_to_activation(network.value("output_activation", "None")) };
	case  32: return new SwiftNetMLP<32>{ q, network["n_input_dims"], network["n_output_dims"], network["n_hidden_layers"], string_to_activation(network.value("activation", "ReLU")),string_to_activation(network.value("output_activation", "None")) };
	case  64: return new SwiftNetMLP<64>{ q, network["n_input_dims"], network["n_output_dims"], network["n_hidden_layers"], string_to_activation(network.value("activation", "ReLU")),string_to_activation(network.value("output_activation", "None")) };
	case 128: return new SwiftNetMLP<128>{ q, network["n_input_dims"], network["n_output_dims"], network["n_hidden_layers"], string_to_activation(network.value("activation", "ReLU")),string_to_activation(network.value("output_activation", "None")) };
	default: throw std::runtime_error{"SwiftNetMLP only supports 16, 32, 64, and 128 neurons, but got ..."};
	}
}


void test1() {

	const int batch_size = std::pow(2, 20);
	const int output_width = 128;
	const int WIDTH = 64;

	const float scale = 1e-3f;

	queue q = queue();

	DeviceMem<bf16> inputs = DeviceMem<bf16>(batch_size * WIDTH, q);
	DeviceMem<float> output = DeviceMem<float>(batch_size * output_width, q);
	DeviceMem<float> target = DeviceMem<float>(batch_size * output_width, q);
	DeviceMem<bf16> grads = DeviceMem<bf16>(batch_size * output_width, q);
	DeviceMem<float> losses = DeviceMem<float>(batch_size * output_width, q);

	inputs.initialize_constant(bf16(1.0f));
	output.initialize_constant(0.0f);
	target.initialize_constant(8.0f);
	grads.initialize_constant(bf16(0.0f));
	losses.initialize_constant(0.0f);

	for (int i = 0; i < batch_size * WIDTH; i++) {
		inputs.data()[i] = (i / WIDTH) % 16;
	}
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

	//auto model = create_from_config<64>(q, config);

	L2Loss loss;
	SGDOptimizer<64> optim = SGDOptimizer<64>(128, 2, 1e-3f, 1e-8f);
	SwiftNetMLP<64> network = SwiftNetMLP<64>(q, 64, 128, 2, Activation::None, Activation::None);
	Trainer<64> train(network, loss, optim);

	train.initialize_params();

	std::cout << "first step \n";

	train.training_step(inputs, output, target, grads, losses, scale);

	for (int i = 0; i < 1; i++) {

		std::cout << "losses : " << losses.data()[i] << std::endl;
		std::cout << "grads : " << grads.data()[i] << std::endl;
		std::cout << "output : " << output.data()[i] << std::endl;
	}

	std::cout << "1 step \n";
	for (int i = 0; i < 1; i++) {
		train.training_step(inputs, output, target, grads, losses, scale);
	}

	for (int i = 0; i < 1; i++) {

		std::cout << "losses : " << losses.data()[i] << std::endl;
		std::cout << "grads : " << grads.data()[i] << std::endl;
		std::cout << "output : " << output.data()[i] << std::endl;
	}

	inputs.free_mem(q);
	output.free_mem(q);
	target.free_mem(q);
	grads.free_mem(q);
	losses.free_mem(q);
}