#include <iostream>
#include <vector>
#include <CL/sycl.hpp>
#include <ext/oneapi/experimental/bfloat16.hpp>
#include <ext/oneapi/matrix/matrix.hpp>
#include "activation.h"

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
using bf16 = sycl::ext::oneapi::bfloat16;
#define TM 8
#define TK 16
#define TN 16


#define SG_SIZE 16
#define WG_SIZE 4*SG_SIZE
#define BATCH_CHUNK 64
template <int WIDTH, int N_ITERS>
void work_group_layer_backward(nd_item<1> it, Activation activation, bf16* act_mem, bf16* weights_layer, float* out_inter, bf16* out, stream outs, bf16* forward_act = nullptr) {

	auto sg = it.get_sub_group();
	int sgId = sg.get_group_id();
	const int N_BLOCKS = WIDTH / 16;
	device_ptr<bf16> w(weights_layer);
	device_ptr<bf16> a(act_mem);

	device_ptr<bf16> f(forward_act);
	device_ptr<float> o(out_inter);


	it.barrier();

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
		
		matrix_activation_backward<float, bf16, bf16,SG_SZ>(it, activation, o + TN * sgId + 8 * l * WIDTH, f + TN * sgId + l * 8 * WIDTH, out + TN * sgId + 8 * l * WIDTH, WIDTH);
		
		



	}
}
template <int WIDTH, int N_ITERS, typename T, bool BACKWARD = false>
void work_group_layer(nd_item<1> item, Activation activation, bf16* act_mem, bf16* weights_layer, T* out, stream outs, bf16* forward_act = nullptr) {
	
	auto sg = item.get_sub_group();
	int sgId = sg.get_group_id();
	const int N_BLOCKS = WIDTH / TK;
	device_ptr<bf16> w(weights_layer);
	device_ptr<bf16> a(act_mem);
	device_ptr<float> o(out);
	device_ptr<bf16> f(forward_act);
	

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
	
		if (BACKWARD) {
			matrix_activation_backward<T,bf16>(item,activation,o+TK*sgId+TM*l*WIDTH,f + TK*sgId + l * TM * WIDTH,WIDTH,outs);
				
		}
		else {
			matrix_activation<T>(item, activation, o + TK * sgId + TM * l * WIDTH, WIDTH, outs);
		}

		joint_matrix_store(sg, result_matrix, o + TN * sgId + TM * l * WIDTH, WIDTH, layout::row_major);

	}
}


template <int WIDTH, int N_ITERS, Activation ACTIVATION>
void kernel_swiftnet_backward(
	nd_item<1> item,
	bf16* loss_gradients,
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

	//On suppose qu'on a déjà fait la backprop dans le dernier layer
	// Hidden Layers
	
	for (int k = 0; k < n_hidden_matmuls; k++) {
		work_group_layer_backward<WIDTH, N_ITERS>(
			item,
			ACTIVATION,
			loss_gradients + WIDTH * WIDTH * batch_number * (n_hidden_matmuls - k) + groupId * WIDTH * WIDTH,
			weights + WIDTH * WIDTH * (n_hidden_matmuls - k),
			out_inter,
			loss_gradients + WIDTH * WIDTH * batch_number * (n_hidden_matmuls - k - 1) + groupId * WIDTH * WIDTH,
			outs,
			forward + WIDTH * WIDTH * batch_number * (n_hidden_matmuls - k - 1) + groupId * WIDTH * WIDTH
		);
	}
	
}


template<int WIDTH, Activation ACTIVATION>
void mlp_swiftnet_backward(
	std::vector<bf16> weights_transposed,
	std::vector<bf16>& deltas,
	std::vector<bf16> forward,
	int batch_size,
	const uint32_t n_hidden_matmuls
) {
	// here, weights are already transposed and packed
	// in deltas, the last layer has already been calculated

	const int N_ITERS = 8;
	int batch_number = batch_size / 64;
	queue q = queue();
	try {

		bf16* weights_device = malloc_shared<bf16>(weights_transposed.size(), q);
		q.memcpy(weights_device, weights_transposed.data(), weights_transposed.size() * sizeof(bf16));

		bf16* deltas_device = malloc_shared<bf16>(deltas.size(), q);
		q.memcpy(deltas_device, deltas.data(), deltas.size() * sizeof(bf16));
		


		float* out_inter = malloc_shared<float>(64 * 64, q);

		bf16* fwd_device = malloc_shared<bf16>(forward.size(), q);
		q.memcpy(fwd_device, forward.data(), forward.size() * sizeof(bf16));
		q.submit([&](handler& h) {
			//Transfer data to device memory
			stream outs(1024, 256, h);
			h.parallel_for(nd_range<1>(batch_size, 64), [=](nd_item<1> item) [[intel::reqd_sub_group_size(SG_SZ)]] {
				kernel_swiftnet_backward<WIDTH, N_ITERS, ACTIVATION>(item, deltas_device, weights_device, fwd_device, out_inter, batch_number, n_hidden_matmuls, outs);
				});
			});
		for (int i = 0; i < deltas.size(); i++) {
			deltas[i] = deltas_device[i];
		}
		q.memcpy(deltas.data(), deltas_device, deltas.size() * sizeof(bf16));
	}
	
	catch (std::exception const& e)
	{
		std::cout << "An exception was caught when performing AMX/XMX matrix multiply.\n";
		std::terminate();
	}
}

//Fix les index pour copier la memoire
template <int WIDTH, int N_ITERS>
void workgroup_load_input_static(nd_item<1> item, bf16* act_shmem, const bf16* input) {
	int localId = item.get_local_id();
	auto sg = item.get_sub_group();
	int sgId = sg.get_group_id();

	int offset = (8 * localId) % WIDTH;
	int row = (16 * localId + sgId * 64) / WIDTH;

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

	int offset = (8 * localId) % WIDTH;
	int row = (16 * localId + sgId * 64) / WIDTH;

	for (int i = 0; i < N_ITERS; i++) {
		for (int j = 0; j < TN; j++) {
			for (int k = 0; k < TM; k++) {
				output_threadblock[TN * sgId + TM * i * WIDTH + j + k * WIDTH] = act_shmem[TN * sgId + TM * i * WIDTH + j + k * WIDTH];
			}
		}
	}
}


template <int WIDTH, int N_ITERS, typename T, Activation activation>
void kernel_swift_mlp(nd_item<1> item,
	const Activation output_activation,
	bf16* input,
	bf16* weights_layer,
	T* out_intermediate_layer,
	bf16* act_shmem,
	T* out,
	const uint32_t output_stride,
	const uint32_t input_width,
	const uint32_t output_width,
	const uint32_t n_hidden_matmuls,
	const layout input_layout,
	const layout output_layout) {



	// Handle first layer because it has different input

	auto wg = item.get_group();
	const int wg_idx = wg.get_group_id();
	const int elem_idx = WIDTH * wg_idx;

	if (input_width == WIDTH) {

		workgroup_load_input_static<WIDTH, N_ITERS>(item, act_shmem + elem_idx * WIDTH, input + elem_idx * WIDTH);
		work_group_layer<WIDTH, N_ITERS, T, false>(item, activation, act_shmem + elem_idx * WIDTH, weights_layer, out_intermediate_layer + elem_idx * WIDTH);
	}
	//else {
	//	workgroup_input_layer_forward_dynamic<WIDTH, N_ITERS, matrix_layout::row_major>(activation,
	//		act_shmem,
	//		input + elem_idx * input_width,
	//		weights_layer,
	//		input_width,
	//		batch_size);

	//}

	// Handle hidden layers all together

	const int first_weight_length = input_width * WIDTH;
	const int hidden_weight_lenght = WIDTH * WIDTH;
	const int last_weight_lenght = WIDTH * output_width;

	for (int k = 0; k < n_hidden_matmuls; k++) {
		work_group_layer<WIDTH, N_ITERS, T, false>(item, activation, act_shmem + elem_idx * WIDTH, weights_layer + first_weight_length + k * hidden_weight_lenght, out_intermediate_layer + elem_idx * WIDTH + +first_weight_length);
	}

	workgroup_write_output_static<WIDTH, N_ITERS>(item, act_shmem, out + elem_idx * WIDTH);

	//// Handle output layer
	//if (out) {
	//	workgroup_last_layer_forward<WIDTH, N_ITERS, OUT_T>(output_activation,
	//		act_shmem,
	//		weights_layer + first_weights_stride + weights_stride * n_hidden_matmuls,
	//		out + elem_idx * output_stride,
	//		output_stride,
	//		output_layout);

	//}
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
			//matrix_activation<float, joint_matrix<float, TM, TN>>(sg, activation, result_matrix);
			joint_matrix_store(sg, result_matrix, o + TN * sgId + TM * l * WIDTH, WIDTH, layout::row_major);
		}
	}
}


template <int WIDTH, int N_ITERS>
void workgroup_last_layer_forward(nd_item<1> item,
	Activation activation,
	bf16* act_mem,
	const bf16* input,
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
	//joint_matrix<sub_group, half, use::b, TK, TN, sycl::ext::intel::experimental::matrix::layout::packed> weight_matrices[N_BLOCKS];
	joint_matrix<sub_group, bf16, use::b, TK, TN, sycl::ext::intel::experimental::matrix::layout::packed> weight_matrix0;
	joint_matrix<sub_group, bf16, use::b, TK, TN, sycl::ext::intel::experimental::matrix::layout::packed> weight_matrix1;
	joint_matrix<sub_group, bf16, use::b, TK, TN, sycl::ext::intel::experimental::matrix::layout::packed> weight_matrix2;
	joint_matrix<sub_group, bf16, use::b, TK, TN, sycl::ext::intel::experimental::matrix::layout::packed> weight_matrix3;
	joint_matrix<sub_group, float, use::accumulator, TM, TN> result_matrix;

	bf16* weights_shmem = act_mem + N_ITERS * 16 * WIDTH;

	const int weights_row = (8 * li) % WIDTH;
	const int weights_col = (8 * li + 8 * 32 * sgId) / WIDTH;

	weights_shmem[weights_row + weights_col * WIDTH] = weights_layer[weights_row + weights_col * WIDTH];

	joint_matrix_load(sg, weight_matrix0, w + 16 * 2 * sgId + 8 * 0 * WIDTH * 2, WIDTH * 2);
	joint_matrix_load(sg, weight_matrix1, w + 16 * 2 * sgId + 8 * 1 * WIDTH * 2, WIDTH * 2);
	joint_matrix_load(sg, weight_matrix2, w + 16 * 2 * sgId + 8 * 2 * WIDTH * 2, WIDTH * 2);
	joint_matrix_load(sg, weight_matrix3, w + 16 * 2 * sgId + 8 * 3 * WIDTH * 2, WIDTH * 2);



	for (int l = 0; l < N_ITERS; l++) {
		joint_matrix_fill(sg, result_matrix, 0.0f);

		joint_matrix_load(sg, act_matrix, a + 16 * 0 + 8 * l * WIDTH, WIDTH);
		result_matrix = joint_matrix_mad(sg, act_matrix, weight_matrix0, result_matrix);
		joint_matrix_load(sg, act_matrix, a + 16 * 1 + 8 * l * WIDTH, WIDTH);
		result_matrix = joint_matrix_mad(sg, act_matrix, weight_matrix1, result_matrix);
		joint_matrix_load(sg, act_matrix, a + 16 * 2 + 8 * l * WIDTH, WIDTH);
		result_matrix = joint_matrix_mad(sg, act_matrix, weight_matrix2, result_matrix);
		joint_matrix_load(sg, act_matrix, a + 16 * 3 + 8 * l * WIDTH, WIDTH);
		result_matrix = joint_matrix_mad(sg, act_matrix, weight_matrix3, result_matrix);

		//matrix_activation<float, joint_matrix<sub_group, float, use::accumulator, TM, TN>>(sg, activation, result_matrix);

		joint_matrix_store(sg, result_matrix, o + 16 * sgId + 8 * l * WIDTH, WIDTH, layout::row_major);

	}
}


template <int WIDTH, typename T, Activation activation>
void mlp_swift_forward(Activation output_activation,
	const std::vector<bf16>& weights,
	const std::vector<bf16>& inputs,
	std::vector<T>& intermediate_output,
	std::vector<T>& output,
	const int output_stride,
	const int n_hidden_layers,
	const int batch_size,
	const int input_width,
	const int output_width)
{
	const int N_BLOCKS = WIDTH / TK;
	const int N_ITERS = WIDTH / TM;

	queue q = queue();

	std::vector<bf16> act(batch_size * WIDTH, bf16(0.0f));

	bf16* inputs_device = malloc_shared<bf16>(inputs.size(), q);
	bf16* weights_layer_device = malloc_shared<bf16>(weights.size(), q);
	T* output_device = malloc_shared<T>(output.size(), q);
	T* intermediate_output_device = malloc_shared<T>(intermediate_output.size(), q);


	int shmem_size = batch_size * WIDTH;

	bf16* act_shmem = malloc_shared<bf16>(shmem_size, q);

	q.memcpy(inputs_device, inputs.data(), inputs.size() * sizeof(bf16));
	q.memcpy(weights_layer_device, weights.data(), weights.size() * sizeof(bf16));
	q.memcpy(intermediate_output_device, intermediate_output.data(), intermediate_output.size() * sizeof(T));
	q.memcpy(act_shmem, act.data(), batch_size * WIDTH * sizeof(bf16));

	q.submit([&](handler& cgh)
		{
			cgh.parallel_for<>(
				nd_range<1>(batch_size * WG_SIZE / BATCH_CHUNK, WG_SIZE),
				[=](nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]]
				{
					kernel_swift_mlp<WIDTH, N_ITERS, T, Activation::None>(item,
						output_activation,
						inputs_device,
						weights_layer_device,
						intermediate_output_device,
						act_shmem,
						output_device,
						output_stride,
						input_width,
						output_width,
						n_hidden_layers,
						layout::col_major,
						layout::col_major);
				});
		}).wait();

		q.memcpy(output.data(), output_device, output.size() * sizeof(T));
		q.memcpy(intermediate_output.data(), intermediate_output_device, intermediate_output.size() * sizeof(T));



}





