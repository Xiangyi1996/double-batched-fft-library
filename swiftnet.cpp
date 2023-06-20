#include <iostream>
#include <vector>
#include <CL/sycl.hpp>
#include <ext/oneapi/experimental/bfloat16.hpp>
#include <ext/oneapi/matrix/matrix.hpp>
#include "activation.h"

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
#define TM 8
#define TK 16
#define TN 16


#define SG_SIZE 32

template <int WIDTH, int N_ITERS, bool BACKWARD = false>
void work_group_layer(nd_item<1> item,Activation activation, half* act_mem, half* weights_layer, float* out,float* forward_act = nullptr) {
	auto sg = item.get_sub_group();
	int sgId = sg.get_group_id();
	int N_BLOCKS = WIDTH / 16;
	device_ptr<half> w(weights_layer);
	device_ptr<half> a(act_mem);
	device_ptr<float> o(out);
	using weight_mat_layout = std::conditional<BACKWARD, matrix_layout::row_major, matrix_layout::col_major>;
	
	joint_matrix<half, TM, TK, matrix_layout::row_major> act_matrix(sg);
	
	joint_matrix<half, TK, TN, weight_mat_layout> weight_matrix0(sg);
	joint_matrix<half, TK, TN, weight_mat_layout> weight_matrix1(sg);
	joint_matrix<half, TK, TN, weight_mat_layout> weight_matrix2(sg);
	joint_matrix<half, TK, TN, weight_mat_layout> weight_matrix3(sg);
	
	joint_matrix<float, TM, TN, matrix_layout::row_major> result_matrix(sg);

	if (BACKWARD) {
		joint_matrix_load(sg, weight_matrix0,w +  16 * 0 * WIDTH + 16 * sgId, WIDTH, matrix_layout::row_major);
		joint_matrix_load(sg, weight_matrix1,w +  16 * 1 * WIDTH + 16 * sgId, WIDTH, matrix_layout::row_major);
		joint_matrix_load(sg, weight_matrix2,w + 16 * 2 * WIDTH + 16 * sgId, WIDTH, matrix_layout::row_major);
		joint_matrix_load(sg, weight_matrix3,w +  16 * 3 * WIDTH + 16 * sgId, WIDTH, matrix_layout::row_major);
	}
	else {
		joint_matrix_load(sg, weight_matrix0,w + WIDTH * 16 * sgId + 16 * 0, WIDTH, matrix_layout::col_major);
		joint_matrix_load(sg, weight_matrix1,w + WIDTH * 16 * sgId + 16 * 1, WIDTH, matrix_layout::col_major);
		joint_matrix_load(sg, weight_matrix2,w + WIDTH * 16 * sgId + 16 * 2, WIDTH, matrix_layout::col_major);
		joint_matrix_load(sg, weight_matrix3,w + WIDTH * 16 * sgId + 16 * 3, WIDTH, matrix_layout::col_major);
	}
	
	for (int l = 0; l < N_ITERS; l++) {
		joint_matrix_fill(sg, result_matrix, 0.0f);

		
		joint_matrix_load(sg, act_matrix, a + 16 * 0 + 8 * l * WIDTH, WIDTH,matrix_layout::row_major);
		result_matrix = joint_matrix_mad(sg, act_matrix, weight_matrix0, result_matrix);
		joint_matrix_load(sg, act_matrix, a + 16 * 1 + 8 * l * WIDTH, WIDTH, matrix_layout::row_major);
		result_matrix = joint_matrix_mad(sg, act_matrix, weight_matrix1, result_matrix);
		joint_matrix_load(sg, act_matrix, a + 16 * 2 + 8 * l * WIDTH, WIDTH, matrix_layout::row_major);
		result_matrix = joint_matrix_mad(sg, act_matrix, weight_matrix2, result_matrix);
		joint_matrix_load(sg, act_matrix, a + 16 * 3 + 8 * l * WIDTH, WIDTH, matrix_layout::row_major);
		result_matrix = joint_matrix_mad(sg, act_matrix, weight_matrix3, result_matrix);
		if (BACKWARD) {
			device_ptr f(forward_act);
			joint_matrix_load(sg, act_matrix, f + 16 * sgId + l * 8 * WIDTH, WIDTH);
			matrix_activation<float, joint_matrix<float, TM, TN>,joint_matrix<half,TM,TK>>(sg, activation, result_matrix,act_matrix);
		}
		else {
			matrix_activation<float, joint_matrix<float, TM, TN>>(sg, activation, result_matrix);
		}
		joint_matrix_store(sg, result_matrix, o + 16 * sgId + 8 * l * WIDTH, WIDTH, matrix_layout::row_major);
	}

}


template <int WIDTH, int N_ITERS, Activation ACTIVATION, typename OUTPUT_LAYOUT>
void kernel_swiftnet_backward(
	nd_item<1> item ,
	half* loss_gradients,
	half* weights,
	float* back_act_gradient,
	half* forward,
	half* act_shmem,
	half* weights_first_layer,
	uint32_t out_stride,
	uint32_t batch_size,
	uint32_t out_width,
	uint32_t n_hidden_matmuls
) {
	auto sg = item.get_sub_group();
	
	int groupId = item.get_group(0);
	int sgId = sg.get_group_id();
	int idx = 16 * groupId * N_ITERS;
	int weights_stride = WIDTH * WIDTH;
	device_ptr loss(loss_gradients);
	device_ptr fwd(forward);
	device_ptr w(weights);
	device_ptr act(act_shmem);
	//Backprop last layer
	if (out_width <= 16) {
		joint_matrix<half, TM, TK, OUTPUT_LAYOUT> act_matrix;
		joint_matrix<half, TK, TN, matrix_layout::row_major> weights_matrix;
		joint_matrix<float, TM, TN> result_matrix;
		joint_matrix<half,TM,TK,matrix_layout::row_major> froward_matrix
		for (int l = 0; l < N_ITERS; l++) {
			joint_matrix_fill(sg, result_matrix, 0.0f);

			joint_matrix_load(sg, act_matrix, loss + (idx * 16 * l) + out_stride, out_stride);

			result_matrix = joint_matrix_mad(sg, act_matrix, weights_matrix, result_matrix);

			joint_matrix_load(forward_matrix, fwd + WIDTH * batch_size * n_hidden_matmuls + 16 * sgId + (idx + l * 16) * WIDTH, WIDTH);

			matrix_activation_bacward<half>(ACTIVATION, result_matrix, forward_matrix);

			joint_matrix_store(act + weights_col + (16 * l) * (WIDTH), result_matrix, WIDTH, matrix_layout::row_major);

		}

	}
	else {
		workgroup_load_input_static<WIDTH, N_ITERS>(act_shmem, out_intermediate + idx * WIDTH);
	}


	// Hidden Layers
	for (int k = 0; k < n_hidden_matmuls; k++) {
		work_group_layer<WIDTH, N_ITERS, true>(item, ACTIVATION, act_shmems, w + weights_stride * (n_hidden_matmuls - k - 1, back_act_gradient + layer_stride * (k + 1) + idx * WIDTH, forward + layer_stride * (n_hidden_matmuls - k - 1) + idx * WIDTH);

	}

	if (loss_gradients != nullptr) {
		work_group_layer<WIDTH, N_ITERS, true>(item, Activation::None, act_shmem, weights_first_layer, loss_gradients + idx * WIDTH);
	}

}


template<int WIDTH,Activation ACTIVATION>
mlp_swiftnet_backward(
	std::vector<half> weights_first_layer,
	std::vector<half> weights,
	std::vector<float> dl_output,
	std::vector<half> temporaries,
	std::vector<half> forward,
	const uint32_t n_hidden_matmuls,
	const uint32_t out_width,
	const uint32_t out_stride,
	const uint32_t batch_size
) {
	int N_BLOCKS = WIDTH / 16;
	const N_ITERS = 16;
	auto selector = gpu_selector();
	queue q = queue(selector);
	try {
		q.submit([&](handler& h) {
			//Transfer data to device memory
			half* weights_device = malloc_device<half>(weights.size(), q);
			half* temp_device = malloc_device<half>(temporaries.size(), q);
			half* weights_1_device = malloc_device<half>(weights_first_layer.size(), q);
			half* out_device = malloc_device<half>(dl_output.size(), q);
			half* fwd_device = malloc_device<half>(forward.size(), q);
			int shmem size = 16 * N_ITERS * WIDTH;
			half* act_shmem = malloc_device<half>(shmem_size, q);
			q.memcpy(weights_device,weights.data(),weights.size()*sizeof(half) );
			q.memcpy(temp_device, temporaries.data(), temporaries.size() * sizeof(half));
			q.memcpy(weights_1_device, weights_first_layer.data(), weights_first_layer.size() * sizeof(half));
			q.memcpy(out_device, dl_output.data(), dl_output.size() * sizeof(half));
			q.memcpy(fwd_device, forward.data(), forward.size() * sizeof(half));
			h.parallel_for(nd_range<1>(batch_size * 2, 128), [=](nd_item<1> item) [[intel::reqd_sub_group_size(32)]] {
				kernel_swiftnet_backward<WIDTH, N_ITERS, ACTIVATION, matrix_layout::row_major>(item,out_device,weights_device,temp_device,fwd_device,act_shmem,weights_1_device,out_stride,batch_size,out_width,n_hidden_matmuls);
				}
			});
	}
	catch (std::exception const& e)
	{
		std::cout << "An exception was caught when performing AMX/XMX matrix multiply.\n";
		std::terminate();
	}
}


template <int WIDTH, int N_ITERS, Activation activation>
kernel_swift_mlp(nd_item<1> item,
	const Activation output_activation,
	const __half* __restrict__ input,
	const __half* __restrict__ weights_layer,
	float* __restrict__ out_intermediate_layer,
	OUT_T* __restrict__ out,
	const uint32_t output_stride,
	const uint32_t batch_size,
	const uint32_t input_width,
	const uint32_t output_width,
	const uint32_t n_hidden_matmuls,
	const matrix_layout input_layout,
	const matrix_layout output_layout) {

	extern __shared__ __half shmem[];
	__half* act_shmem = shmem;
	device_ptr<half> w(weights_layer);

	// Handle first layer because it has different input

	const int elem_idx = 16 * item.get_local_id(0) * N_ITERS;

	if (input_width == WIDTH) {
		work_group_load_input_static<WIDTH, N_ITERS>(act_shmem, input + elem_idx * WIDTH);
		work_group_layer<WIDTH, N_ITERS, false>(item, activation, act_mem, w, out);
	}
	else {
		workgroup_input_layer_forward_dynamic<WIDTH, N_ITERS, matrix_layout::row_major>(activation,
			act_shmem,
			input + elem_idx * input_width,
			weights,
			input_width,
			batch_size);

	}

	// Handle hidden layers all together

	const int first_weight_length = input_width * WIDTH;
	const int hidden_weight_lenght = WIDTH * WIDTH;
	const int last_weight_lenght = WIDTH * ouput_width;

	for (int k = 0; k < m_hidden_matmuls; k++) {
		work_group_layer<WIDTH, N_ITERS, false>(item, activation, act_mem, w + first_weight_length + k * hidden_weight_lenght, out);
	}

	// Handle output layer
	if (output) {
		workgroup_last_layer_forward<WIDTH, N_ITERS, OUT_T>(output_activation,
			act_shmem,
			weights + first_weights_stride + weights_stride * n_hidden_matmuls,
			out + elem_idx * output_stride,
			output_stride,
			output_layout);

	}
}


template<WIDTH, int N_ITERS>
void workgroup_load_input_static(nd_item<1> item, half* act_shmem, half* input) {
	int localId = item.get_local_id();
	auto sg = item.get_sub_group();
	int sgId = sg.get_local_id()[0];

	int offset = (8 * localId) % WIDTH;
	itn row = (8 * localId + sgId * 8 * 32) / WIDTH;

	for (int i = 0; i < N_ITERS; i++) {
		act_shmem[offset + (row + 16 * i) * WIDTH] = input[offset + (row + 16 * i) * WIDTH];
	}
}


template <int WIDTH, int N_ITERS>
workgroup_load_input_static(nd_item<1> item, __half* __restrict__ act_shmem, const __half* __restrict__ input_workgroup) {

	const int li = item.get_local_id(0);
	auto sg = item.get_sub_group();
	int sgId = sg.get_group_id();

	const uint32_t lane_offset = (8 * li) % WIDTH;
	const uint32_t row = (8 * li + sgId * 8 * 32) / WIDTH;

	for (int i = 0; i < N_ITERS; ++i) {
		*(int4*)&act_shmem[lane_offset + (row + 16 * i) * WIDTH] = *(int4*)&input_workgroup[lane_offset + (row + 16 * i) * WIDTH];
	}
}


template <int WIDTH, int N_ITERS>
workgroup_input_layer_forward_dynamic<WIDTH, N_ITERS, matrix_layout::row_major>(nd_item<1> item,
	Activation activation,
	__half* __restrict__ act_shmem,
	const __half* __restrict__ input,
	const __half* __restrict__ weights_layer,
	float __restrict out_intermediate_layer,
	const int input_width,
	const int batch_size)
{
	auto sg = item.get_sub_group();
	int sgId = sg.get_group_id();
	const int li = item.get_local_id(0);
	const int N_BLOCKS = input_width / 16;
	device_ptr<half> w(weights_layer);
	device_ptr<half> a(act_shmem);
	device_ptr<float> o(out_intermediate_layer);

	__half* __restrict__ weights_shmem = act_shmem + 16 * input_width;

	const int n_element_load = N_BLOCKS * 32 * 8;
	const int wi_elem_idx = (li + sgId * 32) * 8;

	const int n_elements_weights = WIDTH * input_width;

	for (int i = wi_elem_idx; i < n_elements_weights; i += n_element_load) {
		*(int4*)&weights_shmem[i] = *(int4*)&weights_this_layer[i];
	}

	joint_matrix<half, TM, TK, matrix_layout::row_major> act_matrix(sg);

	joint_matrix<half, TK, TN, matrix_layout::col_major> weight_matrix(sg);

	joint_matrix<float, TM, TN, matrix_layout::row_major> result_matrix(sg);

	const int n_operations = input_width / 16;

	for (int l = 0; l < N_ITERS; l++) {
		const int n_elems_input = 16 * input_width;
		for (int i = wi_elem_i; i < n_elems_input; idx += n_element_load) {
			*(int4*)&act_shmem[i] = *(int4*)&input_threadblock[l * n_elems_a + i];
		}

		joint_matrix_fill(sg, result_matrix, 0.0f);
		for (int i = 0; i < n_operations; i++) {
			joint_matrix_load(sg, act_matrix, a + 16 * i, input_width, matrix_layout::row_major);
			joint_matrix_load(sg, weight_matrix, w + 16 * i + 16 * sgId * input_width, input_width, matrix_layout::col_major);
			result_matrix = joint_matrix_mad(sg, act_matrix, weight_matrix, result_matrix);
			matrix_activation<float, joint_matrix<float, TM, TN>>(sg, activation, result_matrix);
			joint_matrix_store(sg, result_matrix, o + 16 * sgId + 8 * l * WIDTH, WIDTH, matrix_layout::row_major)
		}
	}

}

template <int WIDTH, int N_ITERS>
workgroup_last_layer_forward(Activation activation,
	__half* __restrict__ act_mem,
	const __half* __restrict__ input,
	const __half* __restrict__ weights_layer,
	float __restrict out,
	const int output_stride) {

	auto sg = item.get_sub_group();
	int sgId = sg.get_group_id();
	const int li = item.get_local_id(0);
	int N_BLOCKS = WIDTH / 16;
	device_ptr<half> w(weights_layer);
	device_ptr<half> a(act_mem);
	device_ptr<float> o(out);

	joint_matrix<half, TM, TK, matrix_layout::row_major> act_matrix(sg);

	joint_matrix<half, TK, TN, weight_mat_layout> weight_matrix0(sg);
	joint_matrix<half, TK, TN, weight_mat_layout> weight_matrix1(sg);
	joint_matrix<half, TK, TN, weight_mat_layout> weight_matrix2(sg);
	joint_matrix<half, TK, TN, weight_mat_layout> weight_matrix3(sg);

	joint_matrix<float, TM, TN, matrix_layout::row_major> result_matrix(sg);


	__half* __restrict__ weights_shmem = act_shmem + N_ITERS * 16 * WIDTH;

	const int weights_row = (8 * li) % WIDTH;
	const int weights_col = (8 * li + 8 * 32 * sdUd) / WIDTH;

	*(int4*)&weights_shmem[weights_row + weights_col * WIDTH] = *(int4*)&weights_this_layer[weights_row + weights_col * WIDTH];


	joint_matrix_load(sg, weight_matrix0, w + WIDTH * 16 * sgId + 16 * 0, WIDTH, matrix_layout::col_major);
	joint_matrix_load(sg, weight_matrix1, w + WIDTH * 16 * sgId + 16 * 1, WIDTH, matrix_layout::col_major);
	joint_matrix_load(sg, weight_matrix2, w + WIDTH * 16 * sgId + 16 * 2, WIDTH, matrix_layout::col_major);
	joint_matrix_load(sg, weight_matrix3, w + WIDTH * 16 * sgId + 16 * 3, WIDTH, matrix_layout::col_major);


	for (int i = sgId; i < N_ITERS; i += N_BLOCKS) {
		joint_matrix_fill(sg, result_matrix, 0.0f);

		joint_matrix_load(sg, act_matrix, a + 16 * 0 + 8 * i * WIDTH, WIDTH, matrix_layout::row_major);
		result_matrix = joint_matrix_mad(sg, act_matrix, weight_matrix0, result_matrix);
		joint_matrix_load(sg, act_matrix, a + 16 * 1 + 8 * i * WIDTH, WIDTH, matrix_layout::row_major);
		result_matrix = joint_matrix_mad(sg, act_matrix, weight_matrix1, result_matrix);
		joint_matrix_load(sg, act_matrix, a + 16 * 2 + 8 * i * WIDTH, WIDTH, matrix_layout::row_major);
		result_matrix = joint_matrix_mad(sg, act_matrix, weight_matrix2, result_matrix);
		joint_matrix_load(sg, act_matrix, a + 16 * 3 + 8 * i * WIDTH, WIDTH, matrix_layout::row_major);
		result_matrix = joint_matrix_mad(sg, act_matrix, weight_matrix3, result_matrix);

		matrix_activation<float, joint_matrix<float, TM, TN>>(sg, activation, result_matrix);

		joint_matrix_store(sg, result_matrix, o + 16 * i * output_stride, output_stride, matrix_layout::row_major);
	}
}

template <int WIDTH, int N_ITERS>
__device__ void workgroup_write_output_static(const __half* __restrict__ act_shmem, __half* __restrict__ output_workgroup) {

	const int li = item.get_local_id(0);
	auto sg = item.get_sub_group();
	int sgId = sg.get_group_id();

	const uint32_t lane_offset = (8 * li) % WIDTH;
	const uint32_t row = (8 * li + sgId * 8 * 32) / WIDTH;

	for (int i = 0; i < N_ITERS; ++i) {
		*(int4*)&output_workgroup[lane_offset + (row + 16 * i) * WIDTH] = *(int4*)&act_shmem[lane_offset + (row + 16 * i) * (WIDTH + SKEW)];
	}
}


template <int WIDTH, typename T, Activation ACTIVATION, bool INFERENCE>
mlp_swift_forward(Activation output_activation,
	const std::vector<half>& weights,
	const std::vector<half>& input,
	std::vector<half>& intermediate_output,
	std::vector<half>* output,
	const int output_stride,
	const int n_hidden_layers,
	const int batch_size,
	const int input_width,
	const int output_width)
{
	size_t NDRangeM = batch_size / TM;
	size_t NDRangeN = input_width / TN;

	int N_BLOCKS = WIDTH / 16;
	const N_ITERS = 16;

	auto selector = gpu_selector();
	queue q = queue(selector)

		q.submit([&](handler& cgh)
			{
				half* weights_device = malloc_device<half>(weights.size(), q);;
				half* inputs_device = malloc_device<half>(inputs.size(), q);
				half* output_device = malloc_device<half>(output.size(), q);
				half* intermediate_output_device = malloc_device<half>(intermediate_output.size(), q);

				int shmem size = 16 * N_ITERS * WIDTH;

				half* act_shmem = malloc_device<half>(shmem_size, q);

				q.memcpy(weights_device, weights.data(), weights.size() * sizeof(half));
				q.memcpy(inputs_device, inputs.data(), inputs.size() * sizeof(half));
				q.memcpy(output_device, output.data(), output.size() * sizeof(half));
				q.memcpy(intermediate_output_device, intermediate_output.data(), intermediate_output.size() * sizeof(half));

				cgh.parallel_for<class imatrix>(
					nd_range<2>(nd_range<1>(batch_size * 2, 128),
						[=](nd_item<1> item) [[intel::reqd_sub_group_size(32)]]
						{
							kernel_swift_mlp<N_BLOCKS, N_ITERS>(spmd_item,
								output_activation,
								input,
								weights,
								intermediate_output,
								output,
								output_stride,
								batch_size,
								input_width,
								output_width,
								n_hidden_layers,
								matrix_layout::col_major,
								matrix_layout::col_majors)
						};

						};



