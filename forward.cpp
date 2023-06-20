#include <iostream>
#include <vector>
#include <CL/sycl.hpp>
#include <ext/oneapi/experimental/bfloat16.hpp>
#include <ext/oneapi/matrix/matrix.hpp>
#include "activation.h"

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

#define SG_SIZE 32

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

template <int WIDTH, int N_ITERS>
workgroup_load_input_static(nd_item<1> item, __half* __restrict__ act_shmem, const __half* __restrict__ input_workgroup) {

	const int li = item.get_local_id(0);
	auto sg = item.get_sub_group();
	int sgId = sg.get_group_id();

	const uint32_t lane_offset = (8 * li) % WIDTH;
	const uint32_t row = (8 * li + sgId * 8 * 32) / WIDTH;

	for (int i = 0; i < N_ITERS; ++i) {
		*(int4*)&act_shmem[lane_offset + (row + 16 * i) * WIDTH ] = *(int4*)&input_workgroup[lane_offset + (row + 16 * i) * WIDTH];
	}
}


template <int WIDTH, int N_ITERS, typename T>
workgroup_input_layer_forward_dynamic<WIDTH, N_ITERS, matrix_layout::row_major>(Activation activation,
	__half* __restrict__ act_shmem,
	const __half* __restrict__ input,
	const __half* __restrict__ weights_layer,
	const int input_width,
	const int batch_size)
{
	const int N_BLOCKS = input_width / 16;

	joint_matrix<half, TM, TK, matrix_layout::row_major> act_matrix(sg);

	joint_matrix<half, TK, TN, weight_mat_layout> weight_matrix0(sg);
	joint_matrix<half, TK, TN, weight_mat_layout> weight_matrix1(sg);
	joint_matrix<half, TK, TN, weight_mat_layout> weight_matrix2(sg);
	joint_matrix<half, TK, TN, weight_mat_layout> weight_matrix3(sg);

	joint_matrix<float, TM, TN, matrix_layout::row_major> result_matrix(sg);


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
mlp_swift_forward(queue q, Activation output_activation,
	const std::vector<float>& weights,
	const std::vector<float>& input,
	std::vector<float>& intermediate_output,
	std::vector<float>* output,
	const int output_stride,
	const int n_hidden_layers,
	const int batch_size,
	const int input_width,
	const int output_width)
{
	size_t NDRangeM = batch_size / TM;
	size_t NDRangeN = input_width / TN;

	q.submit([&](handler& cgh)
		{
			auto accC = bufC.get_access<access::mode::read_write>(cgh);
			auto accA = bufA.get_access<access::mode::read_write>(cgh);
			auto accB = bufB.get_access<access::mode::read_write>(cgh);

			cgh.parallel_for<class imatrix>(
				nd_range<2>({ NDRangeM, NDRangeN * SG_SZ }, { 1, 1 * SG_SZ }),
				[=](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]]

				{
					kernel_swift_mlp(spmd_item,
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

