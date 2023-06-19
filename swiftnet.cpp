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

template <int WIDTH, int N_ITERS, bool BACKWARD = false>
void work_group_layer(nd_item<1> item,Activation activation, half* act_mem, half* weights_layer, float* out,float* forward_act = nullptr) {
	auto sg = it.get_sub_group();
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