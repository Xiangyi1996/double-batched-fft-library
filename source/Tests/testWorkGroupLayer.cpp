#include <iostream>
#include <vector>
#include <CL/sycl.hpp>
#include "activation.h"
#include <mathimf.h>




using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
using bf16 = sycl::ext::oneapi::bfloat16;
#define TM 8
#define TK 16
#define TN 16
#define SYCL_EXT_ONEAPI_MATRIX_V 4
template<typename T>
void toPackedLayout(std::vector<T>& src, std::vector<T>& dest, int stride, int size) {
	int rows = size / stride;
	for (int i = 0; i < rows / 2; i++) {
		for (int j = 0; j < stride; j++) {
			dest[i * stride * 2 + 2 * j] = src[2 * i * stride + j];
			dest[i * stride * 2 + 2 * j + 1] = src[(2 * i + 1) * stride + j];
		}
	}
	return;
}



template <int WIDTH, int N_ITERS, bool BACKWARD = false>
void work_group_layer(nd_item<1> it, Activation activation, bf16* act_mem, bf16* weights_layer, float* out, stream outs, half* forward_act = nullptr) {
	
	auto sg = it.get_sub_group();
	int sgId = sg.get_group_id();
	const int N_BLOCKS = WIDTH / 16;
	device_ptr<bf16> w(weights_layer);
	device_ptr<bf16> a(act_mem);
	device_ptr<float> o(out);
	
	it.barrier();

	joint_matrix<sub_group, bf16, use::a, TM, TK, layout::row_major> act_matrix;
	//joint_matrix<sub_group, half, use::b, TK, TN, sycl::ext::intel::experimental::matrix::layout::packed> weight_matrices[N_BLOCKS];
	joint_matrix<sub_group, bf16, use::b, TK, TN, sycl::ext::intel::experimental::matrix::layout::packed> weight_matrix0;
	joint_matrix<sub_group, bf16, use::b, TK, TN, sycl::ext::intel::experimental::matrix::layout::packed> weight_matrix1;
	joint_matrix<sub_group, bf16, use::b, TK, TN, sycl::ext::intel::experimental::matrix::layout::packed> weight_matrix2;
	joint_matrix<sub_group, bf16, use::b, TK, TN, sycl::ext::intel::experimental::matrix::layout::packed> weight_matrix3;
	joint_matrix<sub_group, float, use::accumulator, TM, TN> result_matrix;
	
	/*joint_matrix<sub_group, bf16, use::b, TK, TN, layout::row_major> weight_test;
	joint_matrix_fill(sg, weight_test, 1.0f);
	joint_matrix_fill(sg, act_matrix, 1.0f);
	joint_matrix_fill(sg, result_matrix, 0.0f);
	result_matrix = joint_matrix_mad(sg, act_matrix, weight_test, result_matrix);
	auto data = get_wi_data(sg, result_matrix);
	outs << data[0] << endl;*/
	


	//joint_matrix_fill(sg, weight_matrix0, bf16(1.0f));
	joint_matrix_load(sg, weight_matrix0, w + 16 * 2 * sgId + 8 * 0 * WIDTH * 2, WIDTH*2);
	joint_matrix_load(sg, weight_matrix1, w + 16 * 2 * sgId + 8 * 1 * WIDTH * 2, WIDTH*2);
	joint_matrix_load(sg, weight_matrix2, w + 16 * 2 * sgId + 8 * 2 * WIDTH * 2, WIDTH*2);
	joint_matrix_load(sg, weight_matrix3, w + 16 * 2 * sgId + 8 * 3 * WIDTH * 2, WIDTH*2);


	/*for (int i = 0; i < N_BLOCKS; i++) {
		joint_matrix_load(sg, weight_matrices[i], w + WIDTH * 16 * sgId + 16 * i, WIDTH);
	}*/


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
		
		
		auto data = get_wi_data(sg, result_matrix);
		
		for (int i = 0; i < data.length(); i++) {
			float x = (float)data[i];
			outs << x;
			elt_activation<float>(Activation::Sine, x);
			outs << " " << x << endl;
		}
		//matrix_activation<float, joint_matrix<sub_group, float, use::accumulator, TM, TN>>(sg, activation, result_matrix);
		joint_matrix_store(sg, result_matrix, o + 16 * sgId + 8 * l * WIDTH, WIDTH, layout::row_major);
		matrix_activation<float>(it, activation, o + 16 * sgId + 8 * l * WIDTH, WIDTH, outs);
		
		
		
	}
	
	
	
	
	

}
//template <int WIDTH, int N_ITERS, bool BACKWARD = false>
//void work_group_layer_substitute(nd_item<1> it, Activation activation, bf16* act_mem, bf16* weights_layer, float* out, stream outs, half* forward_act = nullptr) {
//	if (BACKWARD) {
//		for (int i = 0; i < 64; i++) {
//			for (int j = 0; j < 64; j++) {
//				out[i * 64 + j] += (float)act_mem[i * 64 + k] * (float)weights_layer[k * 64 + j];
//			}
//		}
//		matrix_activation(activation, out);
//	}
//	else {
//		for (int i = 0; i < 64; i++) {
//			for (int j = 0; j < 64; j++) {
//				out[i * 64 + j] += (float)act_mem[i * 64 + k] * (float)weights_layer[j * 64 + k];
//			}
//		}
//		matrix_activation(activation, out, (float *)forward_act);
//	}
//	return;
//}


//void test1() {
//	
//	queue q = queue();
//	std::vector<bf16> act(64 * 64);
//	std::vector<bf16> weight(64 * 64);
//	
//	for (int i = 0; i < 64*64 ; i++) {
//		if (i < 64 * 64) {
//			act[i] = bf16(1.0f);
//		}
//		weight[i] = bf16(1.0f * i);
//		
//	}
//	
//	
//	std::vector<bf16> weight_packed(64 * 64);
//	toPackedLayout<bf16>(weight, weight_packed, 64, 64 * 64);
//	std::vector<float> res(64 * 64,0);
//	
//	bf16* act_mem = malloc_shared<bf16>(64 * 64, q);
//	bf16* weights_layer = malloc_shared<bf16>(64 * 64, q);
//	float* out = malloc_shared<float>(64 * 64, q);
//	q.memcpy(act_mem, act.data(), 64 * 64 * sizeof(bf16));
//	q.memcpy(weights_layer, weight_packed.data(), 64 * 64 * sizeof(bf16));
//	
//	
//	
//	q.submit([&](handler& h) {
//		
//		stream outs(1024, 256, h);
//		h.parallel_for<class dmatrix>(nd_range<1>(64, 64), [=](nd_item<1> it) [[intel::reqd_sub_group_size(16)]]  {
//			auto sg = it.get_sub_group();
//			int sgId = sg.get_group_id();
//			work_group_layer<64, 8>(it, Activation::None, act_mem, weights_layer, out,outs);
//			
//
//			});
//		//q.memcpy(out_res, out, 64 * 64 * sizeof(half));
//		}).wait();
//	
//	for (int i = 0; i < 64; i++) {
//		for (int j = 0; j < 64; j++) {
//			for (int k = 0; k < 64; k++) {
//				res[i * 64 + j] += act[i * 64 + k] * weight[k * 64 + j];
//			}
//		}
//	}
//	for (int i = 0; i < 10; i++) {
//		std::cout << out[i] << " " << res[i] << std::endl;
//	}
//	
//}
void test2() {

	queue q = queue();
	std::vector<bf16> act(64 * 64);
	std::vector<bf16> weight(64 * 64);
	std::vector<float> u(64 * 64, 0.0f);
	for (int i = 0; i < 64 * 64; i++) {
		if (i < 64 * 64) {
			act[i] = bf16(1.0f);
		}
		if (i % 2 != 0) {
			weight[i] = bf16(-1.0f/1000 * i);
		}
		else {
			weight[i] = bf16(1.0f/1000 * i);
		}
		

	}


	std::vector<bf16> weight_packed(64 * 64);
	toPackedLayout<bf16>(weight, weight_packed, 64, 64 * 64);
	std::vector<float> res(64 * 64, 0);
	
	bf16* act_mem = malloc_shared<bf16>(64 * 64, q);
	bf16* weights_layer = malloc_shared<bf16>(64 * 64, q);
	float* out = malloc_shared<float>(64 * 64, q);
	q.memcpy(act_mem, act.data(), 64 * 64 * sizeof(bf16));
	q.memcpy(weights_layer, weight_packed.data(), 64 * 64 * sizeof(bf16));
	q.memcpy(out, u.data(), 64 * 64 * sizeof(float));

	q.submit([&](handler& h) {

		stream outs(1024, 256, h);
		h.parallel_for(nd_range<1>(64, 64), [=](nd_item<1> it) [[intel::reqd_sub_group_size(16)]] {
			
			
			work_group_layer<64, 8>(it, Activation::Sine, act_mem, weights_layer, out, outs);
			
			
			});
		//q.memcpy(out_res, out, 64 * 64 * sizeof(half));
		}).wait();
		//Activation 
		/*for (int i = 0; i < 64 * 64; i++) {
			elt_activation<float>(Activation::Sine, out[i]);
		}*/

		for (int i = 0; i < 64; i++) {
			for (int j = 0; j < 64; j++) {
				for (int k = 0; k < 64; k++) {
					res[i * 64 + j] += act[i * 64 + k] * weight[k * 64 + j];
				}
			}
		}
		std::cout << "Outputs" << std::endl;
		for (int i = 0; i < 10; i++) {
			std::cout << out[i] << " " << sinf(res[i]) <<  std::endl;
		}

}







int main() {
	//std::cout << "Test 1" << std::endl;
	//test1();
	std::cout << "Test 2" << std::endl;
	test2();

	return 0;
}


