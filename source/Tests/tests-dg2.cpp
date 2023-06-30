#include <iostream>
#include <vector>
#include <CL/sycl.hpp>
#include "activation.h"




using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
using bf16 = sycl::ext::oneapi::bfloat16;
#define TM 8
#define TK 16
#define TN 8
#define SYCL_EXT_ONEAPI_MATRIX_V 4

#define SG_SIZE 8
#define WG_SIZE 8*SG_SIZE
#define BATCH_CHUNK 64

template <int WIDTH, int N_ITERS, typename T, bool BACKWARD = false>
void work_group_layer(nd_item<1> it, Activation activation, bf16* act_mem, bf16* weights_layer, T* out) {

	auto sg = it.get_sub_group();
	int sgId = sg.get_group_id();
	const int N_BLOCKS = WIDTH / TK;
	device_ptr<bf16> w(weights_layer);
	device_ptr<bf16> a(act_mem);
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

		if (BACKWARD) {
			//matrix_activation_backward<float, bf16>(it, activation, o + 16 * sgId + 8 * l * WIDTH, f + 16 * sgId + l * 8 * WIDTH, WIDTH, outs);

		}
		else {
			//matrix_activation<float>(it, activation, o + 16 * sgId + 8 * l * WIDTH, WIDTH, outs);
		}
	}
	for (int i = 0; i < N_ITERS; i++) {
		for (int j = 0; j < TN; j++) {
			for (int k = 0; k < TM; k++) {
				act_mem[TN * sgId + TM * i * WIDTH + j + k * WIDTH] = out[TN * sgId + TM * i * WIDTH + j + k * WIDTH];
			}
		}
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
		work_group_layer<WIDTH, N_ITERS, T, false>(item, activation, act_shmem + elem_idx * WIDTH, weights_layer, out + elem_idx * WIDTH);
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
		work_group_layer<WIDTH, N_ITERS, T, false>(item, activation, act_shmem + elem_idx * WIDTH, weights_layer + first_weight_length + k * hidden_weight_lenght, out + elem_idx * WIDTH);
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


template <int WIDTH, typename T>
void mlp_swift_forward(Activation output_activation,
	const std::vector<bf16>& weights,
	const std::vector<bf16>& inputs,
	std::vector<T>& intermediate_output,
	const int output_stride,
	const int n_hidden_layers,
	const int batch_size,
	const int input_width,
	const int output_width)
{


	const int N_BLOCKS = WIDTH / TK;
	const int N_ITERS = WIDTH / TM;

	queue q = queue();


	bf16* inputs_device = malloc_shared<bf16>(inputs.size(), q);
	bf16* weights_layer_device = malloc_shared<bf16>(weights.size(), q);
	T* output_device = malloc_shared<T>(inputs.size(), q);
	T* intermediate_output_device = malloc_shared<T>(intermediate_output.size(), q);


	int shmem_size = WIDTH * WIDTH;

	bf16* act_shmem = malloc_device<bf16>(shmem_size, q);

	q.memcpy(weights_layer_device, weights.data(), weights.size() * sizeof(bf16));
	q.memcpy(inputs_device, inputs.data(), inputs.size() * sizeof(bf16));
	//q.memcpy(output_device, output.data(), output.size() * sizeof(T));
	q.memcpy(intermediate_output_device, intermediate_output.data(), intermediate_output.size() * sizeof(T));

	q.submit([&](handler& cgh)
		{
			cgh.parallel_for<class imatrix>(
				nd_range<1>(batch_size * WG_SIZE / BATCH_CHUNK, WG_SIZE),
				[=](nd_item<1> item) [[intel::reqd_sub_group_size(8)]]
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
		});
	//q.memcpy(output, output_device, output.size() * sizeof(float));
}

void test0() {

	const int batch_size = 128;
	const int WIDTH = 64;
	const int N_ITERS = 8;

	queue q = queue();
	std::vector<bf16> act(batch_size * WIDTH, 0);
	std::vector<bf16> input(batch_size * WIDTH, bf16(0.0f));
	std::vector<bf16> weight(WIDTH * WIDTH, 0);
	std::vector<bf16> packed_weight(WIDTH * WIDTH, 0);
	std::vector<float> res(batch_size * WIDTH, 0);
	std::vector<float> out_res(batch_size * WIDTH, 0);

	for (int i = 0; i < batch_size; i++) {
		for (int j = 0; j < WIDTH; j++) {
			input[i * WIDTH + j] = bf16(1.0f * (WIDTH * i + j));
		}

	}
	for (int i = 0; i < WIDTH; i++) {
		for (int j = 0; j < WIDTH; j++) {
			weight[WIDTH * i + j] = bf16(1.0f * (WIDTH * i + j));
		}
	}
	for (int i = 0; i < WIDTH / 2; i++) {
		for (int j = 0; j < WIDTH; j++) {
			packed_weight[i * WIDTH * 2 + 2 * j] = weight[2 * i * WIDTH + j];
			packed_weight[i * WIDTH * 2 + 2 * j + 1] = weight[(2 * i + 1) * WIDTH + j];
		}
	}
	bf16* input_device = malloc_shared<bf16>(batch_size * WIDTH, q);
	bf16* weights_layer = malloc_shared<bf16>(WIDTH * WIDTH, q);
	float* out = malloc_shared<float>(batch_size * WIDTH, q);
	q.memcpy(input_device, input.data(), batch_size * WIDTH * sizeof(bf16));
	q.memcpy(weights_layer, packed_weight.data(), WIDTH * WIDTH * sizeof(bf16));



	q.submit([&](handler& h) {

		stream outs(1024, 256, h);
		h.parallel_for(nd_range<1>(batch_size * WG_SIZE / BATCH_CHUNK, WG_SIZE), [=](nd_item<1> it) [[intel::reqd_sub_group_size(SG_SIZE)]] {

			auto wg = it.get_group();
			const int wg_idx = wg.get_group_id();
			const int elem_idx = WIDTH * wg_idx;

			bf16 act_mem[WIDTH * WIDTH];

			workgroup_load_input_static<WIDTH, N_ITERS>(it, act_mem + elem_idx * WIDTH, input_device + elem_idx * WIDTH);

			work_group_layer<WIDTH, N_ITERS, float>(it, Activation::None, act_mem + elem_idx * WIDTH, weights_layer, out + elem_idx * WIDTH);

			workgroup_write_output_static<WIDTH, N_ITERS>(it, act_mem + elem_idx * WIDTH, out + elem_idx * WIDTH);

			});


		}).wait();

		for (int i = 0; i < batch_size; i++) {
			for (int j = 0; j < WIDTH; j++) {
				for (int k = 0; k < WIDTH; k++) {
					res[i * WIDTH + j] += (float)input[i * WIDTH + k] * (float)weight[k * WIDTH + j];
				}
			}
		}
		for (int i = WIDTH * WIDTH; i < WIDTH * WIDTH + 10; i++) {
			std::cout << out[i] << " " << res[i] << std::endl;
		}


}

void test1() {

	const int batch_size = 128;
	const int WIDTH = 64;
	const int N_ITERS = 8;

	queue q = queue();
	std::vector<bf16> act(batch_size * WIDTH, 0);
	std::vector<bf16> input(batch_size * WIDTH, bf16(0.0f));
	std::vector<bf16> weight(WIDTH * WIDTH, 0);
	std::vector<bf16> packed_weight(WIDTH * WIDTH, 0);
	std::vector<float> res(batch_size * WIDTH, 0);
	std::vector<float> out_res(batch_size * WIDTH, 0);

	for (int i = 0; i < batch_size; i++) {
		for (int j = 0; j < WIDTH; j++) {
			input[i * WIDTH + j] = bf16(1.0f * (WIDTH * i + j));
		}

	}
	for (int i = 0; i < WIDTH; i++) {
		for (int j = 0; j < WIDTH; j++) {
			weight[WIDTH * i + j] = bf16(1.0f * (WIDTH * i + j));
		}
	}
	for (int i = 0; i < WIDTH / 2; i++) {
		for (int j = 0; j < WIDTH; j++) {
			packed_weight[i * WIDTH * 2 + 2 * j] = weight[2 * i * WIDTH + j];
			packed_weight[i * WIDTH * 2 + 2 * j + 1] = weight[(2 * i + 1) * WIDTH + j];
		}
	}
	bf16* act_mem = malloc_shared<bf16>(batch_size * WIDTH, q);
	bf16* input_device = malloc_shared<bf16>(batch_size * WIDTH, q);
	bf16* weights_layer = malloc_shared<bf16>(WIDTH * WIDTH, q);
	float* out = malloc_shared<float>(batch_size * WIDTH, q);
	q.memcpy(act_mem, act.data(), batch_size * WIDTH * sizeof(bf16));
	q.memcpy(input_device, input.data(), batch_size * WIDTH * sizeof(bf16));
	q.memcpy(weights_layer, packed_weight.data(), WIDTH * WIDTH * sizeof(bf16));



	q.submit([&](handler& h) {

		stream outs(1024, 256, h);
		h.parallel_for(nd_range<1>(batch_size * WG_SIZE / BATCH_CHUNK, WG_SIZE), [=](nd_item<1> it) [[intel::reqd_sub_group_size(SG_SIZE)]] {

			auto wg = it.get_group();
			const int wg_idx = wg.get_group_id();
			const int elem_idx = WIDTH * wg_idx;

			//bf16 act_mem[WIDTH * WIDTH];

			workgroup_load_input_static<WIDTH, N_ITERS>(it, act_mem + elem_idx * WIDTH, input_device + elem_idx * WIDTH);


			work_group_layer<WIDTH, N_ITERS, float>(it, Activation::None, act_mem + elem_idx * WIDTH, weights_layer, out + elem_idx * WIDTH);

			//workgroup_write_output_static<WIDTH,N_ITERS>(it, act_mem + elem_idx * WIDTH, out + elem_idx * WIDTH);

			});


		}).wait();

		for (int i = 0; i < batch_size; i++) {
			for (int j = 0; j < WIDTH; j++) {
				for (int k = 0; k < WIDTH; k++) {
					res[i * WIDTH + j] += (float)input[i * WIDTH + k] * (float)weight[k * WIDTH + j];
				}
			}
		}
		for (int i = WIDTH * WIDTH; i < WIDTH * WIDTH + 10; i++) {
			std::cout << out[i] << " " << res[i] << std::endl;
		}


}




void test2() {

	const int batch_size = 128;
	const int WIDTH = 64;
	const int N_ITERS = 8;
	const int n_hidden_layers = 1;
	const int output_stride = 64;
	const int input_width = 64;
	const int output_width = 64;

	queue q = queue();
	std::vector<bf16> act(batch_size * WIDTH, 0);
	std::vector<bf16> input(batch_size * WIDTH, bf16(0.0f));
	std::vector<bf16> weight((n_hidden_layers+1) * WIDTH * WIDTH, 0);
	std::vector<bf16> packed_weight((n_hidden_layers +1)*WIDTH * WIDTH, 0);
	std::vector<float> res1(batch_size * WIDTH, 0);
	std::vector<float> res2(batch_size * WIDTH, 0);
	std::vector<float> intermediate_output((n_hidden_layers+1)* batch_size * WIDTH, 0);
	std::vector<float> out_res(batch_size * WIDTH, 0);

	for (int i = 0; i < batch_size; i++) {
		for (int j = 0; j < WIDTH; j++) {
			input[i * WIDTH + j] = bf16(1.0f );
		}

	}
	for (int i = 0; i < WIDTH; i++) {
		for (int j = 0; j < WIDTH; j++) {
			weight[WIDTH * i + j] = bf16(1.0f * (WIDTH * i + j));
			weight[WIDTH *WIDTH + WIDTH * i + j] = bf16(1.0f);
		}
	}
	for (int i = 0; i < WIDTH / 2; i++) {
		for (int j = 0; j < WIDTH; j++) {
			packed_weight[ i * WIDTH * 2 + 2 * j] = weight[ 2 * i * WIDTH + j];
			packed_weight[ i * WIDTH * 2 + 2 * j + 1] = weight[(2 * i + 1) * WIDTH + j];
			packed_weight[WIDTH * WIDTH + i * WIDTH * 2 + 2 * j] = weight[WIDTH * WIDTH + 2 * i * WIDTH + j];
			packed_weight[WIDTH * WIDTH + i * WIDTH * 2 + 2 * j + 1] = weight[WIDTH * WIDTH + (2 * i + 1) * WIDTH + j];

		}
	}
	bf16* act_mem = malloc_shared<bf16>(batch_size * WIDTH, q);
	bf16* input_device = malloc_shared<bf16>(batch_size * WIDTH, q);
	bf16* weights_layer = malloc_shared<bf16>((n_hidden_layers + 1) * WIDTH * WIDTH, q);
	float* out = malloc_shared<float>(batch_size * WIDTH, q);
	float* intermediate_output_device = malloc_shared<float>((n_hidden_layers + 1) * batch_size * WIDTH, q);
	q.memcpy(act_mem, act.data(), batch_size * WIDTH * sizeof(bf16));
	q.memcpy(input_device, input.data(), batch_size * WIDTH * sizeof(bf16));
	q.memcpy(weights_layer, packed_weight.data(), (n_hidden_layers + 1) * WIDTH * WIDTH * sizeof(bf16));
	q.memcpy(intermediate_output_device, intermediate_output.data(), (n_hidden_layers + 1) * batch_size * WIDTH * sizeof(float));


	q.submit([&](handler& h) {

		stream outs(1024, 256, h);
		h.parallel_for(nd_range<1>(batch_size * WG_SIZE / BATCH_CHUNK, WG_SIZE), [=](nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {

			kernel_swift_mlp<WIDTH, N_ITERS, float, Activation::None>(item,
				Activation::None,
				input_device,
				weights_layer,
				intermediate_output_device,
				act_mem,
				out,
				output_stride,
				input_width,
				output_width,
				n_hidden_layers,
				layout::col_major,
				layout::col_major);
			});
		//q.memcpy(out_res, out, 64 * 64 * sizeof(half));
		}).wait();

		for (int i = 0; i < batch_size; i++) {
			for (int j = 0; j < WIDTH; j++) {
				for (int k = 0; k < WIDTH; k++) {
					res1[i * WIDTH + j] += (float)input[i * WIDTH + k] * (float)weight[k * WIDTH + j];
				}
			}
		}
		for (int i = 0; i < batch_size; i++) {
			for (int j = 0; j < WIDTH; j++) {
				for (int k = 0; k < WIDTH; k++) {
					res2[i * WIDTH + j] += (float)res1[i * 64 + k] * (float)weight[WIDTH * WIDTH + k * WIDTH + j];
				}
			}
		}
		for (int i = WIDTH * WIDTH; i < WIDTH * WIDTH + 10; i++) {
			std::cout << out[i] << " " << res2[i] << std::endl;
		}



}



void test3() {

	const int batch_size = 128;
	const int WIDTH = 64;
	const int N_ITERS = 8;
	const int n_hidden_layers = 1;
	const int output_stride = 64;
	const int input_width = 64;
	const int output_width = 64;

	std::vector<bf16> input(batch_size * WIDTH, 1);
	std::vector<bf16> weight(WIDTH * WIDTH * (n_hidden_layers + 1), 0);
	std::vector<bf16> packed_weight((n_hidden_layers + 1) * WIDTH * WIDTH, 0);
	std::vector<float> intermediate_output(batch_size * WIDTH, 0);
	std::vector<float> out(batch_size * WIDTH, 0);


	float res1[batch_size * WIDTH];
	float res2[batch_size * WIDTH];

	for (int i = 0; i < batch_size; i++) {
		for (int j = 0; j < WIDTH; j++) {
			input[i * WIDTH + j] = bf16(1.0f);
		}

	}
	for (int i = 0; i < WIDTH; i++) {
		for (int j = 0; j < WIDTH; j++) {
			weight[WIDTH * i + j] = bf16(1.0f);
			weight[WIDTH * WIDTH + WIDTH * i + j] = bf16(1.0f);
		}
	}
	for (int i = 0; i < WIDTH / 2; i++) {
		for (int j = 0; j < WIDTH; j++) {
			packed_weight[i * WIDTH * 2 + 2 * j] = weight[2 * i * WIDTH + j];
			packed_weight[i * WIDTH * 2 + 2 * j + 1] = weight[(2 * i + 1) * WIDTH + j];
			packed_weight[WIDTH * WIDTH + i * WIDTH * 2 + 2 * j] = weight[WIDTH * WIDTH + 2 * i * WIDTH + j];
			packed_weight[WIDTH * WIDTH + i * WIDTH * 2 + 2 * j + 1] = weight[WIDTH * WIDTH + (2 * i + 1) * WIDTH + j];

		}
	}

	mlp_swift_forward<WIDTH, float>(Activation::None,
		packed_weight,
		input,
		intermediate_output,
		output_stride,
		n_hidden_layers,
		batch_size,
		input_width,
		output_width);


	for (int i = 0; i < batch_size; i++) {
		for (int j = 0; j < WIDTH; j++) {
			for (int k = 0; k < WIDTH; k++) {
				res1[i * WIDTH + j] += (float)input[i * WIDTH + k] * (float)weight[k * WIDTH + j];
			}
		}
	}
	for (int i = 0; i < batch_size; i++) {
		for (int j = 0; j < WIDTH; j++) {
			for (int k = 0; k < WIDTH; k++) {
				res2[i * WIDTH + j] += (float)res1[i * WIDTH + k] * (float)weight[WIDTH * WIDTH + k * WIDTH + j];
			}
		}
	}
	for (int i = 0; i < 10; i++) {
		std::cout << out[i] << " " << res2[i] << std::endl;
	}

}

int main() {
	std::cout << "Test 0" << std::endl;
	test0();
	std::cout << "Test 1" << std::endl;
	test1();
	std::cout << "Test 2" << std::endl;
	test2();
	std::cout << "Test 3" << std::endl;
	test3();

	return 0;
}