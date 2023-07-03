#include <iostream>
#include <vector>
#include <CL/sycl.hpp>
#include "activation.h"
#include "SwiftNetMLP.h"




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
void work_group_layer(nd_item<1> item, Activation activation, bf16* act_mem, bf16* weights_layer, T* out, bf16* forward_act = nullptr) {

	auto sg = item.get_sub_group();
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
			//matrix_activation<float>(it, activation, o + TK * sgId + TM * l * WIDTH, WIDTH, outs);
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

template <typename T, int WIDTH>
SwiftNetMLP<T, WIDTH>::SwiftNetMLP(
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
	m_n_hidden_matrices = m_n_hidden_layers ;

	m_weights_matrices.resize(m_net_width * m_inputs_width + (m_net_width * m_net_width) * m_n_hidden_matrices);
	m_weights_matrices_inferences.resize(m_net_width * m_inputs_width + (m_net_width * m_net_width) * m_n_hidden_matrices);
	m_grads_matrices.resize(m_net_width * m_inputs_width + (m_net_width * m_net_width) * m_n_hidden_matrices);


}

template <typename T, int WIDTH>
void SwiftNetMLP<T, WIDTH>::initialize_params() {
	for (int i = 0; i < m_net_width * m_inputs_width; i++) {
		m_weights_matrices[i] = bf16(1.0f);
		m_weights_matrices_inferences[i] = bf16(1.0f);
		m_grads_matrices[i] = bf16(1.0f);
	}
	for (int i = 0; i < m_n_hidden_matrices ; i++) {
		for (int j = 0; j < m_net_width * m_net_width; j++) {

			m_weights_matrices[i * m_net_width * m_net_width + m_net_width * m_inputs_width + j] = bf16(1.0f);
			m_weights_matrices_inferences[i * m_net_width * m_net_width + m_net_width * m_inputs_width + j] = bf16(1.0f);
			m_grads_matrices[i * m_net_width * m_net_width + m_net_width * m_inputs_width + j] = bf16(1.0f);
		}

	}
}

template <typename T, int WIDTH>
std::vector<float> SwiftNetMLP<T, WIDTH>::forward_pass(const std::vector<bf16>& input, std::vector<T>& output) {

	int output_stride = WIDTH;
	int batch_size = input.size()/WIDTH;
	std::vector<float> forward(128 * WIDTH * (m_n_hidden_matrices + 1 ), 0.0f);
	std::cout << m_weights_matrices[0] << std::endl;

	switch (m_activation) {
	case Activation::None:        mlp_swift_forward<WIDTH, T, Activation::None>(m_output_activation, m_weights_matrices, input, forward, output,output_stride, m_n_hidden_matrices, 128, m_inputs_width, m_output_width); break;
	case Activation::Exponential: mlp_swift_forward<WIDTH, T, Activation::Exponential>(m_output_activation, m_weights_matrices, input, forward, output, output_stride, m_n_hidden_matrices, batch_size, m_inputs_width, m_output_width); break;
	case Activation::Sigmoid:     mlp_swift_forward<WIDTH, T, Activation::Sigmoid>(m_output_activation, m_weights_matrices, input, forward, output, output_stride, m_n_hidden_matrices, batch_size, m_inputs_width, m_output_width); break;
	case Activation::ReLU:        mlp_swift_forward<WIDTH, T, Activation::ReLU>(m_output_activation, m_weights_matrices, input, forward, output, output_stride, m_n_hidden_matrices, batch_size, m_inputs_width, m_output_width); break;
	case Activation::LeakyReLU:   mlp_swift_forward<WIDTH, T, Activation::LeakyReLU>(m_output_activation, m_weights_matrices, input, forward, output, output_stride, m_n_hidden_matrices, batch_size, m_inputs_width, m_output_width); break;
	case Activation::Squareplus:  mlp_swift_forward<WIDTH, T, Activation::Squareplus>(m_output_activation, m_weights_matrices, input, forward, output, output_stride, m_n_hidden_matrices, batch_size, m_inputs_width, m_output_width); break;
	case Activation::Softplus:    mlp_swift_forward<WIDTH, T, Activation::Softplus>(m_output_activation, m_weights_matrices, input, forward, output, output_stride, m_n_hidden_matrices, batch_size, m_inputs_width, m_output_width); break;
	case Activation::Tanh:        mlp_swift_forward<WIDTH, T, Activation::Tanh>(m_output_activation, m_weights_matrices, input, forward, output, output_stride, m_n_hidden_matrices, batch_size, m_inputs_width, m_output_width); break;
	default: throw std::runtime_error{"Unsupported activation."};
	}
	
	return forward;
}



////FIXME a voir comment est implemnte la backward pass
//template <typename T, int WIDTH>
//void SwiftNetMLP<T, WIDTH>::backward_pass(
//	stream outs,
//	const Context& ctx,
//	const std::vector<bf16> &input,
//	const std::vector<T> &output,
//	std::vector<bf16> &dL_doutput,
//	std::vector<bf16> *dL_dinput,
//	EGradientMode param_gradients_mode
//) {
//
//	int batch_size = dL_doutput.size();
//
//	std::vector<float> backward_tmp(batch_size * WIDTH, 0);
//
//	GPUMatrixDynamic<T> backward_output_tmp;
//	if (m_output_activation != Activation::None) {
//		backward_output_tmp = { m_padded_output_width, batch_size, outs, dL_doutput.layout() };
//		activation_backward_output_gpu(stream, dL_doutput.n_elements(), m_output_activation, output.data(), dL_doutput.data(), backward_output_tmp.data());
//	}
//
//	// Backprop
//	// - weight_gradient.T = activation * output_gradient.T
//	// - input_gradient = weights.T * output_gradient
//	// - RELU: pre_activation_gradinet = post_activation_gradient if val > 0 else 0
//
//	const float param_gradient_beta = param_gradients_mode == EGradientMode::Accumulate ? 1.0f : 0.0f;
//
//	std::vector<SyncedMultiStream> multi_streams;
//
//	const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
//
//	int split_k_factor = batch_size / std::min((uint32_t)(1 << 12), batch_size);
//
//	const GPUMatrixDynamic<T>& tmp_dL_doutput = m_output_activation == Activation::None ? dL_doutput : backward_output_tmp;
//
//	uint32_t tmp_idx = m_n_hidden_matmuls;
//	uint32_t backward_tmp_idx = 0;
//
//	// Output layer
//	if (param_gradients_mode != EGradientMode::Ignore) {
//		multi_streams.emplace_back(stream, 2);
//		fc_multiply_split_k<LastLayerK>(multi_streams.back().get(1), tmp_dL_doutput, forward.hidden.at(tmp_idx).transposed(), output_gradient_matrix(), split_k_factor, param_gradient_beta);
//	}
//
//	// If the output width is larger than 16 dims, we use cutlass to backpropagate through the last layer
//	// rather than fusing it with our kernel.
//	if (m_output_width > 16) {
//		fc_multiply<FullLayer>(stream, output_weight_matrix(use_inference_params).transposed(), tmp_dL_doutput, forward.hidden.at(tmp_idx), backward_tmp.at(backward_tmp_idx), m_activation, true);
//	}
//
//	// Only let the fully fused kernel compute gradients w.r.t. the input, if the input layer has the same size & layout as the other layers
//	auto dL_dinput_fused = input.m() == forward.hidden.at(0).m() && input.layout() == CM ? dL_dinput : nullptr;
//
//	// ASSUMPTION: weight matrices & forward_tmp matrices are contiguous in memory
//	switch (m_activation) {
//	case Activation::None:        mlp_swiftnet_backward<WIDTH, T, Activation::None>(outs, input_weight_matrix(use_inference_params), weight_matrix_at(use_inference_params, 0), tmp_dL_doutput, backward_tmp.at(backward_tmp_idx), forward.hidden.at(0), dL_dinput_fused, m_n_hidden_matmuls); break;
//	case Activation::Exponential: mlp_swiftnet_backward<WIDTH, T, Activation::Exponential>(outs, input_weight_matrix(use_inference_params), weight_matrix_at(use_inference_params, 0), tmp_dL_doutput, backward_tmp.at(backward_tmp_idx), forward.hidden.at(0), dL_dinput_fused, m_n_hidden_matmuls); break;
//	case Activation::Sigmoid:     mlp_swiftnet_backward<WIDTH, T, Activation::Sigmoid>(outs, input_weight_matrix(use_inference_params), weight_matrix_at(use_inference_params, 0), tmp_dL_doutput, backward_tmp.at(backward_tmp_idx), forward.hidden.at(0), dL_dinput_fused, m_n_hidden_matmuls); break;
//	case Activation::ReLU:        mlp_swiftnet_backward<WIDTH, T, Activation::ReLU>(outs, input_weight_matrix(use_inference_params), weight_matrix_at(use_inference_params, 0), tmp_dL_doutput, backward_tmp.at(backward_tmp_idx), forward.hidden.at(0), dL_dinput_fused, m_n_hidden_matmuls); break;
//	case Activation::LeakyReLU:   mlp_swiftnet_backward<WIDTH, T, Activation::LeakyReLU>(outs, input_weight_matrix(use_inference_params), weight_matrix_at(use_inference_params, 0), tmp_dL_doutput, backward_tmp.at(backward_tmp_idx), forward.hidden.at(0), dL_dinput_fused, m_n_hidden_matmuls); break;
//	case Activation::Squareplus:  mlp_swiftnet_backward<WIDTH, T, Activation::Squareplus>(outs, input_weight_matrix(use_inference_params), weight_matrix_at(use_inference_params, 0), tmp_dL_doutput, backward_tmp.at(backward_tmp_idx), forward.hidden.at(0), dL_dinput_fused, m_n_hidden_matmuls); break;
//	case Activation::Softplus:    mlp_swiftnet_backward<WIDTH, T, Activation::Softplus>(outs, input_weight_matrix(use_inference_params), weight_matrix_at(use_inference_params, 0), tmp_dL_doutput, backward_tmp.at(backward_tmp_idx), forward.hidden.at(0), dL_dinput_fused, m_n_hidden_matmuls); break;
//	case Activation::Tanh:        mlp_swiftnet_backward<WIDTH, T, Activation::Tanh>(outs, input_weight_matrix(use_inference_params), weight_matrix_at(use_inference_params, 0), tmp_dL_doutput, backward_tmp.at(backward_tmp_idx), forward.hidden.at(0), dL_dinput_fused, m_n_hidden_matmuls); break;
//	default: throw std::runtime_error{"Unsupported activation."};
//	}
//
//	tmp_idx -= 1;
//	++backward_tmp_idx;
//
//	// layers
//	for (uint32_t i = 0; i < m_n_hidden_matmuls; ++i) {
//		uint32_t matrix_idx = m_n_hidden_matmuls - i - 1;
//
//		if (param_gradients_mode != EGradientMode::Ignore) {
//			multi_streams.emplace_back(stream, 2);
//			fc_multiply_split_k<FullLayerK>(multi_streams.back().get(1), backward_tmp.at(backward_tmp_idx - 1), forward.hidden.at(tmp_idx).transposed(), gradient_matrix_at(matrix_idx), split_k_factor, param_gradient_beta);
//		}
//
//		tmp_idx -= 1;
//		++backward_tmp_idx;
//	}
//
//	if (param_gradients_mode != EGradientMode::Ignore) {
//		multi_streams.emplace_back(stream, 2);
//		fc_multiply_split_k<FullLayerK>(multi_streams.back().get(1), backward_tmp.at(backward_tmp_idx - 1), input.transposed(), input_gradient_matrix(), split_k_factor, param_gradient_beta);
//	}
//
//
//}



void test4() {

	const int batch_size = 128;
	const int WIDTH = 64;

	std::vector<bf16> inputs(batch_size * WIDTH, bf16(1.0f));
	std::vector<float> output(batch_size * WIDTH, 0.0f);

	auto model = SwiftNetMLP<float, 64>(64, 64, 1, Activation::None, Activation::None);
	model.initialize_params();
	auto forward = model.forward_pass(inputs, output);

	for (int i = 0; i < 10; i++) {
		std::cout << forward[i] << std::endl;
	}
	for (int i = 64 *64; i < 64* 64 + 10; i++) {
		std::cout << forward[i] << std::endl;
	}

}

int main() {

	std::cout << "Test 4" << std::endl;
	test4();

	return 0;
}