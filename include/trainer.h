// #pragma once

// #include "DeviceMem.h"
// #include "Encodings/identity.h"
// #include "L2.h"
// #include "Network.h"
// #include "encoding.h"
// #include "loss.h"
// #include "optimizer.h"
// // #include "Encodings/grid.h"

// template <typename T>
// constexpr float default_loss_scale();
// template <>
// constexpr float default_loss_scale<float>() {
//   return 1.0f;
// }
// template <>
// constexpr float default_loss_scale<sycl::half>() {
//   return 128.0f;
// }

// // float default_loss_scale(Precision p) {
// //         return p == Precision::Fp32 ? default_loss_scale<float>()
// //                                     : default_loss_scale<sycl::half>();
// // }

// template <typename T, typename PARAMS_T = T, typename COMPUTE_T = T>
// class EncodingTrainer {
//  public:
//   EncodingTrainer(Encoding<T>& encoding, Loss& loss, Optimizer& optim) {
//     m_encoding = &encoding;
//     m_loss = &loss;
//     m_optim = &optim;
//   }

//   struct ForwardContext : public Context {
//     GPUMatrix<COMPUTE_T> perturbed_output;
//     GPUMatrix<COMPUTE_T> output;
//     GPUMatrix<COMPUTE_T> dL_doutput;
//     GPUMatrix<float> L;
//     std::unique_ptr<Context> model_ctx;
//   };

//   std::unique_ptr<ForwardContext> training_step(
//       dpct::queue_ptr stream, const GPUMatrixDynamic<T>& input,
//       const GPUMatrix<float>& target,
//       const GPUMatrix<float>* data_pdf = nullptr, bool run_optimizer = true,
//       GPUMatrixDynamic<T>* dL_dinput = nullptr,
//       bool use_inference_params = false,
//       GradientMode param_gradients_mode = GradientMode::Overwrite,
//       const GPUMatrix<COMPUTE_T>* external_dL_dy = nullptr) {
//     const float loss_scale = default_loss_scale<PARAMS_T>();

//     // Execute forward and backward in a CUDA graph for maximum performance.
//     std::unique_ptr<ForwardContext> ctx;
//     {
//       sycl::queue queue;
//       // Execute forward and backward in a CUDA graph for maximum
//       performance.
//       // auto capture_guard = m_graph.capture_guard(stream);
//       ctx = forward(&queue, loss_scale, input, target, data_pdf,
//                     use_inference_params, dL_dinput, external_dL_dy);
//       backward(&queue, *ctx, input, dL_dinput, use_inference_params,
//                param_gradients_mode);
//     }

//     // if (run_optimizer) {
//     // 	optimizer_step(stream, loss_scale);
//     //   m_optim->step(m_network->get_queue(), scale,
//     //   m_network->m_weights_matrices,
//     //               m_network->m_weightsT_matrices,
//     //               m_network->m_grads_matrices, WIDTH);
//     // }

//     return ctx;
//   }

//   std::unique_ptr<ForwardContext> training_step(
//       const GPUMatrixDynamic<T>& input, const GPUMatrix<float>& target,
//       const GPUMatrix<float>* data_pdf = nullptr, bool run_optimizer = true,
//       GPUMatrixDynamic<T>* dL_dinput = nullptr,
//       bool use_inference_params = false,
//       GradientMode param_gradients_mode = GradientMode::Overwrite,
//       const GPUMatrix<COMPUTE_T>* external_dL_dy = nullptr) {
//     return training_step(nullptr, input, target, data_pdf, run_optimizer,
//                          dL_dinput, use_inference_params,
//                          param_gradients_mode, external_dL_dy);
//   }

//   // void training_step(DeviceMem<bf16>& input, DeviceMem<float>& output,
//   //                      DeviceMem<float>& target, DeviceMem<bf16>& grads,
//   //                      DeviceMem<float>& losses, const float scale,
//   //                      const int WIDTH) {
//   //     // const int input_size = input.size();
//   //     // const int batch_size = std::pow(2, 19);

//   //     // eignetlich input
//   //     auto input_gen = GPUMatrix<float>{batch.data(), num_dims_encoded,
//   //     batch_size} auto input_mat =
//   //     GPUMatrixDynamic<T>{m_encoding->padded_output_width(),
//   input_gen.n(),
//   //     /*stream,*/ m_encoding->preferred_output_layout()};

//   //     //auto p = m_encoding->m_forward;
//   //     m_encoding->get_queue().parallel_for<WIDTH>(
//   //         range<1>(input.size()), [=](id<1> idx) { input_mat.data()[idx] =
//   //         input.data()[idx]; }).wait();

//   //     /*m_encoding->forward_pass(input, m_encoding->m_forward,
//   //     m_encoding->m_A_forward,
//   //                             m_encoding->m_B_forward,
//   //                             m_encoding->m_C_forward, output);*/

//   //     // network_input <=> output
//   //     auto network_input =
//   //     GPUMatrixDynamic<T>{m_encoding->padded_output_width(),
//   input_mat.n(),
//   //     /*stream,*/ m_encoding->preferred_output_layout()};
//   //     m_encoding->forward_impl(m_encoding->get_queue(),
//   //       input_mat, &network_input);

//   //     m_loss->evaluate(m_encoding->get_queue(), WIDTH, WIDTH, scale,
//   output,
//   //                      target, grads, losses);

//   //     m_encoding->backward_pass(
//   //         input, grads, m_encoding->m_out_inter,
//   m_encoding->m_deltas_temp,
//   //         m_encoding->m_deltas, m_encoding->m_A_backward,
//   //         m_encoding->m_B_backward, m_encoding->m_C_backward,
//   //         m_encoding->m_A_backward_last_layer,
//   //         m_encoding->m_B_backward_last_layer,
//   //         m_encoding->m_C_backward_last_layer,
//   //         m_encoding->m_D_backward_last_layer,
//   //         m_encoding->m_E_backward_last_layer,
//   //         m_encoding->m_F_backward_last_layer, m_encoding->m_A_dgemm,
//   //         m_encoding->m_B_dgemm, m_encoding->m_C_dgemm,
//   //         m_encoding->m_forward);

//   //     m_optim->step(m_encoding->get_queue(), scale,
//   //     m_encoding->m_weights_matrices,
//   //                   m_encoding->m_weightsT_matrices,
//   //                   m_encoding->m_grads_matrices, WIDTH);
//   //   }

//   void training_step(DeviceMem<bf16>& input, DeviceMem<float>& output,
//                      DeviceMem<float>& target, DeviceMem<bf16>& grads,
//                      DeviceMem<float>& losses, const float scale,
//                      const int WIDTH) {
//     // const int input_size = input.size();
//     // const int batch_size = std::pow(2, 19);

//     // eignetlich input
//     // auto input_gen = GPUMatrix<float>{batch.data(), num_dims_encoded,
//     // batch_size}; auto input_mat =
//     // GPUMatrixDynamic<T>{m_encoding->padded_output_width(), input_gen.n(),
//     // /*stream,*/ m_encoding->preferred_output_layout()};

//     auto p = m_encoding->m_forward;
//     // m_encoding->get_queue().parallel_for<WIDTH>(
//     //     range<1>(input.size()), [=](id<1> idx) { p[idx] =
//     input.data()[idx];
//     //     }).wait();

//     /*m_encoding->forward_pass(input, m_encoding->m_forward,
//        m_encoding->m_A_forward, m_encoding->m_B_forward,
//        m_encoding->m_C_forward, output);*/

//     // network_input <=> output
//     // auto network_input =
//     // GPUMatrixDynamic<T>{m_encoding->padded_output_width(), input_mat.n(),
//     // /*stream,*/ m_encoding->preferred_output_layout()};
//     // m_encoding->forward_impl(m_encoding->get_queue(),
//     //  input_mat, &network_input);

//     m_loss->evaluate(m_encoding->get_queue(), WIDTH, WIDTH, scale, output,
//                      target, grads, losses);

//     m_encoding->backward_pass(
//         input, grads, m_encoding->m_out_inter, m_encoding->m_deltas_temp,
//         m_encoding->m_deltas, m_encoding->m_A_backward,
//         m_encoding->m_B_backward, m_encoding->m_C_backward,
//         m_encoding->m_A_backward_last_layer,
//         m_encoding->m_B_backward_last_layer,
//         m_encoding->m_C_backward_last_layer,
//         m_encoding->m_D_backward_last_layer,
//         m_encoding->m_E_backward_last_layer,
//         m_encoding->m_F_backward_last_layer, m_encoding->m_A_dgemm,
//         m_encoding->m_B_dgemm, m_encoding->m_C_dgemm, m_encoding->m_forward);

//     m_optim->step(
//         m_encoding->get_queue(), scale, m_encoding->m_weights_matrices,
//         m_encoding->m_weightsT_matrices, m_encoding->m_grads_matrices,
//         WIDTH);
//   }

//   std::unique_ptr<ForwardContext> forward(
//       dpct::queue_ptr stream, const float loss_scale,
//       const GPUMatrixDynamic<T>& input, const GPUMatrix<float>& target,
//       const GPUMatrix<float>* data_pdf = nullptr,
//       bool use_inference_params = false, bool prepare_input_gradients =
//       false, const GPUMatrix<COMPUTE_T>* external_dL_dy = nullptr) {
//     const uint32_t batch_size = input.n();

//     auto forward = std::make_unique<ForwardContext>();
//     // TODO: m_model->padded_output_width()
//     sycl::queue queue;
//     forward->output =
//         GPUMatrix<COMPUTE_T>{m_encoding->padded_output_width(), batch_size};
//     forward->model_ctx =
//         m_encoding->forward_impl(&queue, input, &forward->output,
//                                  use_inference_params,
//                                  prepare_input_gradients);

//     // if (m_perturbation_sigma > 0) {
//     // 	GPUMatrix<float> perturbation{m_model->padded_output_width(),
//     // batch_size, stream}; 	forward->perturbed_output =
//     // GPUMatrix<COMPUTE_T>{m_model->padded_output_width(), batch_size,
//     stream};

//     // 	const uint32_t n_elements = perturbation.n_elements();
//     // 	generate_random_logistic<float>(stream, m_rng, n_elements,
//     // perturbation.data(), 0.0f, m_perturbation_sigma);
//     //         stream->submit([&](sycl::handler &cgh) {
//     //             auto forward_output_data_ct1 = forward->output.data();
//     //             auto perturbation_data_ct2 = perturbation.data();
//     //             auto forward_perturbed_output_data_ct3 =
//     //                 forward->perturbed_output.data();

//     //             cgh.parallel_for(
//     //                 sycl::nd_range<3>(
//     //                     sycl::range<3>(1, 1, n_blocks_linear(n_elements))
//     *
//     //                         sycl::range<3>(1, 1, N_THREADS_LINEAR),
//     //                     sycl::range<3>(1, 1, N_THREADS_LINEAR)),
//     //                 [=](sycl::nd_item<3> item_ct1) {
//     //                     add(n_elements, forward_output_data_ct1,
//     //                         perturbation_data_ct2,
//     //                         forward_perturbed_output_data_ct3, item_ct1);
//     //                 });
//     //         });
//     //             }

//     // auto& loss_input = m_perturbation_sigma > 0 ?
//     forward->perturbed_output :
//     // forward->output;

//     // forward->L = GPUMatrix<float>{m_model->padded_output_width(),
//     batch_size,
//     // stream};

//     // if (external_dL_dy) {
//     // 	CHECK_THROW(external_dL_dy->m() ==
//     m_model->padded_output_width());
//     // 	CHECK_THROW(external_dL_dy->n() == batch_size);

//     // 	forward->dL_doutput =
//     GPUMatrix<COMPUTE_T>{external_dL_dy->data(),
//     // m_model->padded_output_width(), batch_size}; } else {
//     // 	CHECK_THROW(input.n() == target.n());
//     // 	CHECK_THROW(m_model->output_width() == target.m());

//     // 	forward->dL_doutput =
//     // GPUMatrix<COMPUTE_T>{m_model->padded_output_width(), batch_size,
//     stream};
//     // 	m_loss->evaluate(stream, loss_scale, loss_input, target,
//     forward->L,
//     // forward->dL_doutput, data_pdf);
//     // }

//     return forward;
//   }

//   std::unique_ptr<ForwardContext> forward(
//       const float loss_scale, const GPUMatrixDynamic<T>& input,
//       const GPUMatrix<float>& target,
//       const GPUMatrix<float>* data_pdf = nullptr,
//       bool use_inference_params = false, bool prepare_input_gradients =
//       false, const GPUMatrix<COMPUTE_T>* external_dL_dy = nullptr) {
//     return forward(nullptr, loss_scale, input, target, data_pdf,
//                    use_inference_params, prepare_input_gradients,
//                    external_dL_dy);
//   }

//   void backward(dpct::queue_ptr stream, const ForwardContext& ctx,
//                 const GPUMatrixDynamic<T>& input,
//                 GPUMatrixDynamic<T>* dL_dinput = nullptr,
//                 bool use_inference_params = false,
//                 GradientMode param_gradients_mode = GradientMode::Overwrite)
//                 {
//     m_encoding->backward_impl(stream, *ctx.model_ctx, input, ctx.output,
//                               ctx.dL_doutput, dL_dinput,
//                               use_inference_params, param_gradients_mode);
//   }

//   void backward(const ForwardContext& ctx, const GPUMatrixDynamic<T>& input,
//                 GPUMatrixDynamic<T>* dL_dinput = nullptr,
//                 bool use_inference_params = false,
//                 GradientMode param_gradients_mode = GradientMode::Overwrite)
//                 {
//     backward(nullptr, ctx, input, dL_dinput, use_inference_params,
//              param_gradients_mode);
//   }

//   Encoding<T>* m_encoding;
//   Loss* m_loss;
//   Optimizer* m_optim;

//   void initialize_params() { m_encoding->initialize_params(); }
// };

// class Trainer {
//  public:
//   Trainer(Network& network, Loss& loss, Optimizer& optim) {
//     m_network = &network;
//     m_loss = &loss;
//     m_optim = &optim;
//   }

//   void training_step(DeviceMem<bf16>& input, DeviceMem<float>& output,
//                      DeviceMem<float>& target, DeviceMem<bf16>& grads,
//                      DeviceMem<float>& losses, const float scale,
//                      const int WIDTH, int forward_only = 1,
//                      int run_inference = 0) {
//     // const int input_size = input.size();
//     // const int batch_size = std::pow(2, 19);

//     auto p = m_network->m_forward;
//     m_network->get_queue()
//         .parallel_for<>(range<1>(input.size()),
//                         [=](id<1> idx) { p[idx] = input.data()[idx]; })
//         .wait();

//     if (run_inference) {
//       forward_only = 1;
//       m_network->inference(input, m_network->m_forward,
//       m_network->m_A_forward,
//                            m_network->m_B_forward, m_network->m_C_forward,
//                            output);

//     } else {
//       m_network->forward_pass(input, m_network->m_forward,
//                               m_network->m_A_forward, m_network->m_B_forward,
//                               m_network->m_C_forward, output);
//     }
//     if (forward_only) {
//       return;
//     }

//     m_loss->evaluate(m_network->get_queue(), WIDTH, WIDTH, scale, output,
//                      target, grads, losses);

//     m_network->backward_pass(
//         input, grads, m_network->m_out_inter, m_network->m_deltas,
//         m_network->m_A_backward, m_network->m_B_backward,
//         m_network->m_C_backward, m_network->m_A_backward_last_layer,
//         m_network->m_B_backward_last_layer,
//         m_network->m_C_backward_last_layer,
//         m_network->m_D_backward_last_layer,
//         m_network->m_E_backward_last_layer,
//         m_network->m_F_backward_last_layer, m_network->m_A_dgemm,
//         m_network->m_B_dgemm, m_network->m_C_dgemm, m_network->m_forward);
//     // no optimisation as we run benchmarking
//     // m_optim->step(m_network->get_queue(), scale,
//     // m_network->m_weights_matrices,
//     //               m_network->m_weightsT_matrices,
//     //               m_network->m_grads_matrices, WIDTH);
//   }
//   Network* m_network;
//   Loss* m_loss;
//   Optimizer* m_optim;

//   void initialize_params(int use_constant = 0) {
//     m_network->initialize_params(use_constant);
//   }

//  private:
// };
