// #include <iostream>
// #include <sycl/sycl.hpp>
// #include <vector>

// #include "L2.h"
// #include "SwiftNetMLP.h"
// #include "activation.h"
// #include "common.h"
// #include "mkl.h"
// #include "mkl_omp_offload.h"
// #include "oneapi/mkl.hpp"
// #include "sgd.h"
// #include "trainer.h"
// // #include "config.h"

// using namespace sycl;
// using namespace sycl::ext::oneapi::experimental::matrix;
// using bf16 = sycl::ext::oneapi::bfloat16;

// #define TM 8
// #define TK 16
// #define TN 8

// #define SG_SIZE 8
// #define WG_SIZE 8 * SG_SIZE
// #define BATCH_CHUNK 64

// class MultilayerPerceptron {
//   struct WeightMatrix {
//     int inputDim;
//     int outputDim;
//     std::vector<float> w;

//     WeightMatrix(int inputDim_, int outputDim_, float initialWeightScale_) {
//       w.clear();
//       inputDim = inputDim_;
//       outputDim = outputDim_;
//       for (int k = 0; k < inputDim * outputDim; ++k) {
//         w.push_back(2 * initialWeightScale_ * (rand() / double(RAND_MAX)) -
//                     initialWeightScale_);
//       }
//     }
//     WeightMatrix(int inputDim_, int outputDim_, bool constantInit,
//                  float constant_) {
//       w.clear();
//       inputDim = inputDim_;
//       outputDim = outputDim_;
//       for (int k = 0; k < inputDim * outputDim; ++k) {
//         w.push_back(constant_);
//       }
//     }
//   };

//   struct Layer {
//     int dim;
//     std::vector<float> in;
//     std::vector<float> out;
//     std::vector<float> err;

//     Layer(int dim_) {
//       dim = dim_;
//       for (int k = 0; k < dim; ++k) {
//         in.push_back(0);
//         out.push_back(0);
//         err.push_back(0);
//       }
//     }
//   };

//  public:
//   int H;
//   int inputDimension;
//   int outputDimension;

//   struct TrainingElement {
//     std::vector<float> in;
//     std::vector<float> out;

//     TrainingElement(std::vector<float> in_, std::vector<float> out_) {
//       in = in_;
//       out = out_;
//     }
//   };

//   MultilayerPerceptron(int inputDimension_, int outputDimension_) {
//     inputDimension = inputDimension_;
//     outputDimension = outputDimension_;

//     layers.push_back(Layer(inputDimension));
//     H = 1;
//   }

//   std::vector<WeightMatrix> weights;
//   std::vector<Layer> layers;
//   std::vector<TrainingElement> trainingSet;

//   ~MultilayerPerceptron() {}

//   void addHiddenLayer(int dimension_) {
//     layers.push_back(Layer(dimension_));
//     H++;
//   }

//   void init() {
//     layers.push_back(Layer(outputDimension));
//     H++;

//     resetWeights();

//     WeightMatrix* weightMatrix;
//     for (int h = 0; h < H - 1; ++h) {
//       weightMatrix = &(weights[h]);
//     }
//   }

//   void init(float constant) {
//     layers.push_back(Layer(outputDimension));
//     H++;
//     resetWeights(constant);
//     WeightMatrix* weightMatrix;
//     for (int h = 0; h < H - 1; ++h) {
//       weightMatrix = &(weights[h]);
//     }
//   }

//   void resetWeights(float constant) {
//     weights.clear();
//     int h;
//     int dim0, dim1;
//     for (h = 0; h < H - 1; ++h) {
//       dim0 = layers[h].dim;
//       dim1 = layers[h + 1].dim;
//       weights.push_back(WeightMatrix(dim0, dim1, true, constant));
//     }
//   }

//   void resetWeights() {
//     weights.clear();
//     int h;
//     int dim0, dim1;
//     for (h = 0; h < H - 1; ++h) {
//       dim0 = layers[h].dim;
//       dim1 = layers[h + 1].dim;
//       weights.push_back(WeightMatrix(dim0, dim1, 1.0f));
//     }
//   }

//   void calcLayerInput(int h_) {
//     if (h_ > 0 && h_ < H) {
//       WeightMatrix* w = &(weights[h_ - 1]);
//       int i, j;
//       for (i = 0; i < layers[h_].dim; ++i) {
//         layers[h_].in[i] = 0;
//         for (j = 0; j < layers[h_ - 1].dim; ++j) {
//           layers[h_].in[i] += layers[h_ - 1].out[j] * w->w[i * w->inputDim +
//           j];
//         }
//       }
//     }
//   }

//   void calcLayerOutput(int h_) {
//     for (int i = 0; i < layers[h_].dim; ++i) {
//       layers[h_].out[i] = nonef(layers[h_].in[i]);
//     }
//   }

//   std::vector<float> classify(std::vector<float> x_) {
//     int h;
//     int i;
//     if (x_.size() == inputDimension) {
//       for (i = 0; i < inputDimension; ++i) {
//         layers[0].out[i] = x_[i];
//       }
//       for (h = 1; h < H; ++h) {
//         calcLayerInput(h);
//         calcLayerOutput(h);
//       }
//       return layers[H - 1].out;
//     }
//     return x_;
//   }

//   void calcLayerError(int h_) {
//     int i, j;
//     WeightMatrix* w = &(weights[h_]);
//     for (i = 0; i < layers[h_].dim; ++i) {
//       float sum = 0;
//       for (j = 0; j < layers[h_ + 1].dim; ++j) {
//         sum += w->w[i * w->inputDim + j] * layers[h_ + 1].err[j];
//       }
//       layers[h_].err[i] = dnonefdx(layers[h_].in[i]) * sum;
//     }
//   }

//   void updateWeights(int h_, float eta_) {
//     WeightMatrix* w = &(weights[h_ - 1]);
//     int i, j;
//     float dw;
//     for (i = 0; i < w->outputDim; ++i) {
//       for (j = 0; j < w->inputDim; ++j) {
//         dw = eta_ * (layers[h_].err[j] * layers[h_ - 1].out[i]);
//         w->w[j * w->inputDim + i] += dw;
//       }
//     }
//   }

//   float psi(float x_) {
//     float a = 0.5f;
//     return 1.0f / (1 + exp(-a * x_));
//   }

//   float dpsidx(float x_) { return psi(x_) * (1 - psi(x_)); }

//   float nonef(float x_) { return x_; }

//   float dnonefdx(float x_) { return 1; }

//   void setTrainingSet(std::vector<TrainingElement> trainingSet_) {
//     trainingSet = trainingSet_;
//   }

//   float train(float eta_) {
//     float trainingSetError = 0;
//     int t, i, h;
//     TrainingElement* te;
//     for (t = 0; t < trainingSet.size(); ++t) {
//       te = &(trainingSet[t]);
//       std::vector<float> x = te->in;
//       std::vector<float> y_desired = te->out;
//       std::vector<float> y_actual = classify(x);
//       float err = 0;
//       for (i = 0; i < y_actual.size(); ++i) {
//         err += pow(y_desired[i] - y_actual[i], 2);
//       }
//       trainingSetError += err * err;
//       for (i = 0; i < layers[H - 1].dim; ++i) {
//         layers[H - 1].err[i] = y_desired[i] - y_actual[i];
//       }
//       for (h = H - 2; h >= 0; h--) {
//         calcLayerError(h);
//       }
//       for (h = 1; h < H; ++h) {
//         updateWeights(h, eta_);
//       }
//     }
//     return sqrt(trainingSetError);
//   }

//   void copyWeights(std::vector<bf16> trainer_weights, int n) {
//     int input_width = 64;
//     int output_width = 64;
//     int width = 64;
//     // Input Weights

//     for (int i = 0; i < input_width; i++) {
//       for (int j = 0; j < width; j++) {
//         weights[0].w[j * width + i] = trainer_weights[toPackedLayoutCoord(
//             i * width + j, input_width, width)];
//       }
//     }

//     for (int k = 1; k < n; k++) {
//       for (int i = 0; i < width; i++) {
//         for (int j = 0; j < width; j++) {
//           weights[k].w[j * width + i] =
//               trainer_weights[input_width * width + (k - 1) * width * width +
//                               toPackedLayoutCoord(i * width + j, width,
//                               width)];
//         }
//       }
//     }
//     for (int i = 0; i < width; i++) {
//       for (int j = 0; j < output_width; j++) {
//         weights[n].w[j * output_width + i] =
//             trainer_weights[input_width * width + (n - 1) * width * width +
//                             toPackedLayoutCoord(i * output_width + j, width,
//                                                 output_width)];
//       }
//     }
//   }
// };

// void test_exactitude() {
//   // REFERENCE

//   MultilayerPerceptron my_mlp(64, 64);
//   my_mlp.addHiddenLayer(64);
//   my_mlp.addHiddenLayer(64);

//   // SWIFTNET
//   const int batch_size =
//       64;  // POur l'istant on teste avec un batch size de 1 ( 'est � dire
//            //   que
//   // les 64 �l�ments sont tous les m�mes)
//   const int output_width = 64;
//   const int WIDTH = 64;
//   const int intermediate_output_size = batch_size * WIDTH * 2;
//   const int layer_length = WIDTH * batch_size;
//   const int n_hidden_matrices = 1;
//   const int net_width = 64;
//   const int inputs_width = 64;

//   const float scale = 1e-3f;
//   /*device dev = device(gpu_selector_v);

//   std::vector<device> subdev = {};

//   subdev = dev.create_sub_devices<sycl::info::partition_property::
//       partition_by_affinity_domain>(sycl::info::partition_affinity_domain::numa);*/

//   queue q = queue();

//   DeviceMem<bf16> inputs = DeviceMem<bf16>(batch_size * WIDTH, q);
//   DeviceMem<float> output = DeviceMem<float>(batch_size * output_width, q);
//   DeviceMem<float> target = DeviceMem<float>(batch_size * output_width, q);
//   DeviceMem<bf16> grads = DeviceMem<bf16>(batch_size * output_width, q);
//   DeviceMem<float> losses = DeviceMem<float>(batch_size * output_width, q);

//   float* forward_f = malloc_device<float>(intermediate_output_size, q);
//   float* forward =
//       malloc_device<float>(batch_size * (WIDTH + output_width + WIDTH * 2),
//       q);
//   int shmem_size = batch_size * WIDTH * 4;
//   const size_t alignment = 4096;

//   //   auto act_mem = sycl::aligned_alloc_device<bf16>(alignment, shmem_size,
//   //   q); auto act_mem_temp =
//   //       sycl::aligned_alloc_device<float>(alignment, shmem_size, q);

//   auto A_forward =
//       sycl::aligned_alloc_device<float>(alignment, layer_length, q);
//   auto B_forward =
//       sycl::aligned_alloc_device<float>(alignment, output_width * 64, q);
//   auto C_forward = sycl::aligned_alloc_device<float>(
//       alignment, output_width * batch_size, q);

//   float* out_inter =
//       malloc_device<float>(batch_size * WIDTH * (n_hidden_matrices + 1), q);
//   auto deltas_temp = sycl::aligned_alloc_device<float>(
//       alignment, output_width * batch_size, q);
//   DeviceMem<bf16> deltas(output_width * batch_size, q);

//   auto A_backward =
//       sycl::aligned_alloc_device<float>(alignment, WIDTH * batch_size, q);
//   auto B_backward = sycl::aligned_alloc_device<float>(
//       alignment, batch_size * output_width, q);
//   auto C_backward =
//       sycl::aligned_alloc_device<float>(alignment, WIDTH * output_width, q);

//   auto A_backward_last_layer =
//       sycl::aligned_alloc_device<float>(alignment, grads.size(), q);
//   auto B_backward_last_layer =
//       sycl::aligned_alloc_device<float>(alignment, output_width * WIDTH, q);
//   auto C_backward_last_layer =
//       sycl::aligned_alloc_device<float>(alignment, WIDTH * batch_size, q);
//   auto D_backward_last_layer =
//       sycl::aligned_alloc_device<float>(alignment, WIDTH * batch_size, q);
//   auto E_backward_last_layer =
//       sycl::aligned_alloc_device<float>(alignment, batch_size * WIDTH, q);
//   auto F_backward_last_layer =
//       sycl::aligned_alloc_device<float>(alignment, WIDTH * WIDTH, q);

//   auto A_dgemm =
//       sycl::aligned_alloc_device<float>(alignment, batch_size * WIDTH, q);
//   auto B_dgemm =
//       sycl::aligned_alloc_device<float>(alignment, batch_size * WIDTH, q);
//   auto C_dgemm = sycl::aligned_alloc_device<float>(alignment, WIDTH * WIDTH,
//   q);

//   L2Loss loss;
//   SGDOptimizer optim = SGDOptimizer(64, 2, 1e-3f, 1e-8f);
//   SwiftNetMLP<64> network = SwiftNetMLP<64>(q, 64, 64, 2, Activation::None,
//                                             Activation::None, batch_size);
//   Trainer train(network, loss, optim);

//   train.initialize_params();
//   my_mlp.init(1e-4f);
//   std::vector<bf16> w_swift(64 * 64 * 3);
//   q.memcpy(w_swift.data(), train.m_network->m_weights_matrices.data(),
//            64 * 64 * 3 * sizeof(bf16));
//   q.wait();
//   my_mlp.copyWeights(w_swift, 2);
//   /*for (int i = 0; i < 100; i++) {
//       std::cout << my_mlp.weights[2].w[i] << std::endl;

//   }*/

//   std::vector<float> x(64);
//   for (int i = 0; i < 64; i++) {
//     x[i] = i * 1e-2f;
//   }
//   std::vector<float> res_ref = my_mlp.classify(x);
//   std::cout << res_ref[0] << std::endl;

//   std::vector<MultilayerPerceptron::TrainingElement> training_set(
//       1,
//       MultilayerPerceptron::TrainingElement(x,
//       std::vector<float>(64, 1.0f)));
//   my_mlp.setTrainingSet(training_set);
//   my_mlp.train(1e-3f);

//   inputs.initialize_constant(0.01f, q);
//   output.initialize_constant(0.0f, q);
//   target.initialize_constant(1.0f, q);
//   grads.initialize_constant(bf16(0.0f), q);
//   losses.initialize_constant(0.0f, q);
//   q.parallel_for<>(range<1>(inputs.size()), [=](id<1> idx) {
//     forward[idx] = (float)inputs.data()[idx];
//   });
//   network.initialize_params();
//   network.forward_pass(inputs, forward, A_forward, B_forward, C_forward,
//                        output);

//   std::cout << "Performing one training step" << std::endl;
//   train.training_step(inputs, output, target, grads, losses, scale, 64);

//   std::vector<float> res(64 * 3, 0.0f);

//   for (int j = 0; j < 64; j++) {
//     for (int k = 0; k < 64; k++) {
//       res[j] += x[k] * w_swift[toPackedLayoutCoord(k * 64 + j, 64, 64)];
//     }
//   }
//   for (int j = 0; j < 64; j++) {
//     for (int k = 0; k < 64; k++) {
//       res[j + 64] +=
//           res[k] * w_swift[64 * 64 + toPackedLayoutCoord(k * 64 + j, 64,
//           64)];
//     }
//   }
//   for (int j = 0; j < 64; j++) {
//     for (int k = 0; k < 64; k++) {
//       res[j + 64 * 2] +=
//           res[k + 64] *
//           w_swift[64 * 64 * 2 + toPackedLayoutCoord(k * 64 + j, 64, 64)];
//     }
//   }
//   std::vector<float> fwd(batch_size * (WIDTH + output_width + WIDTH * 2));
//   std::vector<float> output_vec(batch_size * output_width);
//   q.memcpy(fwd.data(), forward,
//            batch_size * (WIDTH + output_width + WIDTH * 2) * sizeof(float));
//   q.wait();
//   output.copy_to_host(output_vec, q);

//   std::cout << "Second layer" << std::endl;
//   for (int j = 0; j < 64; j++) {
//     std::cout << my_mlp.layers[2].out[j] << ": "
//               << fwd[j + 2 * batch_size * WIDTH] << ", ";
//   }
//   std::cout << std::endl;

//   std::cout << "Third layer" << std::endl;

//   for (int j = 0; j < 64; j++) {
//     std::cout << my_mlp.layers[3].out[j] << ": "
//               << fwd[j + 3 * batch_size * WIDTH] << ", ";
//   }
//   std::cout << std::endl;

//   std::cout << "Output" << std::endl;
//   for (const float& element : output_vec) {
//     std::cout << element << " ";
//   }
//   std::cout << std::endl;
//   std::cout << " grads " << std::endl;
//   std::vector<bf16> grad(3 * 64 * 64);
//   q.memcpy(grad.data(), train.m_network->m_grads_matrices.data(),
//            3 * 64 * 64 * sizeof(bf16));
//   std::vector<bf16> l_grad(64 * 64);
//   q.memcpy(l_grad.data(), grads.data(), 64 * 64 * sizeof(bf16));
//   q.wait();
//   for (int i = 0; i < 64; i++) {
//     std::cout << my_mlp.layers[3].err[i] * my_mlp.layers[2].out[0] << " ; "
//               << grad[2 * 64 * 64 + i] << " ; " << l_grad[i] << std::endl;
//     std::cout << my_mlp.layers[3].err[i] * my_mlp.layers[2].out[0] << " ; "
//               << grad[2 * 64 * 64 + 64 + i] << " ; " << l_grad[64 + i]
//               << std::endl;
//     std::cout << my_mlp.layers[3].err[i] * my_mlp.layers[2].out[0] << " ; "
//               << grad[2 * 64 * 64 + 2 * 64 + i] << " ; " << l_grad[2 * 64 +
//               i]
//               << std::endl;
//     std::cout << my_mlp.layers[3].err[i] * my_mlp.layers[2].out[0] << " ; "
//               << grad[2 * 64 * 64 + 3 * 64 + i] << " ; " << l_grad[3 * 64 +
//               i]
//               << std::endl;
//   }
//   std::cout << " weights " << std::endl;

//   q.memcpy(w_swift.data(), train.m_network->m_weights_matrices.data(),
//            64 * 64 * 3 * sizeof(bf16));
//   q.wait();
//   for (int i = 0; i < 10; i++) {
//     for (int j = 0; j < 10; j++) {
//       std::cout << my_mlp.weights[0].w[j * 64 + i] << " ; "
//                 << w_swift[toPackedLayoutCoord(i * 64 + j, 64, 64) +
//                            0 * batch_size * WIDTH]
//                 << std::endl;
//     }
//   }
//   /*inputs.initialize_constant(bf16(2.0f), q);

//   output.initialize_constant(0.0f, q);
//   target.initialize_constant(10.0f, q);
//   grads.initialize_constant(bf16(0.0f), q);
//   losses.initialize_constant(0.0f, q);
//   q.parallel_for<>(range<1>(inputs.size()), [=](id<1> idx) {
//       forward[idx] = (float)inputs.data()[idx];
//       });
//   std::cout << "2" << std::endl;
//   train.training_step(inputs,
//       forward,
//       act_mem,
//       act_mem_temp,
//       A_forward,
//       B_forward,
//       C_forward,
//       out_inter,
//       deltas_temp,
//       deltas,
//       A_backward,
//       B_backward,
//       C_backward,
//       A_backward_last_layer,
//       B_backward_last_layer,
//       C_backward_last_layer,
//       D_backward_last_layer,
//       E_backward_last_layer,
//       F_backward_last_layer,
//       A_dgemm,
//       B_dgemm,
//       C_dgemm,
//       output,
//       target,
//       grads,
//       losses,
//       scale,
//       64);*/
//   // Affichage des weights de notre MLP apr�s le training

//   /*std::vector<bf16> w_swift(64 * 64);
//   q.memcpy(w_swift.data(), train.m_network->m_weights_matrices.data() ,
//   64
//   * 64
//   * sizeof(bf16)); q.wait(); for (int i = 0; i < 100; i++) { std::cout <<
//   my_mlp.weights[0].w[i] << std::endl; std::cout << w_swift[i] <<
//   std::endl;

//   } */

//   inputs.free_mem(q);
//   output.free_mem(q);
//   target.free_mem(q);
//   grads.free_mem(q);
//   losses.free_mem(q);
//   //   free(act_mem, q);
//   //   free(act_mem_temp, q);
//   free(out_inter, q);
//   free(deltas_temp, q);
//   free(A_forward, q);
//   free(B_forward, q);
//   free(C_forward, q);
//   deltas.free_mem(q);
//   free(A_backward, q);
//   free(B_backward, q);
//   free(C_backward, q);
//   free(A_backward_last_layer, q);
//   free(B_backward_last_layer, q);
//   free(C_backward_last_layer, q);
//   free(D_backward_last_layer, q);
//   free(E_backward_last_layer, q);
//   free(F_backward_last_layer, q);
//   free(A_dgemm, q);
//   free(B_dgemm, q);
//   free(C_dgemm, q);
// }

// void test_encoding() {
//   // REFERENCE

//   L2Loss loss;
//   SGDOptimizer optim = SGDOptimizer(64, 2, 1e-3f, 1e-8f);

//   // SWIFTNET
//   const int batch_size = 8;
//   const int output_width = 2;
//   const int input_width = 2;

//   queue q = queue();

//   //   DeviceMem<float> input_f = DeviceMem<float>(batch_size * input_width,
//   q);
//   //   DeviceMem<float> output_f = DeviceMem<float>(batch_size *
//   output_width,
//   //   q); DeviceMem<float> target_f = DeviceMem<float>(batch_size *
//   //   output_width, q);

//   // Encoding<float>* encoding = create_grid_encoding_templated_2<float>(2,
//   // nullptr);
//   Encoding<float>* encoding = new IdentityEncoding<float>(input_width);
//   EncodingTrainer<float, float, float> train(*encoding, loss, optim);

//   train.initialize_params();  // encoding has none

//   GPUMatrix<float> input_float(input_width, batch_size);
//   GPUMatrix<float> output_float(input_width, batch_size);
//   GPUMatrix<float> target_float(input_width, batch_size);

//   input_float.initialize_constant(0.01f);
//   output_float.initialize_constant(0.00f);
//   target_float.initialize_constant(1.0f);
//   std::cout << "Performing one forward step" << std::endl;

//   std::cout << "Input before fwd pass" << std::endl;
//   input_float.print();
//   std::cout << "Output before fwd pass" << std::endl;
//   output_float.print();
//   std::cout << "Target before fwd pass" << std::endl;
//   target_float.print();

//   std::unique_ptr<Context> model_ctx =
//       encoding->forward_impl(&q, input_float, &output_float);

//   std::cout << "Input after fwd pass" << std::endl;
//   input_float.print();
//   std::cout << "Output after fwd pass" << std::endl;
//   output_float.print();
//   std::cout << "Target after fwd pass" << std::endl;
//   target_float.print();

//   //   std::cout << "Training step" << std::endl;

//   // dummy matrices

//   //   DeviceMem<float> dli_f = DeviceMem<float>(batch_size * input_width,
//   q);
//   //   DeviceMem<float> dlo_f = DeviceMem<float>(batch_size * output_width,
//   q);
//   //   GPUMatrix<float> dl_input(dli_f.data(), batch_size * input_width,
//   //                             batch_size * output_width);
//   //   GPUMatrix<float> dl_output(dlo_f.data(), batch_size * input_width,
//   //                              batch_size * output_width);
//   //   auto ctx = train.training_step(input_float, target_float);

//   //   std::cout << "Input after fwd pass" << std::endl;
//   //   input_float.print();
//   //   std::cout << "Output after fwd pass" << std::endl;
//   //   output_float.print();
//   //   std::cout << "Target after fwd pass" << std::endl;
//   //   target_float.print();

//   //   std::cout << "Training step done" << std::endl;

//   //   std::vector<float> output_train(output_float.cols() *
//   //   output_float.rows()); if (output_train.empty() || !ctx->output.data())
//   {
//   //     std::cerr << "Error: No training data! Empty..." << std::endl;
//   //     exit(0);
//   //   }

//   //   std::cout << "Performed encoded training step" << std::endl;
// }

// int main() {
//   test_encoding();
//   return 0;
// }