#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

#include "L2.h"
#include "SwiftNetMLP.h"
#include "activation.h"
#include "common.h"
#include "mkl.h"
#include "mkl_omp_offload.h"
#include "oneapi/mkl.hpp"
#include "sgd.h"
#include "trainer.h"
// #include "config.h"

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
using bf16 = sycl::ext::oneapi::bfloat16;

#define TM 8
#define TK 16
#define TN 8

#define INPUT_WIDTH 64
#define OUTPUT_WIDTH 64

#define SG_SIZE 8
#define WG_SIZE 8 * SG_SIZE
#define BATCH_CHUNK 64

class MultilayerPerceptron {
  struct WeightMatrix {
    int inputDim;
    int outputDim;
    std::vector<float> w;

    WeightMatrix(int inputDim_, int outputDim_, float initialWeightScale_) {
      w.clear();
      inputDim = inputDim_;
      outputDim = outputDim_;
      for (int k = 0; k < inputDim * outputDim; ++k) {
        w.push_back(2 * initialWeightScale_ * (rand() / double(RAND_MAX)) -
                    initialWeightScale_);
      }
    }
    WeightMatrix(int inputDim_, int outputDim_, bool constantInit,
                 float constant_) {
      w.clear();
      inputDim = inputDim_;
      outputDim = outputDim_;
      for (int k = 0; k < inputDim * outputDim; ++k) {
        w.push_back(constant_);
      }
    }
  };

  struct Layer {
    int dim;
    std::vector<float> in;
    std::vector<float> out;
    std::vector<float> err;

    Layer(int dim_) {
      dim = dim_;
      for (int k = 0; k < dim; ++k) {
        in.push_back(0);
        out.push_back(0);
        err.push_back(0);
      }
    }
  };

 public:
  int H;
  int inputDimension;
  int outputDimension;

  struct TrainingElement {
    std::vector<float> in;
    std::vector<float> out;

    TrainingElement(std::vector<float> in_, std::vector<float> out_) {
      in = in_;
      out = out_;
    }
  };

  MultilayerPerceptron(int inputDimension_, int outputDimension_) {
    inputDimension = inputDimension_;
    outputDimension = outputDimension_;

    layers.push_back(Layer(inputDimension));
    H = 1;
  }

  std::vector<WeightMatrix> weights;
  std::vector<Layer> layers;
  std::vector<TrainingElement> trainingSet;

  ~MultilayerPerceptron() {}

  void addHiddenLayer(int dimension_) {
    layers.push_back(Layer(dimension_));
    H++;
  }

  void init() {
    layers.push_back(Layer(outputDimension));
    H++;

    resetWeights();

    WeightMatrix* weightMatrix;
    for (int h = 0; h < H - 1; ++h) {
      weightMatrix = &(weights[h]);
    }
  }

  void init(float constant) {
    layers.push_back(Layer(outputDimension));
    H++;
    resetWeights(constant);
    WeightMatrix* weightMatrix;
    for (int h = 0; h < H - 1; ++h) {
      weightMatrix = &(weights[h]);
    }
  }

  void resetWeights(float constant) {
    weights.clear();
    int h;
    int dim0, dim1;
    for (h = 0; h < H - 1; ++h) {
      dim0 = layers[h].dim;
      dim1 = layers[h + 1].dim;
      weights.push_back(WeightMatrix(dim0, dim1, true, constant));
    }
  }

  void resetWeights() {
    weights.clear();
    int h;
    int dim0, dim1;
    for (h = 0; h < H - 1; ++h) {
      dim0 = layers[h].dim;
      dim1 = layers[h + 1].dim;
      weights.push_back(WeightMatrix(dim0, dim1, 1.0f));
    }
  }

  void calcLayerInput(int h_) {
    if (h_ > 0 && h_ < H) {
      WeightMatrix* w = &(weights[h_ - 1]);
      int i, j;
      for (i = 0; i < layers[h_].dim; ++i) {
        layers[h_].in[i] = 0;
        for (j = 0; j < layers[h_ - 1].dim; ++j) {
          //   if (i == 0) {
          //     std::cout << w->w[i * w->inputDim + j] << ", ";
          //   }
          layers[h_].in[i] += layers[h_ - 1].out[j] * w->w[i * w->inputDim + j];
        }
      }
      //   std::cout << std::endl;
    }
  }

  void calcLayerOutput(int h_, int use_linear = 0) {
    for (int i = 0; i < layers[h_].dim; ++i) {
      layers[h_].out[i] = nonef(layers[h_].in[i]);
      //   if (use_linear) {
      //     layers[h_].out[i] = nonef(layers[h_].in[i]);
      //   } else {
      //     layers[h_].out[i] = relu(layers[h_].in[i]);
      //   }
    }
  }

  std::vector<float> classify(std::vector<float> x_) {
    int h;
    int i;
    if (x_.size() == inputDimension) {
      for (i = 0; i < inputDimension; ++i) {
        layers[0].out[i] = x_[i];
      }
      for (h = 1; h < H; ++h) {
        int use_linear = (h == (H - 1));
        calcLayerInput(h);
        calcLayerOutput(h, use_linear);
      }
      return layers[H - 1].out;
    } else {
      std::cout << "Input has to be the same as width." << std::endl;
    }

    return x_;
  }

  void calcLayerError(int h_) {
    int i, j;
    WeightMatrix* w = &(weights[h_]);
    for (i = 0; i < layers[h_].dim; ++i) {
      float sum = 0;
      for (j = 0; j < layers[h_ + 1].dim; ++j) {
        sum += w->w[i * w->inputDim + j] * layers[h_ + 1].err[j];
      }
      layers[h_].err[i] = dnonefdx(layers[h_].in[i]) * sum;
    }
  }

  void updateWeights(int h_, float eta_) {
    WeightMatrix* w = &(weights[h_ - 1]);
    int i, j;
    float dw;
    for (i = 0; i < w->outputDim; ++i) {
      for (j = 0; j < w->inputDim; ++j) {
        dw = eta_ * (layers[h_].err[j] * layers[h_ - 1].out[i]);
        w->w[j * w->inputDim + i] += dw;
      }
    }
  }

  float psi(float x_) {
    float a = 0.5f;
    return 1.0f / (1 + exp(-a * x_));
  }

  float dpsidx(float x_) { return psi(x_) * (1 - psi(x_)); }

  float nonef(float x_) { return x_; }
  float relu(float x_) { return (x_ > 0.0) ? x_ : 0.0; }
  float dnonefdx(float x_) { return 1; }

  void setTrainingSet(std::vector<TrainingElement> trainingSet_) {
    trainingSet = trainingSet_;
  }

  float train(float eta_) {
    float trainingSetError = 0;
    int t, i, h;
    TrainingElement* te;
    for (t = 0; t < trainingSet.size(); ++t) {
      te = &(trainingSet[t]);
      std::vector<float> x = te->in;
      std::vector<float> y_desired = te->out;
      std::vector<float> y_actual = classify(x);
      float err = 0;
      for (i = 0; i < y_actual.size(); ++i) {
        err += pow(y_desired[i] - y_actual[i], 2);
      }
      trainingSetError += err * err;
      for (i = 0; i < layers[H - 1].dim; ++i) {
        layers[H - 1].err[i] = y_desired[i] - y_actual[i];
      }
      for (h = H - 2; h >= 0; h--) {
        calcLayerError(h);
      }
      for (h = 1; h < H; ++h) {
        updateWeights(h, eta_);
      }
    }
    return sqrt(trainingSetError);
  }

  void copyWeights(std::vector<bf16> trainer_weights, int n) {
    int input_width = INPUT_WIDTH;
    int output_width = OUTPUT_WIDTH;
    int width = 64;
    // Input Weights

    for (int i = 0; i < input_width; i++) {
      for (int j = 0; j < width; j++) {
        // Uncomment this line to generate test cases for
        // python/test_to_packed_layout_coord std::cout << "(" << i * width + j
        // << "," << input_width << ", " << width
        //           << ", "
        //           << toPackedLayoutCoord(i * width + j, input_width, width)
        //           << ")" << std::endl;
        weights[0].w[j * input_width + i] = trainer_weights[toPackedLayoutCoord(
            i * width + j, input_width, width)];
      }
    }

    for (int k = 1; k < n; k++) {
      for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
          weights[k].w[j * width + i] =
              trainer_weights[input_width * width + (k - 1) * width * width +
                              toPackedLayoutCoord(i * width + j, width, width)];
        }
      }
    }
    for (int i = 0; i < width; i++) {
      for (int j = 0; j < output_width; j++) {
        weights[n].w[j * width + i] =
            trainer_weights[input_width * width + (n - 1) * width * width +
                            toPackedLayoutCoord(i * output_width + j, width,
                                                output_width)];
        // std::cout << "Idx: " << j * width + i << std::endl;
      }
    }

    // std::cout << "Weights 0 (" << weights[0].w.size() << ")" << std::endl;
    // int layer = 0;
    // for (int i = 0; i < weights[layer].w.size(); i++) {
    //   std::cout << weights[layer].w[i] << ", ";
    // }
    // std::cout << std::endl;
    // std::cout << "Weights 1 (" << weights[1].w.size() << ")" << std::endl;
    // layer = 1;
    // for (int i = 0; i < weights[layer].w.size(); i++) {
    //   std::cout << weights[layer].w[i] << ", ";
    // }
    // std::cout << std::endl;
  }
};

void test_exactitude() {
  // REFERENCE

  MultilayerPerceptron my_mlp(INPUT_WIDTH, OUTPUT_WIDTH);
  my_mlp.addHiddenLayer(64);
  my_mlp.addHiddenLayer(64);

  // SWIFTNET
  const int batch_size = 64;
  const int output_width = OUTPUT_WIDTH;
  const int WIDTH = 64;
  const int m_n_hidden_layers = 2;
  const int net_width = 64;

  const float scale = 1e-3f;

  queue q = queue();

  DeviceMem<bf16> inputs = DeviceMem<bf16>(INPUT_WIDTH * batch_size, q);
  DeviceMem<float> output = DeviceMem<float>(batch_size * output_width, q);
  DeviceMem<float> target = DeviceMem<float>(batch_size * output_width, q);
  DeviceMem<bf16> grads = DeviceMem<bf16>(batch_size * output_width, q);
  DeviceMem<float> losses = DeviceMem<float>(batch_size * output_width, q);

  L2Loss loss;
  SGDOptimizer optim =
      SGDOptimizer(OUTPUT_WIDTH, m_n_hidden_layers, 1e-3f, 1e-8f);
  SwiftNetMLP<64> network =
      SwiftNetMLP<64>(q, INPUT_WIDTH, output_width, m_n_hidden_layers,
                      //   Activation::ReLU, Activation::None, batch_size);
                      Activation::None, Activation::None, batch_size);
  Trainer train(network, loss, optim);

  train.initialize_params();
  my_mlp.init(1e-4f);
  //   std::cout << "Network size: " << network.m_weights_matrices.size()
  //             << std::endl;
  std::vector<bf16> w_swift(network.m_weights_matrices.size());
  q.memcpy(w_swift.data(), train.m_network->m_weights_matrices.data(),
           network.m_weights_matrices.size() * sizeof(bf16));
  q.wait();
  //   std::cout << " weights" << std::endl;

  //   for (int i = 0; i < 10; i++) {
  //     for (int j = 0; j < 10; j++) {
  //   std::cout << my_mlp.weights[0].w[j * 64 + i] << " ; "
  //             << w_swift[toPackedLayoutCoord(i * 64 + j, 64, 64) +
  //                        0 * batch_size * WIDTH]
  //             << std::endl;
  //       std::cout << w_swift[toPackedLayoutCoord(i * 64 + j, 64, 64) +
  //                            0 * batch_size * WIDTH]
  //                 << ", ";
  //     }
  //   }
  //   for (int i = 0; i < w_swift.size(); i++) {
  //     std::cout << w_swift[i] << ", ";
  //   }
  //   std::cout << std::endl;

  my_mlp.copyWeights(w_swift, 2);
  std::vector<float> x(INPUT_WIDTH);
  for (int i = 0; i < INPUT_WIDTH; i++) {
    x[i] = 1e-1f;
    // x[i] = 1.0f;
  }
  std::vector<float> res_ref = my_mlp.classify(x);

  std::vector<MultilayerPerceptron::TrainingElement> training_set(
      1, MultilayerPerceptron::TrainingElement(
             x, std::vector<float>(OUTPUT_WIDTH, 1.0f)));
  my_mlp.setTrainingSet(training_set);
  my_mlp.train(1e-3f);

  inputs.initialize_constant(0.1f, q);
  output.initialize_constant(0.0f, q);
  target.initialize_constant(1.0f, q);
  grads.initialize_constant(bf16(0.0f), q);
  losses.initialize_constant(0.0f, q);

  train.training_step(inputs, output, target, grads, losses, scale, 64, 0);

  std::vector<float> fwd(
      batch_size * (INPUT_WIDTH + OUTPUT_WIDTH + WIDTH * m_n_hidden_layers));
  q.memcpy(fwd.data(), network.m_forward,
           batch_size * (WIDTH + output_width + WIDTH * m_n_hidden_layers) *
               sizeof(float));
  q.wait();

  std::cout
      << "====================================================================="
         "=================================================================="
      << std::endl;
  std::cout << "Layer 0" << std::endl;
  std::cout
      << "====================================================================="
         "=================================================================="
      << std::endl;
  for (int j = 0; j < INPUT_WIDTH; j++) {
    std::cout << "Idx " << j << " - " << my_mlp.layers[0].out[j] << ": "
              << fwd[j] << std::endl;
  }
  std::cout
      << "====================================================================="
         "=================================================================="
      << std::endl;
  std::cout << "Layer 1" << std::endl;
  std::cout
      << "====================================================================="
         "=================================================================="
      << std::endl;

  for (int j = 0; j < WIDTH; j++) {
    std::cout << "Idx " << j << " - " << my_mlp.layers[1].out[j] << ": "
              << fwd[j + batch_size * INPUT_WIDTH] << std::endl;
  }

  std::cout
      << "====================================================================="
         "=================================================================="
      << std::endl;
  std::cout << "Layer 2" << std::endl;
  std::cout
      << "====================================================================="
         "=================================================================="
      << std::endl;
  for (int j = 0; j < output_width; j++) {
    std::cout << "Idx " << j << " - " << my_mlp.layers[2].out[j] << ": "
              << fwd[j + (batch_size * INPUT_WIDTH) + batch_size * WIDTH * 1]
              << std::endl;
  }
  std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
               "+++++++++"
               "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
               "+++++++++"
            << std::endl;

  std::vector<bf16> grad(train.m_network->m_grads_matrices.size());
  q.memcpy(grad.data(), train.m_network->m_grads_matrices.data(),
           train.m_network->m_grads_matrices.size() * sizeof(bf16))
      .wait();

  //   std::cout << " grads " << std::endl;
  //   for (int i = 0; i < grad.size(); i++) {
  //     if (i == INPUT_WIDTH * WIDTH) {
  //       std::cout << "===================" << std::endl;
  //     }
  //     std::cout << grad[i] << ", ";
  //   }
  //   std::cout << std::endl;

  std::cout << "Grad compare " << std::endl;
  std::cout << "Layer 0" << std::endl;
  for (int i = 0; i < INPUT_WIDTH; i++) {
    for (int j = 0; j < WIDTH; j++) {
      std::cout << i << ", " << j
                << ": "
                // << my_mlp.layers[1].err[j] * my_mlp.layers[0].out[i] << ": "
                << grad[i + j] << std::endl;
    }
  }
  std::cout << "Layer 1" << std::endl;
  for (int i = 0; i < WIDTH; i++) {
    for (int j = 0; j < WIDTH; j++) {
      std::cout << i << ", " << j
                << ": "
                // << my_mlp.layers[2].err[j] * my_mlp.layers[1].out[i] << ": "
                << grad[INPUT_WIDTH * WIDTH + i + j] << std::endl;
    }
  }
  std::cout << "Layer 2" << std::endl;
  for (int i = 0; i < WIDTH; i++) {
    for (int j = 0; j < OUTPUT_WIDTH; j++) {
      std::cout << i << ", " << j
                << ": "
                // << my_mlp.layers[3].err[j] * my_mlp.layers[2].out[i] << ": "
                << grad[INPUT_WIDTH * WIDTH + WIDTH * WIDTH + i + j]
                << std::endl;
    }
  }
  //   std::cout << " weights after backprop" << std::endl;
  //   q.memcpy(w_swift.data(), train.m_network->m_weights_matrices.data(),
  //            train.m_network->m_weights_matrices.size() * sizeof(bf16));
  //   q.wait();
  //   for (int i = 0; i < 10; i++) {
  //     for (int j = 0; j < 10; j++) {
  //       std::cout << my_mlp.weights[0].w[j * 64 + i] << " ; "
  //                 << w_swift[toPackedLayoutCoord(i * 64 + j, 64, 64) +
  //                            0 * batch_size * WIDTH]
  //                 << std::endl;
  //       //   std::cout << w_swift[toPackedLayoutCoord(i * 64 + j, 64, 64) +
  //       //                        0 * batch_size * WIDTH]
  //       //             << ", " << std::endl;
  //     }
  // }

  inputs.free_mem(q);
  output.free_mem(q);
  target.free_mem(q);
  grads.free_mem(q);
  losses.free_mem(q);
}
int main() {
  test_exactitude();
  return 0;
}