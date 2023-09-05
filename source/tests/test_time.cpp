#include <CL/sycl.hpp>
#include <cmath>
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
#include <chrono>

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
      if (use_linear) {
        layers[h_].out[i] = nonef(layers[h_].in[i]);
      } else {
        layers[h_].out[i] = relu(layers[h_].in[i]);
      }
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

    // std::cout << "Weights 0: " << std::endl;
    // int layer = 0;
    // for (int i = 0; i < weights[layer].w.size(); i++) {
    //   std::cout << weights[layer].w[i] << ", ";
    // }
    // std::cout << std::endl;
    // std::cout << "Weights 1: " << std::endl;
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
  // my_mlp.addHiddenLayer(64);

  // SWIFTNET
  //   const int batch_size = 64;
  const int batch_size = std::pow(2, 10);
  const int output_width = OUTPUT_WIDTH;
  const int WIDTH = 64;
  const int intermediate_output_size = batch_size * WIDTH * 2;
  const int m_n_hidden_layers = 1;
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
                      Activation::ReLU, Activation::None, batch_size);
  Trainer train(network, loss, optim);

  train.initialize_params();
  my_mlp.init(1e-4f);

  std::vector<bf16> w_swift(network.m_weights_matrices.size());
  q.memcpy(w_swift.data(), train.m_network->m_weights_matrices.data(),
           network.m_weights_matrices.size() * sizeof(bf16));
  q.wait();

  my_mlp.copyWeights(w_swift, 1);
  std::vector<float> x(INPUT_WIDTH);
  for (int i = 0; i < INPUT_WIDTH; i++) {
    //   x[i] = i * 1e-2f;
    x[i] = 1.0f;
  }

  auto start_time = std::chrono::high_resolution_clock::now();
  int iter_steps = 1000;
  // Code to profile (Section 1)
  std::vector<float> res_ref;
  for (int i = 0; i < iter_steps; ++i) {
    for (int j = 0; j < batch_size; ++j) {
      // Some computation
      res_ref = my_mlp.classify(x);
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);

  std::cout << "MLP Execution Time: " << duration.count() << " ms" << std::endl;

  inputs.initialize_constant(1.0f, q);
  output.initialize_constant(0.0f, q);
  target.initialize_constant(1.0f, q);
  grads.initialize_constant(bf16(0.0f), q);
  losses.initialize_constant(0.0f, q);

  start_time = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iter_steps; ++i) {
    train.training_step(inputs, output, target, grads, losses, scale, 64, 1);
  }
  end_time = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                                   start_time);

  std::cout << "DPCPP Execution Time: " << duration.count() << " ms"
            << std::endl;
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