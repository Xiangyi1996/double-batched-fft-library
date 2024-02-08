/**
 * @file doctest_swiftnet.cpp
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Tests for the Swiftnet class.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "doctest/doctest.h"
#include <filesystem>

#include "SwiftNetMLP.h"
#include "io.h"
#include "result_check.h"
#include <Eigen/Dense>

// ReLU function
double relu(double x) { return x > 0 ? x : 0; }

// Derivative of the ReLU function
double drelu(double x) { return x > 0 ? 1 : 0; }

// Mean squared error loss
double mse(Eigen::VectorXd y, Eigen::VectorXd y_pred) { return (y - y_pred).array().pow(2).mean(); }

// Convert Eigen::VectorXd to std::vector
template <typename T> std::vector<T> eigenToStdVector(const Eigen::VectorXd &eigenVector) {
    std::vector<T> stdVector(eigenVector.data(), eigenVector.data() + eigenVector.size());
    return stdVector;
}

template <typename T> class MLP {
  private:
    std::vector<Eigen::MatrixXd> weights;
    int n_hidden_layers;

  public:
    MLP<T>(int inputDim, int hiddenDim, int outputDim, int n_hidden_layers, bool linspace_weights)
        : n_hidden_layers{n_hidden_layers} {
        // first layer
        weights.push_back(Eigen::MatrixXd::Ones(hiddenDim, inputDim) * 0.1);

        // hidden layers
        for (int i = 0; i < n_hidden_layers - 3; i++) {
            weights.push_back(Eigen::MatrixXd::Ones(hiddenDim, hiddenDim) * 0.1);
        }
        // last layer
        weights.push_back(Eigen::MatrixXd::Ones(outputDim, hiddenDim) * 0.1);

        if (linspace_weights) {
            double start = -1.0;
            double end = 1.0;
            for (int i = 0; i < weights.size(); i++) {
                set_weights_linspace(weights[i], start, end);
            }
        }
    }

    void set_weights_linspace(Eigen::MatrixXd &W, double start, double end) {
        int rows = W.rows();
        int cols = W.cols();
        double increment = (end - start) / ((rows * cols) - 1);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                W(i, j) = start + (i + j * cols) * increment;
            }
        }
    }

    std::vector<T> getUnpackedWeights() {
        std::vector<T> all_weights;

        for (Eigen::MatrixXd &weight : weights) {
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> transposed_weight =
                weight.transpose();
            all_weights.insert(all_weights.end(), transposed_weight.data(),
                               transposed_weight.data() + transposed_weight.size());
        }

        return all_weights;
    }

    std::vector<T> forward(Eigen::VectorXd x) {
        std::vector<Eigen::VectorXd> layer_outputs(weights.size());
        // Input -> First hidden layer
        layer_outputs[0] = (weights[0] * x).unaryExpr(&relu);
        // Hidden layers -> Output layer (execept last layer)
        for (int i = 1; i < weights.size() - 1; ++i) {
            layer_outputs[i] = (weights[i] * layer_outputs[i - 1]).unaryExpr(&relu);
        }
        // Last layer no activation function
        layer_outputs.back() = weights.back() * layer_outputs[weights.size() - 2];

        // // assuming all width in input, hidden and out are the same
        // Eigen::VectorXd all_layers = Eigen::VectorXd::Zero(layer_outputs[0].size() * weights.size());
        // for (int i = 0; i < weights.size(); i++) {
        //     all_layers.segment(i * layer_outputs[0].size(), layer_outputs[0].size()) = layer_outputs[i];
        // }

        return eigenToStdVector<T>(layer_outputs.back());
    }

    void backward(Eigen::VectorXd input, Eigen::VectorXd target, std::vector<std::vector<T>> &flattened_grads,
                  std::vector<std::vector<T>> &loss_grads) {
        std::vector<Eigen::VectorXd> A(n_hidden_layers);
        A[0] = input; // A represents each layer's data
        // Forward Pass
        for (int i = 0; i < n_hidden_layers - 1; i++) {

            Eigen::VectorXd Ci = weights[i] * A[i];

            if (i < n_hidden_layers - 2) {
                A[i + 1] = Ci.unaryExpr(&relu); // Within hidden layers, relu activation function is assumed.
            } else {
                A[i + 1] = Ci; // In the output layer, no activation function is assumed
            }
        }

        // Compute loss
        double loss = (A.back() - target).squaredNorm() / target.size();

        // Compute gradients
        std::vector<Eigen::VectorXd> D(n_hidden_layers);
        D.back() = 2 * (A.back() - target) / target.size();

        std::vector<Eigen::MatrixXd> G(n_hidden_layers - 1);
        G.back() = D.back() * A[n_hidden_layers - 2].transpose();

        // Backward pass
        for (int i = n_hidden_layers - 1; i > 1; i--) {
            D[i - 1] = weights[i - 1].transpose() * D[i];
            D[i - 1] = D[i - 1].array() * A[i - 1].unaryExpr(&drelu).array();

            G[i - 2] = D[i - 1] * A[i - 2].transpose();
        }

        for (int i = 0; i < G.size(); i++) {
            std::vector<T> grad_vector =
                eigenToStdVector<T>(Eigen::Map<Eigen::VectorXd>(G[i].data(), G[i].rows() * G[i].cols()));
            flattened_grads.push_back(grad_vector);
        }
        for (int i = 1; i < D.size(); i++) {
            loss_grads.push_back(eigenToStdVector<T>(D[i]));
        }
    }
};

template <typename T, int WIDTH>
void test_inference_1layer(sycl::queue &q, const int input_width, const int output_width, const int batch_size) {

    constexpr int n_hidden_layers = 1;
    constexpr float input_val = 1.0f;
    // setting Network<T>::WeightInitMode::constant_pos sets the weights to this value
    constexpr float weight_val = 0.01f;

    SwiftNetMLP<T, WIDTH> network(q, input_width, output_width, n_hidden_layers, Activation::ReLU, Activation::None,
                                  Network<T>::WeightInitMode::constant_pos);

    DeviceMatrix<T> network_output(batch_size, network.get_output_width(), q);
    DeviceMatrix<T> network_input(batch_size, network.get_input_width(), q);

    network_input.fill(input_val);

    network.inference(network_input, network_output, {});

    q.wait();

    std::vector<double> out_ref(network_output.size());
    for (int output_idx = 0; output_idx < out_ref.size(); output_idx++) {

        const int nonzero_value = (output_idx % network.get_output_width()) < output_width ? 1 : 0;
        out_ref[output_idx] =
            nonzero_value * weight_val * input_width * input_val * network.get_network_width() * weight_val;
    }
    CHECK(areVectorsWithinTolerance(network_output.copy_to_host(), out_ref, 1e-2));
}

template <typename T, int WIDTH>
void test_forward_1layer(sycl::queue &q, const int input_width, const int output_width, const int batch_size) {

    constexpr int n_hidden_layers = 1;
    constexpr float input_val = 1.0f;
    // setting Network<T>::WeightInitMode::constant_pos sets the weights to this value
    constexpr float weight_val = 0.01f;

    SwiftNetMLP<T, WIDTH> network(q, input_width, output_width, n_hidden_layers, Activation::ReLU, Activation::None,
                                  Network<T>::WeightInitMode::constant_pos);

    DeviceMatrices<T> network_interm_forw(network.get_n_hidden_layers() + 2, batch_size, network.get_input_width(),
                                          batch_size, network.get_network_width(), batch_size,
                                          network.get_output_width(), q);
    DeviceMatrix<T> network_input(batch_size, network.get_input_width(), q);
    network_input.fill(input_val);
    q.wait();

    network.forward_pass(network_input, network_interm_forw, {});

    q.wait();

    std::vector<T> fwd_host = network_interm_forw.copy_to_host();

    // input
    CHECK(isVectorWithinTolerance(
        std::vector<T>(fwd_host.begin(), fwd_host.begin() + batch_size * network.get_input_width()), input_val, 1e-3));

    // intermediate
    const double ref_result = weight_val * input_width * input_val;
    CHECK(isVectorWithinTolerance(
        std::vector<T>(fwd_host.begin() + batch_size * network.get_input_width(),
                       fwd_host.begin() + batch_size * (network.get_input_width() + network.get_network_width())),
        ref_result, 1e-3));

    // output
    for (int i = batch_size * (network.get_input_width() + network.get_network_width()); i < fwd_host.size(); i++) {
        const int output_idx = i - batch_size * (network.get_input_width() + network.get_network_width());
        const int nonzero_value = (output_idx % network.get_output_width()) < output_width ? 1 : 0;
        const double ref_result =
            nonzero_value * weight_val * input_width * input_val * network.get_network_width() * weight_val;
        CHECK(static_cast<double>(fwd_host[i]) == doctest::Approx(ref_result).epsilon(1e-2));
    }
}

// Function which runs forw+backward without loss
// and tests output from backward and intermediate backw result.
// Forward results are tested with other functions
template <typename T, int WIDTH>
void test_backward_1layer(sycl::queue &q, const int input_width, const int output_width, const int n_hidden_layers,
                          const int batch_size, bool linspace_weights) {
    const T input_val = 1.0f;
    Eigen::VectorXd input_ref = Eigen::VectorXd::Ones(input_width) * static_cast<float>(input_val);
    // Eigen::VectorXd input_ref = Eigen::VectorXd::LinSpaced(input_width, -0.5f, 1.0f); // go from 0 till N
    const T target_val = 0.1;

    Eigen::VectorXd target_ref = Eigen::VectorXd::Ones(output_width) * static_cast<float>(target_val);
    std::vector<float> input_vec = eigenToStdVector<float>(input_ref);

    CHECK(input_width == output_width); // this is not a hard requirement, but currently the loop over the
                                        // mlp reference (batch size = 1) assumes this. if this is changed, ensure the
                                        // checks are still correct
    CHECK(input_width == WIDTH);

    SwiftNetMLP<T, WIDTH> network(q, input_width, output_width, n_hidden_layers, Activation::ReLU, Activation::None,
                                  Network<T>::WeightInitMode::constant_pos);
    MLP<float> mlp(input_width, WIDTH, output_width, n_hidden_layers + 2, linspace_weights);
    std::vector<float> unpacked_weights_float = mlp.getUnpackedWeights();

    std::vector<T> unpacked_weights(unpacked_weights_float.size());

    std::transform(unpacked_weights_float.begin(), unpacked_weights_float.end(), unpacked_weights.begin(),
                   [](float val) { return static_cast<T>(val); });

    network.set_weights_matrices(
        io::get_packed_weights<T, WIDTH>(unpacked_weights, n_hidden_layers, input_width, output_width));

    DeviceMatrix<T> network_input(batch_size, input_width, q);
    DeviceMatrices<T> interm_forw(network.get_n_hidden_layers() + 2, batch_size, network.get_input_width(), batch_size,
                                  network.get_network_width(), batch_size, network.get_output_width(), q);
    DeviceMatrices<T> interm_backw(network.get_n_hidden_layers() + 1, batch_size, network.get_network_width(),
                                   batch_size, network.get_network_width(), batch_size, network.get_output_width(), q);
    DeviceMatrix<T> loss(batch_size, network.get_output_width(), q);
    DeviceMatrices<T> network_backward_output(network.get_n_hidden_layers() + 1, network.get_network_width(), WIDTH,
                                              WIDTH, WIDTH, WIDTH, network.get_output_width(), q);

    // network_input.fill(input_val).wait();
    std::vector<T> input_T(input_vec.begin(), input_vec.end());
    // Repeat source vector N times
    std::vector<T> input_full(batch_size * input_T.size());
    auto it = input_full.begin();
    for (int i = 0; i < batch_size; ++i) {
        it = std::copy(input_T.begin(), input_T.end(), it);
    }

    network_input.copy_from_host(input_full).wait();

    interm_forw.fill((T)0).wait();
    interm_backw.fill((T)0).wait();

    std::vector<sycl::event> es = network.forward_pass(network_input, interm_forw, {});
    q.wait();
    std::vector<T> interm_forw_vec = interm_forw.copy_to_host();
    std::vector<T> output_network(interm_forw_vec.end() - (batch_size * output_width), interm_forw_vec.end());

    std::vector<T> dL_doutput(loss.size());

    if (dL_doutput.size() != output_network.size()) {
        throw std::invalid_argument("dL_doutput.size() != output_network.size()");
    }
    for (size_t i = 0; i < output_network.size(); ++i) {
        dL_doutput[i] = 2 * ((output_network[i] - target_val) / output_width);
    }
    loss.copy_from_host(dL_doutput).wait(); // L2 loss
    q.wait();
    network.backward_pass(loss, network_backward_output, interm_backw, interm_forw, {});
    q.wait();

    std::vector<T> interm_backw_host = interm_backw.copy_to_host();
    std::vector<T> grad = network_backward_output.copy_to_host();
    q.wait();

    std::vector<float> fwd_result_ref = mlp.forward(input_ref);

    std::vector<std::vector<float>> weights_ref, loss_grads_ref;
    mlp.backward(input_ref, target_ref, weights_ref, loss_grads_ref);

    // TODO: Scale automatically by batch size
    for (T &element : grad) {
        element /= batch_size;
    }

    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        // note that mlp is only implemented for batch size = 1, thus checking
        // the outputs and grads against that (as the inputs are all the same
        // for all batch size, the grads and output are as well)
        CHECK(areVectorsWithinTolerance(std::vector<T>(interm_forw_vec.end() - WIDTH * batch_idx - WIDTH,
                                                       interm_forw_vec.end() - WIDTH * batch_idx),
                                        fwd_result_ref, 1.0e-2)); // comparing only output

        for (int idx = 0; idx < loss_grads_ref.size(); idx++) {
            CHECK(areVectorsWithinTolerance(
                std::vector<T>(interm_backw_host.begin() + batch_idx * WIDTH + idx * batch_size * WIDTH,
                               interm_backw_host.begin() + WIDTH + batch_idx * WIDTH + idx * batch_size * WIDTH),
                loss_grads_ref[idx], 1.0e-2));
            // here, we don't distinguish between WIDTH, input_width and
            // output_width.If the values are not the same, we need to separate
        }
        for (int i = 0; i < weights_ref.size(); i++) {
            CHECK(areVectorsWithinTolerance(
                std::vector<T>(grad.begin() + WIDTH * WIDTH * i, grad.begin() + WIDTH * WIDTH * i + WIDTH * WIDTH),
                weights_ref[i], 1.0e-2));
            // here, we don't distinguish between WIDTH, input_width and output_width. If the
            // values are not the same, we need to separate
        }
    }
}

// Function which runs forw+backward without loss
// and tests output from backward and intermediate backw result.
// Forward results are tested with other functions
template <typename T, int WIDTH>
void test_training_loaded_file(sycl::queue &q, const std::string filepath, const int input_width,
                               const int output_width, const int batch_size) {
    int HIDDEN_LAYERS = 2;
    SwiftNetMLP<T, WIDTH> network(q, input_width, output_width, HIDDEN_LAYERS, Activation::ReLU, Activation::None,
                                  Network<T>::WeightInitMode::constant_pos);

    std::vector<T> network_params = load_weights_as_packed_from_file<T, WIDTH>(
        filepath + "network_params.csv", HIDDEN_LAYERS, input_width, output_width);
    CHECK(network_params.size() == network.get_weights_matrices().nelements());
    network.set_weights_matrices(network_params);

    std::vector<T> input_ref = loadVectorFromCSV<T>(filepath + "input_network.csv");
    DeviceMatrix<T> network_input(batch_size, input_width, q);
    network_input.copy_from_host(input_ref).wait();

    DeviceMatrices<T> interm_forw(network.get_n_hidden_layers() + 2, batch_size, network.get_input_width(), batch_size,
                                  network.get_network_width(), batch_size, network.get_output_width(), q);
    DeviceMatrices<T> interm_backw(network.get_n_hidden_layers() + 1, batch_size, network.get_network_width(),
                                   batch_size, network.get_network_width(), batch_size, network.get_output_width(), q);
    DeviceMatrix<T> loss(batch_size, network.get_output_width(), q);
    DeviceMatrices<T> network_backward_output(network.get_n_hidden_layers() + 1, network.get_network_width(), WIDTH,
                                              WIDTH, WIDTH, WIDTH, network.get_output_width(), q);
    DeviceMatrix<T> network_output(batch_size, WIDTH, q);

    interm_forw.fill((T)0).wait();
    interm_backw.fill((T)0).wait();

    std::vector<sycl::event> es = network.forward_pass(network_input, interm_forw, {});
    q.wait();

    std::vector<T> interm_forw_vec = interm_forw.copy_to_host();
    std::vector<T> output_network(interm_forw_vec.end() - (batch_size * output_width), interm_forw_vec.end());

    std::vector<T> output_ref = loadVectorFromCSV<T>(filepath + "output_network.csv");

    CHECK(areVectorsWithinTolerance(output_network, output_ref, 1.0e-2));

    std::vector<T> loss_ref = loadVectorFromCSV<T>(filepath + "loss.csv");
    loss.copy_from_host(loss_ref).wait();

    network.backward_pass(loss, network_backward_output, interm_backw, interm_forw, {});

    q.wait();

    std::vector<T> grads = network_backward_output.copy_to_host();
    // TODO: Scale automatically by batch size
    for (T &element : grads) {
        element /= batch_size;
    }

    std::vector<T> grads_ref = loadVectorFromCSV<T>(filepath + "network_params_grad.csv");
    std::cout << "Grad size: " << grads.size() << ", ref: " << grads_ref.size() << std::endl;
    for (int i = 0; i < 64; i++) {
        std::cout << i << ": grad: " << grads[i] << " - ref:" << grads_ref[i] << std::endl;
    }

    CHECK(areVectorsWithinTolerance(grads, grads_ref, 1.0e-2));
}

TEST_CASE("Swiftnet - Constructor") {

    sycl::queue q;
    typedef sycl::ext::oneapi::bfloat16 T;

    // No need to test width template parameter since it is statically asserted in swiftnetmlp class
    // No need to test type template parameter since it is statically asserted in Network class
    SUBCASE("Supported 1") { CHECK_NOTHROW(SwiftNetMLP<T, 16>(q, 16, 16, 4, Activation::ReLU, Activation::None)); }
    SUBCASE("Supported 2") { CHECK_NOTHROW(SwiftNetMLP<T, 32>(q, 32, 32, 4, Activation::ReLU, Activation::None)); }
    SUBCASE("Supported 3") { CHECK_NOTHROW(SwiftNetMLP<T, 64>(q, 64, 64, 4, Activation::ReLU, Activation::None)); }
    SUBCASE("Supported 4") { CHECK_NOTHROW(SwiftNetMLP<T, 128>(q, 128, 128, 4, Activation::ReLU, Activation::None)); }
    SUBCASE("Pad input 1") { CHECK_NOTHROW(SwiftNetMLP<T, 64>(q, 16, 64, 4, Activation::ReLU, Activation::None)); }
    SUBCASE("Pad input 2") { CHECK_NOTHROW(SwiftNetMLP<T, 64>(q, 1, 64, 4, Activation::ReLU, Activation::None)); }
    SUBCASE("Pad output 1") { CHECK_NOTHROW(SwiftNetMLP<T, 64>(q, 64, 1, 4, Activation::ReLU, Activation::None)); }
    SUBCASE("Pad output 2") { CHECK_NOTHROW(SwiftNetMLP<T, 64>(q, 64, 16, 4, Activation::ReLU, Activation::None)); }
    SUBCASE("Unsupported layers 1") {
        CHECK_THROWS(SwiftNetMLP<T, 64>(q, 64, 64, 0, Activation::ReLU, Activation::None));
    }
    SUBCASE("Unsupported layers 2") {
        CHECK_THROWS(SwiftNetMLP<T, 64>(q, 64, 64, -1, Activation::ReLU, Activation::None));
    }

    SUBCASE("Unsupported input width 1") {
        CHECK_THROWS(SwiftNetMLP<T, 64>(q, -1, 64, 4, Activation::ReLU, Activation::None));
    }

    SUBCASE("Unsupported output width 1") {
        CHECK_THROWS(SwiftNetMLP<T, 64>(q, 64, -1, 4, Activation::ReLU, Activation::None));
    }
    SUBCASE("Unsupported activation 1") {
        CHECK_THROWS(SwiftNetMLP<T, 64>(q, 64, 64, 4, Activation::None, Activation::None));
    }
    SUBCASE("Unsupported activation 2") {
        CHECK_THROWS(SwiftNetMLP<T, 64>(q, 64, 64, 4, Activation::Tanh, Activation::None));
    }
    SUBCASE("Unsupported activation 3") {
        CHECK_THROWS(SwiftNetMLP<T, 64>(q, 64, 64, 4, Activation::Sigmoid, Activation::None));
    }
    SUBCASE("Unsupported output activation 1") {
        CHECK_THROWS(SwiftNetMLP<T, 64>(q, 64, 64, 4, Activation::ReLU, Activation::ReLU));
    }
    SUBCASE("Unsupported output activation 2") {
        CHECK_THROWS(SwiftNetMLP<T, 64>(q, 64, 64, 4, Activation::ReLU, Activation::Tanh));
    }
    SUBCASE("Unsupported output activation 3") {
        CHECK_THROWS(SwiftNetMLP<T, 64>(q, 64, 64, 4, Activation::ReLU, Activation::Sigmoid));
    }
}

/// TODO: check if the weights are actually 0 whereever they should be.
TEST_CASE("Swiftnet - Zero Padding") {
    sycl::queue q;
    typedef sycl::ext::oneapi::bfloat16 T;
    SUBCASE("Input 1-64") {
        SwiftNetMLP<T, 64> network(q, 1, 64, 4, Activation::ReLU, Activation::None);
        CHECK(network.get_input_width() == 64);
        CHECK(network.get_network_width() == 64);
        CHECK(network.get_output_width() == 64);
    }
    SUBCASE("Input 1-16") {
        SwiftNetMLP<T, 16> network(q, 1, 16, 4, Activation::ReLU, Activation::None);
        CHECK(network.get_input_width() == 16);
        CHECK(network.get_network_width() == 16);
        CHECK(network.get_output_width() == 16);
    }
    SUBCASE("Input 17-32") {
        SwiftNetMLP<T, 32> network(q, 17, 32, 4, Activation::ReLU, Activation::None);
        CHECK(network.get_input_width() == 32);
        CHECK(network.get_network_width() == 32);
        CHECK(network.get_output_width() == 32);
    }
    SUBCASE("Input 17-128") {
        SwiftNetMLP<T, 128> network(q, 17, 128, 4, Activation::ReLU, Activation::None);
        CHECK(network.get_input_width() == 128);
        CHECK(network.get_network_width() == 128);
        CHECK(network.get_output_width() == 128);
    }
    SUBCASE("Output 1-64") {
        SwiftNetMLP<T, 64> network(q, 64, 1, 4, Activation::ReLU, Activation::None);
        CHECK(network.get_input_width() == 64);
        CHECK(network.get_network_width() == 64);
        CHECK(network.get_output_width() == 64);
    }
    SUBCASE("Output 1-16") {
        SwiftNetMLP<T, 16> network(q, 16, 1, 4, Activation::ReLU, Activation::None);
        CHECK(network.get_input_width() == 16);
        CHECK(network.get_network_width() == 16);
        CHECK(network.get_output_width() == 16);
    }
    SUBCASE("Output 17-32") {
        SwiftNetMLP<T, 32> network(q, 32, 17, 4, Activation::ReLU, Activation::None);
        CHECK(network.get_input_width() == 32);
        CHECK(network.get_network_width() == 32);
        CHECK(network.get_output_width() == 32);
    }
    SUBCASE("Output 17-128") {
        SwiftNetMLP<T, 128> network(q, 128, 17, 4, Activation::ReLU, Activation::None);
        CHECK(network.get_input_width() == 128);
        CHECK(network.get_network_width() == 128);
        CHECK(network.get_output_width() == 128);
    }
}

TEST_CASE("Swiftnet - weights init") {
    sycl::queue q(sycl::gpu_selector_v);
    typedef sycl::ext::oneapi::bfloat16 T;

    SUBCASE("Default positive, No Pad") {
        SwiftNetMLP<T, 64> network(q, 64, 64, 4, Activation::ReLU, Activation::None,
                                   Network<T>::WeightInitMode::constant_pos);
        CHECK_NOTHROW(network.get_weights_matrices());
        CHECK(network.get_weights_matrices().GetNumberOfMatrices() == 5);
        for (int iter = 0; iter < 5; iter++) {
            CHECK(network.get_weights_matrices().GetView(iter).m() == 64);
            CHECK(network.get_weights_matrices().GetView(iter).n() == 64);
        }

        CHECK(areVectorsWithinTolerance(network.get_weights_matrices().copy_to_host(),
                                        std::vector<T>(network.get_weights_matrices().nelements(), 0.01), 1e-3));
    }

    SUBCASE("Default positive, Output Pad") {
        SwiftNetMLP<T, 64> network(q, 64, 63, 4, Activation::ReLU, Activation::None,
                                   Network<T>::WeightInitMode::constant_pos);
        CHECK_NOTHROW(network.get_weights_matrices());
        CHECK(network.get_weights_matrices().GetNumberOfMatrices() == 5);

        for (int iter = 0; iter < 4; iter++) {
            CHECK(network.get_weights_matrices().GetView(iter).m() == 64);
            CHECK(network.get_weights_matrices().GetView(iter).n() == 64);
        }
        CHECK(network.get_weights_matrices().Back().m() == 64);
        CHECK(network.get_weights_matrices().Back().n() == 64);
    }

    SUBCASE("Overwrite, No Pad") {
        SwiftNetMLP<T, 64> network(q, 64, 64, 4, Activation::ReLU, Activation::None,
                                   Network<T>::WeightInitMode::constant_pos);
        CHECK_NOTHROW(network.get_weights_matrices());
        CHECK(network.get_weights_matrices().GetNumberOfMatrices() == 5);
        std::vector<T> new_weights(network.get_weights_matrices().nelements(), 1.23);
        network.set_weights_matrices(new_weights);
        for (int iter = 0; iter < 5; iter++) {
            CHECK(network.get_weights_matrices().GetView(iter).m() == 64);
            CHECK(network.get_weights_matrices().GetView(iter).n() == 64);
        }

        CHECK(areVectorsWithinTolerance(network.get_weights_matrices().copy_to_host(),
                                        std::vector<T>(network.get_weights_matrices().nelements(), 1.23), 1e-3));
    }
}

TEST_CASE("Swiftnet - zero pad forward_pass WIDTH 64") {
    sycl::queue q(sycl::gpu_selector_v);

    auto test_function = [=](const int input_width, const int output_width, sycl::queue &q) {
        typedef sycl::ext::oneapi::bfloat16 T;
        constexpr int WIDTH = 64;
        test_forward_1layer<T, WIDTH>(q, input_width, output_width, 8);
    };

    SUBCASE("No Pad") {
        constexpr int input_width = 64;
        constexpr int output_width = 64;
        test_function(input_width, output_width, q);
    }
    SUBCASE("Input Pad") {
        constexpr int input_width = 3;
        constexpr int output_width = 64;
        test_function(input_width, output_width, q);
    }
    SUBCASE("Output Pad") {
        constexpr int input_width = 64;
        constexpr int output_width = 7;
        test_function(input_width, output_width, q);
    }
    SUBCASE("Input and Output Pad") {
        constexpr int input_width = 3;
        constexpr int output_width = 5;
        test_function(input_width, output_width, q);
    }
}

TEST_CASE("Swiftnet - zero pad inference WIDTH 64") {
    sycl::queue q(sycl::gpu_selector_v);

    auto test_function = [=](const int input_width, const int output_width, sycl::queue &q) {
        typedef sycl::ext::oneapi::bfloat16 T;
        constexpr int WIDTH = 64;
        test_inference_1layer<T, WIDTH>(q, input_width, output_width, 8);
    };

    SUBCASE("No Pad") {
        constexpr int input_width = 64;
        constexpr int output_width = 64;
        test_function(input_width, output_width, q);
    }
    SUBCASE("Input Pad") {
        constexpr int input_width = 3;
        constexpr int output_width = 64;
        test_function(input_width, output_width, q);
    }
    SUBCASE("Output Pad") {
        constexpr int input_width = 64;
        constexpr int output_width = 7;
        test_function(input_width, output_width, q);
    }
    SUBCASE("Input and Output Pad") {
        constexpr int input_width = 3;
        constexpr int output_width = 5;
        test_function(input_width, output_width, q);
    }
}

TEST_CASE("Swiftnet - Batch Sizes forward") {
    sycl::queue q(sycl::gpu_selector_v);

    auto test_function = [=](const int batch_size, sycl::queue &q) {
        typedef sycl::ext::oneapi::bfloat16 T;
        constexpr int WIDTH = 64;
        test_forward_1layer<T, WIDTH>(q, WIDTH, WIDTH, batch_size);
    };

    SUBCASE("Batch size 8") { CHECK_NOTHROW(test_function(8, q)); }
    SUBCASE("Batch size 512") { CHECK_NOTHROW(test_function(512, q)); }
    SUBCASE("Batch size 16") { CHECK_NOTHROW(test_function(16, q)); }
    SUBCASE("Batch size 1") { CHECK_THROWS(test_function(1, q)); }
    SUBCASE("Batch size 13") { CHECK_THROWS(test_function(13, q)); }
}

TEST_CASE("Swiftnet - Net Widths forward") {
    // only testing constructor. values tested later
    sycl::queue q(sycl::gpu_selector_v);

    auto test_function = [=](const int width, sycl::queue &q) {
        typedef sycl::ext::oneapi::bfloat16 T;
        if (width == 16)
            test_forward_1layer<T, 16>(q, 16, 16, 8);
        else if (width == 32)
            test_forward_1layer<T, 32>(q, 32, 32, 8);
        else if (width == 64)
            test_forward_1layer<T, 64>(q, 64, 64, 8);
        else if (width == 128)
            test_forward_1layer<T, 128>(q, 128, 128, 8);
        else
            throw std::invalid_argument("Unsupported width");
    };

    SUBCASE("WIDTH 16") { CHECK_NOTHROW(test_function(16, q)); }
    SUBCASE("WIDTH 32") { CHECK_NOTHROW(test_function(32, q)); }
    SUBCASE("WIDTH 64") { CHECK_NOTHROW(test_function(64, q)); }
    SUBCASE("WIDTH 128") { CHECK_NOTHROW(test_function(128, q)); }
}

#ifdef TEST_BWD
TEST_CASE("Swiftnet - backward different widths") {
    // only testing constructor. values tested later
    sycl::queue q(sycl::gpu_selector_v);
    const int n_hidden_layers = 1;
    auto test_function = [=](const int width, sycl::queue &q, bool linspace_weights) {
        typedef sycl::ext::oneapi::bfloat16 T;
        if (width == 16)
            test_backward_1layer<T, 16>(q, 16, 16, n_hidden_layers, 8, linspace_weights);
        else if (width == 32)
            test_backward_1layer<T, 32>(q, 32, 32, n_hidden_layers, 8, linspace_weights);
        else if (width == 64)
            test_backward_1layer<T, 64>(q, 64, 64, n_hidden_layers, 8, linspace_weights);
        else if (width == 128)
            test_backward_1layer<T, 128>(q, 128, 128, n_hidden_layers, 8, linspace_weights);
        else
            throw std::invalid_argument("Unsupported width");
    };

    // SUBCASE("WIDTH 16") { CHECK_NOTHROW(test_function(16, q, true)); }
    SUBCASE("WIDTH 16") { CHECK_NOTHROW(test_function(16, q, false)); }
    // SUBCASE("WIDTH 32") { CHECK_NOTHROW(test_function(32, q)); }
    // SUBCASE("WIDTH 64") { CHECK_NOTHROW(test_function(64, q)); }
    // SUBCASE("WIDTH 128") { CHECK_NOTHROW(test_function(128, q)); }
}

TEST_CASE("Swiftnet - backward different batch_size") {
    // only testing constructor. values tested later
    sycl::queue q(sycl::gpu_selector_v);
    const int n_hidden_layers = 1;

    auto test_function = [=](const int width, sycl::queue &q, bool linspace_weights) {
        typedef sycl::ext::oneapi::bfloat16 T;
        if (width == 16)
            test_backward_1layer<T, 16>(q, 16, 16, n_hidden_layers, 128, linspace_weights);
        else if (width == 32)
            test_backward_1layer<T, 32>(q, 32, 32, n_hidden_layers, 64, linspace_weights);
        else if (width == 64)
            test_backward_1layer<T, 64>(q, 64, 64, n_hidden_layers, 32, linspace_weights);
        else if (width == 128)
            test_backward_1layer<T, 128>(q, 128, 128, n_hidden_layers, 16, linspace_weights);
        else
            throw std::invalid_argument("Unsupported width");
    };

    SUBCASE("WIDTH 16") { CHECK_NOTHROW(test_function(16, q, false)); }
    // SUBCASE("WIDTH 32") { CHECK_NOTHROW(test_function(32, q)); }
    // SUBCASE("WIDTH 64") { CHECK_NOTHROW(test_function(64, q)); }
    // SUBCASE("WIDTH 128") { CHECK_NOTHROW(test_function(128, q)); }
}
#endif
