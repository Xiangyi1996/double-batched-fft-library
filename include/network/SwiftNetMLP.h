/**
 * @file SwiftNetMLP.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Implementation of a fused MLP class.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <CL/sycl.hpp>
#include <functional>
#include <iostream>
#include <json/json.hpp>
#include <unordered_map>
#include <vector>

#include "DeviceMatrix.h"
#include "Network.h"
#include "common.h"
// #include "kernel.h"
#include "kernel_esimd.h"

template <typename T, int WIDTH> class SwiftNetMLP : public Network<T> {
  public:
    /**
     * Constructor for the SwiftNetMLP class.
     *
     * @param q                  SYCL queue for command submission.
     * @param input_width        Width of the input data.
     * @param output_width       Width of the output data.
     * @param n_hidden_layers    Number of hidden layers.
     * @param activation         Activation function for hidden layers.
     * @param output_activation  Activation function for the output layer.
     * @tparam WIDTH             Width of the matrices.
     */
    SwiftNetMLP(sycl::queue &q, const int input_width, const int output_width, const int n_hidden_layers,
                Activation activation, Activation output_activation,
                const Network<T>::WeightInitMode mode = Network<T>::WeightInitMode::none)
        : Network<T>(q, n_hidden_layers, input_width, WIDTH, output_width, mode), m_activation{activation},
          m_output_activation{output_activation}, kernels_(GenerateKernels()) {

        SanityCheck();
    }

    ~SwiftNetMLP() {}

    /**
     * Perform a forward pass of the SwiftNetMLP model.
     *
     * @param input The input data on the device.
     * @param forward Pointer to the forward intermediate array.
     * The output is stored at the end of the array 'forward'
     */
    std::vector<sycl::event> forward_pass(const DeviceMatrix<T> &input, DeviceMatrices<T> &intermediate_output_forward,
                                          const std::vector<sycl::event> &deps) override {
        SanityCheckForward(input, intermediate_output_forward);

        // Perform forward pass based on activation function

        return kernels_->forward_impl(Network<T>::get_queue(), Network<T>::get_weights_matrices().GetViews(),
                                      input.GetView(), intermediate_output_forward.GetViews(),
                                      Network<T>::get_n_hidden_layers(), deps);
    }

    /**
     * Perform a forward pass of the SwiftNetMLP model.
     *
     * @param input The input data on the device.
     * @param forward Pointer to the forward intermediate array. In inference this is not used for intermediate values.
     * The output is stored at the end of the array 'forward'
     */
    std::vector<sycl::event> inference(const DeviceMatrix<T> &input, DeviceMatrix<T> &output,
                                       const std::vector<sycl::event> &deps) override {
        SanityCheckInference(input, output);
        return kernels_->inference_impl(Network<T>::get_queue(), Network<T>::get_weights_matrices().GetViews(),
                                        input.GetView(), output.GetViews(), Network<T>::get_n_hidden_layers(), deps);
    }

    /**
     * Perform the backward pass of the neural network.
     *
     * @param grads The gradients on the device. Input for the backward pass
     * @param out_inter Intermediate array for storing outputs. This is filled as part of the backward pass
     * @param forward Pointer to the forward intermediate array which was filled in the forw pass
     */
    std::vector<sycl::event> backward_pass(const DeviceMatrix<T> &input, DeviceMatrices<T> &output,
                                           DeviceMatrices<T> &intermediate_output_backward,
                                           const DeviceMatrices<T> &intermediate_output_forward,
                                           const std::vector<sycl::event> &deps) override {
        SanityCheckBackward(input, output, intermediate_output_backward, intermediate_output_forward);

        return kernels_->backward_impl(Network<T>::get_queue(), Network<T>::get_weightsT_matrices().GetViews(),
                                       input.GetView(), output.GetViews(), intermediate_output_backward.GetViews(),
                                       intermediate_output_forward.GetViews(), Network<T>::get_n_hidden_layers(), deps);
    }

  private:
    /// Generate the relevant kernel class. Has to be called in constructor
    std::unique_ptr<Kernels<T>> GenerateKernels() {
        int tn = 16; // check based on m_q
        // TODO: make this nice. TN value should be part of the
        if (Network<T>::get_queue().get_device().template get_info<sycl::info::device::vendor>().find("Intel") !=
                std::string::npos &&
            Network<T>::get_queue().get_device().is_gpu()) {
            if (Network<T>::get_queue().get_device().template get_info<sycl::info::device::name>().find(
                    "Data Center GPU") != std::string::npos)
                tn = 16;
            else if (Network<T>::get_queue().get_device().template get_info<sycl::info::device::name>() == "DG2")
                tn = 8;
            else
                throw std::invalid_argument("Only runs on PVC and DG");
        } else
            throw std::invalid_argument("Cannot run on anything else but intel gpus");

        // check which createKernels function we should call based on the device.
        switch (tn) {
        case 16:
            return tinydpcppnn::kernels::esimd::createKernels<T, WIDTH, 16>(
                Network<T>::get_input_width(), Network<T>::get_output_width(), m_activation, m_output_activation);
            break;
        case 8:
            throw std::invalid_argument("Cannot run on devices with TN==8");
            // return tinydpcppnn::kernels::esimd::createKernels<T, WIDTH, 8>(
            //     Network<T>::get_input_width(), Network<T>::get_output_width(), m_activation, m_output_activation);
            break;
        default:
            throw std::invalid_argument("TN != {8,16}");
        }
    }

    // TODO: does this have to be virtual?
    virtual void SanityCheck() const override {
        static_assert(WIDTH == 16 || WIDTH == 32 || WIDTH == 64 || WIDTH == 128);
        static_assert(std::is_same<T, sycl::ext::oneapi::bfloat16>::value || std::is_same<T, sycl::half>::value);

        if (m_activation != Activation::ReLU) {
            throw std::runtime_error("m_activation must be ReLU for now.");
        }
        if (m_output_activation != Activation::None) {
            throw std::runtime_error("m_output_activation must be None for now.");
        }
    }

    void SanityCheckInference(const DeviceMatrix<T> &input, DeviceMatrix<T> &output) const {
        if ((input.m() % 8) != 0) throw std::invalid_argument("Batch size is not divisible by 8.");
        if (input.n() < Network<T>::get_input_width()) throw std::invalid_argument("Input array too small");
        if (output.m() != input.m() || output.n() < Network<T>::get_output_width())
            throw std::invalid_argument("Output array too small");
    }

    void SanityCheckForward(const DeviceMatrix<T> &input, DeviceMatrices<T> &intermediate_output_forward) const {
        // Static assertion and assertion checks
        if ((input.m() % 8) != 0) throw std::invalid_argument("Batch size is not divisible by 8.");
        if (input.n() != Network<T>::get_input_width())
            throw std::invalid_argument("Input array too small. Input n: " + std::to_string(input.n()) +
                                        " != input width " + std::to_string(Network<T>::get_input_width()));
        if (intermediate_output_forward.input_m() != input.m() || intermediate_output_forward.input_n() != input.n())
            throw std::invalid_argument("intermediate_output_forward array too small for input");
        if (intermediate_output_forward.middle_m() != input.m() || intermediate_output_forward.middle_n() != WIDTH)
            throw std::invalid_argument("intermediate_output_forward array too small for middle results");
        if (intermediate_output_forward.output_m() != input.m() ||
            intermediate_output_forward.output_n() != Network<T>::get_output_width())
            throw std::invalid_argument("intermediate_output_forward array too small for output");
        if (intermediate_output_forward.GetNumberOfMatrices() != Network<T>::get_n_hidden_layers() + 2)
            throw std::invalid_argument("not enough matrices in intermediate_output_forward array");
    }

    void SanityCheckBackward(const DeviceMatrix<T> &input, DeviceMatrices<T> &output,
                             DeviceMatrices<T> &intermediate_output_backward,
                             const DeviceMatrices<T> &intermediate_output_forward) const {
        if ((input.m() % 8) != 0) throw std::invalid_argument("Batch size is not divisible by 8.");
        if (input.m() != intermediate_output_forward.input_m() || input.n() != Network<T>::get_output_width())
            throw std::invalid_argument("Input array in backward pass too small");

        if (intermediate_output_forward.input_m() != input.m() || intermediate_output_forward.middle_m() != input.m() ||
            intermediate_output_forward.output_m() != input.m())
            throw std::invalid_argument("intermediate_output_forward m too small");
        if (intermediate_output_forward.input_n() != Network<T>::get_input_width() ||
            intermediate_output_forward.middle_n() != WIDTH ||
            intermediate_output_forward.output_n() != Network<T>::get_output_width())
            throw std::invalid_argument("intermediate_output_forward n too small");
        if (intermediate_output_forward.GetNumberOfMatrices() != Network<T>::get_n_hidden_layers() + 2)
            throw std::invalid_argument("intermediate_output_forward not enough matrices");

        if (intermediate_output_backward.input_m() != input.m() ||
            intermediate_output_backward.middle_m() != input.m() ||
            intermediate_output_backward.output_m() != input.m())
            throw std::invalid_argument("intermediate_output_backward m too small");
        if (intermediate_output_backward.input_n() != WIDTH || intermediate_output_backward.middle_n() != WIDTH ||
            intermediate_output_backward.output_n() != Network<T>::get_output_width())
            throw std::invalid_argument("intermediate_output_backward n too small");
        if (intermediate_output_backward.GetNumberOfMatrices() != Network<T>::get_n_hidden_layers() + 1)
            throw std::invalid_argument("intermediate_output_backward not enough matrices");

        if (output.input_m() != Network<T>::get_input_width() || output.input_n() != WIDTH)
            throw std::invalid_argument("Output of backward pass too small for input.");
        if (output.middle_m() != WIDTH || output.middle_n() != WIDTH)
            throw std::invalid_argument("Output of backward pass too small for middle.");
        if (output.output_m() != WIDTH || output.output_n() != Network<T>::get_output_width())
            throw std::invalid_argument("Output of backward pass too small for output.");
    }

    Activation m_activation;
    Activation m_output_activation;
    std::unique_ptr<Kernels<T>> kernels_;
};
