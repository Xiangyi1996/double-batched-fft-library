#ifndef SWIFTNET_H
#define SWIFTNET_H

#include <CL/sycl.hpp>
#include <iostream>
#include <json/json.hpp>
#include <vector>

#include "DeviceMem.h"
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
                Activation activation, Activation output_activation)
        : Network<T>(q, n_hidden_layers, input_width, WIDTH, output_width), m_activation{activation},
          m_output_activation{output_activation} {

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
    std::vector<sycl::event> forward_pass(const DeviceMem<T> &input, DeviceMem<T> &output, const size_t batch_size,
                                          const std::vector<sycl::event> &deps) override {
        // Static assertion and assertion checks
        if ((batch_size % 8) != 0) throw std::invalid_argument("Batch size is not divisible by 8.");

        // Perform forward pass based on activation function
        switch (m_activation) {
        case Activation::None:
            return tinydpcppnn::kernels::esimd::forward_impl_general<T, CType, WIDTH, WIDTH, WIDTH, Activation::None,
                                                                     Activation::None, false, 16>(
                Network<T>::get_queue(), Network<T>::get_weights_matrices().data(), input.data(), output.data(),
                Network<T>::get_n_hidden_layers(), batch_size, deps);
            break;
        case Activation::ReLU:
            return tinydpcppnn::kernels::esimd::forward_impl_general<T, CType, WIDTH, WIDTH, WIDTH, Activation::ReLU,
                                                                     Activation::None, false, 16>(
                Network<T>::get_queue(), Network<T>::get_weights_matrices().data(), input.data(), output.data(),
                Network<T>::get_n_hidden_layers(), batch_size, deps);
            break;
        default:
            throw std::invalid_argument("Activation not supported in forward pass");
        }
    }

    /**
     * Perform a forward pass of the SwiftNetMLP model.
     *
     * @param input The input data on the device.
     * @param forward Pointer to the forward intermediate array. In inference this is not used for intermediate values.
     * The output is stored at the end of the array 'forward'
     */
    std::vector<sycl::event> inference(const DeviceMem<T> &input, DeviceMem<T> &intermediate_output_forward,
                                       const size_t batch_size, const std::vector<sycl::event> &deps) override {
        if ((batch_size % 8) != 0) throw std::invalid_argument("Batch size is not divisible by 8.");

        switch (m_activation) {
        case Activation::None:
            return tinydpcppnn::kernels::esimd::forward_impl_general<T, CType, WIDTH, WIDTH, WIDTH, Activation::None,
                                                                     Activation::None, true, 16>(
                Network<T>::get_queue(), Network<T>::get_weights_matrices().data(), input.data(),
                intermediate_output_forward.data(), Network<T>::get_n_hidden_layers(), batch_size, deps);
            break;
        case Activation::ReLU:
            return tinydpcppnn::kernels::esimd::forward_impl_general<T, CType, WIDTH, WIDTH, WIDTH, Activation::ReLU,
                                                                     Activation::None, true, 16>(
                Network<T>::get_queue(), Network<T>::get_weights_matrices().data(), input.data(),
                intermediate_output_forward.data(), Network<T>::get_n_hidden_layers(), batch_size, deps);
            break;
        default:
            throw std::runtime_error{"Unsupported activation."};
        }
    }

    /**
     * Perform the backward pass of the neural network.
     *
     * @param grads The gradients on the device. Input for the backward pass
     * @param out_inter Intermediate array for storing outputs. This is filled as part of the backward pass
     * @param forward Pointer to the forward intermediate array which was filled in the forw pass
     */
    std::vector<sycl::event> backward_pass(const DeviceMem<T> &input, DeviceMem<T> &output, DeviceMem<T> &intermediate_output_backward,
                                           const DeviceMem<T> &intermediate_output_forward, const size_t batch_size,
                                           const std::vector<sycl::event> &deps) override {
        if ((batch_size % 8) != 0) throw std::invalid_argument("Batch size is not divisible by 8.");

        // Choose appropriate mlp_swiftnet_backward based on activation
        // We are onyl doinh output_activation==none right now
        switch (m_activation) {
        case Activation::None:
            return tinydpcppnn::kernels::esimd::backward_impl_general<T, CType, WIDTH, WIDTH, WIDTH, Activation::None,
                                                                      Activation::None, 16>(
                Network<T>::get_queue(), Network<T>::get_weightsT_matrices().data(), input.data(),
                output.data(), intermediate_output_backward.data(),
                intermediate_output_forward.data(), Network<T>::get_n_hidden_layers(), batch_size, deps);
            break;
        case Activation::ReLU:
            return tinydpcppnn::kernels::esimd::backward_impl_general<T, CType, WIDTH, WIDTH, WIDTH, Activation::ReLU,
                                                                      Activation::None, 16>(
                Network<T>::get_queue(), Network<T>::get_weightsT_matrices().data(), input.data(),
                output.data(), intermediate_output_backward.data(),
                intermediate_output_forward.data(), Network<T>::get_n_hidden_layers(), batch_size, deps);

        default:
            throw std::invalid_argument("Activation not yet implemented in backward_pass");
        }
    }

    std::vector<sycl::event> training(const DeviceMem<T> &input, const DeviceMem<T> &target,
                                      DeviceMem<T> &intermediate_output_backward,
                                      DeviceMem<T> &intermediate_output_forward, const size_t batch_size,
                                      const std::vector<sycl::event> &deps) override {
        throw std::logic_error("Fused not yet implemented.");
    }

  private:
    virtual void SanityCheck() const override {
        static_assert(WIDTH == 16 || WIDTH == 32 || WIDTH == 64 || WIDTH == 128);

        if (m_activation != Activation::ReLU) {
            throw std::runtime_error("m_activation must be ReLU for now.");
        }
        if (m_output_activation != Activation::None) {
            throw std::runtime_error("m_output_activation must be None for now.");
        }
    }

    Activation m_activation;
    Activation m_output_activation;

    using CType = typename tinydpcppnn::kernels::esimd::helpers::XMXCType<T>::CType;
};

#endif
