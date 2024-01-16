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
    std::vector<sycl::event> forward_pass(const DeviceMatrix<T> &input, DeviceMatrix<T> &intermediate_output_forward,
                                          const std::vector<sycl::event> &deps) override {
        SanityCheckForward(input, intermediate_output_forward);
        const size_t batch_size = input.m();

        // input = batch_size * input_width 1) W = input_width*WIDTH, (n_hidden_layer-1)*WIDTH*WIDTH, WIDTH*output_width
        // intermediate_output_forward = batch_size * (input_width + n_hidden_layer*WIDTH + output_width)

        // Perform forward pass based on activation function
        switch (m_activation) {
        case Activation::None:
            return tinydpcppnn::kernels::esimd::forward_impl_general<T, CType, WIDTH, WIDTH, WIDTH, Activation::None,
                                                                     Activation::None, false, 16>(
                Network<T>::get_queue(), Network<T>::get_weights_matrices().GetViews(),
                input.GetView() /*input.data()*/, intermediate_output_forward.data(), Network<T>::get_n_hidden_layers(),
                deps);
            break;
        case Activation::ReLU:
            return tinydpcppnn::kernels::esimd::forward_impl_general<T, CType, WIDTH, WIDTH, WIDTH, Activation::ReLU,
                                                                     Activation::None, false, 16>(
                Network<T>::get_queue(), Network<T>::get_weights_matrices().GetViews(),
                input.GetView() /*input.data()*/, intermediate_output_forward.data(), Network<T>::get_n_hidden_layers(),
                deps);
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
    std::vector<sycl::event> inference(const DeviceMatrix<T> &input, DeviceMatrix<T> &output,
                                       const std::vector<sycl::event> &deps) override {
        SanityCheckInference(input, output);
        const size_t batch_size = input.m();

        switch (m_activation) {
        case Activation::None:
            return tinydpcppnn::kernels::esimd::forward_impl_general<T, CType, WIDTH, WIDTH, WIDTH, Activation::None,
                                                                     Activation::None, true, 16>(
                Network<T>::get_queue(), Network<T>::get_weights_matrices().GetViews(),
                input.GetView() /*input.data()*/, output.data(), Network<T>::get_n_hidden_layers(), deps);
            break;
        case Activation::ReLU:
            return tinydpcppnn::kernels::esimd::forward_impl_general<T, CType, WIDTH, WIDTH, WIDTH, Activation::ReLU,
                                                                     Activation::None, true, 16>(
                Network<T>::get_queue(), Network<T>::get_weights_matrices().GetViews(),
                input.GetView() /*input.data()*/, output.data(), Network<T>::get_n_hidden_layers(), deps);
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
    std::vector<sycl::event> backward_pass(const DeviceMatrix<T> &input, DeviceMatrix<T> &output,
                                           DeviceMatrix<T> &intermediate_output_backward,
                                           const DeviceMatrix<T> &intermediate_output_forward,
                                           const std::vector<sycl::event> &deps) override {
        SanityCheckBackward(input, output, intermediate_output_backward, intermediate_output_forward);
        const size_t batch_size = input.m();

        // Choose appropriate mlp_swiftnet_backward based on activation
        // We are onyl doinh output_activation==none right now
        switch (m_activation) {
        case Activation::None:
            /*return tinydpcppnn::kernels::esimd::backward_impl_general<T, CType, WIDTH, WIDTH, WIDTH, Activation::None,
                                                                      Activation::None, 16>(
                Network<T>::get_queue(), Network<T>::get_weightsT_matrices().data(), input.data(), output.data(),
                intermediate_output_backward.data(), intermediate_output_forward.data(),
                Network<T>::get_n_hidden_layers(), batch_size, deps);*/
            break;
        case Activation::ReLU:
            /*return tinydpcppnn::kernels::esimd::backward_impl_general<T, CType, WIDTH, WIDTH, WIDTH, Activation::ReLU,
                                                                      Activation::None, 16>(
                Network<T>::get_queue(), Network<T>::get_weightsT_matrices().data(), input.data(), output.data(),
                intermediate_output_backward.data(), intermediate_output_forward.data(),
                Network<T>::get_n_hidden_layers(), batch_size, deps);*/

        default:
            throw std::invalid_argument("Activation not yet implemented in backward_pass");
        }
    }

  private:
    class KernelParams {
        friend class hash;

      public:
        KernelParams(const int input_width, const int output_width, const Activation act, const Activation bw_act,
                     const int TN)
            : input_width_(input_width), output_width_(output_width), act_(act), backw_act_(bw_act), TN_(TN) {}

        bool operator==(const KernelParams &rhs) const {
            return (input_width_ == rhs.input_width_) && (output_width_ == rhs.output_width_) && (act_ == rhs.act_) &&
                   (backw_act_ == rhs.backw_act_) && (TN_ == rhs.TN_);
        }

      private:
        int input_width_;
        int output_width_;
        Activation act_;
        Activation backw_act_;
        int TN_;
        // TODO: add device (GPU, CPU, etc.), and kernel type (SYCL, ESIMD, CPU)

        static constexpr std::array<int, 1> possible_io_widths = {WIDTH};
        static constexpr std::array<Activation, 2> possible_activations = {Activation::ReLU, Activation::None};
        static constexpr std::array<Activation, 1> possible_bw_activations = {Activation::None};
        static constexpr std::array<int, 2> possible_TN = {8, 16};
    };

    struct hash {
        std::size_t operator()(const KernelParams &p) const {
            size_t h = std::hash<int>{}(p.input_width_);
            h = hash_combine(h, std::hash<int>{}(p.output_width_));
            h = hash_combine(h, std::hash<int>{}(p.act_));
            h = hash_combine(h, std::hash<int>{}(p.backw_act_));
            return hash_combine(h, std::hash<int>{}(p.TN_));
        }

      private:
        // as in boost::hash_combine
        static inline size_t hash_combine(const size_t h1, const size_t h2) {
            return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
        }
    };

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

    void SanityCheckForward(const DeviceMatrix<T> &input, DeviceMatrix<T> &intermediate_output_forward) const {
        // Static assertion and assertion checks
        if ((input.m() % 8) != 0) throw std::invalid_argument("Batch size is not divisible by 8.");
        if (input.n() < Network<T>::get_input_width()) throw std::invalid_argument("Input array too small");
        if (intermediate_output_forward.m() != input.m() ||
            intermediate_output_forward.n() < (Network<T>::get_input_width() + Network<T>::get_output_width() +
                                               WIDTH * Network<T>::get_n_hidden_layers()))
            throw std::invalid_argument("Output array too small");
    }

    void SanityCheckBackward(const DeviceMatrix<T> &input, DeviceMatrix<T> &output,
                             DeviceMatrix<T> &intermediate_output_backward,
                             const DeviceMatrix<T> &intermediate_output_forward) const {
        if ((input.m() % 8) != 0) throw std::invalid_argument("Batch size is not divisible by 8.");
        if (input.m() != input.m() || input.n() < Network<T>::get_output_width())
            throw std::invalid_argument("Input array in backward pass too small");
        if (intermediate_output_forward.m() != input.m() ||
            intermediate_output_forward.n() < (Network<T>::get_input_width() + Network<T>::get_output_width() +
                                               WIDTH * Network<T>::get_n_hidden_layers()))
            throw std::invalid_argument("intermediate_output_forward array too small");
        if (intermediate_output_backward.m() != input.m() ||
            intermediate_output_backward.n() <
                (Network<T>::get_output_width() + WIDTH * Network<T>::get_n_hidden_layers()))
            throw std::invalid_argument("intermediate_output_backward array too small");
        if (output.m() < WIDTH || output.n() < (Network<T>::get_n_hidden_matrices() * WIDTH +
                                                Network<T>::get_input_width() + Network<T>::get_output_width()))
            throw std::invalid_argument("Output of backward pass too small.");
    }

    Activation m_activation;
    Activation m_output_activation;
    // std::unordered_map<KernelParams, Kernels<T>, hash> kernels_; // store all kernel functions

    using CType = typename tinydpcppnn::kernels::esimd::helpers::XMXCType<T>::CType;
};
