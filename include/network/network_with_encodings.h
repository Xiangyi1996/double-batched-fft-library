/**
 * @file network_with_encodings.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Implementation of a network with an encoding.
 * TODO: somehow consolidate this as a type of network. Requires to rethink our network class.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "Network.h"
#include "SwiftNetMLP.h"
#include "encoding_factory.h"

template <typename SrcType, typename DstType>
DeviceMatrix<DstType> convert_matrix(const DeviceMatrix<SrcType> &src, sycl::queue &q) {
    // Create a new DeviceMatrix of the destination type with the
    // same dimensions as the source matrix.
    DeviceMatrix<DstType> dest(src.rows(), src.cols(), q);

    // Get the pointers to the underlying data of the source and destination matrices.
    SrcType const *src_data = src.data();
    DstType *dest_data = dest.data();

    // Compute the number of elements to convert.
    size_t num_elements = src.size();

    // Launch the SYCL kernel to perform the conversion.
    q.parallel_for(num_elements, [=](sycl::id<1> idx) {
         dest_data[idx] = static_cast<DstType>(src_data[idx]); // Conversion from SrcType to DstType
     }).wait(); // Wait for the kernel to complete before returning the new matrix.

    return dest;
}

template <typename T_enc, typename T_net> class NetworkWithEncoding {
  public:
    NetworkWithEncoding() = delete;

    NetworkWithEncoding(std::shared_ptr<Encoding<T_enc>> encoding, std::shared_ptr<Network<T_net>> network)
        : encoding_(encoding), network_(network) {
        SanityCheck();
    }

    ~NetworkWithEncoding() {}

    std::vector<sycl::event> inference(const DeviceMatrix<float> &input, DeviceMatrix<T_net> &network_input,
                                       DeviceMatrix<T_enc> &encoding_output, DeviceMatrix<T_net> &network_output,
                                       const std::vector<sycl::event> &deps) {
        /// TODO: implemente proper usage of deps. Requires proper implementation of forward_impl
        /// in encodings which takes it as input and returns new dependencies.

        const int batch_size = input.m();
        const int network_input_width = network_->get_input_width();

        if (network_output.m() != batch_size || encoding_output.m() != batch_size)
            throw std::invalid_argument("Wrong dimensions.");
        if (encoding_output.n() != network_input_width) throw std::invalid_argument("Wrong dimensions.");
        if (network_output.n() != network_->get_output_width()) throw std::invalid_argument("Wrong dimensions.");

        auto ctxt = encoding_->forward_impl(&network_->get_queue(), input, &encoding_output);
        network_->get_queue().wait();

        network_input.copy_from_device(encoding_output.data());
        network_->get_queue().wait();
        network_->inference(network_input, network_output, {});
        network_->get_queue().wait();

        return {};
    }

    DeviceMatrixView<T_net> forward_pass(const DeviceMatrix<float> &input, DeviceMatrix<T_net> &network_input,
                                         DeviceMatrix<T_enc> &encoding_output,
                                         DeviceMatrices<T_net> &intermediate_forward,
                                         const std::vector<sycl::event> &deps) {
        /// TODO: implemente proper usage of deps. Requires proper implementation of forward_impl
        /// in encodings which takes it as input and returns new dependencies.

        const int batch_size = input.m();
        const int network_input_width = network_->get_input_width();

        if (intermediate_forward.input_m() != batch_size || encoding_output.m() != batch_size)
            throw std::invalid_argument("Wrong dimensions.");
        if (encoding_output.n() != network_input_width) throw std::invalid_argument("Wrong dimensions.");

        auto ctxt = encoding_->forward_impl(&network_->get_queue(), input, &encoding_output);
        network_->get_queue().wait();
        network_input.copy_from_device(encoding_output.data());
        network_->forward_pass(network_input, intermediate_forward, {});
        network_->get_queue().wait();

        return intermediate_forward.Back();
    }

    std::vector<sycl::event> backward_pass(const DeviceMatrix<T_net> &input_backward,
                                           DeviceMatrices<T_net> &network_gradient,
                                           DeviceMatrices<T_net> &intermediate_backward,
                                           const DeviceMatrices<T_net> &intermediate_forward,
                                           const std::vector<sycl::event> &deps,
                                           const DeviceMatrix<T_enc> &input_encoding, DeviceMatrix<T_net> &dL_dinput) {
        auto dL_dinput_view = dL_dinput.GetView(); // Store the view in a local variable
        auto event = network_->backward_pass(input_backward, network_gradient, intermediate_backward,
                                             intermediate_forward, deps, dL_dinput_view);
        network_->get_queue().wait();

        if (encoding_->n_params()) {
            const int batch_size = input_backward.m();
            std::unique_ptr<Context> model_ctx = nullptr;

            DeviceMatrix<float> output_float(batch_size, encoding_->padded_output_width(), network_->get_queue());
            output_float.fill(0.0f).wait();
            DeviceMatrix<float> dL_dinput_float = convert_matrix<T_net, float>(dL_dinput, network_->get_queue());
            encoding_->backward_impl(&network_->get_queue(), *model_ctx, input_encoding, output_float, dL_dinput_float);
            network_->get_queue().wait();
        }
        return event;
    }

    // functions which simplify the usage by generating the intermediate arrays
    DeviceMatrix<T_net> GenerateIntermediateForwardMatrix(const size_t batch_size) {
        const uint32_t tmp_n_cols = network_->get_input_width() +
                                    network_->get_network_width() * network_->get_n_hidden_layers() +
                                    network_->get_output_width();
        return std::move(DeviceMatrix<T_net>(batch_size, tmp_n_cols, network_->get_queue()));
    }

    DeviceMatrix<T_net> GenerateIntermediateBackwardMatrix(const size_t batch_size) {
        const uint32_t tmp_n_cols =
            network_->get_network_width() * network_->get_n_hidden_layers() + network_->get_output_width();
        return std::move(DeviceMatrix<T_net>(batch_size, tmp_n_cols, network_->get_queue()));
    }
    DeviceMatrix<T_enc> GenerateEncodingOutputMatrix(const size_t batch_size) {
        const uint32_t tmp_n_cols = network_->get_input_width();
        return std::move(DeviceMatrix<T_enc>(batch_size, tmp_n_cols, network_->get_queue()));
    }
    DeviceMatrix<T_net> GenerateForwardOutputMatrix(const size_t batch_size) {
        const uint32_t tmp_n_cols = network_->get_output_width();
        return std::move(DeviceMatrix<T_net>(batch_size, tmp_n_cols, network_->get_queue()));
    }
    DeviceMatrix<T_net> GenerateBackwardOutputMatrix() {
        const uint32_t tmp_n_rows = network_->get_network_width();
        const uint32_t tmp_n_cols = network_->get_n_hidden_matrices() * network_->get_network_width() +
                                    network_->get_input_width() + network_->get_output_width();
        return std::move(DeviceMatrix<T_net>(tmp_n_rows, tmp_n_cols, network_->get_queue()));
    }

    std::shared_ptr<Encoding<T_enc>> get_encoding() { return encoding_; }

    std::shared_ptr<Network<T_net>> get_network() { return network_; }

    void set_network_params(std::vector<T_net> network_params) { network_->set_weights_matrices(network_params); }

    void set_encoding_params(DeviceMem<T_enc> &encoding_params_device_mem,
                             DeviceMem<T_enc> &encoding_gradients_device_mem,
                             std::vector<T_enc> *encoding_params = nullptr) {
        encoding_->set_params_helper(encoding_params_device_mem, encoding_gradients_device_mem, encoding_params);
    }

    void initialize_params(DeviceMem<T_enc> &encoding_params_device_mem,
                           DeviceMem<T_enc> &encoding_gradients_device_mem,
                           std::vector<T_enc> *encoding_params = nullptr,
                           std::vector<T_net> *network_params = nullptr) {
        set_encoding_params(encoding_params_device_mem, encoding_gradients_device_mem, encoding_params);
        if (network_params != nullptr) {
            set_network_params(*network_params);
        }
    }

  private:
    void SanityCheck() const {
        /// TODO: check that the queues of the encoding and network coincide.

        if (encoding_->padded_output_width() != network_->get_input_width())
            throw std::invalid_argument("Encoding output dim and network input dim mismatch. Expected: " +
                                        std::to_string(encoding_->padded_output_width()) +
                                        " for encoding padded output width, but got network input width: " +
                                        std::to_string(network_->get_input_width()));
    }

    std::shared_ptr<Encoding<T_enc>> encoding_;
    std::shared_ptr<Network<T_net>> network_;
};

template <typename T_enc, typename T_net, int WIDTH>
std::shared_ptr<NetworkWithEncoding<T_enc, T_net>>
create_network_with_encoding(sycl::queue &q, const int output_width, const int n_hidden_layers, Activation activation,
                             Activation output_activation, const json &encoding_config) {
    // input width is encoding_config as EncodingParams::N_DIMS_TO_ENCODE
    std::shared_ptr<Encoding<T_enc>> enc = create_encoding<T_enc>(encoding_config);
    if (enc->output_width() < WIDTH) {
        enc->set_padded_output_width(WIDTH);
    }
    std::shared_ptr<SwiftNetMLP<T_net, WIDTH>> net = std::make_shared<SwiftNetMLP<T_net, WIDTH>>(
        q, WIDTH, output_width, n_hidden_layers, activation, output_activation);
    return std::make_shared<NetworkWithEncoding<T_enc, T_net>>(enc, net);
}
