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
        return network_->inference(network_input, network_output, {});
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

        // throw std::logic_error(
        //     "Returned view does not make any sense. Storage is in block major but view uses row-major");
        // return intermediate_forward.GetView(batch_size, network_->get_unpadded_output_width(), 0,
        //                                     network_->get_input_width() +
        //                                         network_->get_network_width() * network_->get_n_hidden_layers());
    }

    std::vector<sycl::event> backward_pass(const DeviceMatrix<T_net> &input_backward, DeviceMatrices<T_net> &output,
                                           DeviceMatrices<T_net> &intermediate_backward,
                                           const DeviceMatrices<T_net> &intermediate_forward,
                                           const std::vector<sycl::event> &deps) {
        return network_->backward_pass(input_backward, output, intermediate_backward, intermediate_forward, deps);
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

  private:
    void SanityCheck() const {
        /// TODO: check that the queues of the encoding and network coincide.
        // Check that the dimensions of network and encoding match
    }

    std::shared_ptr<Encoding<T_enc>> encoding_;
    std::shared_ptr<Network<T_net>> network_;
};

template <typename T_enc, typename T_net, int WIDTH>
std::shared_ptr<NetworkWithEncoding<T_enc, T_net>>
create_network_with_encoding(sycl::queue &q, const int input_width, const int output_width, const int n_hidden_layers,
                             Activation activation, Activation output_activation, const json &encoding_config) {

    std::shared_ptr<SwiftNetMLP<T_net, WIDTH>> net = std::make_shared<SwiftNetMLP<T_net, WIDTH>>(
        q, input_width, output_width, n_hidden_layers, activation, output_activation);
    std::shared_ptr<Encoding<T_enc>> enc = create_encoding<T_enc>(encoding_config);
    return std::make_shared<NetworkWithEncoding<T_enc, T_net>>(enc, net);
}
