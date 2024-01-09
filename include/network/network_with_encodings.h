#pragma once
#include "Network.h"
#include "encoding_factory.h"

template <typename T> class NetworkWithEncoding {
  public:
    NetworkWithEncoding() = delete;
    NetworkWithEncoding(std::shared_ptr<Encoding<T>> encoding, std::shared_ptr<Network<T>> network)
        : encoding_(encoding), network_(network) {

        SanityCheck();
    }

    ~NetworkWithEncoding() {}

    std::vector<sycl::event> inference(const DeviceMatrix<float> &input, DeviceMatrix<T> &encoding_output,
                                       DeviceMatrix<T> &network_output, const std::vector<sycl::event> &deps) {
        const int batch_size = input.m();
        const int network_input_width = network_->get_inputs_width();

        if (network_output.m() != batch_size || encoding_output.m() != batch_size)
            throw std::invalid_argument("Wrong dimensions.");
        if (encoding_output.n() != network_input_width) throw std::invalid_argument("Wrong dimensions.");
        if (network_output.n() != network_->get_output_width()) throw std::invalid_argument("Wrong dimensions.");

        auto new_deps = encoding_->forward_impl(&network_->get_queue(), input, &encoding_output, deps);
        return network_->inference(encoding_output, network_output, batch_size, new_deps);
    }

    DeviceMatrixView<T> forward_pass(const DeviceMatrix<float> &input, DeviceMatrix<T> &encoding_output,
                                     DeviceMatrix<T> &intermediate_forward, const std::vector<sycl::event> &deps) {
        const int batch_size = input.m();
        const int network_input_width = network_->get_input_width();

        if (intermediate_forward.m() != batch_size || encoding_output.m() != batch_size)
            throw std::invalid_argument("Wrong dimensions.");
        if (encoding_output.n() != network_input_width) throw std::invalid_argument("Wrong dimensions.");

        auto new_deps = encoding_->forward_impl(&network_->get_queue(), input, &encoding_output, deps);
        network_->forward_pass(encoding_output, intermediate_forward, batch_size, new_deps);
        network_->get_queue().wait();
        return intermediate_forward.GetView(batch_size, network_->get_unpadded_output_width(), 0,
                                            network_->get_input_width() +
                                                network_->get_network_width() * network_->get_n_hidden_layers());
    }

    std::vector<sycl::event> backward_pass(const DeviceMatrix<T> &input_backward, DeviceMatrix<T> &output,
                                           DeviceMatrix<T> &intermediate_backward,
                                           const DeviceMatrix<T> &intermediate_forward, const int batch_size,
                                           const std::vector<sycl::event> &deps) {
        return network_->backward_pass(input_backward, output, intermediate_backward, intermediate_forward, batch_size,
                                       deps);
    }

    // functions which simplify the usage by generating the intermediate arrays
    DeviceMatrix<T> GenerateIntermediateForwardMatrix(const size_t batch_size) {
        const uint32_t tmp_n_cols = network_->get_input_width() +
                                    network_->get_network_width() * network_->get_n_hidden_layers() +
                                    network_->get_output_width();
        return std::move(DeviceMatrix<T>(batch_size, tmp_n_cols, network_->get_queue()));
    }

    DeviceMatrix<T> GenerateIntermediateBackwardMatrix(const size_t batch_size) {
        const uint32_t tmp_n_cols =
            network_->get_network_width() * network_->get_n_hidden_layers() + network_->get_output_width();
        return std::move(DeviceMatrix<T>(batch_size, tmp_n_cols, network_->get_queue()));
    }
    DeviceMatrix<T> GenerateEncodingOutputMatrix(const size_t batch_size) {
        const uint32_t tmp_n_cols = network_->get_input_width();
        return std::move(DeviceMatrix<T>(batch_size, tmp_n_cols, network_->get_queue()));
    }
    DeviceMatrix<T> GenerateForwardOutputMatrix(const size_t batch_size) {
        const uint32_t tmp_n_cols = network_->get_output_width();
        return std::move(DeviceMatrix<T>(batch_size, tmp_n_cols, network_->get_queue()));
    }
    DeviceMatrix<T> GenerateBackwardOutputMatrix() {
        const uint32_t tmp_n_rows = network_->get_network_width();
        const uint32_t tmp_n_cols = network_->get_n_hidden_matrices() * network_->get_network_width() +
                                    network_->get_input_width() + network_->get_output_width();
        return std::move(DeviceMatrix<T>(tmp_n_rows, tmp_n_cols, network_->get_queue()));
    }

    std::shared_ptr<Network<T>> get_network() { return network_; }
    std::shared_ptr<Encoding<T>> get_encoding() { return encoding_; }

  private:
    void SanityCheck() const {
        /// TODO: check that the queues of the encoding and network coincide.
        // Check that the dimensions of network and encoding match
    }

    std::shared_ptr<Encoding<T>> encoding_;
    std::shared_ptr<Network<T>> network_;
};

template <typename T, int WIDTH>
std::shared_ptr<NetworkWithEncoding<T>>
create_networkwith_encoding(sycl::queue &q, const int input_width, const int output_width, const int n_hidden_layers,
                            Activation activation, Activation output_activation, const int batch_size,
                            std::string encoding_name,
                            const std::unordered_map<std::string, std::string> &encoding_config) {
    /// TODO: make a network, make an encoding return the result.
    std::shared_ptr<SwiftNetMLP<T, WIDTH>> net = std::make_shared<SwiftNetMLP<T, WIDTH>>(
        q, input_width, output_width, n_hidden_layers, activation, output_activation);
    std::shared_ptr<Encoding<T>> enc = create_encoding<T>(encoding_name, encoding_config);
    return std::make_shared<NetworkWithEncoding<T>>(net, enc);
}
