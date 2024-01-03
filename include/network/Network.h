#pragma once

#include "DeviceMem.h"
#include "common.h"

// Base class for neural network
template <typename T> class Network {
  public:
    Network(sycl::queue &q, const int n_hidden_layers, const int inputs_width, const int network_width,
            const int output_width)
        : m_q(q), n_hidden_layers_(n_hidden_layers), inputs_width_(inputs_width), network_width_(network_width),
          output_width_(output_width),
          m_grads_matrices(network_width_ * (inputs_width_ + network_width_ * (n_hidden_layers_ - 1) + output_width_),
                           m_q),
          m_weights_matrices(network_width_ * (inputs_width_ + network_width_ * (n_hidden_layers_ - 1) + output_width_),
                             m_q),
          m_weightsT_matrices(
              network_width_ * (inputs_width_ + network_width_ * (n_hidden_layers_ - 1) + output_width_), m_q) {
        SanityCheck();
    }

    virtual ~Network() {}

    // Perform forward pass through the network
    virtual std::vector<sycl::event> forward_pass(const DeviceMem<T> &input, DeviceMem<T> &intermediate_output_forward,
                                                  const size_t batch_size, const std::vector<sycl::event> &deps) = 0;

    // Perform inference through the network
    virtual std::vector<sycl::event> inference(const DeviceMem<T> &input, DeviceMem<T> &output, const size_t batch_size,
                                               const std::vector<sycl::event> &deps) = 0;

    // Perform backward pass through the network
    virtual std::vector<sycl::event> backward_pass(const DeviceMem<T> &grads,
                                                   DeviceMem<T> &intermediate_output_backward,
                                                   const DeviceMem<T> &intermediate_output_forward,
                                                   const size_t batch_size, const std::vector<sycl::event> &deps) = 0;

    virtual std::vector<sycl::event> training(const DeviceMem<T> &input, const DeviceMem<T> &target,
                                              DeviceMem<T> &intermediate_output_backward,
                                              DeviceMem<T> &intermediate_output_forward, const size_t batch_size,
                                              const std::vector<sycl::event> &deps) = 0;

    /// @brief Get the SYCL queue associated with the network
    queue &get_queue() { return m_q; }

    virtual void set_weights_matrices(const std::vector<T> &weights) {
        m_weights_matrices.copy_from_host(weights);
        TransposeWeights(m_weights_matrices, m_weightsT_matrices);
    };

    ///@brief initializes the weight matrices to pre-set values.
    ///@todo: remove this from the network class.
    void initialize_weights_matrices(int mode) {
        // Initialize weights matrices with uniform random values, you can choose a
        // different initialization ( see in DeviceMem.cpp )

        if (mode == 1)
            initialize_arange(m_weights_matrices, network_width_);
        else if (mode == 2)
            initialize_constant(m_weights_matrices, 0.01);
        else if (mode == 3)
            initialize_constant(m_weights_matrices, -0.01);
        else
            initialize_he_normal(m_weights_matrices, inputs_width_);

        TransposeWeights(m_weights_matrices, m_weightsT_matrices);
    }

    virtual DeviceMem<T> &get_grads_matrices() { return m_grads_matrices; }
    virtual const DeviceMem<T> &get_weights_matrices() const { return m_weights_matrices; }
    virtual const DeviceMem<T> &get_weightsT_matrices() const { return m_weightsT_matrices; }

    virtual int get_n_hidden_layers() const { return n_hidden_layers_; }
    /// @brief returns hidden layers - 1
    /// @return n_hidden_layers - 1
    virtual int get_n_hidden_matrices() const { return n_hidden_layers_ - 1; }
    virtual int get_inputs_width() const { return inputs_width_; }
    virtual int get_network_width() const { return network_width_; }
    virtual int get_output_width() const { return output_width_; }

  private:
    virtual void SanityCheck() const {
        if (inputs_width_ <= 0) {
            std::string errorMessage =
                "Input width of " + std::to_string(inputs_width_) + " is not supported. Value must be larger than 0.";
            throw std::runtime_error(errorMessage);
        }

        if (output_width_ <= 0) {
            std::string errorMessage =
                "Output width of " + std::to_string(output_width_) + " is not supported. Value must be larger than 0.";
            throw std::runtime_error(errorMessage);
        }

        if (inputs_width_ > network_width_) {
            std::string errorMessage = "Input width of " + std::to_string(inputs_width_) +
                                       " is not supported. Value must be <= network width (" +
                                       std::to_string(network_width_) + ").";
            throw std::runtime_error(errorMessage);
        }

        if (output_width_ > network_width_) {
            std::string errorMessage = "Input width of " + std::to_string(output_width_) +
                                       " is not supported. Value must be <= network width (" +
                                       std::to_string(network_width_) + ").";
            throw std::runtime_error(errorMessage);
        }

        if (n_hidden_layers_ <= 0) {
            std::string errorMessage = "N hidden layers is " + std::to_string(output_width_) +
                                       " but must be >= 1, i.e., 1 hidden layer and 1 output layer.";
            throw std::runtime_error(errorMessage);
        }

        if (network_width_ != 16 && network_width_ != 32 && network_width_ != 64 && network_width_ != 128)
            throw std::invalid_argument("Network width has to be a power of 2 between 16 and 128.");

        if (network_width_ != inputs_width_ || network_width_ != output_width_)
            throw std::invalid_argument(
                "Right now, only networks with same input, layer and output width are allowed.");
    }

    // we are making this static void to avoid copying of this class to the device
    static void TransposeWeightMatrix(T const *const in, const int M, const int N, T *const out, sycl::queue &q) {
        q.parallel_for(sycl::range<1>(M * N), [=](auto idx) {
            const int i = idx / N;
            const int j = idx % N;
            out[toPackedLayoutCoord(j * M + i, N, M)] = in[toPackedLayoutCoord(i * N + j, M, N)];
        });
    }

    void TransposeWeights(const DeviceMem<T> &weights, DeviceMem<T> &weightsT) {
        const size_t nelems = get_inputs_width() * get_network_width() +
                              get_n_hidden_matrices() * get_network_width() * get_network_width() +
                              get_network_width() * get_output_width();
        assert(weights.size() >= nelems);
        assert(weightsT.size() >= nelems);

        // input matrix transpose
        TransposeWeightMatrix(weights.data(), get_network_width(), get_inputs_width(), weightsT.data(), m_q);
        // hidden matrices transpose
        for (int matiter = 0; matiter < get_n_hidden_matrices(); matiter++) {
            const size_t offset =
                get_network_width() * get_inputs_width() + matiter * get_network_width() * get_network_width();
            TransposeWeightMatrix(weights.data() + offset, get_network_width(), get_network_width(),
                                  weightsT.data() + offset, m_q);
        }
        // output matrix transpose
        const size_t offset = get_network_width() * get_inputs_width() +
                              get_n_hidden_matrices() * get_network_width() * get_network_width();
        TransposeWeightMatrix(weights.data() + offset, get_network_width(), get_output_width(),
                              weightsT.data() + offset, m_q);

        m_q.wait();
    }

    /**
     * Initialize the device memory with values drawn from a normal distribution.
     *
     * This function initializes the device memory with random values drawn from a
     * normal distribution with the specified standard deviation.
     *
     * @param dev   The standard deviation of the normal distribution.
     * @param q     The SYCL queue for parallel computation.
     */
    static void initialize_normal(DeviceMem<T> &vec, const double dev) {
        std::default_random_engine gen;
        std::normal_distribution<double> distrib(0.0, dev);
        std::vector<T> data(vec.size());
        for (int i = 0; i < vec.size(); i++) {
            data[i] = (T)distrib(gen);
        }
        vec.copy_from_host(data).wait();
    }

    /**
     * Initialize the device memory with values sampled from a uniform distribution.
     *
     * This function generates random values sampled from a uniform distribution
     * within the specified scale and initializes the device memory with those
     * values.
     *
     * @param q       The SYCL queue for memory operations.
     * @param scale   The scale of the uniform distribution.
     */
    static void initialize_uniform(DeviceMem<T> &vec, const double scale) {
        std::default_random_engine gen;
        std::uniform_real_distribution<double> distrib(0.0, scale);
        std::vector<T> data(vec.size());

        for (int i = 0; i < vec.size(); i++) {
            data[i] = (T)distrib(gen);
        }
        vec.copy_from_host(data).wait();
    }

    // Initialize memory with constant values
    static void initialize_constant(DeviceMem<T> &vec, const T &constant) { vec.fill(constant); }

    /**
     * Initialize the device memory with values sampled from a He normal
     * distribution.
     *
     * This function generates random values sampled from a He normal distribution
     * and initializes the device memory with those values.
     *
     * @param input_width   The width of the input.
     * @param q             The SYCL queue for memory operations.
     */
    static void initialize_he_normal(DeviceMem<T> &vec, const int width) {
        const double dev = std::sqrt(2.0 / width);
        initialize_normal(vec, dev);
    }

    static void initialize_arange(DeviceMem<T> &vec, const int range) {
        std::vector<T> tmp_host(vec.size());
        for (size_t blockiter = 0; blockiter < vec.size(); blockiter += range * range) {
            for (int rowiter = 0; rowiter < range; rowiter++) {
                for (int coliter = 0; coliter < range; coliter++) {
                    tmp_host[blockiter + rowiter * range + coliter] =
                        static_cast<T>((rowiter + 1 - (range / 2)) * 0.01);
                }
            }
        }

        vec.copy_from_host(tmp_host).wait();
    }

    sycl::queue &m_q;

    const int n_hidden_layers_;
    const int inputs_width_;
    const int network_width_;
    const int output_width_;

    DeviceMem<T> m_grads_matrices;
    DeviceMem<T> m_weights_matrices;
    DeviceMem<T> m_weightsT_matrices;
};
