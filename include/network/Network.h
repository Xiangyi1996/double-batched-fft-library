#pragma once

#include "DeviceMatrix.h"
#include "common.h"

// completely generic Network.
template <typename T> class NetworkBase {
  public:
    NetworkBase() {}

    virtual ~NetworkBase() {}

    // Perform forward pass through the network
    virtual std::vector<sycl::event> forward_pass(const DeviceMatrix<T> &input,
                                                  DeviceMatrix<T> &intermediate_output_forward,
                                                  const std::vector<sycl::event> &deps) = 0;

    // Perform inference through the network
    virtual std::vector<sycl::event> inference(const DeviceMatrix<T> &input, DeviceMatrix<T> &output,
                                               const std::vector<sycl::event> &deps) = 0;

    ///@brief input are the derivatives of the losses
    /// output are the updates of the weights for the optimization step.
    /// intermediate arrays are not used after this function
    virtual std::vector<sycl::event> backward_pass(const DeviceMatrix<T> &input, DeviceMatrix<T> &output,
                                                   DeviceMatrix<T> &intermediate_output_backward,
                                                   const DeviceMatrix<T> &intermediate_output_forward,
                                                   const std::vector<sycl::event> &deps) = 0;

    // virtual std::vector<sycl::event> training(const DeviceMatrix<T> &input, const DeviceMatrix<T> &target,
    //                                           DeviceMatrix<T> &intermediate_output_backward,
    //                                           DeviceMatrix<T> &intermediate_output_forward,
    //                                           const std::vector<sycl::event> &deps) = 0;
};

// Network Base class for all networks with weights matrices and width restrictions
template <typename T> class Network : public NetworkBase<T> {

  public:
    enum WeightInitMode { arange, constant_pos, constant_negative, he_normal, none };

    Network(sycl::queue &q, const int n_hidden_layers, const int input_width, const int network_width,
            const int output_width, const WeightInitMode mode)
        : m_q(q), input_width_(PadWidths(input_width, network_width)),
          output_width_(PadWidths(output_width, network_width)), original_output_width_(output_width),
          n_hidden_layers_(NonNegative(n_hidden_layers)), network_width_(NonNegative(network_width)),
          m_weights_matrices(network_width_,
                             (get_input_width() + network_width_ * (n_hidden_layers_ - 1) + get_output_width()), m_q),
          m_weightsT_matrices(network_width_,
                              (get_input_width() + network_width_ * (n_hidden_layers_ - 1) + get_output_width()), m_q) {

        SanityCheck();
        initialize_weights_matrices(input_width, output_width, mode);
    }

    virtual ~Network() {}

    /// @brief Get the SYCL queue associated with the network
    sycl::queue &get_queue() { return m_q; }

    virtual void set_weights_matrices(const std::vector<T> &weights) {
        m_weights_matrices.copy_from_host(weights);
        TransposeWeights(m_weights_matrices, m_weightsT_matrices);
    };

    // this is the result from the backward pass. Not sure why this is here to be honest.
    virtual const DeviceMatrix<T> &get_weights_matrices() const { return m_weights_matrices; }
    virtual const DeviceMatrix<T> &get_weightsT_matrices() const { return m_weightsT_matrices; }

    virtual int get_n_hidden_layers() const { return n_hidden_layers_; }
    /// @brief returns hidden layers - 1
    /// @return n_hidden_layers - 1
    virtual int get_n_hidden_matrices() const { return n_hidden_layers_ - 1; }
    virtual int get_network_width() const { return network_width_; }
    virtual uint32_t get_input_width() const { return input_width_; }
    virtual uint32_t get_output_width() const { return output_width_; }
    virtual uint32_t get_unpadded_output_width() const { return original_output_width_; }

  private:
    virtual void SanityCheck() const {
        if (get_input_width() <= 0) {
            std::string errorMessage = "Input width of " + std::to_string(get_input_width()) +
                                       " is not supported. Value must be larger than 0.";
            throw std::runtime_error(errorMessage);
        }

        if (get_output_width() <= 0) {
            std::string errorMessage = "Output width of " + std::to_string(get_output_width()) +
                                       " is not supported. Value must be larger than 0.";
            throw std::runtime_error(errorMessage);
        }

        if (get_input_width() > network_width_) {
            std::string errorMessage = "Input width of " + std::to_string(get_input_width()) +
                                       " is not supported. Value must be <= network width (" +
                                       std::to_string(network_width_) + ").";
            throw std::runtime_error(errorMessage);
        }

        if (get_output_width() > network_width_) {
            std::string errorMessage = "Input width of " + std::to_string(get_output_width()) +
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

        if (network_width_ != get_input_width() || network_width_ != get_output_width())
            throw std::invalid_argument("Only networks with same input, layer and output width are allowed.");
    }

    ///@brief Helper function which sets values of the weights matrices to 0 if
    /// the actual input/output width was padded to the network-allowed input/output width.
    void ZeroWeightsPadding(const int unpadded_input_width, const int unpadded_output_width) {
        if (unpadded_input_width > get_input_width() || unpadded_output_width > get_output_width())
            throw std::invalid_argument("Padded weights width cannot be less than the unpadded.");

        T *const weights = m_weights_matrices.data();
        /// we need to copy everything here since we do not want to have an implicit copy of 'this'
        const int padded_input_width = get_input_width();
        const int network_width = get_network_width();
        const int padded_output_width = get_output_width();
        const int output_offset =
            get_n_hidden_matrices() * network_width * network_width + network_width * padded_input_width;

        // input matrix: set rows to 0.
        if (unpadded_input_width != padded_input_width) {
            m_q.parallel_for(
                   padded_input_width * network_width,
                   [=](auto idx) {
                       const int i = idx / network_width; // rows
                       const int j = idx % network_width; // cols

                       if (i >= unpadded_input_width)
                           weights[toPackedLayoutCoord(i * network_width + j, padded_input_width, network_width)] =
                               static_cast<T>(0);
                   })
                .wait();
        }

        // output matrix set columns to 0
        if (unpadded_output_width != padded_output_width) {
            m_q.parallel_for(padded_output_width * network_width,
                             [=](auto idx) {
                                 const int i = idx / padded_output_width; // rows
                                 const int j = idx % padded_output_width; // cols

                                 if (j >= unpadded_output_width)
                                     weights[output_offset + toPackedLayoutCoord(i * padded_output_width + j,
                                                                                 network_width, padded_output_width)] =
                                         static_cast<T>(0);
                             })
                .wait();
        }
    }

    // we are making this static void to avoid copying of this class to the device
    static void TransposeWeightMatrix(T const *const in, const int M, const int N, T *const out, sycl::queue &q) {
        q.parallel_for(sycl::range<1>(M * N), [=](auto idx) {
            const int i = idx / N;
            const int j = idx % N;
            out[toPackedLayoutCoord(j * M + i, N, M)] = in[toPackedLayoutCoord(i * N + j, M, N)];
        });
    }

    void TransposeWeights(const DeviceMatrix<T> &weights, DeviceMatrix<T> &weightsT) {
        const size_t nelems = get_input_width() * get_network_width() +
                              get_n_hidden_matrices() * get_network_width() * get_network_width() +
                              get_network_width() * get_output_width();
        assert(weights.size() >= nelems);
        assert(weightsT.size() >= nelems);

        // input matrix transpose
        TransposeWeightMatrix(weights.data(), get_network_width(), get_input_width(), weightsT.data(), m_q);
        // hidden matrices transpose
        for (int matiter = 0; matiter < get_n_hidden_matrices(); matiter++) {
            const size_t offset =
                get_network_width() * get_input_width() + matiter * get_network_width() * get_network_width();
            TransposeWeightMatrix(weights.data() + offset, get_network_width(), get_network_width(),
                                  weightsT.data() + offset, m_q);
        }
        // output matrix transpose
        const size_t offset = get_network_width() * get_input_width() +
                              get_n_hidden_matrices() * get_network_width() * get_network_width();
        TransposeWeightMatrix(weights.data() + offset, get_network_width(), get_output_width(),
                              weightsT.data() + offset, m_q);

        m_q.wait();
    }

    ///@brief initializes the weight matrices to pre-set values.
    /// Note that the network has in general no interest in knowing anything
    /// about padding. Only when we initialize the weight matrix are we setting
    /// certain elements to 0.
    ///@todo: remove this from the network class.
    void initialize_weights_matrices(const int unpadded_input_width, const int unpadded_output_width,
                                     WeightInitMode mode) {

        if (mode == WeightInitMode::arange)
            initialize_arange(m_weights_matrices, network_width_);
        else if (mode == WeightInitMode::constant_pos)
            initialize_constant(m_weights_matrices, 0.01);
        else if (mode == WeightInitMode::constant_negative)
            initialize_constant(m_weights_matrices, -0.01);
        else if (mode == WeightInitMode::he_normal)
            initialize_he_normal(m_weights_matrices, network_width_);
        else if (mode == WeightInitMode::none) // init to 0
            initialize_constant(m_weights_matrices, 0);
        else
            throw std::invalid_argument("Invalid weights initialization mode.");

        ZeroWeightsPadding(unpadded_input_width, unpadded_output_width);

        TransposeWeights(m_weights_matrices, m_weightsT_matrices);
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
    static void initialize_normal(DeviceMatrix<T> &vec, const double dev) {
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
    static void initialize_uniform(DeviceMatrix<T> &vec, const double scale) {
        std::default_random_engine gen;
        std::uniform_real_distribution<double> distrib(0.0, scale);
        std::vector<T> data(vec.size());

        for (int i = 0; i < vec.size(); i++) {
            data[i] = (T)distrib(gen);
        }
        vec.copy_from_host(data).wait();
    }

    // Initialize memory with constant values
    static void initialize_constant(DeviceMatrix<T> &vec, const T &constant) { vec.fill(constant); }

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
    static void initialize_he_normal(DeviceMatrix<T> &vec, const int width) {
        const double dev = std::sqrt(2.0 / width);
        initialize_normal(vec, dev);
    }

    static void initialize_arange(DeviceMatrix<T> &vec, const int range) {
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

    static int PadWidths(const int width, const int base) {
        if (width <= 0) throw std::invalid_argument("width <= 0 cannot be padded.");
        if (base <= 0) throw std::invalid_argument("base <= 0 cannot be used for padding.");

        return ((width + base - 1) / base) * base;
    }

    static int NonNegative(const int number) {
        if (number < 0) throw std::invalid_argument("Use non-negative number.");

        return number;
    }

    sycl::queue &m_q;

    const uint32_t input_width_;
    const uint32_t output_width_;
    const uint32_t original_output_width_; // unpadded
    const int n_hidden_layers_;
    const int network_width_;

    DeviceMatrix<T> m_weights_matrices;
    DeviceMatrix<T> m_weightsT_matrices;
};
