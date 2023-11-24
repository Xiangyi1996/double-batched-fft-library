
#include "SwiftNetMLP.h"
#include "common.h"
#include "kernel.h"
#include "mkl.h"
#include "mkl_omp_offload.h"
#include "oneapi/mkl.hpp"
#include "trainer.h"
#include <CL/sycl.hpp>

using bf16 = sycl::ext::oneapi::bfloat16;

/**
 * Constructor for the SwiftNetMLP class.
 *
 * @param q                  SYCL queue for command submission.
 * @param input_width        Width of the input data.
 * @param output_width       Width of the output data.
 * @param n_hidden_layers    Number of hidden layers.
 * @param activation         Activation function for hidden layers.
 * @param output_activation  Activation function for the output layer.
 * @param batch_size         Batch size of the data.
 * @tparam WIDTH             Width of the matrices.
 */
template <int WIDTH>
SwiftNetMLP<WIDTH>::SwiftNetMLP(queue q, int input_width, int output_width, int n_hidden_layers, Activation activation,
                                Activation output_activation)
    : m_inputs_width{input_width}, m_net_width{WIDTH}, m_output_width{output_width}, m_n_hidden_layers{n_hidden_layers},
      m_activation{activation}, m_output_activation{output_activation}, m_inputs_width_padded{WIDTH},
      m_output_width_padded{WIDTH} /*TODO: replace later with smallest, nearest common divisor*/ {
    // Store provided parameters
    m_q = q;
    m_n_hidden_matrices = m_n_hidden_layers - 1;

    check_parameters();

    // As the systolic matrix multiplication works in multiples of 8/16, we cannot have arbitrary input and output
    // width. To get the correct width as defined by input_width and output_width, we pad later with zeros

    // Allocate memory for various matrices
    m_weightsT_matrices.allocate2(m_net_width * m_inputs_width_padded +
                                      (m_net_width * m_net_width) * m_n_hidden_matrices +
                                      m_net_width * m_output_width_padded,
                                  m_q);

    m_weights_matrices.allocate2(m_net_width * m_inputs_width_padded +
                                     (m_net_width * m_net_width) * m_n_hidden_matrices +
                                     m_net_width * m_output_width_padded,
                                 m_q);

    m_weights_matrices_inferences.allocate2(m_net_width * m_inputs_width_padded +
                                                (m_net_width * m_net_width) * m_n_hidden_matrices +
                                                m_net_width * m_output_width_padded,
                                            m_q);

    m_grads_matrices.allocate2(m_net_width * m_inputs_width_padded + (m_net_width * m_net_width) * m_n_hidden_matrices +
                                   m_net_width * m_output_width_padded,
                               m_q);

    // Initialize constants and allocations

    // note that the memory on m_deltas (also called loss sometimes) is
    // "flexible". It doesn't allow m_output_width > WIDTH, as in the
    // last layer backward pass, the m_output_width is first written
}

template <int WIDTH> void SwiftNetMLP<WIDTH>::check_parameters() {
    if (m_inputs_width <= 0) {
        std::string errorMessage =
            "Input width of " + std::to_string(m_inputs_width) + " is not supported. Value must be larger than 0.";
        throw std::runtime_error(errorMessage);
    }

    if (m_output_width <= 0) {
        std::string errorMessage =
            "Output width of " + std::to_string(m_output_width) + " is not supported. Value must be larger than 0.";
        throw std::runtime_error(errorMessage);
    }

    if (m_inputs_width > WIDTH) {
        std::string errorMessage = "Input width of " + std::to_string(m_inputs_width) +
                                   " is not supported. Value must be <= WIDTH (" + std::to_string(WIDTH) + ").";
        throw std::runtime_error(errorMessage);
    }

    if (m_output_width > WIDTH) {
        std::string errorMessage = "Input width of " + std::to_string(m_output_width) +
                                   " is not supported. Value must be <= WIDTH (" + std::to_string(WIDTH) + ").";
        throw std::runtime_error(errorMessage);
    }

    if (m_n_hidden_layers <= 0) {
        std::string errorMessage = "N hidden layers is " + std::to_string(m_output_width) +
                                   " but must be >= 1, i.e., 1 hidden layer and 1 output layer.";
        throw std::runtime_error(errorMessage);
    }

    if (m_activation != Activation::ReLU) {
        throw std::runtime_error("m_activation must be ReLU for now.");
    }
    if (m_output_activation != Activation::None) {
        throw std::runtime_error("m_output_activation must be None for now.");
    }
}
template <int WIDTH> SwiftNetMLP<WIDTH>::~SwiftNetMLP() {}
/**
 * Get a pointer to the gradients matrices.
 *
 * @return A pointer to the gradients matrices.
 */
template <int WIDTH> DeviceMem<bf16> *SwiftNetMLP<WIDTH>::get_grads_matrices() { return &m_grads_matrices; }

/**
 * Get a pointer to the weights matrices.
 *
 * @return A pointer to the weights matrices.
 */
template <int WIDTH> DeviceMem<bf16> *SwiftNetMLP<WIDTH>::get_weights_matrices() { return &m_weights_matrices; }

/**
 * Get a pointer to the transposed weights matrices.
 *
 * @return A pointer to the transposed weights matrices.
 */
template <int WIDTH> DeviceMem<bf16> *SwiftNetMLP<WIDTH>::get_weightsT_matrices() { return &m_weightsT_matrices; }

/**
 * Initialize parameters for the neural network.
 * This function initializes the weights matrices with uniform random values.
 */
template <int WIDTH> void SwiftNetMLP<WIDTH>::initialize_params(int use_easy) {
    // Initialize weights matrices with uniform random values, you can choose a
    // different initialization ( see in DeviceMem.cpp )

    if (use_easy == 1) {
        m_weights_matrices.initialize_arange(m_q, m_inputs_width_padded, m_net_width, m_output_width_padded,
                                             m_n_hidden_matrices);
    } else if (use_easy == 2) {
        m_weights_matrices.initialize_constant(0.01, m_q);
    } else if (use_easy == 3) {
        m_weights_matrices.initialize_constant(-0.01, m_q);
    } else {
        m_weights_matrices.initialize_he_normal(m_inputs_width_padded, m_q);
    }

    zero_pad_weight_matrix();

    m_weights_matrices.make_transposed(m_weightsT_matrices, m_inputs_width_padded, m_net_width, m_output_width_padded,
                                       m_n_hidden_matrices, m_q);
};

template <int WIDTH> void SwiftNetMLP<WIDTH>::zero_pad_weight_matrix() {
    m_weights_matrices.zero_pad_input(m_inputs_width, m_inputs_width_padded, m_net_width, m_q);

    m_weights_matrices.zero_pad_output(m_output_width, m_inputs_width_padded, m_net_width, m_output_width_padded,
                                       m_n_hidden_matrices, m_q);
}

/**
 * Save the neural network parameters to a file.
 *
 * @param filename The name of the file to save the parameters to.
 */
template <int WIDTH> void SwiftNetMLP<WIDTH>::save_to_file(std::string filename) {
    // Open the file for writing
    std::ofstream file;
    file.open(filename);

    // Write parameters to the file
    file << m_inputs_width << "\n";
    file << m_net_width << "\n";
    file << m_output_width << "\n";
    file << m_n_hidden_layers << "\n";
    file << m_n_hidden_matrices << "\n";

    // Write each value of the weights matrices to the file
    for (int i = 0; i < m_weights_matrices.size(); i++) {
        file << m_weights_matrices.data()[i] << "\n";
    }

    // Close the file
    file.close();
    return;
}

/**
 * Load neural network parameters from a file.
 *
 * @param filename The name of the file to load parameters from.
 */
template <int WIDTH> void SwiftNetMLP<WIDTH>::load_from_file(std::string filename) {
    // Open the file for reading
    // std::ifstream file;
    // file.open(filename);
    // std::string line;

    // Read parameters from the file
    // file >> m_inputs_width;
    // file >> m_net_width;
    // file >> m_output_width;
    // file >> m_n_hidden_layers;
    // file >> m_n_hidden_matrices;

    // Read each value from the file and set it as a bf16 value in weights
    // matrices
    std::vector<bf16> data_vec;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
    }

    std::string line;
    while (std::getline(file, line)) {
        try {
            float value = std::stod(line);
            data_vec.push_back((bf16)(value));
        } catch (const std::invalid_argument &e) {
            std::cerr << "Invalid argument: " << e.what() << std::endl;
        } catch (const std::out_of_range &e) {
            std::cerr << "Out of range: " << e.what() << std::endl;
        }
    }

    file.close();

    if (m_weights_matrices.size() != data_vec.size()) {
        std::string errorMessage = "m_weights_matrices.size() " + std::to_string(m_weights_matrices.size()) +
                                   " is not equal loaded size: " + std::to_string(data_vec.size());
        throw std::runtime_error(errorMessage);
    }

    std::vector<bf16> weights_packed(data_vec.size(), 0.0);

    for (int idx = 0; idx < weights_packed.size(); idx++) {
        int i = 0;
        int j = 0;
        // int mat_offset = 0;
        if (idx < m_inputs_width_padded * WIDTH) {
            // std::cout << "idx: " << idx << ", input" << std::endl;

            i = idx / WIDTH; // rows
            j = idx % WIDTH; // cols

            weights_packed[toPackedLayoutCoord(i + j * WIDTH, WIDTH, WIDTH)] = data_vec[idx];
        } else if ((idx >= m_inputs_width_padded * WIDTH) &&
                   (idx < m_inputs_width_padded * WIDTH + m_n_hidden_layers * WIDTH * WIDTH)) {
            int layer = (idx - m_inputs_width_padded * WIDTH) / (WIDTH * WIDTH);
            // std::cout << "idx: " << idx << ", middle and last layer " << layer << std::endl;
            int mat_offset = (idx - (m_inputs_width_padded * WIDTH + layer * WIDTH * WIDTH)) % (WIDTH * WIDTH);
            // std::cout << "Mat offset: " << mat_offset << ", at idx: " << idx << ", and layer " << layer << std::endl;
            i = mat_offset / WIDTH; // rows
            j = mat_offset % WIDTH; // cols

            weights_packed[m_inputs_width_padded * WIDTH + layer * WIDTH * WIDTH +
                           toPackedLayoutCoord(i + j * WIDTH, WIDTH, WIDTH)] = data_vec[idx];
        }
        // else if (idx >= m_inputs_width_padded * WIDTH + m_n_hidden_layers * WIDTH * WIDTH) {
        //     std::cout << "idx: " << idx << ", last layer " << std::endl;
        //     int mat_offset = (idx - m_inputs_width_padded * WIDTH - m_n_hidden_layers * WIDTH * WIDTH) %
        //                      (WIDTH * m_output_width_padded);
        //     i = mat_offset / WIDTH; // rows
        //     j = mat_offset % WIDTH; // cols
        //     std::cout << "Writing to "
        //               << m_inputs_width_padded * WIDTH + m_n_hidden_layers * WIDTH * WIDTH +
        //                      toPackedLayoutCoord(i + j * WIDTH, WIDTH, WIDTH)
        //               << " at idx: " << idx << std::endl;
        //     weights_packed[m_inputs_width_padded * WIDTH + m_n_hidden_layers * WIDTH * WIDTH +
        //                    toPackedLayoutCoord(i + j * WIDTH, WIDTH, WIDTH)] = data_vec[idx];
        // }
    }

    m_weights_matrices.copy_from_host(weights_packed, m_q);
    // m_weights_matrices.copy_from_host(data_vec, m_q);
    m_q.wait();
    // Make the weights matrices transposed using the transposed weights matrices
    m_weights_matrices.make_transposed(m_weightsT_matrices, m_inputs_width_padded, m_net_width, m_output_width,
                                       m_n_hidden_matrices, m_q);
    return;
}

/**
 * Free memory allocated on the device for various arrays.
 *
 * @param q The SYCL queue used for device operations.
 */
template <int WIDTH> void SwiftNetMLP<WIDTH>::free_mem(queue q) {
    m_grads_matrices.free_mem(q); // output after backward pass
    m_weights_matrices.free_mem(q);
    m_weightsT_matrices.free_mem(q);
}

/**
 * Perform a forward pass of the SwiftNetMLP model.
 *
 * @param input The input data on the device.
 * @param forward Pointer to the forward intermediate array.
 * The output is stored at the end of the array 'forward'
 */
template <int WIDTH>
std::vector<sycl::event> SwiftNetMLP<WIDTH>::forward_pass(const DeviceMem<bf16> &input, float *forward,
                                                          const size_t batch_size,
                                                          const std::vector<sycl::event> &deps) {
    // Static assertion and assertion checks
    static_assert(WIDTH % 16 == 0, "Width must be a multiple of 64.");
    if ((batch_size % 8) != 0) throw std::invalid_argument("Batch size is not divisible by 8.");

    // this is necessary for backward pass
    // Get a pointer to the weights matrices data
    bf16 *const Forwardbf16 = reinterpret_cast<bf16 *>(forward);

    // Perform forward pass based on activation function
    switch (m_activation) {
    case Activation::None:
        return tinydpcppnn::kernels::forward_impl_general<bf16, float, WIDTH, WIDTH, WIDTH, Activation::None,
                                                          Activation::None, false, 16>(
            m_q, m_weights_matrices.data(), input.data(), Forwardbf16, m_n_hidden_layers, batch_size, deps);
        break;
    case Activation::ReLU:
        return tinydpcppnn::kernels::forward_impl_general<bf16, float, WIDTH, WIDTH, WIDTH, Activation::ReLU,
                                                          Activation::None, false, 16>(
            m_q, m_weights_matrices.data(), input.data(), Forwardbf16, m_n_hidden_layers, batch_size, deps);
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
template <int WIDTH>
std::vector<sycl::event> SwiftNetMLP<WIDTH>::inference(const DeviceMem<bf16> &input, float *const forward,
                                                       const size_t batch_size, const std::vector<sycl::event> &deps) {
    static_assert(WIDTH % 16 == 0, "Width must be multiple of 16.");
    if ((batch_size % 8) != 0) throw std::invalid_argument("Batch size is not divisible by 8.");

    bf16 *const Forwardbf16 = reinterpret_cast<bf16 *>(forward);

    switch (m_activation) {
    case Activation::None:
        return tinydpcppnn::kernels::forward_impl_general<bf16, float, WIDTH, WIDTH, WIDTH, Activation::None,
                                                          Activation::None, true, 16>(
            m_q, m_weights_matrices.data(), input.data(), Forwardbf16, m_n_hidden_layers, batch_size, deps);
        break;
    case Activation::ReLU:
        return tinydpcppnn::kernels::forward_impl_general<bf16, float, WIDTH, WIDTH, WIDTH, Activation::ReLU,
                                                          Activation::None, true, 16>(
            m_q, m_weights_matrices.data(), input.data(), Forwardbf16, m_n_hidden_layers, batch_size, deps);
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
template <int WIDTH>
std::vector<sycl::event> SwiftNetMLP<WIDTH>::backward_pass(const DeviceMem<bf16> &grads, float *const out_inter,
                                                           float const *const forward, const size_t batch_size,
                                                           const std::vector<sycl::event> &deps) {
    if ((batch_size % 8) != 0) throw std::invalid_argument("Batch size is not divisible by 8.");

    // if ((batch_size % 16) != 0) {
    //     throw std::invalid_argument("Batch size is not divisible by 16.");
    // }

    // if (batch_size < 512) {
    //     throw std::invalid_argument("Batch size must be >= 512, but is " + std::to_string(batch_size));
    // }
    // Compute activation backpropagation using parallel_for
    bf16 const *const Forwardbf16 = reinterpret_cast<const bf16 *>(forward);
    bf16 *const out_interbf16 = reinterpret_cast<bf16 *>(out_inter);

    // Choose appropriate mlp_swiftnet_backward based on activation
    // We are onyl doinh output_activation==none right now
    switch (m_activation) {
    case Activation::None:
        return tinydpcppnn::kernels::backward_impl_general<bf16, float, WIDTH, WIDTH, WIDTH, Activation::None,
                                                           Activation::None, 16>(
            m_q, m_weightsT_matrices.data(), grads.data(), m_grads_matrices.data(), out_interbf16, Forwardbf16,
            m_n_hidden_layers, batch_size, deps);
        break;
    case Activation::ReLU:
        return tinydpcppnn::kernels::backward_impl_general<bf16, float, WIDTH, WIDTH, WIDTH, Activation::ReLU,
                                                           Activation::None, 16>(
            m_q, m_weightsT_matrices.data(), grads.data(), m_grads_matrices.data(), out_interbf16, Forwardbf16,
            m_n_hidden_layers, batch_size, deps);

    default:
        throw std::invalid_argument("Activation not yet implemented in backward_pass");
    }
}

template <int WIDTH>
std::vector<sycl::event> SwiftNetMLP<WIDTH>::training(const DeviceMem<bf16> &input, const DeviceMem<bf16> &target,
                                                      float *const intermediate_output_forward,
                                                      float *const intermediate_output_backward,
                                                      const size_t batch_size, const std::vector<sycl::event> &deps) {
    // Compute activation backpropagation using parallel_for
    bf16 *const intermediate_output_forwardbf16 = reinterpret_cast<bf16 *>(intermediate_output_forward);
    bf16 *const intermediate_output_backwardbf16 = reinterpret_cast<bf16 *>(intermediate_output_backward);

    throw std::invalid_argument("Trainign not yet implemented again.");

    // return mlp_swift_fused<WIDTH, Activation::ReLU, Activation::None>(
    //     m_q, m_weights_matrices.data(), m_weightsT_matrices.data(),
    //     input.data(),            // input to forward pass
    //     target.data(),           // targets for error computation
    //     m_grads_matrices.data(), // gradients output after backward pass
    //     intermediate_output_forwardbf16, intermediate_output_backwardbf16, m_n_hidden_layers, batch_size, deps);

    return {};
}

template <int WIDTH> void SwiftNetMLP<WIDTH>::set_params(float *params) {
    auto p = m_weights_matrices.data();
    int s = m_weights_matrices.size();

    m_q.parallel_for<>(range<1>(s), [=](id<1> idx) { p[idx] = bf16(params[idx]); }).wait();
    m_weights_matrices.make_transposed(m_weightsT_matrices, m_inputs_width_padded, WIDTH, m_output_width,
                                       m_n_hidden_matrices, m_q);
}

template <int WIDTH> void SwiftNetMLP<WIDTH>::set_params(std::vector<bf16> &params) {
    m_weights_matrices.copy_from_host(params, m_q);
    m_weights_matrices.make_transposed(m_weightsT_matrices, m_inputs_width_padded, WIDTH, m_output_width,
                                       m_n_hidden_matrices, m_q);
}

template <int WIDTH> std::vector<bf16> SwiftNetMLP<WIDTH>::get_weights_matrices_as_vector() {
    std::vector<bf16> list_float(m_weights_matrices.size());
    m_weights_matrices.copy_to_host(list_float, m_q);
    return list_float;
}

template <int WIDTH> std::vector<bf16> SwiftNetMLP<WIDTH>::get_weightsT_matrices_as_vector() {
    std::vector<bf16> list_float(m_weightsT_matrices.size());
    m_weightsT_matrices.copy_to_host(list_float, m_q);
    return list_float;
}

template class SwiftNetMLP<64>;
template class SwiftNetMLP<128>;
