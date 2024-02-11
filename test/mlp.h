#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

template <typename T> std::vector<T> repeat_inner_vectors(const std::vector<std::vector<T>> &layer_outputs, int N) {
    // this function is to replicate how to stack over batch sizes in Swiftnet
    std::vector<T> result;

    // Iterate over each vector inside layer_outputs
    for (const auto &inner_vec : layer_outputs) {
        // For each element in the inner vector, repeat it N times
        for (int i = 0; i < N; ++i) {
            for (const T &element : inner_vec) {
                result.push_back(element);
            }
        }
    }

    return result;
}
// Sigmoid function
double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

// Derivative of the sigmoid function
double dsigmoid(double x) {
    double sig = sigmoid(x);
    return sig * (1 - sig);
}

// Mean squared error loss
double mse(Eigen::VectorXd y, Eigen::VectorXd y_pred) { return (y - y_pred).array().pow(2).mean(); }

// Convert Eigen::VectorXd to std::vector
template <typename T> std::vector<T> eigenToStdVector(const Eigen::VectorXd &eigenVector) {
    std::vector<T> stdVector(eigenVector.data(), eigenVector.data() + eigenVector.size());
    return stdVector;
}

template <typename T> void printVector(std::string name, const std::vector<T> &vec) {
    std::cout << name << std::endl;
    for (const T &value : vec) {
        std::cout << value << ", ";
    }
    std::cout << std::endl;
}

template <typename SourceType, typename DestType>
std::vector<DestType> convert_vector(const std::vector<SourceType> &sourceVec) {
    std::vector<DestType> destVec;
    destVec.reserve(sourceVec.size());

    std::transform(sourceVec.begin(), sourceVec.end(), std::back_inserter(destVec),
                   [](const SourceType &val) { return static_cast<DestType>(val); });

    return destVec;
}

template <typename T> std::vector<T> stack_vector(const std::vector<T> &vec, size_t N) {
    // this function is to replicate how to stack over batch sizes in Swiftnet
    std::vector<T> result;
    result.reserve(vec.size() * N); // Reserve space to avoid reallocation
    for (size_t i = 0; i < N; ++i) {
        // Append a copy of the vector to the end of the result vector
        result.insert(result.end(), vec.begin(), vec.end());
    }
    return result;
}

// Define the Matrix struct with basic matrix-vector multiplication
template <typename T> struct Matrix {
    std::vector<std::vector<T>> data;

    // Constructor for the matrix of the given dimension with an initial value
    Matrix(std::size_t rows, std::size_t cols, T initialValue = T()) : data(rows, std::vector<T>(cols, initialValue)) {}

    // Matrix-vector multiplication
    std::vector<T> operator*(const std::vector<T> &vec) const {
        if (data.empty() || data[0].size() != vec.size()) {
            throw std::invalid_argument("Matrix and vector dimensions do not match for multiplication.");
        }

        std::size_t rows = data.size();
        std::size_t cols = data[0].size();
        std::vector<T> result(rows, T());

        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) {
                result[i] += data[i][j] * vec[j];
            }
        }

        return result;
    }

    // Pretty print function for the matrix
    void print(int width = 10) const {
        for (const auto &row : data) {
            for (const T &val : row) {
                std::cout << std::setw(width) << val << " ";
            }
            std::cout << '\n';
        }
    }

    // Set linspace weights for the matrix
    void set_weights_linspace(T start, T end) {
        std::size_t total_elements = rows() * cols();
        T increment = (end - start) / (total_elements - 1);

        for (std::size_t i = 0; i < total_elements; ++i) {
            std::size_t row = i / cols();
            std::size_t col = i % cols();
            data[row][col] = start + i * increment;
        }
    }

    std::size_t rows() const { return data.size(); }
    std::size_t cols() const { return data.empty() ? 0 : data[0].size(); }
};

// Utility functions for linear and ReLU activations and their derivatives
double relu(double x) { return std::max(0.0, x); }

double linear(double x) {
    return x; // Identity function
}

double drelu(double x) { return x > 0 ? 1 : 0; }

double dlinear(double x) { return 1; }

// MLP class using 'Matrix' struct for matrix operations
template <typename T> class MLP {
  private:
    std::vector<Matrix<T>> weights;
    int n_hidden_layers;
    int batch_size;

    int inputDim;
    int outputDim;

    std::string activation;
    std::string output_activation;
    // Utility function to initialize the weights randomly

    void initialize_random_weights(double weight_val_scaling_factor) {
        std::mt19937 gen(42);

        double xavier_stddev = std::sqrt(2.0 / (inputDim + outputDim));
        std::uniform_real_distribution<> dis(-weight_val_scaling_factor * xavier_stddev,
                                             weight_val_scaling_factor * xavier_stddev);

        for (auto &weight_matrix : weights) {
            for (auto &row : weight_matrix.data) {
                for (T &val : row) {
                    val = static_cast<T>(dis(gen));
                }
            }
        }
    }

  public:
    MLP(int inputDim_, int hiddenDim, int outputDim_, int n_hidden_layers_, int batch_size_, std::string activation_,
        std::string output_activation_, std::string weight_init_mode)
        : n_hidden_layers(n_hidden_layers_), activation(activation_), output_activation(output_activation_),
          inputDim(inputDim_), outputDim(outputDim_), batch_size(batch_size_) {
        /// TODO: normalisation of batch_size is only necessary because this is the implementation for batches of
        /// size 1. Later, make input tensor  not just a vector but a matrix (rows being batch_size dim). Then we
        /// don't need this

        // Initialize first layer weights
        weights.push_back(Matrix<T>(hiddenDim, inputDim));

        // Initialize hidden layers weights
        for (int i = 0; i < n_hidden_layers - 2; ++i) {
            weights.push_back(Matrix<T>(hiddenDim, hiddenDim));
        }

        // Initialize output layer weights
        weights.push_back(Matrix<T>(outputDim, hiddenDim));
        double weight_val = 0.1;
        // linspace initialization of weights
        if (weight_init_mode == "linspace") {
            for (int i = 0; i < weights.size(); i++) {
                weights[i].set_weights_linspace(static_cast<T>(-weight_val * (i + 1)),
                                                static_cast<T>(weight_val * (i + 1)));
            }
        } else if (weight_init_mode == "random") {
            initialize_random_weights(1.0);
        } else {
            // default initialization of weights (to small random values or zeros)
            for (auto &weight_matrix : weights) {
                for (auto &row : weight_matrix.data) {
                    for (T &val : row) {
                        val = static_cast<T>(weight_val); // or any small number to initialize
                    }
                }
            }
        }
    }
    std::vector<std::vector<T>> forward(const std::vector<T> &x, bool get_interm_fwd) {
        std::vector<std::vector<T>> layer_outputs(n_hidden_layers + 1);
        layer_outputs[0] = x;

        // Input to hidden layers
        for (int i = 0; i < n_hidden_layers - 1; i++) {
            // Multiply with weights (assuming matrix-vector multiplication)
            layer_outputs[i + 1] = weights[i] * layer_outputs[i];

            // Apply activation function for hidden layers
            for (T &val : layer_outputs[i + 1]) {
                val = activation == "relu" ? relu(val) : linear(val);
            }
        }

        // Hidden layer to output layer
        layer_outputs[n_hidden_layers] = weights[n_hidden_layers - 1] * layer_outputs[n_hidden_layers - 1];

        // Apply activation function to the output layer
        for (T &val : layer_outputs[n_hidden_layers]) {
            val = output_activation == "relu" ? relu(val) : linear(val);
        }

        if (get_interm_fwd) {
            return layer_outputs;
        } else {
            return {layer_outputs[n_hidden_layers]}; // Return final layer output
        }
    }

    // Implement the backward pass to compute gradients
    void backward(const std::vector<T> &input, const std::vector<T> &target, std::vector<Matrix<T>> &weight_gradients,
                  std::vector<std::vector<T>> &loss_grads, std::vector<T> &loss, T loss_scale) {
        // Forward pass to get intermediate activations
        auto layer_outputs = forward(input, true);

        // Vectors to hold the gradients of the loss with respect to the activations
        std::vector<std::vector<T>> delta(layer_outputs.begin() + 1, layer_outputs.end());

        // Calculate the gradient for the output layer
        // Also, compute the MSE loss for the given batch
        for (std::size_t i = 0; i < delta.back().size(); ++i) {
            T error = (layer_outputs.back()[i] - target[i]);
            loss.push_back(error * error / (delta.back().size() * batch_size)); // Squared error for MSE
            delta.back()[i] = loss_scale * 2 * error / (delta.back().size());   // dLoss/dOutput
            if (output_activation == "relu") {
                delta.back()[i] *= drelu(layer_outputs.back()[i]); // ReLU derivative
            } else {
                delta.back()[i] *= dlinear(layer_outputs.back()[i]); // Linear derivative
            }
        }

        // Go through layers in reverse order to propagate the error
        for (int i = n_hidden_layers - 2; i >= 0; --i) {
            // Calculate delta for next layer (i.e., previous in terms of forward pass)
            std::vector<T> new_delta(weights[i + 1].cols(), T(0));
            for (std::size_t col = 0; col < weights[i + 1].cols(); ++col) {
                for (std::size_t row = 0; row < weights[i + 1].rows(); ++row) {
                    new_delta[col] += delta[i + 1][row] * weights[i + 1].data[row][col];
                }
            }

            // Apply derivative of the activation function
            for (std::size_t j = 0; j < layer_outputs[i + 1].size(); ++j) {
                if (activation == "relu") {
                    new_delta[j] *= drelu(layer_outputs[i + 1][j]);
                } else {
                    new_delta[j] *= dlinear(layer_outputs[i + 1][j]);
                }
            }
            delta[i] = new_delta;
        }

        for (int i = 0; i < delta.size(); i++) {
            auto loss_grad_el = delta[i];
            for (int idx = 0; idx < loss_grad_el.size(); idx++) {
                loss_grad_el[idx] /= batch_size;
            }
            loss_grads.push_back(loss_grad_el);
        }

        for (int i = n_hidden_layers - 1; i >= 0; i--) {

            // Initialize gradient matrix for next layer weights
            Matrix<T> layer_gradient(weights[i].rows(), weights[i].cols());
            // Gradient for this layer's weights
            for (std::size_t row = 0; row < layer_gradient.rows(); ++row) {
                for (std::size_t col = 0; col < layer_gradient.cols(); ++col) {
                    // note that layer_output[0] is input, thus this is the
                    // one from the previous layer
                    layer_gradient.data[row][col] = delta[i][col] * layer_outputs[i][row];
                }
            }
            weight_gradients[i] = layer_gradient;
        }
    }
    std::vector<T> getUnpackedWeights() const {
        std::vector<T> all_weights;
        for (const Matrix<T> &weight_matrix : weights) {

            for (const std::vector<T> &row : weight_matrix.data) {
                all_weights.insert(all_weights.end(), row.begin(), row.end());
            }
        }

        return all_weights;
    }
};