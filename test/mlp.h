#include <Eigen/Dense>

// ReLU function
double relu(double x) { return x > 0 ? x : 0; }

// Derivative of the ReLU function
double drelu(double x) { return x > 0 ? 1 : 0; }

// Mean squared error loss
double mse(Eigen::VectorXd y, Eigen::VectorXd y_pred) { return (y - y_pred).array().pow(2).mean(); }

// Convert Eigen::VectorXd to std::vector
template <typename T> std::vector<T> eigenToStdVector(const Eigen::VectorXd &eigenVector) {
    std::vector<T> stdVector(eigenVector.data(), eigenVector.data() + eigenVector.size());
    return stdVector;
}

template <typename T> class MLP {
  private:
    std::vector<Eigen::MatrixXd> weights;
    int n_hidden_layers;
    int batch_size;

  public:
    MLP<T>(int inputDim, int hiddenDim, int outputDim, int n_hidden_layers, int batch_size, bool linspace_weights)
        : n_hidden_layers{n_hidden_layers}, batch_size{batch_size} {
        /// TODO: normalisation of batch_size is only necessary because this is the implementation for batches of
        /// size 1. Later, make input tensor  not just a vector but a matrix (rows being batch_size dim). Then we don't
        /// need this

        // first layer
        weights.push_back(Eigen::MatrixXd::Ones(hiddenDim, inputDim) * 0.1);

        // hidden layers
        for (int i = 0; i < n_hidden_layers - 3; i++) {
            weights.push_back(Eigen::MatrixXd::Ones(hiddenDim, hiddenDim) * 0.1);
        }
        // last layer
        weights.push_back(Eigen::MatrixXd::Ones(outputDim, hiddenDim) * 0.1);

        if (linspace_weights) {
            double start = -1.0;
            double end = 1.0;
            for (int i = 0; i < weights.size(); i++) {
                set_weights_linspace(weights[i], start, end);
            }
        }
    }

    void set_weights_linspace(Eigen::MatrixXd &W, double start, double end) {
        int rows = W.rows();
        int cols = W.cols();
        double increment = (end - start) / ((rows * cols) - 1);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                W(i, j) = start + (i + j * cols) * increment;
            }
        }
    }

    std::vector<T> getUnpackedWeights() {
        std::vector<T> all_weights;

        for (Eigen::MatrixXd &weight : weights) {
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> transposed_weight =
                weight.transpose();
            all_weights.insert(all_weights.end(), transposed_weight.data(),
                               transposed_weight.data() + transposed_weight.size());
        }

        return all_weights;
    }

    std::vector<T> forward(Eigen::VectorXd x) {
        std::vector<Eigen::VectorXd> layer_outputs(weights.size());
        // Input -> First hidden layer
        layer_outputs[0] = (weights[0] * x).unaryExpr(&relu);
        // Hidden layers -> Output layer (execept last layer)
        for (int i = 1; i < weights.size() - 1; ++i) {
            layer_outputs[i] = (weights[i] * layer_outputs[i - 1]).unaryExpr(&relu);
        }
        // Last layer no activation function
        layer_outputs.back() = weights.back() * layer_outputs[weights.size() - 2];

        // // assuming all width in input, hidden and out are the same
        // Eigen::VectorXd all_layers = Eigen::VectorXd::Zero(layer_outputs[0].size() * weights.size());
        // for (int i = 0; i < weights.size(); i++) {
        //     all_layers.segment(i * layer_outputs[0].size(), layer_outputs[0].size()) = layer_outputs[i];
        // }

        return eigenToStdVector<T>(layer_outputs.back());
    }

    void backward(Eigen::VectorXd input, Eigen::VectorXd target, std::vector<std::vector<T>> &flattened_grads,
                  std::vector<std::vector<T>> &loss_grads) {
        std::vector<Eigen::VectorXd> A(n_hidden_layers);
        A[0] = input; // A represents each layer's data
        // Forward Pass
        for (int i = 0; i < n_hidden_layers - 1; i++) {

            Eigen::VectorXd Ci = weights[i] * A[i];

            if (i < n_hidden_layers - 2) {
                A[i + 1] = Ci.unaryExpr(&relu); // Within hidden layers, relu activation function is assumed.
            } else {
                A[i + 1] = Ci; // In the output layer, no activation function is assumed
            }
        }

        // Compute loss
        double loss = (A.back() - target).squaredNorm() / target.size();

        // Compute gradients
        std::vector<Eigen::VectorXd> D(n_hidden_layers);
        D.back() = 2 * (A.back() - target) / target.size();

        std::vector<Eigen::MatrixXd> G(n_hidden_layers - 1);
        G.back() = D.back() * A[n_hidden_layers - 2].transpose();

        // Backward pass
        for (int i = n_hidden_layers - 1; i > 1; i--) {
            D[i - 1] = weights[i - 1].transpose() * D[i];
            D[i - 1] = D[i - 1].array() * A[i - 1].unaryExpr(&drelu).array();

            G[i - 2] = D[i - 1] * A[i - 2].transpose();
        }

        for (int i = 0; i < G.size(); i++) {
            std::vector<T> grad_vector =
                eigenToStdVector<T>(Eigen::Map<Eigen::VectorXd>(G[i].data(), G[i].rows() * G[i].cols()));
            flattened_grads.push_back(grad_vector);
        }
        for (int i = 1; i < D.size(); i++) {
            loss_grads.push_back(eigenToStdVector<T>(D[i] / batch_size));
        }
    }
};