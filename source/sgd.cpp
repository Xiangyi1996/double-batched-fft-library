#include "sgd.h"



void sgd_step(id<1> idx,
    const int n_elements,
    int output_width,
    int n_hidden_layers,
    const float loss_scale,
    const float learning_rate,
    const float l2_reg,
    bf16* weights,
    bf16* gradients,
    int WIDTH
    ) {
    int matrices_number = idx / (WIDTH * WIDTH);
    int matrices_offset = idx % (WIDTH * WIDTH);
    int packed_idx_matrices = 0;

    if (matrices_number < n_hidden_layers) {
        packed_idx_matrices = toPackedLayoutCoord(matrices_offset, WIDTH, WIDTH);
    }
    else {
        packed_idx_matrices = toPackedLayoutCoord(matrices_offset, WIDTH, output_width);
    }

    const int packed_idx = matrices_number * WIDTH * WIDTH + packed_idx_matrices;
    const bf16 weight = weights[packed_idx];
    float gradient = gradients[idx];

    gradient += l2_reg * weight;

    const bf16 new_weight = weight - learning_rate * gradient;

    weights[packed_idx] = new_weight;
}

void sgd_stepT(id<1> idx,
    const int n_elements,
    int output_width,
    int n_hidden_layers,
    const float loss_scale,
    const float learning_rate,
    const float l2_reg,
    bf16* weightsT,
    bf16* gradients,
    int WIDTH
) {
    const int i = idx / WIDTH;
    const int j = idx % WIDTH;

    const int T_idx = WIDTH * j + i;

    const int matrices_number = T_idx / (WIDTH * WIDTH);
    const int matrices_offset = T_idx % (WIDTH * WIDTH);
    int packed_idx_matrices = 0;

    if (matrices_number < n_hidden_layers) {
        int packed_idx_matrices = fromPackedLayoutCoord(matrices_offset, WIDTH, WIDTH);
    }
    else {
        int packed_idx_matrices = fromPackedLayoutCoord(matrices_offset, output_width, WIDTH);
    }

    const int packed_idx = matrices_number * WIDTH * WIDTH + packed_idx_matrices;
    const bf16 weightT = weightsT[packed_idx];
    float gradient = gradients[idx] / loss_scale;

    gradient += l2_reg * weightT;

    const bf16 new_weightT = weightT - learning_rate * gradient;

    weightsT[packed_idx] = new_weightT;
}


SGDOptimizer::SGDOptimizer(int output_rows, int n_hidden_layers, float learning_rate, float l2_reg) {
    m_output_rows = output_rows;
    m_n_hidden_layers = n_hidden_layers;
    m_learning_rate = learning_rate;
    m_l2_reg = l2_reg;
}

void SGDOptimizer::step(queue q, float loss_scale, DeviceMem<bf16>& weights, DeviceMem<bf16>& weightsT, DeviceMem<bf16>& gradients, int WIDTH)  {

    const int n_elements = weights.size();
    float learning_rate = m_learning_rate;
    float l2_reg = m_l2_reg;
    const int output_rows = m_output_rows;
    const int n_hidden_layers = m_n_hidden_layers;

    q.parallel_for<>(range<1>(n_elements), [=](id<1> idx) {
        sgd_step(idx, n_elements, output_rows, n_hidden_layers, loss_scale, learning_rate, l2_reg, weights.data(), gradients.data(), WIDTH);
        }).wait();

        q.parallel_for<>(range<1>(n_elements), [=](id<1> idx) {
            sgd_stepT(idx, n_elements, output_rows, n_hidden_layers, loss_scale, learning_rate, l2_reg, weightsT.data(), gradients.data(), WIDTH);
            }).wait();


}

void SGDOptimizer::set_learning_rate(const float learning_rate) {
    m_learning_rate = learning_rate;
}
