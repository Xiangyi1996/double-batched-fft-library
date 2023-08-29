#pragma once
#include "optimizer.h"
#include "common.h"
#include <vector>
//#include "L2.h"
class AdamOptimizer : public Optimizer {
public:

    void step(queue q, float loss_scale, DeviceMem<bf16>& weights, DeviceMem<bf16>& weightsT, DeviceMem<bf16>& gradients, int WIDTH) override;

    void set_learning_rate(const float learning_rate);

private:
    DeviceMem<float> m_first_moments;
    DeviceMem<float> m_second_moments;

    int m_output_rows;
    int m_n_hidden_layers;
    float m_learning_rate = 1e-3f;
    float m_l2_reg = 1e-8f;
};
