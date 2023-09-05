#pragma once

#include "DeviceMem.h"
#include "L2.h"
#include "Network.h"
#include "loss.h"
#include "optimizer.h"

class Trainer {
 public:
  Trainer(Network& network, Loss& loss, Optimizer& optim) {
    m_network = &network;
    m_loss = &loss;
    m_optim = &optim;
  }

  void training_step(DeviceMem<bf16>& input, DeviceMem<float>& output,
                     DeviceMem<float>& target, DeviceMem<bf16>& grads,
                     DeviceMem<float>& losses, const float scale,
                     const int WIDTH, int forward_only = 1) {
    // const int input_size = input.size();
    // const int batch_size = std::pow(2, 19);

    auto p = m_network->m_forward;
    m_network->get_queue().parallel_for<>(
        range<1>(input.size()), [=](id<1> idx) { p[idx] = input.data()[idx]; });

    m_network->forward_pass(input, m_network->m_forward, m_network->m_A_forward,
                            m_network->m_B_forward, m_network->m_C_forward,
                            output);
    if (forward_only) {
      return;
    }
    m_loss->evaluate(m_network->get_queue(), WIDTH, WIDTH, scale, output,
                     target, grads, losses);

    m_network->backward_pass(
        input, grads, m_network->m_out_inter, m_network->m_deltas_temp,
        m_network->m_deltas, m_network->m_A_backward, m_network->m_B_backward,
        m_network->m_C_backward, m_network->m_A_backward_last_layer,
        m_network->m_B_backward_last_layer, m_network->m_C_backward_last_layer,
        m_network->m_D_backward_last_layer, m_network->m_E_backward_last_layer,
        m_network->m_F_backward_last_layer, m_network->m_A_dgemm,
        m_network->m_B_dgemm, m_network->m_C_dgemm, m_network->m_forward);

    m_optim->step(m_network->get_queue(), scale, m_network->m_weights_matrices,
                  m_network->m_weightsT_matrices, m_network->m_grads_matrices,
                  WIDTH);
  }
  Network* m_network;
  Loss* m_loss;
  Optimizer* m_optim;

  void initialize_params() { m_network->initialize_params(); }

 private:
};
