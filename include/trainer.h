#pragma once

#include "DeviceMem.h"
#include "Network.h"

class Trainer {
  public:
    Trainer(Network &network) { m_network = &network; }

    std::vector<sycl::event> training_step(DeviceMem<bf16> &input, DeviceMem<bf16> &losses, int run_inference,
                                           float *out_inter_forw, float *out_inter_backw, int batch_size,
                                           const std::vector<sycl::event> &dependencies) {

        if (run_inference) {
            return m_network->inference(input, out_inter_forw, batch_size, dependencies);
        } else {
            // return m_network->training(input, target, m_network->m_forward, m_network->m_out_inter, deps);
            auto deps = m_network->forward_pass(input, out_inter_forw, batch_size, dependencies);
            // auto e = m_loss->evaluate(m_network->get_queue(), scale, output, target, grads, losses);
            // deps = {e};

            deps = m_network->backward_pass(losses, out_inter_backw, out_inter_forw, batch_size, deps);

            // no optimisation as we run benchmarking
            // m_optim->step(m_network->get_queue(), scale,
            // m_network->m_weights_matrices,
            //               m_network->m_weightsT_matrices,
            //               m_network->m_grads_matrices, WIDTH);

            return deps;
        }
    }

    Network *m_network;

    void initialize_params(int use_constant = 0) { m_network->initialize_params(use_constant); }

  private:
};