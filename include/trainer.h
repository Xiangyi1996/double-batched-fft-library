#pragma once

#include "DeviceMem.h"
#include "Network.h"

template <typename T> class Trainer {
  public:
    Trainer(Network<T> *network) : m_network(network) {}

    std::vector<sycl::event> training_step(const DeviceMem<T> &input, DeviceMem<T> &losses,
                                           DeviceMem<T> &out_inter_forw, DeviceMem<T> &out_inter_backw,
                                           const int batch_size, const std::vector<sycl::event> &dependencies) {

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

  private:
    Network *m_network;
};