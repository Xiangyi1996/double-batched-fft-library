// #pragma once

// #include "DeviceMem.h"
// #include "Encodings/identity.h"
// #include "Network.h"
// #include "encoding.h"
// #include "l2.h"
// #include "loss.h"
// #include "optimizer.h"

// class Trainer {
//   public:
//     Trainer(Network &network, Loss &loss, Optimizer &optim) {
//         m_network = &network;
//         m_loss = &loss;
//         m_optim = &optim;
//     }

//     std::vector<sycl::event> training_step(DeviceMem<bf16> &input, DeviceMem<float> &output, DeviceMem<bf16> &target,
//                                            DeviceMem<bf16> &grads, DeviceMem<bf16> &losses, const float scale,
//                                            const int WIDTH, int run_inference, std::vector<sycl::event> &deps) {

//         if (run_inference) {
//             return m_network->inference(input, m_network->m_forward, deps);
//         } else {
//             return m_network->training(input, target, m_network->m_forward, m_network->m_out_inter, deps);
//             // deps = m_network->forward_pass(input, m_network->m_forward, deps);
//             //  auto e = m_loss->evaluate(m_network->get_queue(), WIDTH, WIDTH, scale, m_network->GetOutput(),
//             //update
//             //  to use m_forward instead of output
//             //                   target, grads, losses, deps);
//             //  deps = {e};

//             // deps = m_network->backward_pass(
//             //     grads, m_network->m_out_inter, m_network->m_forward, deps);

//             // no optimisation as we run benchmarking
//             // m_optim->step(m_network->get_queue(), scale,
//             // m_network->m_weights_matrices,
//             //               m_network->m_weightsT_matrices,
//             //               m_network->m_grads_matrices, WIDTH);

//             // return deps;
//         }

//         return deps;
//     }

//     Network *m_network;
//     Loss *m_loss;
//     Optimizer *m_optim;

//     void initialize_params(int use_constant = 0) { m_network->initialize_params(use_constant); }

//   private:
// };