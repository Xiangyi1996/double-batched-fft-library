#pragma once
#include "SwiftNetMLP.h"
#include "loss.h"
#include "optimizer.h"
#include "trainer.h"

template<WIDTH>
struct TrainableModel {
	queue m_q;
	Loss* loss;
	Optimizer* optimizer;
	SwiftNetMLP<WIDTH>* network;
	Trainer<WIDTH> trainer;
};

template<WIDTH>
inline TrainableModel create_from_config(
	queue q,
	json config
) {

	queue m_q{ q };
	Loss* loss{create_loss<network_precision_t>(config.value("loss", json::object()))};
	Optimizer* optimizer{create_optimizer<network_precision_t>(config.value("optimizer", json::object()))};
	SwiftNetMLP<WIDTH>* network{create_network(queue q, const json& network)}
	auto trainer = Trainer<WIDTH>(network, loss, optimizer);
	return { loss, optimizer, trainer };
}
