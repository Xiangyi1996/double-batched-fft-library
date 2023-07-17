#pragma once
#include "SwiftNetMLP.h"
#include "loss.h"
#include "optimizer.h"
#include "trainer.h"
#include "L2.h"

template<int WIDTH>
struct TrainableModel {
	queue m_q;
	Loss* loss;
	/*Optimizer* optimizer;
	SwiftNetMLP<WIDTH>* network;
	Trainer<WIDTH> trainer;*/
};

template<int WIDTH>
TrainableModel<WIDTH> create_from_config(
	queue q,
	json config
) {

	queue m_q{ q };
	Loss* loss = create_loss(config.value("loss", json::object()));
	//Optimizer* optimizer{create_optimizer(config.value("optimizer", json::object()))};
	//SwiftNetMLP<WIDTH>* network{create_network<WIDTH>(q, config.value("network", json::object()))};
	//auto trainer = Trainer<WIDTH>(*network, *loss, *optimizer);
	//return { m_q, loss, optimizer,network,  trainer };
	return{ m_q, loss };
}
