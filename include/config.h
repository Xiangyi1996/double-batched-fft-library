#pragma once
#include "SwiftNetMLP.h"
#include "loss.h"
#include "optimizer.h"
#include "trainer.h"
#include "L1.h"
#include "L2.h"
#include "RelativeL1.h"
#include "RelativeL2.h"
#include "cross_entropy.h"
#include "adam.h"
#include "sgd.h"


template<int WIDTH>
struct TrainableModel {
	queue m_q;
	Loss* loss;
	Optimizer* optimizer;
	SwiftNetMLP<WIDTH>* network;
	Trainer<WIDTH> trainer;
};

template<int WIDTH>
TrainableModel<WIDTH> create_from_config(
	queue q,
	json config
) {
	queue m_q{ q };
	std::string loss_type = config.value("loss", json::object()).value("otype", "RelativeL2");
	std::string optimizer_type = config.value("optimizer", json::object()).value("otype", "sgd");

	Loss* loss;
	Optimizer* optimizer;
	if (isequalstring(loss_type, "L2")) {
		loss = new L2Loss();
	}
	else if (isequalstring(loss_type, "RelativeL2")) {
		loss = new RelativeL2Loss();
	}
	else if (isequalstring(loss_type, "L1")) {
		loss = new L1Loss();
	}
	else if (isequalstring(loss_type, "RelativeL1")) {
		loss = new RelativeL1Loss();
	}
	else if (isequalstring(loss_type, "CrossEntropy")) {
		loss = new CrossEntropyLoss();
	}
	else {
		throw std::runtime_error{"Invalid loss type: "};
	}
	
	if (isequalstring(optimizer_type, "Adam")) {
		optimizer = new AdamOptimizer<WIDTH>();
	}

	else if (isequalstring(optimizer_type, "SGD")) {
		optimizer = new SGDOptimizer<WIDTH>(config.value("optimizer", json::object()).value("output_width", 64) , config.value("optimizer", json::object()).value("n_hidden_layer", 2) , config.value("optimizer", json::object()).value("learning_rate", 1e-3f) , config.value("optimizer", json::object()).value("l2_reg", 1e-8f) );
	}
	else {
		throw std::runtime_error{"Invalid optimizer type: "};
	}
	//Loss* loss = create_loss(config.value("loss", json::object()));
	//Optimizer* optimizer{create_optimizer(config.value("optimizer", json::object()))};
	SwiftNetMLP<WIDTH>* network{create_network<WIDTH>(q, config.value("network", json::object()))};
	auto trainer = Trainer<WIDTH>(*network, *loss, *optimizer);
	return { m_q, loss, optimizer,network,  trainer };

}
