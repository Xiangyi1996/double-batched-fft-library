#pragma once

#include <stdint.h>
#include <json/json.hpp>
using json = nlohmann::json;

class Optimizer {
public:
	virtual ~Optimizer() {}

	virtual void step(float loss_scale, std::vector<bf16>& weights, std::vector<bf16>& weightsT, std::vector<bf16>& gradients) const  = 0;

};
