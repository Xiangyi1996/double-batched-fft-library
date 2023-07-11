#pragma once

#include <stdint.h>
#include <json/json.hpp>
#include "DeviceMem.h"

using json = nlohmann::json;

class Optimizer {
public:
	virtual ~Optimizer() {}

	virtual void step(queue q, float loss_scale, DeviceMem<bf16>& weights, DeviceMem<bf16>& weightsT, DeviceMem<bf16>& gradients) const = 0;

};

