#pragma once

#include <stdint.h>
#include <json/json.hpp>
#include "DeviceMem.h"

using json = nlohmann::json;
using bf16 = sycl::ext::oneapi::bfloat16;

class Optimizer {
public:
	virtual ~Optimizer() {}

	virtual void step(queue q, float loss_scale, DeviceMem<bf16>& weights, DeviceMem<bf16>& weightsT, DeviceMem<bf16>& gradients, int WIDTH) const = 0;

};

