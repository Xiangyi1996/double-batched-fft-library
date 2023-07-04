#pragma once



#include <stdint.h>
#include <json/json.hpp>
using json = nlohmann::json;
template <typename T>
class Optimizer {
public:
	virtual ~Optimizer() {}
	

	virtual void step(float loss_scale, std::vector<bf16>& weights, std::vector<T>& gradients) const  = 0;
};



