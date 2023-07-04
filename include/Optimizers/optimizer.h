#pragma once



#include <stdint.h>
#include <json/json.hpp>
using json = nlohmann::json;
template <typename T>
class Optimizer {
public:
	virtual ~Optimizer() {}
	virtual void update_hyperparams(const json& params) = 0;
	virtual json hyperparams() const = 0;
	

	virtual void step(float loss_scale, float* weights_full_precision, T* weights, const T* gradients) = 0;
	virtual float learning_rate() const = 0;
	virtual void set_learning_rate(float val) = 0;
	virtual uint32_t step() const = 0;
	virtual uint32_t n_weights() const = 0;
	virtual T* custom_weights() const = 0;
};



