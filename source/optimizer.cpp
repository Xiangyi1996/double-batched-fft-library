#include "optimizer.h"

#include "adam.h"
#include "average.h"
#include "lookahead.h"
#include "sgd.h"


Optimizer<T>* create_optimizer(const json& optimizer) {
	std::string optimizer_type = optimizer.value("otype", "Adam");

	if (equals_case_insensitive(optimizer_type, "Adam")) {
		return new AdamOptimizer{ };
	}
	else if (equals_case_insensitive(optimizer_type, "Average")) {
		return new AverageOptimizer{};
	}
	else if (equals_case_insensitive(optimizer_type, "Lookahead")) {
		return new LookaheadOptimizer{  };
	}
	else if (equals_case_insensitive(optimizer_type, "SGD")) {
		return new SGDOptimizer{ };
	}
}