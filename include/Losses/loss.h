#pragma once
#include <cmath.h>

#define SG_SIZE 8
#define WG_SIZE 8*SG_SIZE
#define BATCH_CHUNK 64

template<typename T>
class Loss {
public:
	
	virtual void evaluate(
		const int dims,
		const int stride,
		const float scale,
		const std::vector<T> pred,
		const std::vector<float> target,
		std::vector<T> grads,
		std::vector<float> values
		) const = 0;
};

