#pragma once
#includ <cmath.h>

template<typename T>
class Loss {
public:
	
	virtual void evaluate(
		queue q,
		const int dims,
		const int stride,
		const float scale,
		const std::vector<T> pred,
		const std::vector<float> target,
		std::vector<T> grads,
		std::vector<float> values
		) const = 0;
};

