#pragma once
//#include <cmath.h>

#define SG_SIZE 8
#define WG_SIZE 8*SG_SIZE
#define BATCH_CHUNK 64

#define TM 8
#define TK 16
#define TN 8
#define SYCL_EXT_ONEAPI_MATRIX_V 4

template<typename T>
class Loss {
public:
	
	virtual void evaluate(
		const int dims,
		const int stride,
		const float scale,
		std::vector<T>& pred,
		std::vector<T>& target,
		std::vector<T>& grads,
		std::vector<T>& values
		) const = 0;
};

