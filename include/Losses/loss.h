#pragma once
//#include <cmath.h>

#define SG_SIZE 8
#define WG_SIZE 8*SG_SIZE
#define BATCH_CHUNK 64

#define TM 8
#define TK 16
#define TN 8
#define SYCL_EXT_ONEAPI_MATRIX_V 4

class Loss {
public:
	
	virtual void evaluate(
		const int dims,
		const int stride,
		const float scale,
		std::vector<float>& pred,
		std::vector<float>& target,
		std::vector<bf16>& grads,
		std::vector<float>& values
		) const = 0;
};
