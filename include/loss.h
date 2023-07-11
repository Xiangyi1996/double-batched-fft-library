#pragma once
//#include <cmath.h>
#include "DeviceMem.h"

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
		queue q,
		const int dims,
		const int stride,
		const float scale,
		DeviceMem<float>& pred,
		DeviceMem<float>& target,
		DeviceMem<bf16>& grads,
		DeviceMem<float>& values
	) const = 0;
};

