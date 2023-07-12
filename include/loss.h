#pragma once
//#include <cmath.h>
#include "DeviceMem.h"
#include <json/json.hpp>

#define SG_SIZE 8
#define WG_SIZE 8*SG_SIZE
#define BATCH_CHUNK 64

#define TM 8
#define TK 16
#define TN 8
#define SYCL_EXT_ONEAPI_MATRIX_V 4

using bf16 = sycl::ext::oneapi::bfloat16;
using json = nlohmann::json;

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

Loss* create_loss(const json& params);

