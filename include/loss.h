#pragma once
// #include <cmath.h>
#include "DeviceMem.h"

#define SYCL_EXT_ONEAPI_MATRIX_V 4

using bf16 = sycl::ext::oneapi::bfloat16;

class Loss {
  public:
    virtual void evaluate(queue q, const int dims, const int stride, const float scale, DeviceMem<float> &pred,
                          DeviceMem<float> &target, DeviceMem<bf16> &grads, DeviceMem<float> &values) = 0;
};
