#pragma once

#include "loss.h"

class CrossEntropyLoss : public Loss {
  public:
    void evaluate(queue q, const int dims, const int stride, const float scale, DeviceMem<float> &preds,
                  DeviceMem<float> &targets, DeviceMem<bf16> &grads, DeviceMem<float> &values) override;
};
