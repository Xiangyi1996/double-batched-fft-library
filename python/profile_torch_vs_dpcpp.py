import numpy as np
import torch
import intel_extension_for_pytorch  # required for SwiftNet
import pytest
import time

from utils import create_models

# BATCH_SIZE = 2**6 # 64
BATCH_SIZE = 2**20  # 64
DEVICE_NAME = "xpu"


def test_fwd(model, name, iters=1000):
    # Generate random input data for testing
    torch.manual_seed(123)
    input_data = torch.randn(BATCH_SIZE, model.input_width, dtype=torch.float32).to(
        DEVICE_NAME
    )
    model.to(DEVICE_NAME)
    print("Warming up for 100 iters for JIT compilation etc")
    for _ in range(100):
        y = model(input_data)

    print("Measuring time")
    start_time = time.time()
    for idx in range(iters):
        if idx % 100 == 0:
            print(f"{name} on {DEVICE_NAME} at {idx}.")

        y = model(input_data)

    end_time = time.time()
    duration = (end_time - start_time) * 1000
    print(
        f"{name}: Execution Time: {duration:.2f} ms for {iters} iterations on {DEVICE_NAME}."
    )


if __name__ == "__main__":
    input_width = 64
    output_width = 64
    n_hidden_layers = 4
    activation_func = "relu"
    output_func = "linear"

    model_dpcpp, model_torch = create_models(
        input_width,
        n_hidden_layers,
        output_width,
        activation_func,
        output_func,
        batch_size=BATCH_SIZE,
        device_name=DEVICE_NAME,
    )
    test_fwd(model_dpcpp, "dpcpp")
    test_fwd(model_torch, "torch")
    model_dpcpp.free_memory()
