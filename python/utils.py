import numpy as np
import torch
import intel_extension_for_pytorch  # required for SwiftNet

from mlp import MLP
from modules import SwiftNet
from tiny_nn import Activation


WIDTH = 64
BATCH_SIZE = 64
DEVICE_NAME = "cpu"


def get_dpcpp_activation(name):
    # activation = Activation.ReLU
    if name == "relu":
        activation = Activation.ReLU
    elif name == "tanh":
        activation = Activation.Tanh
    elif name == "sigmoid":
        activation = Activation.Sigmoid
    elif name == "linear":
        activation = Activation.Linear
    else:
        raise NotImplementedError(f"Activation: {name} not defined")

    return activation


def create_models(
    input_size,
    hidden_size,
    output_size,
    activation_func,
    output_func,
    batch_size=BATCH_SIZE,
    device_name=DEVICE_NAME,
):
    # Create and test CustomMLP
    hidden_sizes = [WIDTH] * hidden_size
    model_torch = MLP(
        input_size,
        hidden_sizes,
        output_size,
        activation_func,
        output_func,
        use_batchnorm=False,
    ).to(device_name)
    model_torch.eval()

    # Create and test model_dpcpp
    activation = get_dpcpp_activation(activation_func)
    output_activation = get_dpcpp_activation(output_func)

    model_dpcpp = SwiftNet(
        batch_size,
        WIDTH,
        input_size,
        output_size,
        hidden_size,
        activation=activation,
        output_activation=output_activation,
        device=device_name,
    )
    model_dpcpp.eval()

    # Set weights of model_torch to the ones of model_dpcpp
    params = []
    for param in model_dpcpp.parameters():
        if len(param.shape) > 1:
            params.append(param.data)
    weights = {
        "input_weights": params[0] if params else None,
        "middle_weights": model_dpcpp.get_reshaped_params(),
    }
    model_torch.set_weights(weights)
    return model_dpcpp, model_torch
