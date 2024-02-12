import numpy as np

# import torch

# import intel_extension_for_pytorch  # required for tinyn_nn (SwiftNet inside)

from mlp import MLP
from modules import NetworkWithInputEncoding, Network
from tiny_nn import Activation

import os
import pickle
import zipfile

WIDTH = 64


def create_models(
    input_size,
    hidden_size,
    output_size,
    activation_func,
    output_func,
    device_name,
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
    # model_torch.eval()

    network_config = {
        "activation": activation_func,
        "output_activation": output_func,
        "n_neurons": WIDTH,
        "n_hidden_layers": hidden_size,
    }
    model_dpcpp = Network(
        n_input_dims=input_size,
        n_output_dims=output_size,
        network_config=network_config,
        device=device_name,
    )
    # encoding_config = {
    #     "otype": "Identity",
    #     "n_dims_to_encode": str(input_size),
    #     "scale": "1.0",
    #     "offset": "0.0",
    # }
    # model_dpcpp = NetworkWithInputEncoding(
    #     n_input_dims=input_size,
    #     n_output_dims=output_size,
    #     network_config=network_config,
    #     encoding_config=encoding_config,
    #     device=device_name,
    #     flipped_input=True,
    # )
    # model_dpcpp.eval()

    # Set weights of model_torch to the ones of model_dpcpp

    weights = model_dpcpp.get_reshaped_params()
    model_torch.set_weights(weights)
    model_torch.to(device_name)

    return model_dpcpp, model_torch
