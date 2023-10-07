import numpy as np
import torch
import intel_extension_for_pytorch  # required for SwiftNet

from mlp import MLP
from modules import SwiftNet, NetworkWithEncoding
from tiny_nn import Activation

import os
import pickle
import gdown
import zipfile

WIDTH = 64


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
    batch_size,
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

    # Create and test model_dpcpp
    activation = get_dpcpp_activation(activation_func)
    output_activation = get_dpcpp_activation(output_func)

    # model_dpcpp = SwiftNet(
    #     batch_size=batch_size,
    #     width=WIDTH,
    #     input_width=input_size,
    #     output_width=output_size,
    #     n_hidden_layers=hidden_size,
    #     activation=activation,
    #     output_activation=output_activation,
    #     device=device_name,
    # )
    model_dpcpp = NetworkWithEncoding(
        batch_size=batch_size,
        width=WIDTH,
        input_width=input_size,
        output_width=output_size,
        n_hidden_layers=hidden_size,
        activation=activation,
        output_activation=output_activation,
        device=device_name,
    )
    # model_dpcpp.eval()

    # Set weights of model_torch to the ones of model_dpcpp

    weights = model_dpcpp.get_reshaped_params()

    model_torch.set_weights(weights)
    return model_dpcpp, model_torch


def download_samples():
    # Check if 'data' directory exists in the current path
    if not os.path.exists("data"):
        # Create the 'data' directory
        print("Downloading from gdrive")

        url = "https://drive.google.com/file/d/11QhrgOcxSehaHOzMHjn_bXU_z2LuKqw-/view?usp=sharing"
        gdown.download_folder(url, quiet=False, use_cookies=False)
        # Here, we'll assume you have 'tinynn_samples.zip' and want to unzip it
        with zipfile.ZipFile("tinynn_samples.zip", "r") as zip_ref:
            # Extract all contents of 'data.zip' into the 'data' directory
            zip_ref.extractall()
        print("Downloaded data")


def load_from_folder(folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)

    if not os.path.exists(file_path):
        download_samples()

    with open(file_path, "rb") as file:
        data = torch.load(file)

    return data
