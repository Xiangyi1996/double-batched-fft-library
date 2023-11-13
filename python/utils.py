import numpy as np
import torch
import intel_extension_for_pytorch  # required for tinyn_nn (SwiftNet inside)

from mlp import MLP
from modules import NetworkWithInputEncoding
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
    encoding_config = {
        "otype": "Identity",
        "n_dims_to_encode": str(input_size),
        "scale": "1.0",
        "offset": "0.0",
    }
    model_dpcpp = NetworkWithInputEncoding(
        n_input_dims=input_size,
        n_output_dims=output_size,
        network_config=network_config,
        encoding_config=encoding_config,
        device=device_name,
        flipped_input=True,
    )
    # model_dpcpp.eval()

    # Set weights of model_torch to the ones of model_dpcpp

    weights = model_dpcpp.get_reshaped_params()

    model_torch.set_weights(weights)
    return model_dpcpp, model_torch


# def download_samples():
#     import gdown

#     # Check if 'data' directory exists in the current path
#     if not os.path.exists("data"):
#         # Create the 'data' directory
#         print("Downloading from gdrive")

#         url = "https://drive.google.com/file/d/11QhrgOcxSehaHOzMHjn_bXU_z2LuKqw-/view?usp=sharing"
#         gdown.download_folder(url, quiet=False, use_cookies=False)
#         # Here, we'll assume you have 'tinynn_samples.zip' and want to unzip it
#         with zipfile.ZipFile("tinynn_samples.zip", "r") as zip_ref:
#             # Extract all contents of 'data.zip' into the 'data' directory
#             zip_ref.extractall()
#         print("Downloaded data")


# def load_from_folder(folder_path, file_name):
#     file_path = os.path.join(folder_path, file_name)

#     if not os.path.exists(file_path):
#         download_samples()

#     with open(file_path, "rb") as file:
#         data = torch.load(file)

#     return data
