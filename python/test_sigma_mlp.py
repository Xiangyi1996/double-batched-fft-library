import torch
from torch import nn
import tinycudann as tcnn
import numpy as np
from copy import deepcopy as dc
import time
from utils import load_from_folder

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
FOLDER_PATH = f"{CUR_DIR}/tinynn_samples/"


class TestSigmaMlp:
    def test_sigma_mlp(self):
        print("===test_sigma_mlp===")
        params = load_from_folder(FOLDER_PATH, "sigma_mlp_params.pth")
        sigma_mlp = tcnn.Network(
            n_input_dims=params["n_input_dims"],
            n_output_dims=params["n_output_dims"],
            network_config=params["network_config"],
        )
        sigma_mlp.load_state_dict(params["state_dict"])

        input_tensor = params["input"]
        output_tensor_ref = params["output"]

        output_tensor = sigma_mlp(input_tensor)

        absolute_error = torch.mean(abs(output_tensor - output_tensor_ref))
        mean_vals = torch.mean(output_tensor_ref)

        relative_error = absolute_error / mean_vals

        print(f"absolute error: {absolute_error:.4f}")
        print(f"relative_error: {relative_error:.4f}")

        assert absolute_error < 1e-5
        assert relative_error < 1e-5


if __name__ == "__main__":
    x = TestSigmaMlp()
    x.test_sigma_mlp()
