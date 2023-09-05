import torch
import intel_extension_for_pytorch
from modules import SwiftNet
from tiny_nn import Activation
import pytest

DEVICE = "cpu"


# DEVICE = "xpu"
def test_fwd():
    batch_size = 64
    width = 64
    input_width = 64
    output_width = 64
    n_hidden_layers = 2
    activation = Activation.Linear
    output_activation = Activation.Linear
    network = SwiftNet(
        batch_size,
        width,
        input_width,
        output_width,
        n_hidden_layers,
        activation,
        output_activation,
        device=DEVICE,
    )
    print("FWD pass")
    output = network.forward(torch.ones([batch_size, input_width]).to(DEVICE))
    print(output)
    print("FWD pass end")
    network.free_memory()
    assert (
        abs((output - torch.ones([batch_size, input_width]).to(DEVICE) * 1.3401).mean())
        < 1e-3
    ), "Output values not correct. Either weights changed, or issue with fwd_pass."


if __name__ == "__main__":
    test_fwd()
