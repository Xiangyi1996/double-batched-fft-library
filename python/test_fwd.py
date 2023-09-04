import torch
import intel_extension_for_pytorch
from modules import SwiftNet
from tiny_nn import Activation

# DEVICE = "cpu"
DEVICE = "xpu"

if __name__ == "__main__":
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
    # output =  network.forward(torch.tensor([0.1] * (input_width * batch_size), device="xpu"))
    print(output)
    print("FWD pass end")
    network.free_memory()
