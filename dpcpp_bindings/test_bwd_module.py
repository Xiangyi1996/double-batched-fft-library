import torch
import intel_extension_for_pytorch
from modules import SwiftNet
from tiny_nn import Activation
import numpy as np

if __name__ == "__main__":
    batch_size = 64
    width = 64
    input_width = 64
    output_width = 64
    n_hidden_layers = 2
    activation = Activation.ReLU
    output_activation = Activation.Linear

    network = SwiftNet(
        batch_size,
        width,
        input_width,
        output_width,
        n_hidden_layers,
        activation,
        output_activation,
    )

    x = torch.linspace(0, 10, steps=4096)
    y = 2 * x

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)

    for n in range(500):
        y_pred = network(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"{n} - Loss: {loss}")

    network.free_memory()
