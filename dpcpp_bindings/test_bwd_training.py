import torch
import intel_extension_for_pytorch
from modules import SwiftNet
from tiny_nn import Activation
import numpy as np


if __name__ == "__main__":
    batch_size = 64
    width = 64
    input_width = 2
    output_width = 1
    n_hidden_layers = 1

    DATA_AMOUNT = 400
    x = [torch.linspace(0.01, 10, steps=DATA_AMOUNT)] * input_width
    x = torch.stack((x)).T
    dim_reduce = torch.ones(input_width, output_width)
    y = 2 * x @ dim_reduce

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

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    for n in range(1000):
        all_loss = []
        for idx in range(x.shape[0] // batch_size):
            y_pred = network(x[idx * batch_size : (idx + 1) * batch_size, ...])
            loss = loss_fn(
                y_pred,
                y[idx * batch_size : (idx + 1) * batch_size],
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_loss.append(loss.detach().numpy())
        print(f"{n} - Loss: {np.mean(np.array(all_loss))}")

    network.free_memory()
