import torch
import intel_extension_for_pytorch
from modules import SwiftNet
from tiny_nn import Activation
import numpy as np


class Embedding(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Embedding, self).__init__()

        self.embedding = torch.nn.Linear(input_dim, output_dim)
        for param in self.embedding.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.embedding(x)
        return x


if __name__ == "__main__":
    batch_size = 64
    width = 64
    input_width = 2
    output_width = 64
    n_hidden_layers = 1

    activation = Activation.ReLU
    output_activation = Activation.Linear
    embedding = Embedding(input_width, width)

    network = SwiftNet(
        batch_size,
        width,
        width,
        output_width,
        n_hidden_layers,
        activation,
        output_activation,
    )

    # x = torch.linspace(0, 10, steps=batch_size * input_width)
    x1 = torch.linspace(0, 10, steps=64)
    x2 = torch.linspace(0, 10, steps=64)
    x = torch.stack((x1, x2)).T

    x = embedding(x)
    y = network(x.flatten())
    # y = network(x)
    print(f"Zero vals: {int(sum(y == 0))}")
    # for val in y:
    #     print(val)
    # y = x[:, 0] + x[:, 1]
    # # y = 2 * x

    # loss_fn = torch.nn.MSELoss()
    # optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    # for n in range(500):
    #     all_loss = []
    #     for idx in range(x.shape[0] // batch_size):
    #         y_pred = network(
    #             x[idx * batch_size : (idx + 1) * batch_size, ...].flatten()
    #         )
    #         loss = loss_fn(y_pred, y[idx * batch_size : (idx + 1) * batch_size])
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         all_loss.append(loss.detach().numpy())
    #     print(f"{n} - Loss: {np.mean(np.array(all_loss))}")

    # network.free_memory()
