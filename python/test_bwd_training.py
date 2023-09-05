import torch
import intel_extension_for_pytorch
from modules import SwiftNet
from tiny_nn import Activation
import numpy as np
import pytest

DEVICE = "cpu"

BATCH_SIZE = 64
DATA_AMOUNT = 400

TOLERANCE = 1e-3


def perform_training(
    width, input_width, output_width, n_hidden_layers, tolerance=TOLERANCE
):
    activation = Activation.ReLU
    output_activation = Activation.Linear

    network = SwiftNet(
        BATCH_SIZE,
        width,
        input_width,
        output_width,
        n_hidden_layers,
        activation,
        output_activation,
        device=DEVICE,
    )

    x = [torch.linspace(0.01, 10, steps=DATA_AMOUNT)] * input_width
    x = (torch.stack((x)).T).to(DEVICE)
    dim_reduce = torch.ones(input_width, output_width)

    y = (2 * x @ dim_reduce).to(DEVICE)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    for n in range(1000):
        all_loss = []
        for idx in range(x.shape[0] // BATCH_SIZE):
            y_pred = network(x[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE, ...])
            loss = loss_fn(
                y_pred,
                y[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE],
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_loss.append(loss.detach().numpy())
        loss_mean = np.mean(np.array(all_loss))
        print(f"{n} - Loss: {loss_mean}")
        if loss_mean < tolerance:
            network.free_memory()
            return loss_mean

    network.free_memory()
    return loss_mean


def test_training():
    # widths = [16, 32, 64]
    widths = [64]
    # input_widths = [2**idx for idx in range(7)]
    # output_widths = [2**idx for idx in range(7)]
    input_widths = [1, 16, 32, 64]
    output_widths = [1, 16, 32, 64]
    n_hidden_layers_all = [1, 2, 4]
    counter = 0

    loss_exceeding_threshold = 0
    print("Loop over all test configs. This will take some time")
    for width in widths:
        for input_width in input_widths:
            for output_width in output_widths:
                for n_hidden_layers in n_hidden_layers_all:
                    print(
                        f"Training loop ({counter}/{len(widths)*len(input_widths)*len(output_widths)*len(n_hidden_layers_all)})"
                    )
                    print(
                        f"Running for width: {width}, input_width: {input_width}, output_width: {output_width}, and n_hidden_layers: {n_hidden_layers}"
                    )
                    loss_mean = perform_training(
                        width, input_width, output_width, n_hidden_layers
                    )
                    if loss_mean > TOLERANCE:
                        loss_exceeding_threshold += 1
                        print(
                            f"Loss was: {loss_mean} for width: {width}, input_width: {input_width}, output_width: {output_width}, and n_hidden_layers: {n_hidden_layers}"
                        )

                    counter += 1
    assert (
        loss_exceeding_threshold == 0
    ), f"Training error was larger than {TOLERANCE}. Check for errors or increase TOLERANCE."


if __name__ == "__main__":
    test_training()
    # width = 64
    # input_width = 2
    # output_width = 1
    # n_hidden_layers = 1
    # perform_training(width, input_width, output_width, n_hidden_layers)
