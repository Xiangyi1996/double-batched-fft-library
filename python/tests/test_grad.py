import numpy as np
import torch
import intel_extension_for_pytorch  # required for SwiftNet
import pytest

from modules import SwiftNet
from tiny_nn import Activation


# Define the parameters for the grid search
input_sizes = [64]  # Only 64 working as of now
output_funcs = ["linear"]  # only linear working as of now.
output_sizes = [1, 2, 8, 16, 64]
activation_funcs = ["relu", "linear", "sigmoid", "tanh"]
hidden_layer_counts = [1, 2, 3]

# BATCH_SIZE = 64
BATCH_SIZE = 1
DEVICE_NAME = "cpu"


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


class CustomMSELoss(torch.nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()

    def forward(self, predicted, target):
        # Calculate the mean squared error
        mse = torch.mean((predicted - target) ** 2)
        return mse


def train_model(model, x_train, y_train, n_steps):
    batch_size = BATCH_SIZE
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = CustomMSELoss()
    # loss_fn = torch.nn.MSELoss()
    y_predicted_all = []
    grads = []
    losses = []
    for n in range(n_steps):
        all_loss = []
        for idx in range(x_train.shape[0] // batch_size):
            y_pred = model(x_train[0, 0])
            loss = loss_fn(
                y_pred,
                # y_train[idx * batch_size : (idx + 1) * batch_size].flatten(),
                y_train[0, 0].flatten(),
            )
            # y_predicted_all.append(y_pred.detach().cpu().numpy())
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
        # loss_mean = np.mean(np.array(all_loss))
        # losses.append(loss_mean)
        # print(f"{n} - Loss: {loss_mean}")

    return losses, y_predicted_all, grads


def test_grad(
    input_size,
    hidden_size,
    output_size,
    activation_func,
    output_func,
    iterations=1,
):
    x_train = (
        torch.tensor(BATCH_SIZE * [0.2 for _ in range(input_size)])
        .to(DEVICE_NAME)
        .reshape(BATCH_SIZE, -1)
    )
    y_train = torch.ones([BATCH_SIZE, output_size]).to(DEVICE_NAME)
    for _ in range(iterations):
        # Create and test model_dpcpp
        activation = get_dpcpp_activation(activation_func)
        output_activation = get_dpcpp_activation(output_func)

        model_dpcpp = SwiftNet(
            1,
            64,
            input_size,
            output_size,
            hidden_size,
            activation=activation,
            output_activation=output_activation,
            device=DEVICE_NAME,
        )
        n_steps = 1
        loss_dpcpp, y_dpcpp, grads_dpcpp = train_model(
            model_dpcpp, x_train, y_train, n_steps
        )


if __name__ == "__main__":
    input_width = 64
    output_width = 64
    n_hidden_layers = 1
    activation_func = "linear"
    output_func = "linear"

    # test_fwd(input_width, n_hidden_layers, output_width, activation_func, output_func)
    # print("Passed fwd test")

    test_grad(
        input_width,
        n_hidden_layers,
        output_width,
        activation_func,
        output_func,
    )
