import numpy as np
import torch
import intel_extension_for_pytorch  # required for SwiftNet
import pytest

from mlp import MLP
from modules import SwiftNet
from tiny_nn import Activation

# Define the parameters for the grid search
input_sizes = [1, 2, 8, 16, 64, 128]
output_sizes = [1, 2, 8, 16, 64, 128]
activation_funcs = ["relu", "linear", "sigmoid", "tanh"]
output_funcs = [torch.nn.functional.relu, torch.sigmoid, torch.tanh, None]
hidden_layer_counts = [1, 2, 3, 4, 5]

WIDTH = 64
BATCH_SIZE = 64


def train_model(model, x_train, y_train, batch_size, n_steps):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    y_predicted_all = []
    grads = []
    losses = []
    for n in range(n_steps):
        all_loss = []
        for idx in range(x_train.shape[0] // batch_size):
            y_pred = model(x_train[idx * batch_size : (idx + 1) * batch_size, ...])
            loss = loss_fn(
                y_pred,
                y_train[idx * batch_size : (idx + 1) * batch_size],
            )
            y_predicted_all.append(y_pred.detach().cpu().numpy())
            optimizer.zero_grad()
            loss.backward()
            grads_all = []
            for param in model.parameters():
                if param.requires_grad:
                    gradient = param.grad
                    if len(gradient.shape) == 1:
                        grad = model.get_reshaped_params(gradient)
                    else:
                        grad = gradient
                    grads_all.append(grad)
            grads.append(grads_all)
            optimizer.step()
            all_loss.append(loss.detach().numpy())
        loss_mean = np.mean(np.array(all_loss))
        losses.append(loss_mean)
        # print(f"{n} - Loss: {loss_mean}")

    return losses, y_predicted_all, grads


def test_grad(model_dpcpp, model_torch, x_train, y_train, batch_size, iterations=1):
    for _ in range(iterations):
        n_steps = 100
        loss_dpcpp, y_dpcpp, grads_dpcpp = train_model(
            model_dpcpp, x_train, y_train, batch_size, n_steps
        )
        loss_torch, y_torch, grads_torch = train_model(
            model_torch, x_train, y_train, batch_size, n_steps
        )

        last_idx = int(n_steps * 0.8)
        assert (
            np.mean(
                abs(np.array(loss_dpcpp[last_idx:]) - np.array(loss_torch[last_idx:]))
            )
            < 1e-3
        )
        assert (
            np.mean(abs(np.array(y_dpcpp[last_idx:]) - np.array(y_torch[last_idx:])))
            < 1e-2
        )


def test_custom_mlp_vs_model_dpcpp(
    input_size, hidden_size, output_size, activation_func, output_func
):
    # Create and test CustomMLP
    hidden_sizes = [WIDTH] * hidden_size
    model_torch = MLP(
        input_size, hidden_sizes, output_size, activation_func, output_func
    )
    model_torch.eval()

    # Create and test model_dpcpp

    # activation = Activation.ReLU
    activation = Activation.Linear
    output_activation = Activation.Linear

    model_dpcpp = SwiftNet(
        batch_size,
        width,
        input_width,
        output_width,
        n_hidden_layers,
        activation,
        output_activation,
        device=device_name,
    )
    model_dpcpp.eval()

    # Generate random input data for testing
    input_data = torch.randn(32, input_size)

    # Ensure both models have the same weights
    assert torch.allclose(custom_mlp(input_data), model_dpcpp(input_data), atol=1e-5)


def test_forward(
    model_dpcpp,
    model_torch,
    input_width,
    batch_size,
    iterations=10,
):
    for _ in range(iterations):
        x_input = torch.rand(
            [batch_size, input_width], device=device, dtype=torch.float32
        )

        y_dpcpp = model_dpcpp(x_input)
        y_torch = model_torch(x_input)

        assert (
            abs(y_dpcpp - y_torch).mean() < 1e-3
        ), f"Fwd error is too large: {abs(y_dpcpp - y_torch).mean()}"


if __name__ == "__main__":
    device_name = "cpu"
    # device = torch.device("xpu")
    device = torch.device(device_name)

    # batch_size = 2**12
    batch_size = 64
    width = 64
    input_width = 64
    output_width = 64
    n_hidden_layers = 2
    # activation = Activation.ReLU
    activation = Activation.Linear
    output_activation = Activation.Linear

    model_dpcpp = SwiftNet(
        batch_size,
        width,
        input_width,
        output_width,
        n_hidden_layers,
        activation,
        output_activation,
        device=device_name,
    )

    params = []
    for param in model_dpcpp.parameters():
        if len(param.shape) > 1:
            params.append(param.data)
    weights = {
        "input_weights": params[0] if params else None,
        "middle_weights": model_dpcpp.get_reshaped_params(),
    }
    hidden_sizes = [64] * n_hidden_layers
    activation_func = "linear"
    output_activation = None

    model_torch = MLP(
        input_width, hidden_sizes, output_width, activation_func, output_activation
    ).to(device)
    # print(model)
    model_torch.set_weights(weights)

    test_forward(model_dpcpp, model_torch, input_width, batch_size)

    x_train = (
        torch.tensor(batch_size * [2e-1 for idx in range(input_width)])
        .to(device_name)
        .reshape(batch_size, -1)
    )

    y_train = torch.ones([batch_size, output_width]).to(device_name)
    test_grad(model_dpcpp, model_torch, x_train, y_train, batch_size=batch_size)

    model_dpcpp.free_memory()
