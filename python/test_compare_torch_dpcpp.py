import numpy as np
import torch
import intel_extension_for_pytorch  # required for SwiftNet
import pytest
import time
from utils import create_models
import csv

# Define the parameters for the grid search
input_sizes = [1, 2, 8, 16, 64]
output_funcs = ["linear", "relu"]
output_sizes = [1, 2, 8, 16, 64]
# activation_funcs = ["relu", "linear", "sigmoid", "tanh"]
activation_funcs = ["relu", "linear"]
hidden_layer_counts = [1, 2, 3, 4, 5]

BATCH_SIZE = 64
DEVICE_NAME = "xpu"


class CustomMSELoss(torch.nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()

    def forward(self, predicted, target):
        # Calculate the mean squared error
        mse = torch.mean((predicted - target) ** 2)
        return mse


def train_model(model, x_train, y_train, n_steps, save_grads=0):
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
            y_pred = model(x_train[idx * batch_size : (idx + 1) * batch_size, ...])
            # try:
            # y_pred = model(
            #     x_train[idx * batch_size : (idx + 1) * batch_size, ...].flatten()
            # )
            # except:
            #     y_pred = model(
            #         x_train[idx * batch_size : (idx + 1) * batch_size, ...]
            #     ).flatten()
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

            if save_grads:
                name = "torch" if save_grads == 1 else "dpcpp"
                if isinstance(grads_all, list):
                    grads_all = grads_all[0]
                flattened_gradients = [
                    tensor.flatten().tolist() for tensor in grads_all
                ]

                file_path = f"python/{name}_grads.csv"

                # Write the flattened gradients to the CSV file
                with open(file_path, mode="w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerows(flattened_gradients)

                file_path = f"python/{name}_loss.csv"
                with open(file_path, mode="w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerows([[loss.tolist()]])

            grads.append(grads_all)
            optimizer.step()
            all_loss.append(loss.detach().numpy())
        loss_mean = np.mean(np.array(all_loss))
        losses.append(loss_mean)
        # print(f"{n} - Loss: {loss_mean}")

    return losses, y_predicted_all, grads


@pytest.mark.parametrize(
    "input_size, hidden_size, output_size, activation_func, output_func",
    [
        (input_size, hidden_size, output_size, activation_func, output_func)
        for input_size in input_sizes
        for output_size in output_sizes
        for activation_func in activation_funcs
        for output_func in output_funcs
        for hidden_size in hidden_layer_counts
    ],
)
def test_grad(
    input_size,
    hidden_size,
    output_size,
    activation_func,
    output_func,
    iterations=1,
):
    for iter_ in range(iterations):
        if iter_ == 0:
            # easiest, debug test
            x_train = (
                torch.tensor(BATCH_SIZE * [0.001 for _ in range(input_size)])
                .to(DEVICE_NAME)
                .reshape(BATCH_SIZE, -1)
            )
            y_train = torch.ones([BATCH_SIZE, output_size]).to(DEVICE_NAME) * 0.1
        else:
            x_train = torch.rand([BATCH_SIZE, input_size]).to(DEVICE_NAME)
            y_train = torch.rand([BATCH_SIZE, output_size]).to(DEVICE_NAME)

        # Need to generate new model, because weights are updated in one loop.
        model_dpcpp, model_torch = create_models(
            input_size,
            hidden_size,
            output_size,
            activation_func,
            output_func,
            BATCH_SIZE,
            DEVICE_NAME,
        )

        n_steps = 1  # if this is too large, there will be accumulated error (weights aren't the same, thus the loss is not the same etc)
        loss_dpcpp, y_dpcpp, grads_dpcpp = train_model(
            model_dpcpp, x_train, y_train, n_steps, save_grads=0
        )
        loss_torch, y_torch, grads_torch = train_model(
            model_torch, x_train, y_train, n_steps, save_grads=0
        )
        grads_dpcpp = grads_dpcpp[0][0]
        grads_torch = grads_torch[0]
        total_diff = []
        time.sleep(1)
        for layer in range(len(grads_dpcpp)):
            rel_diff_in_layer = abs(
                abs(grads_torch[layer]).sum() - abs(grads_dpcpp[layer]).sum()
            ) / (abs(grads_torch[layer]).sum())
            total_diff.append(rel_diff_in_layer)
            print(
                f"Layer {layer+1}: {rel_diff_in_layer*100:.2f}% (sum: ",
                f"{abs(grads_torch[layer]).sum():.4f}, and {abs(grads_dpcpp[layer]).sum():.4f})",
            )
            if rel_diff_in_layer > 0.05:
                print("Torch")
                print(grads_torch[layer])
                print("DPCPP")
                print(grads_dpcpp[layer])
            assert (
                rel_diff_in_layer < 0.05
            ), f"Difference larger than 5%: {rel_diff_in_layer* 100:.2f}%"
        print(f"Average difference: {100*np.mean(np.array(total_diff)):.2f}%")


@pytest.mark.parametrize(
    "input_size, hidden_size, output_size, activation_func, output_func",
    [
        (input_size, hidden_size, output_size, activation_func, output_func)
        for input_size in input_sizes
        for hidden_size in hidden_layer_counts
        for output_size in output_sizes
        for activation_func in activation_funcs
        for output_func in output_funcs
    ],
)
def test_fwd(input_size, hidden_size, output_size, activation_func, output_func):
    # Generate random input data for testing
    torch.manual_seed(123)
    input_data = (
        torch.randn(input_size, BATCH_SIZE, dtype=torch.float32).to(DEVICE_NAME) * 0
        + 0.01
        # torch.randn(BATCH_SIZE, input_size, dtype=torch.float32).to(DEVICE_NAME)
    )
    model_dpcpp, model_torch = create_models(
        input_size,
        hidden_size,
        output_size,
        activation_func,
        output_func,
        batch_size=BATCH_SIZE,
        device_name=DEVICE_NAME,
    )
    model_torch.to(DEVICE_NAME)
    model_dpcpp.to(DEVICE_NAME)
    y_torch = model_torch(input_data.T)
    y_dpcpp = model_dpcpp(input_data)
    # print("Torch: ", y_torch[0, :])
    # print("DPCPP: ", y_dpcpp[0, :])
    # print(
    #     f"diff: {y_torch[0, :] - y_dpcpp[0, :]}, average: {abs(y_torch[0, :] - y_dpcpp[0, :]).mean()}"
    # )

    print("Torch: ", y_torch)
    print("DPCPP: ", y_dpcpp)
    print(
        f"diff: {y_torch[0, :] - y_dpcpp[0, :]}, average: {abs(y_torch - y_dpcpp).mean()}"
    )

    # Ensure both models have the same weights
    # assert torch.allclose(
    #     y_torch, y_dpcpp, atol=1e-1
    # ), f"Forward error is too large {y_torch}, {y_dpcpp}"
    assert (
        abs(y_torch.sum() - y_dpcpp.sum()) / (abs(y_torch).sum()) < 0.01
    ), f"Forward error is too large {y_torch}, {y_dpcpp}"
    model_dpcpp.free_memory()


if __name__ == "__main__":
    input_width = 80
    output_width = 64
    n_hidden_layers = 1
    # activation_func = "linear"
    activation_func = "relu"
    output_func = "linear"
    # output_func = "sigmoid"

    test_fwd(input_width, n_hidden_layers, output_width, activation_func, output_func)
    print("Passed fwd test")

    # test_grad(
    #     input_width,
    #     n_hidden_layers,
    #     output_width,
    #     activation_func,
    #     output_func,
    # )
