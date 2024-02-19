import torch
import intel_extension_for_pytorch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from mlp import MLP
import matplotlib.pyplot as plt
from modules import Network, NetworkWithInputEncoding
from utils import create_models
import copy

torch.set_printoptions(precision=10)

# CALC_DPCPP = False
CALC_DPCPP = True
CALC_REF = True
# CALC_REF = False

# USE_ADAM = False
USE_ADAM = True


def report_grad_differences(name, grad1, grad2, is_close):
    indices_not_close = torch.nonzero(~is_close)
    print(f"Gradients not close for parameter: {name}")
    for index in map(tuple, indices_not_close):
        print(
            f"In param: {name} at index {index} grad1 is {grad1[index]} but grad2 is {grad2[index]}"
        )


def report_param_differences(name, param1_data, param2_data, is_close):
    indices_not_close = torch.nonzero(~is_close)
    print(f"Parameters not close for: {name}")
    for index in map(tuple, indices_not_close):
        print(
            f"In param: {name} at index {index}, model_torch is {param1_data[index]}, model_dpcpp is {param2_data[index]}"
        )


def report_output_differences(outputs1, outputs2, is_close):
    indices_not_close = torch.nonzero(~is_close)
    for index in map(tuple, indices_not_close):
        print(
            f"In index {index}, outputs1 is {outputs1[index]}, but outputs2 is {outputs2[index]}"
        )

    assert torch.all(is_close), "Outputs are not close"


class SimpleSGDOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)
        super(SimpleSGDOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for idx, p in enumerate(group["params"]):
                if p.grad is None:
                    print("p.grad is none")
                    continue
                grad = p.grad.data
                p.data = p.data - group["lr"] * grad
        return loss


# Define a simple linear function for the dataset
def true_function(x):
    return 0.5 * x


# Create a synthetic dataset based on the true function
input_size = 64
output_size = 64
num_samples = 8
batch_size = 8

# Random inputs
inputs_single = torch.linspace(-1, 1, steps=num_samples)
inputs_training = inputs_single.repeat(input_size, 1).T

# Corresponding labels with some noise
noise = torch.randn(num_samples, output_size) * 0.1

labels_training = true_function(inputs_training)
# Create a DataLoader instance for batch processing
dataset = TensorDataset(inputs_training, labels_training)
dataloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=False
)  # if we shuffle, the loss is not identical

# Instantiate the network
device = "xpu"
model_dpcpp, model_torch = create_models(
    input_size,
    [64],
    output_size,
    "relu",
    "linear",
    "xpu",
)

model_torch.to(torch.bfloat16)
# Define a loss function and an optimizer
criterion = torch.nn.MSELoss()
if USE_ADAM:
    optimizer1 = torch.optim.Adam(model_dpcpp.parameters(), lr=1e-3)
    optimizer2 = torch.optim.Adam(model_torch.parameters(), lr=1e-3)
else:
    optimizer1 = SimpleSGDOptimizer(model_dpcpp.parameters(), lr=1e-3)
    optimizer2 = SimpleSGDOptimizer(model_torch.parameters(), lr=1e-3)

# Lists for tracking loss and epochs
epoch_losses1 = []
epoch_losses2 = []
epoch_count = []

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    running_loss1 = 0.0
    running_loss2 = 0.0
    print(f"Epoch: {epoch}")
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        if CALC_DPCPP:
            inputs1, labels1 = inputs.clone().to(device), labels.clone().to(device).to(
                torch.bfloat16
            )
            # Forward pass
            outputs1 = model_dpcpp(inputs1)
            loss1 = criterion(outputs1, labels1)
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()
            running_loss1 += loss1.item()
        if CALC_REF:
            inputs2, labels2 = inputs.clone().to(device).to(
                torch.bfloat16
            ), labels.clone().to(device).to(torch.bfloat16)
            outputs2 = model_torch(inputs2)
            loss2 = criterion(outputs2, labels2)
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()
            running_loss2 += loss2.item()

    if CALC_DPCPP:
        epoch_loss1 = running_loss1 / len(dataloader)
        print(f"Epoch {epoch+1}, Loss dpcpp: {epoch_loss1}")
    if CALC_REF:
        epoch_loss2 = running_loss2 / len(dataloader)
        print(f"Epoch {epoch+1}, Loss torch: {epoch_loss2}")
    print("================================")
    epoch_losses1.append(epoch_loss1)
    epoch_losses2.append(epoch_loss2)
    epoch_count.append(epoch + 1)

print("Finished Training")

# Plot the loss over epochs
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(epoch_count, epoch_losses1, label="Training Loss Torch")
plt.plot(epoch_count, epoch_losses2, label="Training Loss DPCPP")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Over Epochs")
plt.legend()

# Plot the ground truth and the learned function
plt.subplot(1, 2, 2)
plt.scatter(
    inputs_training.cpu().numpy(),
    labels_training.cpu().numpy(),
    s=8,
    label="Ground Truth",
)
if CALC_DPCPP:
    with torch.no_grad():
        learned_function_dpcpp = model_dpcpp(inputs1.to(device)).cpu()
    # print(inputs1.shape)
    # print(learned_function_dpcpp.shape)
    plt.scatter(
        inputs_training.cpu().numpy()[0, :],
        learned_function_dpcpp.cpu().numpy()[0, :],
        s=8,
        label="Learned Function dpcpp",
    )
if CALC_REF:
    with torch.no_grad():
        learned_function_torch = model_torch(inputs_training.to(device)).cpu()
    plt.scatter(
        inputs_training.cpu().numpy(),
        learned_function_torch.cpu().numpy(),
        s=8,
        label="Learned Function torch",
    )
plt.xlabel("Input")
plt.ylabel("Output")
plt.title("Ground Truth vs Learned Function")
plt.legend()
plt.ylim(-1, 1)
plt.tight_layout()

# Save the figure instead of showing it
plt.savefig("loss_and_function.png")
