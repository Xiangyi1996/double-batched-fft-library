import gc
import warnings

import torch

import intel_extension_for_pytorch

import tiny_nn as tnn
from tiny_nn import Activation


def to_packed_layout_coord(idx, rows, cols):
    i = idx // cols
    j = idx % cols
    if (i % 2) == 0:
        return i * cols + 2 * j
    else:
        return (i - 1) * cols + 2 * j + 1


def from_packed_layout_coord(idx, rows, cols):
    # Not really used.
    i = idx // (cols * 2)
    j = idx % (cols * 2)
    if (j % 2) == 0:
        return (i * 2) * cols + j // 2
    else:
        return (i * 2 + 1) * cols + (j - 1) // 2


def free_temporary_memory():
    # Ensure all Python objects (potentially pointing
    # to temporary TNN allocations) are cleaned up.
    gc.collect()
    tnn.free_temporary_memory()


class _module_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, native_tcnn_module, input, params):
        output = native_tcnn_module.fwd(input, params)
        ctx.save_for_backward(input, output, params)
        ctx.native_tcnn_module = native_tcnn_module
        return output

    @staticmethod
    def backward(ctx, doutput):
        input, params, output = ctx.saved_tensors

        with torch.no_grad():
            grad = ctx.native_tcnn_module.bwd(input, doutput, params)
        # 3 inputs to forward, so need 3 grads
        return None, None, grad


# class Embedding(torch.nn.Module):
#     def __init__(self, input_dim, output_dim, requires_grad=False):
#         super(Embedding, self).__init__()

#         self.embedding = torch.nn.Linear(input_dim, output_dim, bias=False)
#         if not requires_grad:
#             # Initialize the parameters with the specified value
#             torch.nn.init.constant_(self.embedding.weight, 0.1)

#         for param in self.embedding.parameters():
#             param.requires_grad = requires_grad

#     def forward(self, x):
#         x = self.embedding(x)
#         return x


class Module(torch.nn.Module):
    def __init__(self, device="xpu"):
        super(Module, self).__init__()
        self.device = device

        self.tnn_module = self.create_module()
        initial_params = self.tnn_module.initial_params()

        self.params = torch.nn.Parameter(initial_params.to(device), requires_grad=True)

    def get_reshaped_params(self, weights=None):
        all_weights = []
        if weights is None:
            weights = self.params

        input_matrix = torch.zeros(self.width, self.input_width)

        for i in range(self.input_width):
            for j in range(self.width):
                idx = to_packed_layout_coord(
                    i * self.width + j, self.input_width, self.width
                )
                input_matrix[j, i] = weights[idx]

        len_input_matrix = input_matrix.shape[0] * input_matrix.shape[1]
        hidden_layer_size = self.width * self.width
        hidden_matrices = []

        for nth_hidden in range(self.n_hidden_layers - 1):
            hidden_matrix = torch.zeros(self.width, self.width)

            for i in range(self.width):
                for j in range(self.width):
                    idx = to_packed_layout_coord(
                        i * self.width + j, self.width, self.width
                    )
                    hidden_matrix[j, i] = weights[
                        len_input_matrix + nth_hidden * hidden_layer_size + idx
                    ]
            hidden_matrices.append(hidden_matrix)

        output_matrix = torch.zeros(self.output_width, self.width)
        for i in range(self.width):
            for j in range(self.output_width):
                idx = to_packed_layout_coord(
                    i * self.output_width + j, self.width, self.output_width
                )
                output_matrix[j, i] = weights[
                    len_input_matrix
                    + (self.n_hidden_layers - 1) * hidden_layer_size
                    + idx
                ]
        all_weights.append(input_matrix)
        all_weights.extend(hidden_matrices)
        all_weights.append(output_matrix)
        return all_weights

    def create_module(self):
        return tnn.create_network(
            self.width,
            self.input_width,
            self.output_width,
            self.n_hidden_layers,
            self.activation,
            self.output_activation,
            self.batch_size,
            self.device,
        )

    def forward(self, x):
        x = x.reshape(-1, 1)  # flatten for tiny nn
        output = _module_function.apply(self.tnn_module, x, self.params)

        output = output.reshape(self.batch_size, -1).to(self.device)

        # zero_vals = int((output == 0).sum())
        # if zero_vals > 2:
        #     print(f"{zero_vals} values are exactly zero. Check if intended behaviour.")

        return output

    def free_memory(self):
        self.tnn_module.free_memory()


class SwiftNet(Module):
    # TODO: Add Swiftnet with Encoder here, which has a different native module (makign it faster than inputing an embedding in python)
    def __init__(
        self,
        batch_size=64,  # needs to be % 64 == 0
        width=64,  # needs to be 16, 32, 64
        input_width=1,
        output_width=1,
        n_hidden_layers=1,
        activation=Activation.ReLU,
        output_activation=Activation.ReLU,
        device="xpu",
    ):
        self.batch_size = batch_size
        self.width = width
        self.input_width = input_width
        self.output_width = output_width
        self.n_hidden_layers = n_hidden_layers
        self.activation = activation
        self.output_activation = output_activation

        super().__init__(device=device)
