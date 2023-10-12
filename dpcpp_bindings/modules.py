import gc
import warnings

import torch

import intel_extension_for_pytorch

import tiny_nn as tnn
from tiny_nn import Activation


def get_dpcpp_activation(name):
    if name.lower() == "relu":
        activation = Activation.ReLU
    elif name.lower() == "tanh":
        activation = Activation.Tanh
    elif name.lower() == "sigmoid":
        activation = Activation.Sigmoid
    elif name.lower() == "linear" or name.lower() == "none":
        activation = Activation.Linear
    else:
        raise NotImplementedError(f"Activation: {name} not defined")

    return activation


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
        input, output, params = ctx.saved_tensors

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
        if self.tnn_module.n_params():
            initial_params = self.tnn_module.initial_params(1)

            self.params = torch.nn.Parameter(
                initial_params.to(device), requires_grad=True
            )
        else:
            print(
                "No params initialised, as n_params = 0. This is correct for Encodings."
            )
            self.params = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def get_reshaped_params(self, weights=None):
        all_weights = []
        if weights is None:
            weights = self.params

        n_input_dims = self.width  # because we pad
        input_matrix = torch.zeros(self.width, n_input_dims)

        for i in range(n_input_dims):
            for j in range(self.width):
                idx = to_packed_layout_coord(
                    i * self.width + j, n_input_dims, self.width
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

        output_matrix = torch.zeros(self.n_output_dims, self.width)
        for i in range(self.width):
            for j in range(self.n_output_dims):
                idx = to_packed_layout_coord(
                    i * self.n_output_dims + j, self.width, self.n_output_dims
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


class Network(Module):
    def __init__(
        self,
        batch_size=64,
        n_input_dims=1,
        n_output_dims=1,
        network_config=None,
        device="xpu",
    ):
        if network_config is None:
            self.network_config = {
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        else:
            self.network_config = network_config

        self.batch_size = batch_size
        self.width = self.network_config["n_neurons"]
        self.n_input_dims = n_input_dims
        self.n_output_dims = n_output_dims
        self.n_hidden_layers = self.network_config["n_hidden_layers"]
        self.activation = get_dpcpp_activation(self.network_config["activation"])
        self.output_activation = get_dpcpp_activation(
            self.network_config["output_activation"]
        )

        super().__init__(device=device)

    def create_module(self):
        return tnn.create_network(
            self.width,
            self.n_input_dims,
            self.n_output_dims,
            self.n_hidden_layers,
            self.activation,
            self.output_activation,
            self.batch_size,
            self.device,
        )


class NetworkWithInputEncoding(Module):
    def __init__(
        self,
        batch_size=64,
        n_input_dims=1,
        n_output_dims=1,
        network_config=None,
        encoding_config=None,
        device="xpu",
    ):
        if network_config is None:
            self.network_config = {
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        else:
            self.network_config = network_config

        self.batch_size = batch_size
        self.width = self.network_config["n_neurons"]
        self.n_input_dims = n_input_dims
        self.n_output_dims = n_output_dims
        self.n_hidden_layers = self.network_config["n_hidden_layers"]
        self.activation = get_dpcpp_activation(self.network_config["activation"])
        self.output_activation = get_dpcpp_activation(
            self.network_config["output_activation"]
        )

        self.encoding_config = encoding_config
        if self.encoding_config is None:
            self.encoding_config = {
                "otype": "Identity",
                "n_dims_to_encode": str(self.n_input_dims),
                "scale": "1.0",
                "offset": "0.0",
            }
        self.encoding_name = self.encoding_config["otype"]

        if "n_dims_to_encode" not in self.encoding_config:
            self.encoding_config["n_dims_to_encode"] = str(self.n_input_dims)

        for value in self.encoding_config.values():
            assert isinstance(value, str), "Not all values are of type str"

        super().__init__(device=device)

    def create_module(self):
        return tnn.create_networkwithencoding(
            self.width,
            self.n_input_dims,
            self.n_output_dims,
            self.n_hidden_layers,
            self.activation,
            self.output_activation,
            self.batch_size,
            self.encoding_name,
            self.encoding_config,
            self.device,
        )


class Encoding(Module):
    def __init__(
        self,
        batch_size=64,
        n_input_dims=1,
        encoding_config=None,
        device="xpu",
    ):
        self.batch_size = batch_size
        self.n_input_dims = n_input_dims

        self.encoding_config = encoding_config
        if self.encoding_config is None:
            self.encoding_config = {
                "otype": "Identity",
                "n_dims_to_encode": str(self.n_input_dims),
                "scale": "1.0",
                "offset": "0.0",
            }
        self.encoding_name = self.encoding_config["otype"]

        if "n_dims_to_encode" not in self.encoding_config:
            self.encoding_config["n_dims_to_encode"] = str(self.n_input_dims)

        for value in self.encoding_config.values():
            assert isinstance(value, str), "Not all values are of type str"

        super().__init__(device=device)

        self.n_output_dims = self.tnn_module.n_output_dims()

    def create_module(self):
        return tnn.create_encoding(
            self.n_input_dims,
            self.batch_size,
            self.encoding_name,
            self.encoding_config,
            self.device,
        )
