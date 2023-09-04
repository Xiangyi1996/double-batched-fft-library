import gc
import warnings

import torch

import intel_extension_for_pytorch
import tiny_nn as tnn
from tiny_nn import Activation


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


class Embedding(torch.nn.Module):
    def __init__(self, input_dim, output_dim, requires_grad=False):
        super(Embedding, self).__init__()

        self.embedding = torch.nn.Linear(input_dim, output_dim)
        if not requires_grad:
            # Initialize the parameters with the specified value
            torch.nn.init.constant_(self.embedding.weight, 0.1)
            torch.nn.init.constant_(self.embedding.bias, 0.0)

        for param in self.embedding.parameters():
            # param.requires_grad = True
            param.requires_grad = requires_grad

    def forward(self, x):
        x = self.embedding(x)
        return x


class Module(torch.nn.Module):
    def __init__(self, device="xpu"):
        super(Module, self).__init__()

        self.tnn_module = self.create_module()
        initial_params = self.tnn_module.initial_params()
        # self.params = torch.nn.Parameter(initial_params.to("xpu"), requires_grad=True)

        self.params = torch.nn.Parameter(initial_params, requires_grad=True)

        self.embedding = Embedding(
            self.input_width, self.input_swiftnet_width, requires_grad=False
        ).to(device)
        self.decoder = Embedding(
            self.out_swiftnet_width, self.output_width, requires_grad=False
        ).to(device)

    def create_module(self):
        return tnn.create_network(
            self.width,
            self.input_swiftnet_width,
            self.out_swiftnet_width,
            self.n_hidden_layers,
            self.activation,
            self.output_activation,
            self.batch_size,
        )

    def forward(self, x):
        x = self.embedding(x)
        x = x.reshape(-1, 1)  # flatten for tiny nn
        output = _module_function.apply(
            self.tnn_module,
            x,
            self.params,
        )

        output = output.reshape(self.batch_size, -1)
        output = self.decoder(output)

        zero_vals = int((output == 0).sum())
        if zero_vals > 2:
            print(f"{zero_vals} values are exactly zero. Check if intended behaviour.")

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
        out_swiftnet_width=64,
        input_swiftnet_width=64,
        device="xpu",
    ):
        self.batch_size = batch_size
        self.width = width
        self.input_width = input_width
        self.output_width = output_width
        self.n_hidden_layers = n_hidden_layers
        self.activation = activation
        self.output_activation = output_activation
        self.input_swiftnet_width = input_swiftnet_width
        self.out_swiftnet_width = out_swiftnet_width

        super().__init__(device=device)
