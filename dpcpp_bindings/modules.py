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


class Module(torch.nn.Module):
    def __init__(self):
        super(Module, self).__init__()

        self.tnn_module = self.create_module()

        initial_params = self.tnn_module.initial_params()
        # self.params = torch.nn.Parameter(initial_params.to("xpu"), requires_grad=True)

        self.params = torch.nn.Parameter(initial_params, requires_grad=True)
        # self.register_parameter(name="model_params", param=self.params)

    def create_module(self):
        return tnn.create_network(
            self.width,
            self.input_width,
            self.output_width,
            self.n_hidden_layers,
            self.activation,
            self.output_activation,
            self.batch_size,
        )

    def forward(self, x):
        # return self.tnn_module.fwd(x, self.params)
        output = _module_function.apply(
            self.tnn_module,
            x,
            self.params,
        )
        return output

    def free_memory(self):
        self.tnn_module.free_memory()


class SwiftNet(Module):
    def __init__(
        self,
        batch_size=64,  # needs to be % 64 == 0
        width=64,  # needs to be 16, 32, 64
        input_width=1,
        output_width=1,
        n_hidden_layers=1,
        activation=Activation.ReLU,
        output_activation=Activation.ReLU,
    ):
        self.batch_size = batch_size
        self.width = width
        self.input_width = input_width
        self.output_width = output_width
        self.n_hidden_layers = n_hidden_layers
        self.activation = activation
        self.output_activation = output_activation
        super().__init__()
