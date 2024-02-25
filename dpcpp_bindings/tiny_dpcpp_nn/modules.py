import torch
import math
import intel_extension_for_pytorch

from tiny_dpcpp_nn_pybind_module import (
    Activation,
    create_network,
    create_encoding,
    create_networkwithencoding,
)


def pad_tensor_to_width(tensor, width):
    batch_size, input_dim = tensor.shape
    if input_dim >= width:
        raise ValueError("input_dim must be less than width to pad")

    padding_size = width - input_dim
    pad_tensor = torch.nn.functional.pad(tensor, (0, padding_size), "constant", 0)
    return pad_tensor


def unpad_tensor_to_input_dim(padded_tensor, output_dim):
    batch_size, current_width = padded_tensor.shape
    if output_dim > current_width:
        raise ValueError(
            "input_dim must be less than or equal to the current width of the tensor"
        )

    unpadded_tensor = padded_tensor[:, :output_dim]
    return unpadded_tensor


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


def pack_vector(unpacked_vector, n_hidden_layers, input_dim, output_dim, width):
    # Initialize the starting index of the current matrix within the packed vector
    current_idx = 0
    # The vector to hold all unpacked matrices
    packed_matrices = []

    # Unpack the input matrix
    total_elements = input_dim * width
    packed_input_matrix = [0] * total_elements
    for idx in range(total_elements):
        packed_index = to_packed_layout_coord(idx, input_dim, width)
        packed_input_matrix[idx] = unpacked_vector[current_idx + packed_index]
    packed_matrices.append(packed_input_matrix)
    current_idx += total_elements

    # Unpack N hidden matrices
    for _ in range(n_hidden_layers):
        total_elements = width * width
        packed_hidden_matrix = [0] * total_elements
        for idx in range(total_elements):
            packed_index = to_packed_layout_coord(idx, width, width)
            packed_hidden_matrix[idx] = unpacked_vector[current_idx + packed_index]
        packed_matrices.append(packed_hidden_matrix)
        current_idx += total_elements

    # Unpack the output matrix
    total_elements = width * output_dim
    packed_output_matrix = [0] * total_elements
    for idx in range(total_elements):
        packed_index = to_packed_layout_coord(idx, width, output_dim)
        packed_output_matrix[idx] = unpacked_vector[current_idx + packed_index]
    packed_matrices.append(packed_output_matrix)

    # Return the unpacked matrices as a flat list (one long vector in unpacked format)
    return [item for sublist in packed_matrices for item in sublist]


def unpack_vector(packed_vector, n_hidden_layers, input_dim, output_dim, width):
    # Initialize the starting index of the current matrix within the packed vector
    current_idx = 0
    # The vector to hold all unpacked matrices
    unpacked_matrices = []

    # Unpack the input matrix
    total_elements = input_dim * width
    unpacked_input_matrix = [0] * total_elements
    for idx in range(total_elements):
        unpacked_index = from_packed_layout_coord(idx, input_dim, width)
        unpacked_input_matrix[idx] = packed_vector[current_idx + unpacked_index]
    unpacked_matrices.append(unpacked_input_matrix)
    current_idx += total_elements

    # Unpack N hidden matrices
    for _ in range(n_hidden_layers):
        total_elements = width * width
        unpacked_hidden_matrix = [0] * total_elements
        for idx in range(total_elements):
            unpacked_index = from_packed_layout_coord(idx, width, width)
            unpacked_hidden_matrix[idx] = packed_vector[current_idx + unpacked_index]
        unpacked_matrices.append(unpacked_hidden_matrix)
        current_idx += total_elements

    # Unpack the output matrix
    total_elements = width * output_dim
    unpacked_output_matrix = [0] * total_elements
    for idx in range(total_elements):
        unpacked_index = from_packed_layout_coord(idx, width, output_dim)
        unpacked_output_matrix[idx] = packed_vector[current_idx + unpacked_index]
    unpacked_matrices.append(unpacked_output_matrix)

    # Return the unpacked matrices as a flat list (one long vector in unpacked format)
    return [item for sublist in unpacked_matrices for item in sublist]


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


class _module_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, native_tcnn_module, input, params):
        batch_size = input.shape[0]
        output = native_tcnn_module.fwd(input, params)
        output = output.reshape(batch_size, -1).to(input.device)
        ctx.save_for_backward(input, output, params)
        ctx.native_tcnn_module = native_tcnn_module
        return output

    @staticmethod
    def backward(ctx, doutput):
        _, _, params = ctx.saved_tensors
        # print("doutput: ", doutput)
        loss_scale = 128.0  # because half precision
        # loss_scale = 1  # because half precision
        doutput = doutput.to(dtype=torch.float) * loss_scale
        with torch.no_grad():
            grad = ctx.native_tcnn_module.bwd(doutput, params)
        grad = grad.to("xpu")
        grad = None if grad is None else (grad / loss_scale)

        # 3 inputs to forward, so need 3 grads
        grad = (
            torch.tensor(unpack_vector(grad, 0, 64, 64, 64))
            .to("xpu")
            .to(torch.bfloat16)
        )  # grad is in unpacked format, but we need pack it.
        return (None, None, grad)


class Module(torch.nn.Module):
    def __init__(self, device="xpu"):
        super(Module, self).__init__()
        self.device = device

        self.tnn_module = self.create_module()

        if self.tnn_module.n_params():
            # CONSTANT = True
            CONSTANT = False
            if CONSTANT:
                assert (self.tnn_module.n_params() / 2) % 2 == 0
                torch_params = torch.hstack(
                    [
                        torch.linspace(
                            -0.2,
                            0.1,
                            int(self.tnn_module.n_params() / 2),
                            dtype=torch.bfloat16,
                        ),
                        torch.linspace(
                            -0.4,
                            0.2,
                            int(self.tnn_module.n_params() / 2),
                            dtype=torch.bfloat16,
                        ),
                    ]
                )
            else:
                std = math.sqrt(2.0 / float(self.width + self.width))

                torch_params = torch.normal(
                    torch.zeros(self.tnn_module.n_params()), std
                ).to(torch.bfloat16)

            # Set the initial parameters based on the transformed torch_params
            initial_params = self.tnn_module.initial_params(torch_params)

            # Creating the torch.nn.Parameter object with the initialized tensor
            self.params = torch.nn.Parameter(
                initial_params.to(device), requires_grad=True
            )
        else:
            print(
                "No params initialised, as n_params = 0. This is correct for Encodings (apart from grid encodings)."
            )
            self.params = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def get_reshaped_params(
        self, weights=None, datatype=torch.float, is_packed_format=True
    ):
        all_weights = []
        if weights is None:
            weights = self.params
        # for idx, val in enumerate(self.params):
        #     print(f"{idx}: {val}")

        n_input_dims = (
            self.width if self.n_input_dims <= self.width else self.n_input_dims
        )  # because we pad
        input_matrix = (
            torch.zeros(self.width, n_input_dims).to(datatype).to(self.device)
        )

        for i in range(n_input_dims):
            for j in range(self.width):
                if is_packed_format:
                    idx = to_packed_layout_coord(
                        i * self.width + j, n_input_dims, self.width
                    )
                else:
                    idx = i * self.width + j
                input_matrix[j, i] = weights[idx]

        len_input_matrix = input_matrix.shape[0] * input_matrix.shape[1]
        hidden_layer_size = self.width * self.width
        hidden_matrices = []

        for nth_hidden in range(self.n_hidden_layers - 1):
            hidden_matrix = (
                torch.zeros(self.width, self.width).to(datatype).to(self.device)
            )

            for i in range(self.width):
                for j in range(self.width):
                    if is_packed_format:
                        idx = to_packed_layout_coord(
                            i * self.width + j, self.width, self.width
                        )
                    else:
                        idx = i * self.width + j
                    hidden_matrix[j, i] = weights[
                        len_input_matrix + nth_hidden * hidden_layer_size + idx
                    ]
            hidden_matrices.append(hidden_matrix)

        output_matrix = (
            torch.zeros(self.n_output_dims, self.width).to(datatype).to(self.device)
        )
        for i in range(self.width):
            for j in range(self.n_output_dims):
                if is_packed_format:
                    idx = to_packed_layout_coord(
                        i * self.n_output_dims + j, self.width, self.n_output_dims
                    )
                else:
                    idx = i * self.n_output_dims + j
                output_matrix[j, i] = weights[
                    len_input_matrix
                    + (self.n_hidden_layers - 1) * hidden_layer_size
                    + idx
                ]
        all_weights.append(input_matrix)
        all_weights.extend(hidden_matrices)
        # all_weights.append(hidden_matrices)
        all_weights.append(output_matrix)

        return all_weights

    def forward(self, x):
        # Input is batch_size, feature_size
        # print("Input weights: ", self.params)

        # padded_tensor = pad_tensor_to_width(x, self.width)
        output = _module_function.apply(
            # self.tnn_module, padded_tensor, self.params.to(torch.float)
            self.tnn_module,
            x,
            self.params,
        )
        return output


class Network(Module):
    def __init__(
        self,
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

        return create_network(
            self.n_input_dims,
            self.n_output_dims,
            self.n_hidden_layers,
            self.activation,
            self.output_activation,
            self.device,
        )


class NetworkWithInputEncoding(Module):
    def __init__(
        self,
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
                "n_dims_to_encode": self.n_input_dims,
                "scale": 1.0,
                "offset": 0.0,
            }
        self.encoding_name = self.encoding_config["otype"]

        if "n_dims_to_encode" not in self.encoding_config:
            self.encoding_config["n_dims_to_encode"] = self.n_input_dims

        super().__init__(device=device)

    def create_module(self):

        return create_networkwithencoding(
            self.n_input_dims,
            self.n_output_dims,
            self.n_hidden_layers,
            self.activation,
            self.output_activation,
            self.encoding_config,
            self.device,
            self.width,
        )


class Encoding(Module):
    def __init__(
        self,
        n_input_dims=1,
        encoding_config=None,
        device="xpu",
    ):
        self.n_input_dims = n_input_dims
        self.encoding_config = encoding_config
        if self.encoding_config is None:
            self.encoding_config = {
                "otype": "Identity",
                "n_dims_to_encode": self.n_input_dims,
                "scale": 1.0,
                "offset": 0.0,
            }
        self.encoding_name = self.encoding_config["otype"]

        if "n_dims_to_encode" not in self.encoding_config:
            self.encoding_config["n_dims_to_encode"] = self.n_input_dims

        super().__init__(device=device)

    def create_module(self):
        return create_encoding(
            self.n_input_dims,
            self.encoding_name,
            self.encoding_config,
            self.device,
        )
