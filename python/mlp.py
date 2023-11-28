import torch
import torch.nn.functional as F
import copy
import numpy as np

BIAS = False


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_sizes,
        output_size,
        activation_func="relu",
        output_activation=None,
        use_batchnorm=False,
        save_outputs=False,
    ):
        super().__init__()
        self.save_outputs = save_outputs
        # Used for gradecheck and naming consistency with modules.py (Swiftnet)
        self.input_width = input_size
        self.output_width = output_size
        # if input_size < 16:
        #     print("Currently we do manual encoding for input size < 16.")
        #     hidden_sizes.insert(0, 64)
        self.layers = torch.nn.ModuleList()
        assert isinstance(activation_func, str) or None
        self.activation_func = activation_func
        self.output_activation = output_activation
        self.use_batchnorm = use_batchnorm
        self.network_width = hidden_sizes[0]

        # Input layer
        input_dim = hidden_sizes[0] if input_size <= 64 else input_size
        self.layers.append(torch.nn.Linear(input_dim, hidden_sizes[0], bias=BIAS))
        # if input_size < 16:
        #     # the encoding in the current implementaiton doesn't have grad.
        #     # Set requires_grad to False for the parameters of the first layer (layers[0])
        #     self.layers[0].weight.requires_grad = False

        if self.use_batchnorm:
            self.layers.append(torch.nn.BatchNorm1d(hidden_sizes[0]))

        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            self.layers.append(
                torch.nn.Linear(hidden_sizes[i - 1], hidden_sizes[i], bias=False)
            )

            # BatchNorm layer for hidden layers (if enabled)
            if self.use_batchnorm:
                self.layers.append(torch.nn.BatchNorm1d(hidden_sizes[i]))

        # Output layer
        self.layers.append(torch.nn.Linear(hidden_sizes[-1], output_size, bias=False))

    def forward(self, x):
        batch_size = x.size(0)
        ones = torch.ones(
            (batch_size, self.layers[0].in_features - self.input_width),
            dtype=x.dtype,
            device=x.device,
        )
        # print(ones)
        x = torch.cat((x, ones), dim=1)

        if self.save_outputs:
            layer_outputs = [x[0, :].cpu().detach().numpy()[None,]]
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                x = self._apply_activation(layer(x), self.output_activation)
            else:
                x = self._apply_activation(layer(x), self.activation_func)
            if self.save_outputs:
                layer_outputs.append(x[0, :].cpu().detach().numpy()[None,])

        if self.save_outputs:
            # Specify the file path where you want to save the CSV file
            file_path = "python/torch.csv"
            np.savetxt(
                file_path,
                np.array(layer_outputs).squeeze(),
                delimiter=",",
                fmt="%f",
            )
        return x

    def _apply_activation(self, x, activation_func):
        if activation_func == "relu":
            return F.relu(x)
        elif activation_func == "leaky_relu":
            return F.leaky_relu(x)
        elif activation_func == "sigmoid":
            return torch.sigmoid(x)
        elif activation_func == "tanh":
            return torch.tanh(x)
        elif (
            (activation_func == "None")
            or (activation_func is None)
            or (activation_func == "linear")
        ):
            return x
        else:
            raise ValueError("Invalid activation function")

    def set_weights(self, parameters):
        # if parameters["input_weights"] is not None:
        #     assert (
        #         self.layers[0].weight.shape
        #         == torch.nn.Parameter(parameters["input_weights"]).shape
        #     )
        #     self.layers[0].weight = copy.deepcopy(
        #         torch.nn.Parameter(parameters["input_weights"])
        #     )
        #     if self.input_width < 16:
        #         # the encoding in the current implementaiton doesn't have grad.
        #         # Set requires_grad to False for the parameters of the first layer (layers[0])
        #         self.layers[0].weight.requires_grad = False

        #     offset = 1
        # else:
        offset = 0

        # if "middle_weights" in parameters:
        for i, weight in enumerate(parameters):
            assert self.layers[i + offset].weight.shape == weight.shape
            self.layers[i + offset].weight = copy.deepcopy(torch.nn.Parameter(weight))

    # def set_weights(self, parameters):
    #     # Convert incoming parameters to bfloat16 before setting weights
    #     for i, weight in enumerate(parameters):
    #         bfloat16_weight = torch.tensor(weight, dtype=torch.bfloat16)
    #         assert self.layers[i].weight.shape == bfloat16_weight.shape
    #         self.layers[i].weight = torch.nn.Parameter(bfloat16_weight)
