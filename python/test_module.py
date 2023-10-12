import torch
import intel_extension_for_pytorch
from modules import Encoding, Network, NetworkWithInputEncoding

BATCH_SIZE = 64
DEVICE_NAME = "xpu"

## Network With Input Encoding
network_with_encoding = NetworkWithInputEncoding(
    n_input_dims=3,
    n_output_dims=16,
    encoding_config={
        "otype": "Identity",
        "scale": "1.0",
        "offset": "1.0",
    },
    network_config={
        "otype": "FullyFusedMLP",
        "activation": "ReLU",
        "output_activation": "None",
        "n_neurons": 64,
        "n_hidden_layers": 1,
    },
    batch_size=64,
    device=DEVICE_NAME,
)
input_network_with_encoding = torch.ones(
    network_with_encoding.n_input_dims, BATCH_SIZE, dtype=torch.float32
).to(DEVICE_NAME)
network_with_encoding(input_network_with_encoding)

## Encoding
dir_encoder = Encoding(
    batch_size=64,
    n_input_dims=3,
    encoding_config={
        "otype": "Identity",
        "scale": "1.0",
        "offset": "1.0",
    },
    device=DEVICE_NAME,
)

input_encoding = torch.ones(
    dir_encoder.n_input_dims, BATCH_SIZE, dtype=torch.float32
).to(DEVICE_NAME)

output_encoding = dir_encoder(input_encoding)

rgb_net = Network(
    n_input_dims=32,
    n_output_dims=3,
    network_config={
        "otype": "FullyFusedMLP",
        "activation": "ReLU",
        "output_activation": "Sigmoid",
        "n_neurons": 64,
        "n_hidden_layers": 2,
    },
    device=DEVICE_NAME,
)
input_network = torch.ones(rgb_net.n_input_dims, BATCH_SIZE, dtype=torch.float32).to(
    DEVICE_NAME
)
rgb_net(input_network)
