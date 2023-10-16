import torch
import intel_extension_for_pytorch
from modules import Encoding, Network, NetworkWithInputEncoding

BATCH_SIZE = 64
DEVICE_NAME = "xpu"

## Network With Input Encoding
network_with_encoding = NetworkWithInputEncoding(
    n_input_dims=3,
    n_output_dims=16,
    # encoding_config={
    #     "otype": "Identity",
    #     "scale": "1.0",
    #     "offset": "0.0",
    # },
    encoding_config={
        "otype": "Grid",
        "n_levels": "16",
        "n_features_per_level": "2",
        "log2_hashmap_size": "15",
        "base_resolution": "16",
        "per_level_scale": "1.5",
    },
    # encoding_config={
    #     "otype": "SphericalHarmonics",
    #     "degree": "4",
    # },
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
output = network_with_encoding(input_network_with_encoding)
print(output)

## Encoding
# dir_encoder = Encoding(
#     batch_size=64,
#     n_input_dims=3,
#     encoding_config={
#         "otype": "Identity",
#         "scale": "1.0",
#         "offset": "0.0",
#         # "otype": "Grid",
#         # "n_levels": "16",
#         # "n_features_per_level": "2",
#         # "log2_hashmap_size": "15",
#         # "base_resolution": "16",
#         # "per_level_scale": "1.5",
#     },
#     device=DEVICE_NAME,
# )

# input_encoding = (
#     torch.ones(dir_encoder.n_input_dims, BATCH_SIZE, dtype=torch.float32).to(
#         DEVICE_NAME
#     )
#     * 0.123
# )

# output_encoding = dir_encoder(input_encoding)
# print("Output encoding: ", output_encoding)

# rgb_net = Network(
#     n_input_dims=32,
#     n_output_dims=3,
#     network_config={
#         "otype": "FullyFusedMLP",
#         "activation": "ReLU",
#         "output_activation": "None",
#         "n_neurons": 64,
#         "n_hidden_layers": 2,
#     },
#     device=DEVICE_NAME,
# )
# input_network = torch.ones(rgb_net.n_input_dims, BATCH_SIZE, dtype=torch.float32).to(
#     DEVICE_NAME
# )
# output_network = rgb_net(input_network)
# print(output_network)
