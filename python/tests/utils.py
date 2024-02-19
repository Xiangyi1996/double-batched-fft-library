import numpy as np

# import torch

# import intel_extension_for_pytorch  # required for tinyn_nn (SwiftNet inside)

from mlp import MLP
from modules import NetworkWithInputEncoding, Network
from tiny_nn import Activation
import torch


def get_grad_params(model):
    grads_all = []
    params_all = []
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            gradient = param.grad
            if len(gradient.shape) == 1:
                grad = model.get_reshaped_params(
                    gradient.clone(), is_packed_format=True
                )
            else:
                grad = gradient.clone()
            grads_all.append(grad)

        if len(param.data.shape) == 1:
            param_reshaped = model.get_reshaped_params(
                param.data.clone(), is_packed_format=True
            )
        else:
            param_reshaped = param.data.clone()
        params_all.append(param_reshaped)
    return grads_all, params_all


def compare_matrices(weights_dpcpp, weights_torch, tol=1e-2):
    total_diff = []
    for layer in range(len(weights_dpcpp)):
        assert weights_dpcpp[layer].shape == weights_torch[layer].shape
        abs_sum_in_layer_dpcpp = 0.0
        abs_sum_in_layer_torch = 0.0
        for row in range(weights_torch[layer].shape[0]):
            for col in range(weights_torch[layer].shape[1]):
                torch_val = weights_torch[layer][row, col]
                dpcpp_val = weights_dpcpp[layer][row, col]
                abs_sum_in_layer_dpcpp += float(abs(dpcpp_val))
                abs_sum_in_layer_torch += float(abs(torch_val))
                if abs(torch_val) < 1e-9 and abs(dpcpp_val) < 1e-9:
                    continue
                rel_diff = abs(abs(torch_val).sum() - abs(dpcpp_val).sum()) / (
                    abs(torch_val).sum()
                )
                total_diff.append(rel_diff.cpu().numpy())
                if rel_diff > tol:
                    print(
                        f"Layer {layer+1} - at [{row}, {col}]: {rel_diff*100:.4f}% (sum: ",
                        f"{torch_val:.7f}, and {dpcpp_val:.7f})",
                    )
                # assert (
                #     rel_diff < tol
                # ), f"Difference larger than {tol*100:.4f}%: {rel_diff* 100:.4f}%"
        print(
            f"Abs sum in layer for torch: {abs_sum_in_layer_torch:.7f}, dpcpp: {abs_sum_in_layer_dpcpp:.7f}"
        )
    print(f"Average difference: {100*np.mean(np.array(total_diff)):.4f}%")


def create_models(
    input_size,
    hidden_sizes,
    output_size,
    activation_func,
    output_func,
    device_name,
):

    # Create and test CustomMLP
    model_torch = MLP(
        input_size,
        hidden_sizes,
        output_size,
        activation_func,
        output_func,
        use_batchnorm=False,
        dtype=torch.bfloat16,
    ).to(device_name)
    # model_torch.eval()

    network_config = {
        "activation": activation_func,
        "output_activation": output_func,
        "n_neurons": hidden_sizes[0],
        "n_hidden_layers": len(hidden_sizes),
    }
    model_dpcpp = Network(
        n_input_dims=input_size,
        n_output_dims=output_size,
        network_config=network_config,
        device=device_name,
    )

    weights = model_dpcpp.get_reshaped_params(datatype=torch.bfloat16)
    # print(weights)
    model_torch.set_weights(weights)
    model_torch.to(device_name)
    grads_dpcpp, params_dpcpp = get_grad_params(model_dpcpp)
    grads_torch, params_torch = get_grad_params(model_torch)
    print("Comparing params set. Error must be 0%")
    compare_matrices(params_dpcpp[0], params_torch, tol=1e-7)
    return model_dpcpp, model_torch
