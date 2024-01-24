# from mlp import MLP
from mlp_benchmark import MLP

import torch
import torch.nn as nn
import numpy as np
import time
import json


# PRECISION = torch.float32
PRECISION = torch.float16

# DEVICE_NAME = "cuda"
DEVICE_NAME = "xpu"

if DEVICE_NAME == "xpu":
    import intel_extension_for_pytorch as ipex  # required for xpu support

    OPTIMISE_MODEL = False  # not supported
else:
    OPTIMISE_MODEL = False


def custom_loss_function(input, target):
    # Implement your custom loss function here
    # This is just a placeholder; replace it with your actual loss computation
    loss = torch.mean((input - target) ** 2)
    return loss


class CustomHalfPrecisionLoss(nn.Module):
    def __init__(self):
        super(CustomHalfPrecisionLoss, self).__init__()

    def forward(self, input, target):
        # Ensure input and target are in half precision
        input = input.to(PRECISION)
        target = target.to(PRECISION)

        # Compute your custom loss using half precision
        loss = custom_loss_function(input, target)

        return loss


def get_times(input_size, hidden_sizes, output_size, batch_sizes, name, iters=10):
    training_times = []
    infer_times = []
    for i in range(iters):
        result = start_training(
            input_size, hidden_sizes, output_size, batch_sizes, name
        )
        training_times.append(result["training_time"])
        infer_times.append(result["infer_time"])

    mean_training_time = np.mean(training_times)
    std_training_time = np.std(training_times)
    mean_inference_time = np.mean(infer_times)
    std_inference_time = np.std(infer_times)

    print(
        f"Training time: {mean_training_time:.4f} [s] (std: {std_training_time:.4f} [s]), Inference time: {mean_inference_time:.4f} [s] (std: {std_inference_time:.4f} [s])"
    )
    return mean_training_time, mean_inference_time


def start_training(
    input_size, hidden_sizes, output_size, batch_sizes, name, print_debug=False
):
    activation_func = "relu"
    output_func = None

    model = (
        MLP(
            input_size,
            hidden_sizes,
            output_size,
            activation_func,
            output_func,
        )
        .to(DEVICE_NAME)
        .to(PRECISION)
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    # Set the new weights using the set_weights method

    model.set_weights(constant_value=0.01)
    # Run the network
    PRINT_INTERVAL = 100
    bench_result = {}

    for batch_size in batch_sizes:
        if print_debug:
            print(f"Running batch size: {np.log2(batch_size)}")
        N_ITERS = 1000
        WARMUP_ITERS = N_ITERS / 4
        PRINT_INTERVAL = 100

        throughputs = []
        model.train()

        if OPTIMISE_MODEL:
            model_torch, optimized_optimizer = ipex.optimize(
                model, optimizer=torch.optim.Adam
            )
        else:
            model_torch = model

        timer_start = time.perf_counter()
        # print("Timer start")
        elapsed_times = []
        time_loss = []

        loss_fn = CustomHalfPrecisionLoss()

        target_tensor = (
            # torch.randn((batch_size, output_size)).to(DEVICE_NAME).to(PRECISION)
            2
            * torch.ones((batch_size, output_size)).to(DEVICE_NAME).to(PRECISION)
        )
        input_tensor = (
            # torch.randn((batch_size, input_size)).to(DEVICE_NAME).to(PRECISION)
            torch.ones((batch_size, input_size))
            .to(DEVICE_NAME)
            .to(PRECISION)
        )
        for i in range(N_ITERS):
            # timer_loss = time.perf_counter()
            # target_tensor = torch.randn((batch_size, output_size)).to(DEVICE_NAME)
            # input_tensor = torch.randn((batch_size, input_size)).to(DEVICE_NAME)

            output_tensor = model_torch(input_tensor)

            timer_loss = time.perf_counter()
            loss = loss_fn(output_tensor, target_tensor)
            time_loss.append(time.perf_counter() - timer_loss)

            # loss.requires_grad = True
            # optimizer.zero_grad()
            loss.backward()
            # fwd_time.append(time.perf_counter() - timer_loss)
            # optimizer.step()

            if i % PRINT_INTERVAL == 0:
                # print(output_tensor[0, 0])
                if name == "benchmark":
                    assert torch.isclose(
                        output_tensor[0, 0],
                        torch.tensor([0.0047], dtype=PRECISION).to(DEVICE_NAME),
                        atol=1e-4,
                    ), "Error in calculation."
                if name == "image":
                    assert torch.isclose(
                        output_tensor[0, 0],
                        torch.tensor([0.1311], dtype=PRECISION).to(DEVICE_NAME),
                        atol=1e-4,
                    ), "Error in calculation."
                if name == "pinns":
                    assert torch.isclose(
                        output_tensor[0, 0],
                        torch.tensor([0.0032], dtype=PRECISION).to(DEVICE_NAME),
                        atol=1e-4,
                    ), "Error in calculation."
                if name == "nerf":
                    assert torch.isclose(
                        output_tensor[0, 0],
                        torch.tensor([0.0537], dtype=PRECISION).to(DEVICE_NAME),
                        atol=1e-4,
                    ), "Error in calculation."
                elapsed_time = time.perf_counter() - timer_start
                throughput = PRINT_INTERVAL * batch_size / elapsed_time
                timer_start = time.perf_counter()
                if i > WARMUP_ITERS:
                    throughputs.append(throughput)
                    elapsed_times.append(elapsed_time / PRINT_INTERVAL)
                if print_debug:
                    print(
                        f"Iteration#{i}: time={int(elapsed_time * 1000000)}[µs] thp={throughput}/s"
                    )
        # print(f"Time for loss calc: {N_ITERS*np.mean(time_loss)}")
        # print(f"Time for fwd calc: {np.mean(fwd_time)}")

        # print(f"Elapsed times: {np.mean(elapsed_times)}")
        training_time = N_ITERS * (np.mean(elapsed_times) - np.mean(time_loss))
        if print_debug:
            print(
                f"Time for {N_ITERS} training: {training_time}[s]"
                # f"Time for {N_ITERS} training: {N_ITERS*(np.mean(elapsed_times) )}[s]"
            )
        mean_training_throughput = np.mean(throughputs[1:])
        if print_debug:
            print(
                f"Finished training benchmark. Mean throughput is {mean_training_throughput}/s. Waiting 10s for GPU to cool down."
            )
        time.sleep(5)
        # print("Continuing")
        # Inference
        model.eval()

        if OPTIMISE_MODEL:
            model_torch = ipex.optimize(model)
        else:
            model_torch = model

        throughputs = []
        timer_start = time.perf_counter()
        elapsed_times = []
        time_loss = []
        with torch.no_grad():
            input_tensor = (
                torch.ones((batch_size, input_size))
                # torch.randn((batch_size, input_size))
                .to(DEVICE_NAME).to(PRECISION)
            )

        for i in range(N_ITERS):
            with torch.no_grad():
                # timer_loss = time.perf_counter()
                # input_tensor = torch.randn((batch_size, input_size)).to(DEVICE_NAME)
                # time_loss.append(time.perf_counter() - timer_loss)

                output_tensor = model_torch(input_tensor)

            if i % PRINT_INTERVAL == 0:
                if name == "benchmark":
                    assert torch.isclose(
                        output_tensor[0, 0],
                        torch.tensor([0.0047], dtype=PRECISION).to(DEVICE_NAME),
                        atol=1e-4,
                    ), "Error in calculation."
                if name == "image":
                    assert torch.isclose(
                        output_tensor[0, 0],
                        torch.tensor([0.1311], dtype=PRECISION).to(DEVICE_NAME),
                        atol=1e-4,
                    ), "Error in calculation."
                if name == "pinns":
                    assert torch.isclose(
                        output_tensor[0, 0],
                        torch.tensor([0.0032], dtype=PRECISION).to(DEVICE_NAME),
                        atol=1e-4,
                    ), "Error in calculation."
                if name == "nerf":
                    assert torch.isclose(
                        output_tensor[0, 0],
                        torch.tensor([0.0537], dtype=PRECISION).to(DEVICE_NAME),
                        atol=1e-4,
                    ), "Error in calculation."
                elapsed_time = time.perf_counter() - timer_start
                throughput = PRINT_INTERVAL * batch_size / elapsed_time
                timer_start = time.perf_counter()
                # print(output_tensor[0, :])
                if i > WARMUP_ITERS:
                    elapsed_times.append(elapsed_time / PRINT_INTERVAL)
                    throughputs.append(throughput)
                if print_debug:
                    print(
                        f"Iteration#{i}: time={int(elapsed_time * 1000000)}[µs] thp={throughput}/s"
                    )

        # print(f"Time for loss calc: {np.mean(time_loss)}")
        # print(f"Elapsed times per iter: {np.mean(elapsed_times)}")
        infer_time = N_ITERS * (np.mean(elapsed_times))
        if print_debug:
            print(
                # f"Time for {N_ITERS} inference: {N_ITERS*(np.mean(elapsed_times) - np.mean(time_loss))}[s]"
                f"Time for {N_ITERS} inference: {infer_time}[s]"
            )
        mean_inference_throughput = np.mean(throughputs[1:])
        if print_debug:
            print(
                f"Finished inference benchmark. Mean throughput is {mean_inference_throughput}/s. Waiting 10s for GPU to cool down."
            )
        # time.sleep(10)

        # Mean throughput (discounting the first one due to XLA compilation)
    bench_result = {
        "batch_size": batch_size,
        "training_throughput": mean_training_throughput,
        "inference_throughput": mean_inference_throughput,
        "training_time": training_time,
        "infer_time": infer_time,
    }

    return bench_result
    # with open("bench_result_pytorch.json", "w") as f:
    #     json.dump(bench_result, f)


if __name__ == "__main__":
    # Benchmark

    # input_size = 64,
    # hidden_sizes = [64] * 4,
    # output_size = 64,
    # batch_sizes = [
    #         2**14,
    #         2**15,
    #         2**16,
    #         2**17,
    #         2**18,
    #         2**19,
    #         2**20,
    #         2**21,
    #         2**22,
    #     ]
    # start_training(input_size, hidden_sizes , output_size , batch_sizes )

    # print("Benchmark")
    # input_size = 64
    # hidden_sizes = [64] * 11
    # output_size = 64
    # batch_sizes = [2**17]
    # train_time, infer_time = get_times(
    #     input_size, hidden_sizes, output_size, batch_sizes, "benchmark"
    # )

    # Image compression
    print("Image compression")
    input_size = 32
    hidden_sizes = [64] * 2
    output_size = 1
    batch_sizes = [2**23]
    train_time, infer_time = get_times(
        input_size, hidden_sizes, output_size, batch_sizes, "image"
    )

    # # Pinns
    # print("Pinns ")
    # input_size = 3
    # hidden_sizes = [64] * 5
    # output_size = 3
    # batch_sizes = [
    #     2**17,
    # ]
    # train_time, infer_time = get_times(
    #     input_size, hidden_sizes, output_size, batch_sizes, "pinns"
    # )

    # # NeRF
    # print("Nerf")

    # input_size = 32
    # hidden_sizes = [64] * 4
    # output_size = 4
    # batch_sizes = [2**20]
    # train_time, infer_time = get_times(
    #     input_size, hidden_sizes, output_size, batch_sizes, "nerf"
    # )
