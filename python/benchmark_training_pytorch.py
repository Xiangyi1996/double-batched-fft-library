# from mlp import MLP
from mlp_benchmark import MLP

import torch
import torch.nn as nn
import numpy as np
import time
import json

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
        input = input.half()
        target = target.half()

        # Compute your custom loss using half precision
        loss = custom_loss_function(input, target)

        return loss


def start_training(
    input_size=64,
    hidden_sizes=[64] * 4,
    output_size=64,
    batch_sizes=[
        2**14,
        2**15,
        2**16,
        2**17,
        2**18,
        2**19,
        2**20,
        2**21,
        2**22,
    ],
):
    activation_func = "relu"
    output_func = None

    model = MLP(
        input_size,
        hidden_sizes,
        output_size,
        activation_func,
        output_func,
    ).to(DEVICE_NAME)

    # Run the network
    PRINT_INTERVAL = 100
    bench_result = {"pytorch": []}

    for batch_size in batch_sizes:
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
        print("Timer start")
        elapsed_times = []
        time_loss = []
        fwd_time = []

        loss_fn = torch.nn.MSELoss()

        target_tensor = torch.randn((batch_size, output_size)).to(DEVICE_NAME)
        input_tensor = torch.randn((batch_size, input_size)).to(DEVICE_NAME)
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
                elapsed_time = time.perf_counter() - timer_start
                throughput = PRINT_INTERVAL * batch_size / elapsed_time
                timer_start = time.perf_counter()
                if i > WARMUP_ITERS:
                    throughputs.append(throughput)
                    elapsed_times.append(elapsed_time / PRINT_INTERVAL)
                print(
                    f"Iteration#{i}: time={int(elapsed_time * 1000000)}[µs] thp={throughput}/s"
                )
        print(f"Time for loss calc: {N_ITERS*np.mean(time_loss)}")
        # print(f"Time for fwd calc: {np.mean(fwd_time)}")

        print(f"Elapsed times: {np.mean(elapsed_times)}")
        print(
            f"Time for {N_ITERS} training: {N_ITERS*(np.mean(elapsed_times) - np.mean(time_loss))}[s]"
            # f"Time for {N_ITERS} training: {N_ITERS*(np.mean(elapsed_times) )}[s]"
        )
        mean_training_throughput = np.mean(throughputs[1:])

        print(
            f"Finished training benchmark. Mean throughput is {mean_training_throughput}/s. Waiting 10s for GPU to cool down."
        )
        time.sleep(2)
        print("Continuing")
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
            input_tensor = torch.randn((batch_size, input_size)).to(DEVICE_NAME)

        for i in range(N_ITERS):
            with torch.no_grad():
                # timer_loss = time.perf_counter()
                # input_tensor = torch.randn((batch_size, input_size)).to(DEVICE_NAME)
                # time_loss.append(time.perf_counter() - timer_loss)

                output_tensor = model_torch(input_tensor)

            if i % PRINT_INTERVAL == 0:
                elapsed_time = time.perf_counter() - timer_start
                throughput = PRINT_INTERVAL * batch_size / elapsed_time
                timer_start = time.perf_counter()

                if i > WARMUP_ITERS:
                    elapsed_times.append(elapsed_time / PRINT_INTERVAL)
                    throughputs.append(throughput)
                print(
                    f"Iteration#{i}: time={int(elapsed_time * 1000000)}[µs] thp={throughput}/s"
                )

        # print(f"Time for loss calc: {np.mean(time_loss)}")
        print(f"Elapsed times per iter: {np.mean(elapsed_times)}")
        print(
            # f"Time for {N_ITERS} inference: {N_ITERS*(np.mean(elapsed_times) - np.mean(time_loss))}[s]"
            f"Time for {N_ITERS} inference: {N_ITERS*(np.mean(elapsed_times))}[s]"
        )
        mean_inference_throughput = np.mean(throughputs[1:])

        print(
            f"Finished inference benchmark. Mean throughput is {mean_inference_throughput}/s. Waiting 10s for GPU to cool down."
        )
        # time.sleep(10)

        # Mean throughput (discounting the first one due to XLA compilation)
        bench_result["pytorch"].append(
            {
                "batch_size": batch_size,
                "training_throughput": mean_training_throughput,
                "inference_throughput": mean_inference_throughput,
            }
        )

    with open("bench_result_pytorch.json", "w") as f:
        json.dump(bench_result, f)


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
    # start_training(input_size, hidden_sizes, output_size, batch_sizes)

    # Image compression
    print("Image compression")
    input_size = 32
    hidden_sizes = [64] * 2
    output_size = 1
    batch_sizes = [2**23]
    start_training(input_size, hidden_sizes, output_size, batch_sizes)

    # Pinns
    print("Pinns ")
    input_size = 3
    hidden_sizes = [64] * 5
    output_size = 3
    batch_sizes = [
        2**17,
    ]
    start_training(input_size, hidden_sizes, output_size, batch_sizes)

    # NeRF
    print("Nerf")

    input_size = 32
    hidden_sizes = [64] * 4
    output_size = 4
    batch_sizes = [2**20]
    start_training(input_size, hidden_sizes, output_size, batch_sizes)
