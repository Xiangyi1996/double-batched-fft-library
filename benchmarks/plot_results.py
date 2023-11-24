import json
import matplotlib.pyplot as plt
import os
import math

FILE_PATH = f"{os.path.dirname(__file__)}/results/"


def plot_result(filename, name):
    # Load JSON data from file
    with open(f"{FILE_PATH}/{filename}.json", "r") as file:
        data = json.load(file)

    # Extract batch sizes, training throughput, and inference throughput
    batch_sizes = [entry["batch_size"] for entry in data["SwiftNet"]]
    training_throughput = [entry["training_throughput"] for entry in data["SwiftNet"]]
    inference_throughput = [entry["inference_throughput"] for entry in data["SwiftNet"]]
    batch_sizes_pow = [int(math.log2(n)) for n in batch_sizes]

    # Plotting the training throughput
    plt.figure(figsize=(8, 6))
    plt.plot(batch_sizes_pow, training_throughput, marker="o")
    plt.xlabel("Batch Size")
    plt.ylabel("Training Throughput")
    plt.title(f"{name} - Training")
    plt.grid(True)
    plt.xticks(batch_sizes_pow, [f"2^{n}" for n in batch_sizes_pow])
    plt.savefig(
        os.path.join(os.path.dirname(__file__), "images", "training_throughput.pdf")
    )

    # Plotting the inference throughput
    plt.figure(figsize=(8, 6))
    plt.plot(batch_sizes_pow, inference_throughput, marker="o")
    plt.xlabel("Batch Size")
    plt.ylabel("Inference Throughput")
    plt.title(f"{name} - Inference")
    plt.grid(True)
    plt.xticks(batch_sizes_pow, [f"2^{n}" for n in batch_sizes_pow])

    plt.savefig(
        os.path.join(os.path.dirname(__file__), "images", "inference_throughput.pdf")
    )


if __name__ == "__main__":
    plot_result("benchmark_training", "Function Approximation")
