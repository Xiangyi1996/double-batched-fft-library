import json
import matplotlib.pyplot as plt
import numpy as np
import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))


def plot_throughput(json_file):
    with open(json_file, "r") as file:
        data = json.load(file)

    # Sort the data by batch size in ascending order
    sorted_data = sorted(data["SwiftNet"], key=lambda x: x["batch_size"])

    batch_sizes = [entry["batch_size"] for entry in sorted_data]
    inference_throughput = [entry["inference_throughput"] for entry in sorted_data]
    training_throughput = [entry["training_throughput"] for entry in sorted_data]

    # Calculate the power of 2 for each batch size
    custom_x_ticks = [int(np.log2(batch)) for batch in batch_sizes]

    # Create two subplots
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(custom_x_ticks, training_throughput, marker="o", linestyle="-", color="r")
    plt.title("Training Throughput")
    plt.xlabel("n (2^n)")
    plt.ylabel("Throughput")
    # plt.yscale("log")  # Set y-axis to logarithmic scale

    # Set custom x-axis ticks as "n" values
    plt.xticks(custom_x_ticks)

    plt.subplot(1, 2, 2)

    plt.plot(custom_x_ticks, inference_throughput, marker="o", linestyle="-", color="b")
    plt.title("Inference Throughput")
    plt.xlabel("n (2^n)")
    plt.ylabel("Throughput")
    # plt.yscale("log")  # Set y-axis to logarithmic scale

    # Set custom x-axis ticks as "n" values
    plt.xticks(custom_x_ticks)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage:
    json_file = f"{CUR_DIR}/../bench_result_ours.json"
    plot_throughput(json_file)
