import torch
import intel_extension_for_pytorch
from modules import SwiftNet
from tiny_nn import Activation
import numpy as np
import time

DEVICE = "cpu"
if __name__ == "__main__":
    width = 64
    input_width = 64
    output_width = 64
    n_hidden_layers = 1

    activation = Activation.ReLU
    output_activation = Activation.Linear

    # for exp in range(20, 22):
    for exp in range(8, 9):
        batch_size = 2**exp
        start = time.perf_counter()
        print(f"Creating batch_size {batch_size}. This may take some time")
        X_train = np.random.rand(batch_size, 64)
        X_train = torch.Tensor(X_train.astype(np.float16)).to(DEVICE)
        y_train = np.random.rand(batch_size, 64)
        y_train = torch.Tensor(y_train.astype(np.float16)).to(DEVICE)
        print(f"Created training data. Took {time.perf_counter() - start}s.")

        network = SwiftNet(
            batch_size,
            width,
            input_width,
            output_width,
            n_hidden_layers,
            activation,
            output_activation,
            device=DEVICE,
        )

        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

        # For some reason the first iteration takes quite some time
        y_pred = network(X_train[0 * batch_size : (0 + 1) * batch_size, ...])

        start_time = time.perf_counter()
        for n in range(1000):
            all_loss = []
            for idx in range(X_train.shape[0] // batch_size):
                y_pred = network(
                    X_train[idx * batch_size : (idx + 1) * batch_size, ...]
                )
                loss = loss_fn(
                    y_pred, y_train[idx * batch_size : (idx + 1) * batch_size]
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            #     all_loss.append(loss.detach().numpy())
            # print(f"{n} - Loss: {np.mean(np.array(all_loss))}")

        end_time = time.perf_counter()
        times = end_time - start_time

        print(f"Batchsize {batch_size}: {times}s")
        network.free_memory()
