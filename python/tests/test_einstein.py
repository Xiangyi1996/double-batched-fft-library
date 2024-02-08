import argparse
import numpy as np
import os
import sys
import torch
import time
import torch
import intel_extension_for_pytorch
from utils import create_models

# SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
# sys.path.insert(0, SCRIPTS_DIR)

from common import read_image, write_image, ROOT_DIR

DATA_DIR = os.path.join(ROOT_DIR, "data")
IMAGES_DIR = os.path.join(ROOT_DIR, "images")

DEVICE_NAME = "cpu"


class Image(torch.nn.Module):
    def __init__(self, filename, device):
        super(Image, self).__init__()
        self.data = read_image(filename)
        self.shape = self.data.shape
        self.data = torch.from_numpy(self.data).float().to(device)

    def forward(self, xs):
        with torch.no_grad():
            # Bilinearly filtered lookup from the image. Not super fast,
            # but less than ~20% of the overall runtime of this example.
            shape = self.shape

            xs = xs * torch.tensor([shape[1], shape[0]], device=xs.device).float()
            indices = xs.long()
            lerp_weights = xs - indices.float()

            x0 = indices[:, 0].clamp(min=0, max=shape[1] - 1)
            y0 = indices[:, 1].clamp(min=0, max=shape[0] - 1)
            x1 = (x0 + 1).clamp(max=shape[1] - 1)
            y1 = (y0 + 1).clamp(max=shape[0] - 1)

            return (
                self.data[y0, x0]
                * (1.0 - lerp_weights[:, 0:1])
                * (1.0 - lerp_weights[:, 1:2])
                + self.data[y0, x1]
                * lerp_weights[:, 0:1]
                * (1.0 - lerp_weights[:, 1:2])
                + self.data[y1, x0]
                * (1.0 - lerp_weights[:, 0:1])
                * lerp_weights[:, 1:2]
                + self.data[y1, x1] * lerp_weights[:, 0:1] * lerp_weights[:, 1:2]
            )


def get_args():
    parser = argparse.ArgumentParser(
        description="Image benchmark using PyTorch bindings."
    )

    parser.add_argument(
        "image", nargs="?", default="data/images/albert.jpg", help="Image to match"
    )
    parser.add_argument(
        "n_steps",
        nargs="?",
        type=int,
        default=10000,
        help="Number of training steps",
    )
    parser.add_argument(
        "result_filename", nargs="?", default="", help="Number of training steps"
    )

    args = parser.parse_args()
    return args


def generate_x_y_grid(image_shape):
    # Variables for saving/displaying image results

    half_dx = 0.5 / image_shape[0]
    half_dy = 0.5 / image_shape[1]
    xs = torch.linspace(half_dx, 1 - half_dx, image_shape[0], device=DEVICE_NAME)
    ys = torch.linspace(half_dy, 1 - half_dy, image_shape[1], device=DEVICE_NAME)
    xv, yv = torch.meshgrid([xs, ys])

    xy = torch.stack((yv.flatten(), xv.flatten())).t()

    x = torch.linspace(0, image_shape[0] - 1, image_shape[0], device=DEVICE_NAME).int()
    y = torch.linspace(0, image_shape[1] - 1, image_shape[1], device=DEVICE_NAME).int()
    xm, ym = torch.meshgrid([x, y])
    xy_coord = torch.stack((ym.flatten(), xm.flatten())).t()

    return xy, xy_coord


def generate_image_from_function(model, image_shape, batch_size):
    # Create a NumPy array to store the image data
    image_data = torch.zeros(image_shape)

    # Generate coordinates for querying the function
    coords, coords_int = generate_x_y_grid(image_shape)

    total_batches = int(np.ceil(coords.shape[0] / batch_size))
    for idx in range(total_batches):
        if idx < (total_batches - 1):
            idx_start = idx * batch_size
            idx_end = (idx + 1) * batch_size
        else:
            idx_start = -batch_size - 1
            idx_end = -1
        coordinates = coords[idx_start:idx_end, ...]
        # Query the function for grayscale values for the entire batch
        grayscale_values = model(coordinates)

        # # Ensure grayscale_values are within [0, 255]
        # grayscale_values = np.clip(grayscale_values, 0, 255)

        # Populate the NumPy array with the grayscale values
        for sub_idx, coordinate in enumerate(coords_int[idx_start:idx_end]):
            x = coordinate[1]
            y = coordinate[0]
            image_data[x, y, 0] = grayscale_values[sub_idx, 0]

    return image_data


if __name__ == "__main__":
    print("================================================================")
    print("This script replicates the behavior of tiny cuda nn ")
    print("================================================================")

    print(f"Using PyTorch version {torch.__version__}")
    args = get_args()

    image = Image(args.image, DEVICE_NAME)
    n_channels = image.data.shape[2]

    # batch_size = 2**18
    # batch_size = 2**17
    batch_size = 2**9
    input_width = 2
    output_width = 1
    n_hidden_layers = 2
    # activation = "leaky_relu"
    # activation = "relu"
    activation = "tanh"
    output_activation = "linear"

    model_dpcpp, model_torch = create_models(
        input_width,
        n_hidden_layers,
        output_width,
        activation,
        output_activation,
        batch_size=batch_size,
        device_name=DEVICE_NAME,
    )
    model = model_dpcpp
    model = model_torch
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Variables for saving/displaying image results
    resolution = image.data.shape[0:2]
    img_shape = resolution + torch.Size([image.data.shape[2]])
    n_pixels = resolution[0] * resolution[1]

    half_dx = 0.5 / resolution[0]
    half_dy = 0.5 / resolution[1]
    xs = torch.linspace(half_dx, 1 - half_dx, resolution[0], device=DEVICE_NAME)
    ys = torch.linspace(half_dy, 1 - half_dy, resolution[1], device=DEVICE_NAME)
    xv, yv = torch.meshgrid([xs, ys])

    xy = torch.stack((yv.flatten(), xv.flatten())).t()

    path = f"data/images/reference.jpg"
    print(f"Writing '{path}'... ", end="")
    write_image(path, image(xy).reshape(img_shape).detach().cpu().numpy())
    print("done.")

    prev_time = time.perf_counter()

    interval = 10

    print(f"Beginning optimization with {args.n_steps} training steps.")

    try:
        batch = torch.rand([batch_size, 2], device=device, dtype=torch.float32)
        traced_image = torch.jit.trace(image, batch)
    except:
        # If tracing causes an error, fall back to regular execution
        print(
            f"WARNING: PyTorch JIT trace failed. Performance will be slightly worse than regular."
        )
        traced_image = image

    for i in range(args.n_steps):
        batch = torch.rand([batch_size, 2], device=DEVICE_NAME, dtype=torch.float32)
        targets = traced_image(batch)
        output = model(batch)

        relative_l2_error = (output - targets.to(output.dtype)) ** 2 / (
            output.detach() ** 2 + 0.01
        )
        loss = relative_l2_error.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % interval == 0:
            loss_val = loss.item()
            elapsed_time = time.perf_counter() - prev_time
            print(f"Step#{i}: loss={loss_val} time={int(elapsed_time*1000000)}[µs]")

            path = f"data/images/{i}.jpg"
            print(f"Writing '{path}'... ", end="")
            with torch.no_grad():
                img_as_array = generate_image_from_function(
                    model,
                    img_shape,
                    batch_size
                    # model, img_shape, batch_size
                )
                write_image(
                    path,
                    img_as_array.clamp(0.0, 1.0).detach().cpu().numpy(),
                )
            print("done.")

            # Ignore the time spent saving the image
            prev_time = time.perf_counter()

            if i > 0 and interval < 1000:
                interval *= 10

    if args.result_filename:
        print(f"Writing '{args.result_filename}'... ", end="")
        with torch.no_grad():
            write_image(
                args.result_filename,
                model(xy).reshape(img_shape).clamp(0.0, 1.0).detach().cpu().numpy(),
            )
        print("done.")

    # model.free_memory()
