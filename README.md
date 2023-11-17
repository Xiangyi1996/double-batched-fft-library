## Introduction
The goal of this repository is to implements an GPU-accelerated tiny neural network framework [original CUDA implementation](https://github.com/NVlabs/tiny-cuda-nn) using Intel hardware. The implementation uses Intel DPC++ compiler and rely on both SYCL language and Intel Level-Zero API.

Because this network is tight, we are able to load both the activation matrices and the weights matrices into the GPU L1 memory (shared memory and registers) that corresponds to the GPU's fast memory.

The computation of the product of matrices is realised thanks to an Intel extension called joint_matrix, that is a high-level wrapper to realise systolic array operations. We also use OneMKL to realise matrices product with bigger dimension when to input or the output are too large to fit the matrix in the L1 memory and use joint_matrix.

## Performance on DG2
![Image](data/performances.png)
We benchmarked the thoughput of our network in training and inference on DG2 GPU. We compared our network with Python Tensorflow on a 4 hidden layers, 64 neurons wide networks.

## Usage
```cpp
#include <config.h>

const int batch_size = std::pow(2, 13);
    const int output_width = 64;
    const int WIDTH = 64;
    const int intermediate_output_size = batch_size * WIDTH * 4;
    const int layer_length = WIDTH * batch_size;
    const int n_hidden_matrices = 4;
    const int net_width = 64;
    const int inputs_width = 64;

    const float scale = 1e-3f;
    device dev = device(gpu_selector_v);

    queue q = queue();

    //uncomment this portion of the code on pvc for better performances
    /*std::vector<device> subdev = {};

    subdev = dev.create_sub_devices<sycl::info::partition_property::
        partition_by_affinity_domain>(sycl::info::partition_affinity_domain::numa);

    queue q = queue(subdev[0]);*/

    DeviceMem<bf16> inputs = DeviceMem<bf16>(batch_size * WIDTH, q);
    DeviceMem<float> output = DeviceMem<float>(batch_size * output_width, q);
    DeviceMem<float> target = DeviceMem<float>(batch_size * output_width, q);
    DeviceMem<bf16> grads = DeviceMem<bf16>(batch_size * output_width, q);
    DeviceMem<float> losses = DeviceMem<float>(batch_size * output_width, q);

    inputs.initialize_constant(bf16(1.0f), q);
    output.initialize_constant(0.0f, q);
    target.initialize_constant(8.0f, q);
    grads.initialize_constant(bf16(0.0f), q);
    losses.initialize_constant(0.0f, q);

    nlohmann::json config = {
    {"loss", {
            {"otype", "L2"}
    }},
    {"optimizer", {
            {"otype", "sgd"},
            {"output_width", 128},
            {"n_hidden_layers", 2},
            {"learning_rate", 1e-3},
            {"l2_reg", 1e-8f}
    }},
    {"network", {
            {"otype", "SwiftNetMLP"},
            {"activation", "None"},
            {"output_activation", "None"},
            {"n_neurons", 64},
            {"n_hidden_layers", 4},
            {"batch_size", 8192}
    }},
    };

    auto model = create_from_config(q, config);

    model.trainer.initialize_params();

    for (int i = 0; i < 1000; i++) {
        std::cout << i << std::endl;
        model.trainer.training_step(inputs,
            output,
            target,
            grads,
            losses,
            scale,
            WIDTH);
    }

    inputs.free_mem(q);
    output.free_mem(q);
    target.free_mem(q);
    grads.free_mem(q);
    losses.free_mem(q);

    model.network->free_mem(q);

    return 0;

```

Build the code as follows to test training for 1000 steps (`benchmarking_training.cpp`).

For validating that the forward and backward passes are calculated correctly, run `unitTest.cpp`. For this, uncomment the `main()` function and comment out the main function in `benchmark_training.cpp`.


## Build

To build the tiny-nn librairy, you can clone the github repo on your machine and put your code in the source folder.
Then you can build the library using :

```
source /usr/share/modules/init/profile.sh && module load intel/oneapi/2023.2.1 && module load intel-comp-rt/agama-ci-devel/682.16 && module load cmake/3.26.0 && module load intel/dpl/2022.2.0
mkdir build && cd build/
cmake -DBUILD_TEST=YES -DoneDPL_DIR=/opt/intel/oneapi/dpl/2022.2.0/lib/cmake/oneDPL/ ..
make 
```

where <options> can be dg2 or pvc depending on the hardware you want to build for.

Note: To make the use of the network, you have to disable the implicit scaling on PVC which can be done by uncommenting the portion of the code indicated in the sample when creating the queue.


## Required Hardware and Framework
Preferred DG2 or PVC with last version of oneAPI.
Mandatory : XMX hardware ( if not DG2 or PVC, pay attention to SG_SIZE and tile sizes ).

## PyTorch extension
tinynn provides pybindings, such that the MLP and encodings can be called as module within Python. Set the PyTorch Extensions up as follows.

### PyBindings

First install pybindings with
```
git submodule update --init --recursive
```

Install the pybindings:
```
cd extern/pybinding11 && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../
make install
```

Install [Intel® Extension for PyTorch*](https://github.com/intel/intel-extension-for-pytorch).
```
python -m pip install torch==2.0.1a0 torchvision==0.15.2a0 intel_extension_for_pytorch==2.0.110+xpu -f https://developer.intel.com/ipex-whl-stable-xpu
```
[Alternatively]: For older versions, run, e.g., `python -m pip install torch==1.13.0a0+git6c9b55e intel_extension_for_pytorch==1.13.120+xpu -f https://developer.intel.com/ipex-whl-stable-xpu`.

After installing the pybindings package, you can now install the pybindings for tiny-nn:
```
cd dpcpp_bindings
pip install -e .
```

### Usage

Example files can be found in the `python` folder to test the functionalities of the tinynn module:

- benchmark_dpcpp.py: Benchmark of dpcpp implementation of tinynn. Reports the speed of training 1000 loops of data with batch size 2^20.
- benchmark_tf.py:  Benchmark standard Multi-Layer Perceptron in Tensorflow. Reports the speed of training 1000 loops of data with batch size 2^20.
- test_fwd.py: Tests forward pass of SwiftNet
- test_bwd_training.py: Tests backward pass of SwiftNet
- test_einstein.py: Tests learning an image of Einstein (x,y pixel coordinate in, grey-value of that point out).

## Benchmark

For the benchmark, tensorflow with XPU support is required:
```
pip install tensorflow==2.13.0
pip install --upgrade intel-extension-for-tensorflow[xpu]
```

# Acknowledgement

- The original implementation of SwiftNet was conducted by Darius Dabert ([DariusDabert](https://github.com/DariusDabert)) and Adrien Tousnakhoff ([Tousnaaa](https://github.com/Tousnaaa))
- The pybindings were developed by Kai Yuan (kai.yuan@intel.com)
- The encodings were developed by xyz
