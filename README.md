## Introduction
The goal of this repository is to implements an GPU-accelerated tiny neural network framework using Intel hardware. The implementation uses Intel DPC++ compiler and rely on both SYCL language and Intel Level-Zero API.

Because this network is tight, we are able to load both the activation matrices and the weights matrices into the GPU L1 memory ( shared memory and registers ) that corresponds to the GPU's fast memory. This framework is based on this technical paper (https://github.com/DariusDabert/tiny-nn/blob/Swifnet-feature/data/fully-fused-mlp-diagram.png)

The computation of the product of matrices is realised thanks to an Intel extension called joint_matrix, that is a high-level wrapper to realise systolic array operations. We also use OneMKL to realise matrices product with bigger dimension when to input or the output are too large to fit the matrix in the L1 memory and use joint_matrix.

## Performance on DG2
Comming soon.

## Usage 
```cpp
#include <config.h>

const int batch_size = 256;
const int output_width = 128;
const int n_iterations = 10;
const int WIDTH = 64;
const float scale = 1e-3f;

queue q = queue();

DeviceMem<bf16> inputs = DeviceMem<bf16>(batch_size * WIDTH, q);
DeviceMem<float> output = DeviceMem<float>(batch_size * output_width, q);
DeviceMem<float> target = DeviceMem<float>(batch_size * output_width, q);
DeviceMem<bf16> grads = DeviceMem<bf16>(batch_size * output_width, q);
DeviceMem<float> losses = DeviceMem<float>(batch_size * output_width, q);

inputs.initialize_constant(bf16(1.0f));
output.initialize_constant(0.0f);
target.initialize_constant(0.0f);
grads.initialize_constant(bf16(0.0f));
losses.initialize_constant(0.0f);

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
		{"activation", "ReLU"},
		{"output_activation", "None"},
		{"n_neurons", 64},
		{"n_hidden_layers", 2},
		{"batch_size", 256}
	}},
	};

auto model = create_from_config<64>(q, config);

model.trainer.initialize_params();

for (int i =0; i< n_iterations; i++){
	// fill input here with your code
	
	model.train.training_step(inputs, output, target, grads, losses, scale);

}

```

## Build

To build the tiny-nn librairy, you can clone the github repo on your machine and then your code inside a main function in the source folder, then you can build the library using :

source /opt/intel/oneapi/setvars.sh intel64
make <options>

Where options stand for the name of the machine you are compiling for. Currently, you can build the librairy for dg2 or pvc.

## Required Hardware and Framework
XMX hardware on GPU or AMX on CPU.
DPC++ with level zero.




