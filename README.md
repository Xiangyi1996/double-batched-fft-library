
The goal of this repository is to implements an GPU-accelerated tiny neural network framework using Intel hardware. The implementation uses Intel DPC++ compiler that rely on both SYCL language and Intel level0 API.

Because this network is tight, we are able to load both the activation matrices and the weights matrices into the GPU L1 memory ( shared memory and registers ) that corresponds to the GPU's fast memory. This framework is based on this technical paper https://raw.githubusercontent.com/NVlabs/tiny-cuda-nn/master/data/readme/fully-fused-mlp-diagram.png.

The computation of the product of matrices is realised thanks to an Intel extension called joint_matrix, that is a high-level wrapper which uses level0 to realise systolic array operations. We also use OneMKL to realise matrices product with bigger dimension when to input or the output are too large to fit the matrix in the L1 memory and use joint_matrix.


