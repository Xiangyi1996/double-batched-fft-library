#include <iostream>
#include <vector>
#include <CL/sycl.hpp>

#include "SwiftNetMLP.h"

#include "common.h"
#include "oneapi/mkl.hpp"
//#include "config.h"

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
using bf16 = sycl::ext::oneapi::bfloat16;



void test1() {

    const int batch_size = std::pow(2, 10);
    const int output_width = 128;
    const int WIDTH = 64;
    const int intermediate_output_size = batch_size * WIDTH * 2;
    const int layer_length = WIDTH * batch_size;
    const int n_hidden_matrices = 1;
    const int net_width = 64;
    const int inputs_width = 64;


    const float scale = 1e-3f;
    /*device dev = device(gpu_selector_v);

    std::vector<device> subdev = {};

    subdev = dev.create_sub_devices<sycl::info::partition_property::
        partition_by_affinity_domain>(sycl::info::partition_affinity_domain::numa);

    queue q = queue(subdev[0]);*/

    queue q = queue();

    DeviceMem<bf16> inputs = DeviceMem<bf16>(batch_size * WIDTH, q);
    DeviceMem<float> output = DeviceMem<float>(batch_size * output_width, q);
    DeviceMem<float> target = DeviceMem<float>(batch_size * output_width, q);
    DeviceMem<bf16> grads = DeviceMem<bf16>(batch_size * output_width, q);
    DeviceMem<float> losses = DeviceMem<float>(batch_size * output_width, q);


    float* forward = malloc_device<float>(batch_size * (WIDTH + output_width + WIDTH * 4), q);
    int shmem_size = batch_size * WIDTH * 2;
    const size_t alignment = 4096;

    auto act_mem = sycl::aligned_alloc_device<bf16>(alignment, shmem_size, q);
    auto act_mem_temp = sycl::aligned_alloc_device<float>(alignment, shmem_size, q);

    auto A_forward = sycl::aligned_alloc_device<float>(alignment, layer_length, q);
    auto B_forward = sycl::aligned_alloc_device<float>(alignment, output_width * 64, q);
    auto C_forward = sycl::aligned_alloc_device<float>(alignment, output_width * batch_size, q);

    float* out_inter = malloc_device<float>(batch_size * WIDTH * (n_hidden_matrices + 1), q);
    auto deltas_temp = sycl::aligned_alloc_device<float>(alignment, output_width * batch_size, q);
    DeviceMem<bf16> deltas(output_width * batch_size, q);

    auto A_backward = sycl::aligned_alloc_device<float>(alignment, WIDTH * batch_size, q);
    auto B_backward = sycl::aligned_alloc_device<float>(alignment, batch_size * output_width, q);
    auto C_backward = sycl::aligned_alloc_device<float>(alignment, WIDTH * output_width, q);

    auto A_backward_last_layer = sycl::aligned_alloc_device<float>(alignment, grads.size(), q);
    auto B_backward_last_layer = sycl::aligned_alloc_device<float>(alignment, output_width * WIDTH, q);
    auto C_backward_last_layer = sycl::aligned_alloc_device<float>(alignment, WIDTH * batch_size, q);
    auto D_backward_last_layer = sycl::aligned_alloc_device<float>(alignment, WIDTH * batch_size, q);
    auto E_backward_last_layer = sycl::aligned_alloc_device<float>(alignment, batch_size * WIDTH, q);
    auto F_backward_last_layer = sycl::aligned_alloc_device<float>(alignment, WIDTH * WIDTH, q);

    auto A_dgemm = sycl::aligned_alloc_device<float>(alignment, batch_size * WIDTH, q);
    auto B_dgemm = sycl::aligned_alloc_device<float>(alignment, batch_size * WIDTH, q);
    auto C_dgemm = sycl::aligned_alloc_device<float>(alignment, WIDTH * WIDTH, q);


    inputs.initialize_constant(bf16(2.0f), q);
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
    {"encoding", {
            {"otype", "HashGrid"},
            {"n_levels", 16},
            {"n_features_per_level", 2},
            {"log2_hashmap_size", 19},
            {"base_resolution", 16},
            {"per_level_scale", 2.0},
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

    //auto model = create_from_config(q, config);

    L2Loss loss;
    SGDOptimizer optim = SGDOptimizer(128, 2, 1e-3f, 1e-8f);
    SwiftNetMLP<64> network = SwiftNetMLP<64>(q, 64, 128, 2, Activation::None, Activation::None, batch_size);
    Trainer train(network, loss, optim);

    train.initialize_params();

    q.parallel_for<>(range<1>(inputs.size()), [=](id<1> idx) {
        forward[idx] = (float)inputs.data()[idx];
        });

    for (int i = 0; i < 10; i++) {
        std::cout << i << std::endl;
        train.training_step(inputs,
            forward,
            A_forward,
            B_forward,
            C_forward,
            out_inter,
            deltas_temp,
            deltas,
            A_backward,
            B_backward,
            C_backward,
            A_backward_last_layer,
            B_backward_last_layer,
            C_backward_last_layer,
            D_backward_last_layer,
            E_backward_last_layer,
            F_backward_last_layer,
            A_dgemm,
            B_dgemm,
            C_dgemm,
            output,
            target,
            grads,
            losses,
            scale,
            64);
    }
}

int main() {

    test1();
    return 0;
}
