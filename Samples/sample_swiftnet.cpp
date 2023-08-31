#include "SwiftNetMLP.h"
#include "trainer.h"
#include "common.h"
#include "config.h"


using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

using bf16 = sycl::ext::oneapi::bfloat16;

int main() {
    const float scale = 1e-3f;

    queue q = queue();

    const int batch_size = 8192; 
    const int output_width = 64;
    const int WIDTH = 64;

    DeviceMem<bf16> inputs = DeviceMem<bf16>(batch_size * WIDTH, q);
    DeviceMem<float> output = DeviceMem<float>(batch_size * output_width, q);
    DeviceMem<float> target = DeviceMem<float>(batch_size * output_width, q);
    DeviceMem<bf16> grads = DeviceMem<bf16>(batch_size * output_width, q);
    DeviceMem<float> losses = DeviceMem<float>(batch_size * output_width, q);

    nlohmann::json config = {
{"loss", {
        {"otype", "L2"}
}},
{"optimizer", {
        {"otype", "sgd"},
        {"output_width", output_width},
        {"n_hidden_layers", 3},
        {"learning_rate", 1e-3},
        {"l2_reg", 1e-8f}
}},
{"network", {
        {"otype", "SwiftNetMLP"},
        {"activation", "ReLU"},
        {"output_activation", "None"},
        {"n_neurons", WIDTH},
        {"n_hidden_layers", 3},
        {"batch_size", batch_size}
}},
    };

    auto model = create_from_config(q, config);

    model.trainer.initialize_params();
    output.initialize_constant(0.0f, q);
    target.initialize_constant(1.0f, q);
    grads.initialize_constant(bf16(0.0f), q);
    losses.initialize_constant(0.0f, q);

    for (int i = 0; i < 1000; i ++){
    model.trainer.training_step(inputs,
        output,
        target,
        grads,
        losses,
        scale,
        64);
    }
    return 0;
}
