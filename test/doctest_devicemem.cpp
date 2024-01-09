#include "DeviceMatrix.h"
#include "DeviceMem.h"
#include "doctest/doctest.h"

using namespace sycl;
using bf16 = sycl::ext::oneapi::bfloat16;

TEST_CASE("DeviceMem") {
    queue q = queue();
    SUBCASE("Default constructor") {
        DeviceMem<float> mem;

        CHECK(mem.size() == 0);
        CHECK(mem.data() == nullptr);
    }

    SUBCASE("Size constructor") {
        size_t size = 100;

        SUBCASE("Valid size uint8") {
            DeviceMem<uint8_t> mem(size);

            CHECK(mem.size() == size);
            CHECK(mem.data() != nullptr);
        }
        SUBCASE("Valid size bf16") {
            DeviceMem<bf16> mem(size);

            CHECK(mem.size() == size);
            CHECK(mem.data() != nullptr);
        }
        SUBCASE("Valid size float") {
            DeviceMem<float> mem(size);

            CHECK(mem.size() == size);
            CHECK(mem.data() != nullptr);
        }

        SUBCASE("Zero size uint8_t") {
            DeviceMem<uint8_t> mem(0);

            CHECK(mem.size() == 0);
            CHECK(mem.data() == nullptr);
        }
        SUBCASE("Zero size bf16") {
            DeviceMem<bf16> mem(0);

            CHECK(mem.size() == 0);
            CHECK(mem.data() == nullptr);
        }
        SUBCASE("Zero size float") {
            DeviceMem<float> mem(0);

            CHECK(mem.size() == 0);
            CHECK(mem.data() == nullptr);
        }
    }
}

// Test the copy_from_host function
TEST_CASE("Testing the DeviceMem copy_from_host") {
    queue q = queue();

    SUBCASE("copy_from_host float") {
        DeviceMem<float> dm(10, q);
        dm.initialize_constant(1.0f, q);
        std::vector<float> data(10, 1);
        dm.copy_from_host(data, 10, q);

        for (int i = 0; i < data.size(); i++) {
            CHECK(data[i] == 1);
        }
    }
}

// Test the copy_to_host function
TEST_CASE("Testing the DeviceMem copy_to_host") {
    queue q = queue();

    DeviceMem<float> dm(10, q);
    std::vector<float> data(10, 1.0f);
    dm.copy_from_host(data, 10, q);
    std::vector<float> data_copy(10, 0);
    dm.copy_to_host(data_copy, 10, q);
    CHECK(data == data_copy);
}

// Test the initialize_normal function
TEST_CASE("Testing the DeviceMem initialisation functions") {
    queue q = queue();

    int input_width = 10;
    int output_width = 10;
    int net_width = 10;
    int hidden_matrices = 1;
    DeviceMem<float> dm(net_width * input_width + (net_width * net_width) * hidden_matrices + net_width * output_width,
                        q);
    DeviceMem<float> transposed(
        net_width * input_width + (net_width * net_width) * hidden_matrices + net_width * output_width, q);
    SUBCASE("Test initialize_normal") {
        CHECK_NOTHROW(dm.initialize_normal(1.0, transposed, input_width, net_width, output_width, hidden_matrices, q));
    }
    SUBCASE("Test initialize_uniform") {
        CHECK_NOTHROW(dm.initialize_uniform(1.0, transposed, input_width, net_width, output_width, hidden_matrices, q));
    }
    SUBCASE("Test make_transposed") {
        CHECK_NOTHROW(dm.make_transposed(transposed, input_width, net_width, output_width, hidden_matrices, q));
    }
    SUBCASE("Test initialize_xavier_unif") {
        CHECK_NOTHROW(dm.initialize_xavier_unif(transposed, input_width, net_width, output_width, hidden_matrices, q));
    }
    SUBCASE("Test initialize_xavier_normal") {
        CHECK_NOTHROW(
            dm.initialize_xavier_normal(transposed, input_width, net_width, output_width, hidden_matrices, q));
    }
    SUBCASE("Test initialize_constant") { CHECK_NOTHROW(dm.initialize_constant(1.0, transposed, q)); }
    SUBCASE("Test initialize_arange") { CHECK_NOTHROW(dm.initialize_arange(q)); }
    SUBCASE("Test initialize_he_normal") {
        CHECK_NOTHROW(dm.initialize_he_normal(transposed, input_width, net_width, output_width, hidden_matrices, q));
    }
}

// Test the allocate_memory function
TEST_CASE("Testing the DeviceMem allocate_memory") {
    DeviceMem<float> dm;
    CHECK_NOTHROW(dm.allocate_memory(10 * sizeof(float)));
}

// Test the resize function
TEST_CASE("Testing the DeviceMem resize") {
    DeviceMem<float> dm;
    dm.resize(10);
    CHECK(dm.get_num_elements() == 10);
}

// Test the enlarge function
TEST_CASE("Testing the DeviceMem enlarge") {
    DeviceMem<float> dm;
    dm.enlarge(10);
    CHECK(dm.get_num_elements() == 10);
}

// Test the memset function
TEST_CASE("Testing the DeviceMem memset") {
    queue q = queue();

    DeviceMem<float> dm(10, q);
    CHECK_NOTHROW(dm.memset(0));
}

// TEST_CASE("Testing the DeviceMem make_transposed") {
//         queue q = queue();

//     DeviceMem<int> dm(4, q); // Create a 2x2 matrix
//     DeviceMem<int> transposed(4, q);

//     // Fill the matrix with known values
//     std::vector<int> data = {1, 2, 3, 4};
//     dm.copy_from_host(data, 4, q);

//     // Transpose the matrix
//     dm.make_transposed(transposed, 2, 2, 2, 0, q);

//     // Copy the result back to the host
//     std::vector<int> result(4, 0);
//     transposed.copy_to_host(result, 4, q);

//     // Check that the result is the transpose of the original matrix
//     std::vector<int> expected = {1, 3, 2, 4};
//     CHECK(result == expected);
// }

TEST_CASE("Zero Padding Input Test") {
    int input_width_padded = 4;
    int output_width_padded = 5;
    queue q = queue();
    // Create a DeviceMatrix with DeviceMem
    DeviceMem<float> dm(input_width_padded * output_width_padded, q);
    dm.initialize_constant(1.0f, q);
    // dm.initialize_arange(q);

    // DeviceMatrix<float, MatrixLayout::RowMajor> DeviceMatrix(dm.data(), output_width_padded, input_width_padded);

    std::vector<float> result(dm.size());
    q.memcpy(result.data(), dm.data(), sizeof(float) * result.size()).wait();
    // DeviceMatrix.print(0);
    // DeviceMatrix.print(1);
    // Zero pad the input
    // for (int i = 0; i < out.size(); i++) {
    //     std::cout << i << ": " << out[i] << std::endl;
    // }
    int input_width = 2;
    dm.zero_pad_input(input_width, input_width_padded, output_width_padded, q);
    q.memcpy(result.data(), dm.data(), sizeof(float) * result.size()).wait();
    // for (int i = 0; i < out.size(); i++) {
    //     std::cout << i << ": " << out[i] << std::endl;
    // }

    // Check the zero padding of the input
    CHECK_EQ(result[0], 1);
    CHECK_EQ(result[1], 1);
    CHECK_EQ(result[2], 1);
    CHECK_EQ(result[3], 1);
    CHECK_EQ(result[4], 1);
    CHECK_EQ(result[5], 1);
    CHECK_EQ(result[6], 1);
    CHECK_EQ(result[7], 0);
    CHECK_EQ(result[8], 1);
    CHECK_EQ(result[9], 0);
    CHECK_EQ(result[10], 0);
    CHECK_EQ(result[11], 0);
    CHECK_EQ(result[12], 0);
    CHECK_EQ(result[13], 0);
    CHECK_EQ(result[14], 0);
    CHECK_EQ(result[15], 0);
    CHECK_EQ(result[16], 0);
    CHECK_EQ(result[17], 0);
    CHECK_EQ(result[18], 0);
    CHECK_EQ(result[19], 0);
}

TEST_CASE("Zero pad output") {
    // Zero pad the output
    int input_width = 3;
    int n_hidden_matrices = 1;

    int output_width = 2;
    int net_width = 5;
    int output_width_padded = 4;
    queue q = queue();
    // Create a DeviceMatrix with DeviceMem
    DeviceMem<float> dm(
        net_width * input_width + (net_width * net_width) * n_hidden_matrices + net_width * output_width_padded, q);
    dm.initialize_constant(1.0f, q);
    // dm.initialize_arange(q);
    std::vector<float> result(output_width_padded * net_width);

    dm.zero_pad_output(output_width, input_width, net_width, output_width_padded, n_hidden_matrices, q);
    q.memcpy(result.data(), dm.data() + net_width * input_width + (net_width * net_width) * n_hidden_matrices,
             sizeof(float) * result.size())
        .wait();
    // for (int i = 0; i < result.size(); i++) {
    //     std::cout << i << ": " << result[i] << std::endl;
    // }
    // Check the zero padding of the output
    CHECK_EQ(result[0], 1);
    CHECK_EQ(result[1], 1);
    CHECK_EQ(result[2], 1);
    CHECK_EQ(result[3], 1);
    CHECK_EQ(result[4], 0);
    CHECK_EQ(result[5], 0);
    CHECK_EQ(result[6], 0);
    CHECK_EQ(result[7], 0);
    CHECK_EQ(result[8], 0);
    CHECK_EQ(result[9], 0);
    CHECK_EQ(result[10], 1);
    CHECK_EQ(result[11], 1);
    CHECK_EQ(result[12], 1);
    CHECK_EQ(result[13], 1);
    CHECK_EQ(result[14], 0);
    CHECK_EQ(result[15], 0);
    CHECK_EQ(result[16], 0);
    CHECK_EQ(result[17], 0);
    CHECK_EQ(result[18], 0);
    CHECK_EQ(result[19], 0);
}