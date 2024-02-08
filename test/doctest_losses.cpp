/**
 * @file doctest_losses.cpp
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Tests for the losses.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

// This file includes tests for all of our loss calculations

#include "doctest/doctest.h"

#include "l1.h"
#include "l2.h"

TEST_CASE("Testing Loss Functionality") {

    // Initialize a sycl::queue object for running kernels
    sycl::queue q;

    // Define initial values for loss testing
    const float loss_scale = 1.0f;
    const int n_elements = 100;

    // Generate some mock data for testing
    std::vector<float> predictions(n_elements, 5.0), targets(n_elements, 10.0);
    std::vector<float> values(n_elements), gradients(n_elements);

    // Initialize DeviceMem for inputs and outputs
    DeviceMem<float> dev_predictions(predictions.data(), n_elements);
    DeviceMem<float> dev_targets(targets.data(), n_elements);
    DeviceMem<float> dev_values(values.data(), n_elements);
    DeviceMem<float> dev_gradients(gradients.data(), n_elements);

    // Create an instance of L2Loss
    L2Loss<float> l2;

    SUBCASE("Testing Evaluate function") {
        // Expected values and gradients calculations
        float expected_value = std::pow(5.0 - 10.0, 2) / n_elements;
        float expected_gradient = 2 * (5.0 - 10.0) / n_elements;

        // Evaluate the loss
        l2.evaluate(q, loss_scale, dev_predictions, dev_targets, dev_values, dev_gradients);

        // Read back data from DeviceMem
        dev_values.readData(q, values.data());
        dev_gradients.readData(q, gradients.data());

        for (int i = 0; i < n_elements; i++) {
            CHECK(values[i] == doctest::Approx(expected_value));
            CHECK(gradients[i] == doctest::Approx(expected_gradient));
        }
    }

    SUBCASE("Testing SanityCheck function") {
        // Create a mismatch in sizes to test asserts
        std::vector<float> incorrect_sizes(n_elements + 1);
        DeviceMem<float> incorrect_values(incorrect_sizes.data(), n_elements + 1);

        // Now, evaluate function must terminate the program due to failed assertion
        CHECK_THROWS_WITH(l2.evaluate(q, loss_scale, dev_predictions, dev_targets, incorrect_values, dev_gradients),
                          "Assert condition failed: values.size() == n_elements");
    }
}