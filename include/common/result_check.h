/**
 * @file result_check.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Basic comparison, load, store functionalities to check correctness of results.
 * TODO: move the load/store functionalities in a different file and put everything in a namespace.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

template <typename T> double GetInfNorm(const std::vector<T> &v) {
    double norm = 0.0;
    for (auto val : v) {
        norm = std::max(norm, std::abs((double)val));
    }

    return norm;
}

template <typename Tl, typename Tr>
std::vector<double> GetAbsDiff(const std::vector<Tl> &lhs, const std::vector<Tr> &rhs) {

    if (lhs.size() != rhs.size()) {
        throw std::invalid_argument("Size mismatch: lhs size = " + std::to_string(lhs.size()) +
                                    ", rhs size = " + std::to_string(rhs.size()));
    }
    std::vector<double> ret(lhs.size(), 0.0);

    for (size_t iter = 0; iter < lhs.size(); iter++) {
        if (!std::isfinite(lhs[iter]) || !std::isfinite(rhs[iter])) throw std::invalid_argument("Infinite numbers");
        ret[iter] = std::abs(lhs[iter] - rhs[iter]);
    }

    return ret;
}

template <typename Tl, typename Tr> std::vector<double> GetAbsDiff(const std::vector<Tl> &lhs, const Tr rhs) {
    std::vector<double> ret(lhs.size(), 0.0);

    for (size_t iter = 0; iter < lhs.size(); iter++) {
        if (!std::isfinite(lhs[iter]) || !std::isfinite(rhs)) throw std::invalid_argument("Infinite numbers");
        ret[iter] = std::abs(lhs[iter] - rhs);
    }

    return ret;
}

template <typename Tval, typename Ttarget>
bool isVectorWithinTolerance(const std::vector<Tval> &value, const Ttarget target, const double tolerance) {

    bool is_same = true;
    double max_diff = 0.0;
    const double inf_diff = GetInfNorm(GetAbsDiff(value, target));
    const double inf_val = GetInfNorm(value);
    if ((double)target == 0.0)
        max_diff = inf_diff;
    else
        max_diff = inf_diff / std::max(std::abs((double)target), inf_val);

    if (max_diff > tolerance) is_same = false;
    if (!is_same) {
        std::cout << "Values are within tolerance = " << std::boolalpha << is_same << std::noboolalpha
                  << ". Max diff = " << max_diff << std::endl;
    }

    return is_same;
}

template <typename Tval, typename Ttarget>
bool areVectorsWithinTolerance(const std::vector<Tval> &value, const std::vector<Ttarget> &target,
                               const double tolerance) {

    bool is_same = true;
    double max_diff = 0.0;
    const double inf_diff = GetInfNorm(GetAbsDiff(value, target));
    const double inf_val = GetInfNorm(value);
    const double inf_tar = GetInfNorm(target);
    if ((double)inf_tar == 0.0)
        max_diff = inf_diff;
    else
        max_diff = inf_diff / std::max(inf_tar, inf_val);

    if (max_diff > tolerance) is_same = false;

    if (!is_same) {
        std::cout << "Values are within tolerance = " << std::boolalpha << is_same << std::noboolalpha
                  << ". Max diff = " << max_diff << std::endl;
    }
    return is_same;
}
