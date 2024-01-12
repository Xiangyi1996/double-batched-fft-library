/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TOR
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** @file   grid_interface.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Base class / abstract interface for controlling and querying
            general aspects of all generic grid encodings.
 */

#pragma once

#include <common.h>
#include <encoding.h>

#include <cstdint>
#include <sycl/sycl.hpp>

namespace tinydpcppnn {
namespace encodings {
namespace grid {

static constexpr uint32_t MAX_N_LEVELS = 128;
struct GridOffsetTable {
    uint32_t data[MAX_N_LEVELS + 1] = {};
    uint32_t size = 0;
};

template <typename T> class GridEncoding : public Encoding<T> {
  public:
    virtual uint32_t n_pos_dims() const = 0;
    virtual uint32_t n_features_per_level() const = 0;

    virtual size_t level_n_params(uint32_t level) const = 0;
    virtual size_t level_params_offset(uint32_t level) const = 0;

    size_t n_params() const { return m_n_params; }

    virtual const GridOffsetTable &grid_offset_table() const = 0;

    float max_level() const { return m_max_level; }

    void set_max_level(float value) { m_max_level = value; }

  protected:
    // Disables lookups of finer levels than this.
    // The default value of 1000 effectively disables the feature
    float m_max_level = 1000.f;
    uint32_t m_n_params;
};

} // namespace grid
} // namespace encodings
} // namespace tinydpcppnn