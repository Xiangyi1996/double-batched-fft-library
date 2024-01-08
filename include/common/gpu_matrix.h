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

/** @file   gpu_matrix.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Matrix whose data resides in GPU (CUDA) memory
 */

#pragma once

#include <DeviceMem.h>
#include <common.h>
#include <common_host.h>
#include <stdint.h>

#include <dpct/dpct.hpp>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

template <typename T, MatrixLayout _layout = MatrixLayout::RowMajor> class GPUMatrix {
  public:
    // Owning its memory as an allocation from a stream's memory arena
    GPUMatrix(const uint32_t m, const uint32_t n, sycl::queue &stream)
        : m_rows(m), m_cols(n), m_q(stream), m_data(sycl::malloc_device<T>(m * n, stream)) {}

    GPUMatrix() = delete;

    GPUMatrix(GPUMatrix<T, _layout> &&other) noexcept { *this = std::move(other); }

    virtual ~GPUMatrix() { sycl::free(m_data, m_q); }

    bool operator==(const GPUMatrix<T, _layout> &rhs) const {
        if (this->rows() != rhs.rows() || this->cols() != rhs.cols()) return false;
        if (this->layout() != rhs.layout()) return false;

        // Check actual data
        std::vector<T> data1 = this->copy_to_host();
        std::vector<T> data2 = rhs.copy_to_host();
        for (size_t i = 0; i < data1.size(); ++i) {
            if (data1[i] != data2[i]) {
                return false;
            }
        }

        return true;
    }

    sycl::event fill(const T val) { return m_q.fill(data(), val, n_elements()); }

    sycl::event copy_to_host(std::vector<T> &out) const {
        if (out.size() < n_elements()) throw std::invalid_argument("Target too small.");
        return m_q.memcpy(out.data(), data(), n_bytes());
    }

    std::vector<T> copy_to_host() const {
        std::vector<T> v(n_elements());
        copy_to_host(v).wait();
        return v;
    }

    sycl::event copy_from_host(const std::vector<T> &vec) {
        if (vec.size() != n_elements()) throw std::invalid_argument("Vector not same size as matrix.");
        return m_q.memcpy(data(), vec.data(), n_bytes());
    }

    MatrixView<T> view() const {
        return {data(), layout() == MatrixLayout::ColumnMajor ? 1u : n(),
                layout() == MatrixLayout::ColumnMajor ? m() : 1u};
    }

    T *data() { return m_data; }
    T const *const data() const { return m_data; }

    uint32_t rows() const { return m_rows; }
    uint32_t m() const { return rows(); }

    uint32_t cols() const { return m_cols; }
    uint32_t n() const { return cols(); }

    uint32_t n_elements() const { return rows() * cols(); }
    size_t n_bytes() const { return n_elements() * sizeof(T); }

    constexpr MatrixLayout layout() const { return _layout; }
    constexpr MatrixLayout transposed_layout() const {
        return _layout == MatrixLayout::RowMajor ? MatrixLayout::ColumnMajor : MatrixLayout::RowMajor;
    }

    // Function to print the matrix values
    void print(int is_packed = 0) const {
        std::vector<T> data(this->cols() * this->rows());
        m_q.memcpy(data.data(), this->data(), data.size() * sizeof(T)).wait();

        std::cout << "Matrix (" << this->rows() << "x" << this->cols() << "):" << std::endl;
        for (uint32_t i = 0; i < this->rows(); ++i) {
            std::cout << "[ ";
            for (uint32_t j = 0; j < this->cols(); ++j) {
                int idx;
                if (_layout == MatrixLayout::ColumnMajor) {
                    idx = j * this->stride() + i;
                } else {
                    idx = i * this->stride() + j;
                }
                if (is_packed) {
                    idx = toPackedLayoutCoord(idx, this->rows(), this->cols());
                }
                std::cout << std::setw(8) << std::setprecision(4) << data[idx];
                if (j < this->cols() - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << " ]" << std::endl;
        }
    }

  private:
    const uint32_t m_rows, m_cols;
    sycl::queue &m_q;
    T *m_data;
};
