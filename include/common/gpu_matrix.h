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

template <typename T> class GPUMatrixDynamic;

template <typename T, MatrixLayout _layout> class GPUMatrix;

class GPUMatrixBase {
  public:
    virtual ~GPUMatrixBase() {}

    virtual size_t n_bytes() const = 0;
    virtual void set_data_unsafe(void *data) = 0;
};

template <typename T> class GPUMatrixDynamic : public GPUMatrixBase {
  public:
    using Type = T;

    // Owning its memory as a DeviceMem<T>
    GPUMatrixDynamic(uint32_t m, uint32_t n, MatrixLayout layout = MatrixLayout::ColumnMajor)
        : m_rows{m}, m_cols{n}, m_layout{layout} {
        sycl::queue stream;
        m_malloc_allocation = std::make_shared<DeviceMem<uint8_t>>(m * n * sizeof(T));
        m_data = (T *)m_malloc_allocation->data();
        set_stride_contiguous();
    }

    GPUMatrixDynamic(uint32_t m, uint32_t n, sycl::queue &stream, MatrixLayout layout = MatrixLayout::ColumnMajor)
        : m_rows{m}, m_cols{n}, m_layout{layout} {
        m_arena_allocation = nullptr;
        m_malloc_allocation = std::make_shared<DeviceMem<uint8_t>>(m * n * sizeof(T), stream);
        m_data = (T *)m_malloc_allocation->data();
        set_stride_contiguous();
    }

    explicit GPUMatrixDynamic(T *data, uint32_t m, uint32_t n, MatrixLayout layout = MatrixLayout::ColumnMajor,
                              uint32_t stride = 0)
        : m_data{data}, m_layout{layout} {
        set(data, m, n, stride);
    }

    GPUMatrixDynamic() : GPUMatrixDynamic{nullptr, 0, 0} {}

    GPUMatrixDynamic<T> &operator=(GPUMatrixDynamic<T> &&other) {
        std::swap(m_data, other.m_data);
        std::swap(m_rows, other.m_rows);
        std::swap(m_cols, other.m_cols);
        std::swap(m_stride, other.m_stride);
        std::swap(m_layout, other.m_layout);
        std::swap(m_malloc_allocation, other.m_malloc_allocation);
        std::swap(m_arena_allocation, other.m_arena_allocation);
        return *this;
    }

    GPUMatrixDynamic(GPUMatrixDynamic<T> &&other) { *this = std::move(other); }

    GPUMatrixDynamic(const GPUMatrixDynamic<T> &other){}; // TODO delete;
    GPUMatrixDynamic<T> &operator=(const GPUMatrixDynamic<T> &other) = delete;

    virtual ~GPUMatrixDynamic() {}

    void set_data_unsafe(void *data) override { m_data = (T *)data; }
    void set_size_unsafe(uint32_t rows, uint32_t cols, uint32_t stride = 0) {
        m_rows = rows;
        m_cols = cols;

        if (stride == 0) {
            set_stride_contiguous();
        } else {
            m_stride = stride;
        }
    }

    void set(T *data, uint32_t rows, uint32_t cols, uint32_t stride = 0) {
        set_data_unsafe(data);
        set_size_unsafe(rows, cols, stride);
    }

    void resize(uint32_t rows, uint32_t cols) {
        if (m_malloc_allocation) {
            m_malloc_allocation.reset();
            m_malloc_allocation = std::make_shared<DeviceMem<uint8_t>>((int)(rows * cols * sizeof(T)));
        } else {
            throw std::runtime_error{"GPUMatrix::resize is not permitted when the underlying memory is "
                                     "not owned. Use GPUMatrix::set instead."};
        }

        set_size_unsafe(rows, cols);
    }

    uint32_t stride_contiguous() const { return m_layout == MatrixLayout::ColumnMajor ? m() : n(); }

    bool is_contiguous() const { return m_stride == stride_contiguous(); }

    void set_stride_contiguous() { m_stride = stride_contiguous(); }

    GPUMatrixDynamic<T> slice(uint32_t offset_rows, uint32_t new_rows, uint32_t offset_cols, uint32_t new_cols) const {
        return GPUMatrixDynamic<T>{
            data() + (layout() == MatrixLayout::ColumnMajor ? (offset_rows + offset_cols * stride())
                                                            : (offset_cols + offset_rows * stride())),
            new_rows,
            new_cols,
            layout(),
            stride(),
            m_malloc_allocation,
            m_arena_allocation,
        };
    }

    GPUMatrixDynamic<T> slice_rows(uint32_t offset, uint32_t size) const { return slice(offset, size, 0, cols()); }

    GPUMatrixDynamic<T> slice_cols(uint32_t offset, uint32_t size) const { return slice(0, rows(), offset, size); }

    GPUMatrixDynamic<T> alias() const { return slice(0, rows(), 0, cols()); }

    MatrixView<T> view() const {
        return {data(), layout() == MatrixLayout::ColumnMajor ? 1u : stride(),
                layout() == MatrixLayout::ColumnMajor ? stride() : 1u};
    }

    uint32_t rows() const { return m_rows; }
    uint32_t fan_out() const { return m_rows; }
    uint32_t m() const { return m_rows; }

    uint32_t cols() const { return m_cols; }
    uint32_t fan_in() const { return m_cols; }
    uint32_t n() const { return m_cols; }

    uint32_t stride() const { return m_stride; }
    PitchedPtr<T> pitched_ptr() { return {data(), stride()}; }
    PitchedPtr<const T> pitched_ptr() const { return {data(), stride()}; }

    uint32_t n_elements() const { return m_rows * m_cols; }
    size_t n_bytes() const override { return n_elements() * sizeof(T); }

    MatrixLayout layout() const { return m_layout; }
    MatrixLayout transposed_layout() const {
        return m_layout == MatrixLayout::RowMajor ? MatrixLayout::ColumnMajor : MatrixLayout::RowMajor;
    }

    T *data() const { return m_data; }

    void memset(int value) {
        CHECK_THROW(data());
        CHECK_THROW(is_contiguous());

        dpct::get_default_queue().memset(data(), value, n_bytes()).wait();
    }

    void memset_async(sycl::queue &stream, int value) {
        CHECK_THROW(data());
        CHECK_THROW(is_contiguous());
        stream.memset(data(), value, n_bytes());
    }

    std::vector<T> to_cpu_vector() {
        CHECK_THROW(data());
        CHECK_THROW(is_contiguous());
        std::vector<T> v(n_elements());
        dpct::get_default_queue().memcpy(v.data(), data(), n_bytes()).wait();
        return v;
    }

    void initialize_constant(float val) {
        CHECK_THROW(data());
        CHECK_THROW(is_contiguous());

        std::vector<T> new_data(n_elements(), (T)val);

        dpct::get_default_queue().memcpy(data(), new_data.data(), n_bytes()).wait();
    }

    GPUMatrixDynamic<T> transposed() const {
        return GPUMatrixDynamic<T>(data(), n(), m(), transposed_layout(), stride(), m_malloc_allocation,
                                   m_arena_allocation);
    }

    GPUMatrix<T, MatrixLayout::RowMajor> rm() const {
        CHECK_THROW(m_layout == MatrixLayout::RowMajor);
        return GPUMatrix<T, MatrixLayout::RowMajor>(data(), m(), n(), stride(), m_malloc_allocation,
                                                    m_arena_allocation);
    }

    GPUMatrix<T, MatrixLayout::ColumnMajor> cm() const {
        CHECK_THROW(m_layout == MatrixLayout::ColumnMajor);
        return GPUMatrix<T, MatrixLayout::ColumnMajor>(data(), m(), n(), stride(), m_malloc_allocation,
                                                       m_arena_allocation);
    }

  private:
    T *m_data;
    uint32_t m_rows, m_cols, m_stride;
    MatrixLayout m_layout;

    // References to corresponding memory allocations. These ensure that
    // m_data does not accidentally become dangling.
    std::shared_ptr<DeviceMem<uint8_t>> m_malloc_allocation;
    std::shared_ptr<void> m_arena_allocation;
};

template <typename T, MatrixLayout _layout = MatrixLayout::ColumnMajor> class GPUMatrix : public GPUMatrixDynamic<T> {
  public:
    static const MatrixLayout static_layout = _layout;
    static const MatrixLayout static_transposed_layout =
        _layout == MatrixLayout::RowMajor ? MatrixLayout::ColumnMajor : MatrixLayout::RowMajor;

    // Owning its memory as a DeviceMem<T>
    GPUMatrix(uint32_t m, uint32_t n) : GPUMatrixDynamic<T>{m, n, static_layout} {}

    // Owning its memory as an allocation from a stream's memory arena
    GPUMatrix(uint32_t m, uint32_t n, sycl::queue &stream) : GPUMatrixDynamic<T>{m, n, stream, static_layout} {}

    GPUMatrix(T *data, uint32_t m, uint32_t n, uint32_t stride = 0)
        : GPUMatrixDynamic<T>{data, m, n, static_layout, stride} {}

    GPUMatrix() : GPUMatrix{nullptr, 0, 0} {}

    GPUMatrix<T, static_layout> &operator=(GPUMatrixDynamic<T> &&other) {
        *((GPUMatrixDynamic<T> *)this) = std::move(other);
        if (static_layout != this->layout()) {
            throw std::runtime_error{"GPUMatrix must be constructed from a GPUMatrixDynamic with matching "
                                     "layout."};
        }
        return *this;
    }

    GPUMatrix(GPUMatrixDynamic<T> &&other) noexcept { *this = std::move(other); }

    GPUMatrix<T, static_layout> &operator=(GPUMatrix<T, static_layout> &&other) noexcept {
        *((GPUMatrixDynamic<T> *)this) = std::move(other);
        return *this;
    }

    GPUMatrix(GPUMatrix<T, static_layout> &&other) noexcept { *this = std::move(other); }

    GPUMatrix(const GPUMatrixDynamic<T> &other) = delete;
    GPUMatrix<T> &operator=(const GPUMatrixDynamic<T> &other) = delete;

    virtual ~GPUMatrix() {}

    GPUMatrix<T, static_layout> slice(uint32_t offset_rows, uint32_t new_rows, uint32_t offset_cols,
                                      uint32_t new_cols) const {
        return ((GPUMatrixDynamic<T> *)this)->slice(offset_rows, new_rows, offset_cols, new_cols);
    }

    GPUMatrix<T, static_layout> slice_rows(uint32_t offset, uint32_t size) const {
        return ((GPUMatrixDynamic<T> *)this)->slice_rows(offset, size);
    }

    GPUMatrix<T, static_layout> slice_cols(uint32_t offset, uint32_t size) const {
        return ((GPUMatrixDynamic<T> *)this)->slice_cols(offset, size);
    }

    GPUMatrix<T, static_layout> alias() const { return ((GPUMatrixDynamic<T> *)this)->alias(); }

    GPUMatrix<T, static_transposed_layout> transposed() const { return ((GPUMatrixDynamic<T> *)this)->transposed(); }

    // Function to print the matrix values
    void print(int is_packed = 0) const {
        std::vector<T> data(this->cols() * this->rows());
        dpct::get_default_queue().memcpy(data.data(), this->data(), data.size() * sizeof(T)).wait();

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
};
