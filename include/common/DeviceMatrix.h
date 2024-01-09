#pragma once

#include "common.h"

#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

template <typename T, MatrixLayout _layout = MatrixLayout::RowMajor> class DeviceMatrix {
  public:
    // Owning its memory as an allocation from a stream's memory arena
    DeviceMatrix(const uint32_t m, const uint32_t n, sycl::queue &stream)
        : m_rows(m), m_cols(n), m_q(stream), m_data(sycl::malloc_device<T>(m * n, stream)) {}
    DeviceMatrix() = delete;

    DeviceMatrix(const DeviceMatrix<T, _layout> &other) : m_rows(other.m_rows), m_cols(other.m_cols), m_q(other.m_q) {
        m_data = sycl::malloc_device<T>(n_elements(), m_q);
        m_q.memcpy(m_data, other.m_data, n_bytes()).wait();
    }
    DeviceMatrix(DeviceMatrix<T, _layout> &&other) noexcept { *this = std::move(other); }

    virtual ~DeviceMatrix() { sycl::free(m_data, m_q); }

    DeviceMatrix<T, _layout> &operator=(const DeviceMatrix<T, _layout> &other) {
        if (m_rows != other.m_rows || m_cols != other.m_cols)
            throw std::invalid_argument("Cannot assign matrices of differing dimensions.");
        if (m_q != other.m_q) throw std::invalid_argument("Cannot assign matrices with differing queues.");

        m_q.memcpy(m_data, other.m_data, n_bytes()).wait();
        return *this;
    }

    bool operator==(const DeviceMatrix<T, _layout> &rhs) const {
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
