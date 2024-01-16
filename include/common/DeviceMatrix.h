#pragma once

#include "common.h"

#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

enum class MatrixLayout { RowMajor = 0, ColumnMajor = 1 };

// view class which can even be used on the device,
// does not own any more
// is always associated with a DeviceMatrix.
// The associated DeviceMatrix owns the memory
// if the associated DeviceMatrix is deleted, behaviour of DeviceMatrixView is undefined
template <typename T> class DeviceMatrixView {
  public:
    DeviceMatrixView() = delete;
    DeviceMatrixView(const size_t m, const size_t n, const size_t stride_col, T *ptr)
        : m_(m), n_(n), stride_col_(stride_col), ptr_(ptr) {}

    // T &operator()(const int i, const int j) { return ptr_[j + i * stride_col_]; }
    T &operator()(const int i, const int j) const { return ptr_[j + i * stride_col_]; }

    T *const GetPointer(const size_t i, const size_t j) const { return ptr_ + j + i * stride_col_; }
    T *const GetPointer() const { return ptr_; }

    DeviceMatrixView<T> GetSubMatrix(const size_t m, const size_t n, const size_t offset_m,
                                     const size_t offset_n) const {
        return DeviceMatrixView<T>(m, n, stride_col_, ptr_ + offset_n + offset_m * stride_col_);
    }

    size_t m() const { return m_; }
    size_t n() const { return n_; }

  private:
    const size_t m_;
    const size_t n_;
    const size_t stride_col_;
    T *const ptr_;
};

template <typename T, MatrixLayout _layout = MatrixLayout::RowMajor> class DeviceMatrix {
  public:
    // Owning its memory as an allocation from a stream's memory arena
    DeviceMatrix(const size_t m, const size_t n, sycl::queue &stream)
        : m_rows(m), m_cols(n), m_q(stream), m_data(sycl::malloc_device<T>(m * n, stream)) {
        static_assert(_layout != MatrixLayout::ColumnMajor);
    }
    DeviceMatrix() = delete;

    DeviceMatrix(const DeviceMatrix<T, _layout> &other) : m_rows(other.m_rows), m_cols(other.m_cols), m_q(other.m_q) {
        m_data = sycl::malloc_device<T>(n_elements(), m_q);
        m_q.memcpy(m_data, other.m_data, n_bytes()).wait();
        static_assert(_layout != MatrixLayout::ColumnMajor);
    }
    DeviceMatrix(DeviceMatrix<T, _layout> &&other)
        : m_rows(other.m_rows), m_cols(other.m_cols), m_q(other.m_q), m_data(other.m_data) {
        other.m_data = nullptr;
        static_assert(_layout != MatrixLayout::ColumnMajor);
    }

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

    template <typename Ts> void copy_from_device(Ts const *const src) {
        T *const ptr = m_data;
        m_q.parallel_for(size(), [=](auto idx) { ptr[idx] = static_cast<T>(src[idx]); }).wait();
    }

    T *data() { return m_data; }
    T const *const data() const { return m_data; }

    size_t rows() const { return m_rows; }
    size_t m() const { return rows(); }

    size_t cols() const { return m_cols; }
    size_t n() const { return cols(); }

    size_t n_elements() const { return rows() * cols(); }
    size_t size() const { return n_elements(); }
    size_t n_bytes() const { return n_elements() * sizeof(T); }

    constexpr MatrixLayout layout() const { return _layout; }

    DeviceMatrixView<T> GetView() { return GetView(m_rows, m_cols, 0, 0); }
    const DeviceMatrixView<T> GetView() const { return GetView(m_rows, m_cols, 0, 0); }

    DeviceMatrixView<T> GetView(const size_t m, const size_t n, const size_t offset_m, const size_t offset_n) {
        if (offset_m + m > m_rows) throw std::invalid_argument("Potential OOB access.");
        if (offset_n + n > m_cols) throw std::invalid_argument("Potential OOB access.");

        return DeviceMatrixView<T>(m, n, m_cols, m_data + offset_n + offset_m * m_cols);
    }

    const DeviceMatrixView<T> GetView(const size_t m, const size_t n, const size_t offset_m,
                                      const size_t offset_n) const {
        if (offset_m + m > m_rows) throw std::invalid_argument("Potential OOB access.");
        if (offset_n + n > m_cols) throw std::invalid_argument("Potential OOB access.");

        return DeviceMatrixView<T>(m, n, m_cols, m_data + offset_n + offset_m * m_cols);
    }

    // Function to print the matrix values
    void print(int is_packed = 0) const {
        std::vector<T> data = copy_to_host();

        std::cout << "Matrix (" << this->rows() << "x" << this->cols() << "):" << std::endl;
        for (size_t i = 0; i < this->rows(); ++i) {
            std::cout << "[ ";
            for (size_t j = 0; j < this->cols(); ++j) {
                size_t idx;
                if (_layout == MatrixLayout::ColumnMajor) {
                    idx = j * rows() + i;
                } else {
                    idx = i * cols() + j;
                }
                if (is_packed) {
                    idx = toPackedLayoutCoord(idx, this->rows(), this->cols());
                }
                std::cout << (double)data[idx];
                if (j < this->cols() - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << " ]" << std::endl;
        }
    }

  private:
    const size_t m_rows, m_cols;
    sycl::queue &m_q;
    T *m_data;
};

template <typename T> class DeviceMatricesView {
  public:
    DeviceMatricesView() = delete;
    DeviceMatricesView(const uint32_t n_matrices, const size_t input_m, const size_t input_n, const size_t middle_m,
                       const size_t middle_n, const size_t output_m, const size_t output_n, T *ptr)
        : n_matrices_(n_matrices), input_m_(input_m), input_n_(input_n), middle_m_(middle_m), middle_n_(middle_n),
          output_m_(output_m), output_n_(output_n), ptr_(ptr) {}

    T *const GetMatrixPointer(const uint32_t matrix) const {
        if (matrix == 0)
            return ptr_;
        else if (matrix < n_matrices_)
            return ptr_ + input_m_ * input_n_ + (matrix - 1) * middle_m_ * middle_n_;
        return nullptr;
    }

  private:
    const uint32_t n_matrices_;
    const size_t input_m_;
    const size_t input_n_;
    const size_t middle_m_;
    const size_t middle_n_;
    const size_t output_m_;
    const size_t output_n_;
    T *const ptr_;
};

/// Class which represents a vector of matrices.
/// This is a host class which does not work on the GPU
/// But it provides interfaces to generate pointers to the MatrixView classes which
/// are meant to be used on the GPU
template <typename T> class DeviceMatrices {

  public:
    DeviceMatrices() = delete;
    DeviceMatrices(const uint32_t n_matrices, const size_t input_m, const size_t input_n, const size_t middle_m,
                   const size_t middle_n, const size_t output_m, const size_t output_n, sycl::queue &q)
        : m_q(q), n_matrices_(n_matrices), input_m_(input_m), input_n_(input_n), middle_m_(middle_m),
          middle_n_(middle_n), output_m_(output_m), output_n_(output_n) {
        if (n_matrices_ < 2) throw std::invalid_argument("need to have at least 2 matrices.");
        matrices_ = sycl::malloc_device<T>(nelements(), m_q);
    }
    DeviceMatrices(const DeviceMatrices<T> &rhs) = delete;
    DeviceMatrices(DeviceMatrices<T> &&rhs) = delete;
    DeviceMatrices<T> &operator=(const DeviceMatrices<T> &rhs) = delete;
    DeviceMatrices<T> &operator=(DeviceMatrices<T> &&rhs) = delete;

    ~DeviceMatrices() { sycl::free(matrices_, m_q); }

    uint32_t GetNumberOfMatrices() const { return n_matrices_; }

    DeviceMatricesView<T> GetViews() const {
        return DeviceMatricesView<T>(n_matrices_, input_m_, input_n_, middle_m_, middle_n_, output_m_, output_n_,
                                     matrices_);
    }

    DeviceMatrixView<T> GetView(const uint32_t idx) const {
        size_t n = middle_n_;
        size_t m = middle_m_;
        if (idx == 0) {
            m = input_m_;
            n = input_n_;
        } else if (idx == n_matrices_ - 1) {
            m = output_m_;
            n = output_n_;
        }
        return DeviceMatrixView<T>(m, n, n, GetMatrixPtr(idx));
    }
    DeviceMatrixView<T> Front() const { return GetView(0); }
    DeviceMatrixView<T> Back() const { return GetView(n_matrices_ - 1); }

    void Transpose(DeviceMatrices<T> &ret) const {
        if (GetNumberOfMatrices() != ret.GetNumberOfMatrices())
            throw std::invalid_argument("Need to have same number of matrices for transpose");

        for (uint32_t iter = 0; iter < GetNumberOfMatrices(); iter++) {
            DeviceMatrices<T>::Transpose(GetView(iter), ret.GetView(iter), m_q);
        }
    }

    void PackedTranspose(DeviceMatrices<T> &ret) const {
        if (GetNumberOfMatrices() != ret.GetNumberOfMatrices())
            throw std::invalid_argument("Need to have same number of matrices for transpose");

        for (uint32_t iter = 0; iter < GetNumberOfMatrices(); iter++) {
            DeviceMatrices<T>::PackedTranspose(GetView(iter), ret.GetView(iter), m_q);
        }
    }

    sycl::event copy_from_host(const std::vector<T> &src) {
        return m_q.memcpy(matrices_, src.data(), nelements() * sizeof(T));
    }

    std::vector<T> copy_to_host() const {
        std::vector<T> ret(nelements());
        m_q.memcpy(ret.data(), matrices_, nelements() * sizeof(T)).wait();
        return ret;
    }

    sycl::event fill(const T val) { return m_q.fill(matrices_, val, nelements()); }

    size_t nelements() const {
        return input_m_ * input_n_ + output_m_ * output_n_ + (n_matrices_ - 2) * middle_m_ * middle_n_;
    }

  private:
    T *GetMatrixPtr(const uint32_t matrix) const {
        if (matrix == 0)
            return matrices_;
        else if (matrix < n_matrices_)
            return matrices_ + input_m_ * input_n_ + (matrix - 1) * middle_m_ * middle_n_;
        else
            throw std::invalid_argument("matrix does not exist");

        return nullptr;
    }

    static void Transpose(const DeviceMatrixView<T> &src, DeviceMatrixView<T> dest, sycl::queue &q) {
        if (src.n() != dest.m() || src.m() != dest.n()) throw std::invalid_argument("Cannot transpose.");
        // TODO: check that the underlying data is actually in the same context.

        T *const new_p = dest.GetPointer();
        T const *const old_p = src.GetPointer();
        const size_t loc_cols = src.n();
        const size_t loc_rows = src.m();
        q.parallel_for(loc_rows * loc_cols, [=](auto idx) {
            const size_t row = idx / loc_cols;
            const size_t col = idx % loc_cols;
            const size_t new_idx = row + col * loc_rows;
            new_p[new_idx] = old_p[idx];
        });
    }

    // TODO, make this work in dependence of the data type.
    // Transposes the data assuming it is in a packed format
    static void PackedTranspose(const DeviceMatrixView<T> &src, DeviceMatrixView<T> dest, sycl::queue &q) {
        if (src.n() != dest.m() || src.m() != dest.n()) throw std::invalid_argument("Cannot transpose.");
        // TODO: check that the underlying data is actually in the same context.

        T *const new_p = dest.GetPointer();
        T const *const old_p = src.GetPointer();
        const size_t loc_cols = src.n();
        const size_t loc_rows = src.m();
        q.parallel_for(loc_rows * loc_cols, [=](auto idx) {
            const size_t i = idx / loc_cols;
            const size_t j = idx % loc_cols;
            new_p[toPackedLayoutCoord(j * loc_rows + i, loc_cols, loc_rows)] =
                old_p[toPackedLayoutCoord(i * loc_cols + j, loc_rows, loc_cols)];
        });
    }

    sycl::queue &m_q;
    const uint32_t n_matrices_;
    const size_t input_m_;
    const size_t input_n_;
    const size_t middle_m_;
    const size_t middle_n_;
    const size_t output_m_;
    const size_t output_n_;
    T *matrices_;
};
