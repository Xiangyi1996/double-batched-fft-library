#ifndef TINYNN_COMMON_H
#define TINYNN_COMMON_H

#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

#include "vec.h"

static constexpr float PI = 3.14159265358979323846f;

#if defined(SYCL_LANGUAGE_VERSION)
#if TCNN_HALF_PRECISION
using network_precision_t = __half;
#else
using network_precision_t = float;
#endif

// Optionally: set the precision to `float` to disable tensor cores and debug
// potential
//             problems with mixed-precision training.
// using network_precision_t = float;
#endif

// When TCNN managed its model parameters, they are always aligned,
// which yields performance benefits in practice. However, parameters
// supplied by PyTorch are not necessarily aligned. The following
// variable controls whether TCNN must deal with unaligned data.
#if defined(TCNN_PARAMS_UNALIGNED)
static constexpr bool PARAMS_ALIGNED = false;
#else
static constexpr bool PARAMS_ALIGNED = true;
#endif

constexpr uint32_t N_THREADS_LINEAR = 128;

enum class Activation {
  ReLU,
  LeakyReLU,
  Exponential,
  Sine,
  Sigmoid,
  Squareplus,
  Softplus,
  Tanh,
  None,
};

enum class GridType {
  Hash,
  Dense,
  Tiled,
};

enum class HashType {
  Prime,
  CoherentPrime,
  ReversedPrime,
  Rng,
};

enum class InterpolationType {
  Nearest,
  Linear,
  Smoothstep,
};

enum class MatrixLayout {
  RowMajor = 0,
  SoA = 0,  // For data matrices TCNN's convention is RowMajor == SoA (struct of
            // arrays)
  ColumnMajor = 1,
  AoS = 1,
};

static constexpr MatrixLayout RM = MatrixLayout::RowMajor;
static constexpr MatrixLayout SoA = MatrixLayout::SoA;
static constexpr MatrixLayout CM = MatrixLayout::ColumnMajor;
static constexpr MatrixLayout AoS = MatrixLayout::AoS;

enum class ReductionType {
  Concatenation,
  Sum,
  Product,
};

struct Context {
  Context() = default;
  virtual ~Context() {}
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;
  Context(Context &&) = delete;
  Context &operator=(Context &&) = delete;
};

// from common.h
template <typename T>
T div_round_up(T val, T divisor) {
  return (val + divisor - 1) / divisor;
}

template <typename T>
T next_multiple(T val, T divisor) {
  return div_round_up(val, divisor) * divisor;
}

template <typename T>
constexpr uint32_t n_blocks_linear(T n_elements,
                                   uint32_t n_threads = N_THREADS_LINEAR) {
  return (uint32_t)div_round_up(n_elements, (T)n_threads);
}

template <typename T>
T previous_multiple(T val, T divisor) {
  return (val / divisor) * divisor;
}

template <typename T>
constexpr bool is_pot(T val) {
  return (val & (val - 1)) == 0;
}

inline constexpr uint32_t next_pot(uint32_t v) {
  --v;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return v + 1;
}

template <typename T>
struct PitchedPtr {
  PitchedPtr() : ptr{nullptr}, stride_in_bytes{sizeof(T)} {}
  PitchedPtr(T *ptr, size_t stride_in_elements, size_t offset = 0,
             size_t extra_stride_bytes = 0)
      : ptr{ptr + offset},
        stride_in_bytes{
            (uint32_t)(stride_in_elements * sizeof(T) + extra_stride_bytes)} {}

  template <typename U>
  explicit PitchedPtr(PitchedPtr<U> other)
      : ptr{(T *)other.ptr}, stride_in_bytes{other.stride_in_bytes} {}

  T *operator()(uint32_t y) const {
    return (T *)((const char *)ptr + y * stride_in_bytes);
  }

  void operator+=(uint32_t y) {
    ptr = (T *)((const char *)ptr + y * stride_in_bytes);
  }

  void operator-=(uint32_t y) {
    ptr = (T *)((const char *)ptr - y * stride_in_bytes);
  }

  explicit operator bool() const { return ptr; }

  T *ptr;
  uint32_t stride_in_bytes;
};

template <typename T>
struct MatrixView {
  MatrixView() : data{nullptr}, stride_i{0}, stride_j{0} {}
  MatrixView(T *data, uint32_t stride_i, uint32_t stride_j)
      : data{data}, stride_i{stride_i}, stride_j{stride_j} {}
  MatrixView(const MatrixView<std::remove_const_t<T>> &other)
      : data{other.data}, stride_i{other.stride_i}, stride_j{other.stride_j} {}

  T &operator()(uint32_t i, uint32_t j = 0) const {
    return data[i * stride_i + j * stride_j];
  }

  void advance(uint32_t m, uint32_t n) { data = &(*this)(m, n); }

  void advance_rows(uint32_t m) { advance(m, 0); }

  void advance_cols(uint32_t n) { advance(0, n); }

  template <uint32_t N>
  tnn::tvec<std::remove_const_t<T>, N> row(uint32_t m) const {
    tnn::tvec<std::remove_const_t<T>, N> result;

    for (uint32_t i = 0; i < N; ++i) {
      result[i] = (*this)(m, i);
    }
    return result;
  }

  template <uint32_t N>
  tnn::tvec<std::remove_const_t<T>, N> col(uint32_t n) const {
    tnn::tvec<std::remove_const_t<T>, N> result;

    for (uint32_t i = 0; i < N; ++i) {
      result[i] = (*this)(i, n);
    }
    return result;
  }

  template <typename U, uint32_t N, size_t A>
  void set_row(uint32_t m, const tnn::tvec<U, N, A> &val) {
    for (uint32_t i = 0; i < N; ++i) {
      (*this)(m, i) = val[i];
    }
  }

  template <typename U, uint32_t N, size_t A>
  void set_col(uint32_t n, const tnn::tvec<U, N, A> &val) {
    for (uint32_t i = 0; i < N; ++i) {
      (*this)(i, n) = val[i];
    }
  }

  explicit operator bool() const { return data; }

  T *data;
  uint32_t stride_i, stride_j;
};

/**
 * @brief Convert index from original matrix layout to packed layout
 *
 * @param idx Index in packed layout
 * @param rows Number of rows in original matrix
 * @param cols Number of columns in original matrix
 * @return Index in packed matrix layout
 */
extern SYCL_EXTERNAL int toPackedLayoutCoord(int idx, int rows, int cols);

/**
 * @brief Convert index from packed layout to original matrix layout
 *
 * @param idx Index in original matrix layout
 * @param rows Number of rows in original matrix
 * @param cols Number of columns in original matrix
 * @return Index in original matrix layout
 */
extern SYCL_EXTERNAL int fromPackedLayoutCoord(int idx, int rows, int cols);

/**
 * @brief Compare two strings case-insensitively
 *
 * @param str1 First string
 * @param str2 Second string
 * @return True if the strings are equal, false otherwise
 */
extern SYCL_EXTERNAL bool isequalstring(const std::string &str1,
                                        const std::string &str2);

#endif
