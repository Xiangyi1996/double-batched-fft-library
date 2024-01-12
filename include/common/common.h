#ifndef TINYNN_COMMON_H
#define TINYNN_COMMON_H

#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

#include "vec.h"

static constexpr float PI = 3.14159265358979323846f;

// When TCNN managed its model parameters, they are always aligned,
// which yields performance benefits in practice. However, parameters
// supplied by PyTorch are not necessarily aligned. The following
// variable controls whether TCNN must deal with unaligned data.
#if defined(TCNN_PARAMS_UNALIGNED)
static constexpr bool PARAMS_ALIGNED = false;
#else
static constexpr bool PARAMS_ALIGNED = true;
#endif

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

/// some common math functions
namespace tinydpcppnn {
namespace math {
template <typename T> T div_round_up(T val, T divisor) { return (val + divisor - 1) / divisor; }

template <typename T> T next_multiple(T val, T divisor) { return div_round_up(val, divisor) * divisor; }

template <typename T> T previous_multiple(T val, T divisor) { return (val / divisor) * divisor; }

inline uint32_t powi(uint32_t base, uint32_t exponent) {
    uint32_t result = 1;
    for (uint32_t i = 0; i < exponent; ++i) {
        result *= base;
    }

    return result;
}

} // namespace math
} // namespace tinydpcppnn

template <typename T> struct PitchedPtr {
    PitchedPtr() : ptr{nullptr}, stride_in_bytes{sizeof(T)} {}
    PitchedPtr(T *ptr, size_t stride_in_elements, size_t offset = 0, size_t extra_stride_bytes = 0)
        : ptr{ptr + offset}, stride_in_bytes{(uint32_t)(stride_in_elements * sizeof(T) + extra_stride_bytes)} {}

    template <typename U>
    explicit PitchedPtr(PitchedPtr<U> other) : ptr{(T *)other.ptr}, stride_in_bytes{other.stride_in_bytes} {}

    T *operator()(uint32_t y) const { return (T *)((const char *)ptr + y * stride_in_bytes); }

    void operator+=(uint32_t y) { ptr = (T *)((const char *)ptr + y * stride_in_bytes); }

    void operator-=(uint32_t y) { ptr = (T *)((const char *)ptr - y * stride_in_bytes); }

    explicit operator bool() const { return ptr; }

    T *ptr;
    uint32_t stride_in_bytes;
};

/**
 * @brief Convert index from original matrix layout to packed layout
 *
 * @param idx Index in packed layout
 * @param rows Number of rows in original matrix
 * @param cols Number of columns in original matrix
 * @return Index in packed matrix layout
 */
extern SYCL_EXTERNAL unsigned toPackedLayoutCoord(const unsigned idx, const unsigned rows, const unsigned cols);

/**
 * @brief Convert index from packed layout to original matrix layout
 *
 * @param idx Index in original matrix layout
 * @param rows Number of rows in original matrix
 * @param cols Number of columns in original matrix
 * @return Index in original matrix layout
 */
extern SYCL_EXTERNAL unsigned fromPackedLayoutCoord(const unsigned idx, const unsigned rows, const unsigned cols);

/**
 * @brief Compare two strings case-insensitively
 *
 * @param str1 First string
 * @param str2 Second string
 * @return True if the strings are equal, false otherwise
 */
extern SYCL_EXTERNAL bool isequalstring(const std::string &str1, const std::string &str2);

#endif
