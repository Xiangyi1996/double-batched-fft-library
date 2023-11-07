/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** @file   common_host.h
 *  @author Thomas Müller and Nikolaus Binder, NVIDIA
 *  @brief  Common utilities that are needed by pretty much every component of this framework.
 */

#pragma once

#include <common.h>
#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>

#include <array>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace tinydpcppnn {
template <typename T> void format_helper(std::ostringstream &os, std::string_view &str, const T &val) {
    std::size_t bracket = str.find('{');
    if (bracket != std::string::npos) {
        std::size_t bracket_close = str.find('}', bracket + 1);
        if (bracket_close != std::string::npos) {
            os << str.substr(0, bracket) << val;
            str = str.substr(bracket_close + 1);
        } else
            throw std::invalid_argument("No closing bracket\n");
    } else
        throw std::invalid_argument("Not enough brackets for arguments\n");
};

template <typename... T> std::string format(std::string_view str, T... vals) {
    std::ostringstream os;
    (format_helper(os, str, vals), ...);
    os << str;
    return os.str();
}
} // namespace tinydpcppnn

enum class LogSeverity {
    Info,
    Debug,
    Warning,
    Error,
    Success,
};

const std::function<void(LogSeverity, const std::string &)> &log_callback();
void set_log_callback(const std::function<void(LogSeverity, const std::string &)> &callback);

template <typename... Ts> void log(LogSeverity severity, const std::string &msg, Ts &&...args) {
    log_callback()(severity, tinydpcppnn::format(msg, std::forward<Ts>(args)...)); // removed fmt, find something else
}

template <typename... Ts> void log_info(const std::string &msg, Ts &&...args) {
    log(LogSeverity::Info, msg, std::forward<Ts>(args)...);
}
template <typename... Ts> void log_debug(const std::string &msg, Ts &&...args) {
    log(LogSeverity::Debug, msg, std::forward<Ts>(args)...);
}
template <typename... Ts> void log_warning(const std::string &msg, Ts &&...args) {
    log(LogSeverity::Warning, msg, std::forward<Ts>(args)...);
}
template <typename... Ts> void log_error(const std::string &msg, Ts &&...args) {
    log(LogSeverity::Error, msg, std::forward<Ts>(args)...);
}
template <typename... Ts> void log_success(const std::string &msg, Ts &&...args) {
    log(LogSeverity::Success, msg, std::forward<Ts>(args)...);
}

bool verbose();
void set_verbose(bool verbose);

#define STRINGIFY(x) #x
#define STR(x) STRINGIFY(x)
#define FILE_LINE __FILE__ ":" STR(__LINE__)
#define CHECK_THROW(x)                                                                                                 \
    do {                                                                                                               \
        if (!(x)) throw std::runtime_error{FILE_LINE " check failed: " #x};                                            \
    } while (0)

//////////////////////////////
// Enum<->string conversion //
//////////////////////////////

Activation string_to_activation(const std::string &activation_name);
std::string to_string(Activation activation);

GridType string_to_grid_type(const std::string &grid_type);
std::string to_string(GridType grid_type);

HashType string_to_hash_type(const std::string &hash_type);
std::string to_string(HashType hash_type);

InterpolationType string_to_interpolation_type(const std::string &interpolation_type);
std::string to_string(InterpolationType interpolation_type);

ReductionType string_to_reduction_type(const std::string &reduction_type);
std::string to_string(ReductionType reduction_type);

//////////////////
// Misc helpers //
//////////////////

int get_device();
void set_device(int device);
int device_count();

std::string get_device_name(int device);

// Hash helpers taken from https://stackoverflow.com/a/50978188
template <typename T> T xorshift(T n, int i) { return n ^ (n >> i); }

inline uint32_t distribute(uint32_t n) {
    uint32_t p = 0x55555555ul; // pattern of alternating 0 and 1
    uint32_t c = 3423571495ul; // random uneven integer constant;
    return c * xorshift(p * xorshift(n, 16), 16);
}

inline uint64_t distribute(uint64_t n) {
    uint64_t p = 0x5555555555555555ull;   // pattern of alternating 0 and 1
    uint64_t c = 17316035218449499591ull; // random uneven integer constant;
    return c * xorshift(p * xorshift(n, 32), 32);
}

template <typename T, typename S>
constexpr typename std::enable_if<std::is_unsigned<T>::value, T>::type rotl(const T n, const S i) {
    const T m = (std::numeric_limits<T>::digits - 1);
    const T c = i & m;
    return (n << c) | (n >> (((T)0 - c) & m)); // this is usually recognized by the compiler to mean rotation
}

template <typename T> size_t hash_combine(std::size_t seed, const T &v) {
    return rotl(seed, std::numeric_limits<size_t>::digits / 3) ^ distribute(std::hash<T>{}(v));
}

std::string to_snake_case(const std::string &str);

std::vector<std::string> split(const std::string &text, const std::string &delim);

template <typename T> std::string join(const T &components, const std::string &delim) {
    std::ostringstream s;
    for (const auto &component : components) {
        if (&components[0] != &component) {
            s << delim;
        }
        s << component;
    }

    return s.str();
}

std::string to_lower(std::string str);
std::string to_upper(std::string str);
inline bool equals_case_insensitive(const std::string &str1, const std::string &str2) {
    return to_lower(str1) == to_lower(str2);
}

struct CaseInsensitiveHash {
    size_t operator()(const std::string &v) const { return std::hash<std::string>{}(to_lower(v)); }
};
struct CaseInsensitiveEqual {
    bool operator()(const std::string &l, const std::string &r) const { return equals_case_insensitive(l, r); }
};

template <typename T> using ci_hashmap = std::unordered_map<std::string, T, CaseInsensitiveHash, CaseInsensitiveEqual>;

template <typename T> std::string type_to_string();

inline std::string bytes_to_string(size_t bytes) {
    std::array<std::string, 7> suffixes = {{"B", "KB", "MB", "GB", "TB", "PB", "EB"}};

    double count = (double)bytes;
    uint32_t i = 0;
    for (; i < suffixes.size() && count >= 1024; ++i) {
        count /= 1024;
    }

    std::ostringstream oss;
    oss.precision(3);
    oss << count << " " << suffixes[i];
    return oss.str();
}

inline bool is_pot(uint32_t num, uint32_t *log2 = nullptr) {
    if (log2) *log2 = 0;
    if (num > 0) {
        while (num % 2 == 0) {
            num /= 2;
            if (log2) ++*log2;
        }
        if (num == 1) {
            return true;
        }
    }

    return false;
}

inline uint32_t powi(uint32_t base, uint32_t exponent) {
    uint32_t result = 1;
    for (uint32_t i = 0; i < exponent; ++i) {
        result *= base;
    }

    return result;
}

class ScopeGuard {
  public:
    ScopeGuard() = default;
    ScopeGuard(const std::function<void()> &callback) : m_callback{callback} {}
    ScopeGuard(std::function<void()> &&callback) : m_callback{std::move(callback)} {}
    ScopeGuard &operator=(const ScopeGuard &other) = delete;
    ScopeGuard(const ScopeGuard &other) = delete;
    ScopeGuard &operator=(ScopeGuard &&other) {
        std::swap(m_callback, other.m_callback);
        return *this;
    }
    ScopeGuard(ScopeGuard &&other) { *this = std::move(other); }
    ~ScopeGuard() {
        if (m_callback) {
            m_callback();
        }
    }

    void disarm() { m_callback = {}; }

  private:
    std::function<void()> m_callback;
};

template <typename T> class Lazy {
  public:
    template <typename F> T &get(F &&generator) {
        if (!m_val) {
            m_val = generator();
        }

        return m_val;
    }

  private:
    T m_val;
};

#if defined(SYCL_LANGUAGE_VERSION) || (defined(__clang__) && defined(SYCL_LANGUAGE_VERSION))

template <typename K, typename T, typename... Types>
inline void linear_kernel(K kernel, uint32_t shmem_size, dpct::queue_ptr stream, T n_elements, Types... args) {
    if (n_elements <= 0) {
        return;
    }
    /*
    DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    stream->parallel_for(sycl::nd_range<3>(n_blocks_linear(n_elements) * sycl::range<3>(1, 1, N_THREADS_LINEAR),
                                           sycl::range<3>(1, 1, N_THREADS_LINEAR)),
                         [=](sycl::nd_item<3> item_ct1) { int a = 3; });
}

template <typename F> void parallel_for_kernel(const size_t n_elements, F fun, const sycl::nd_item<3> &item_ct1) {
    const size_t i = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range(2);
    if (i >= n_elements) return;

    fun(i);
}

template <typename F>
inline void parallel_for_gpu(uint32_t shmem_size, dpct::queue_ptr stream, size_t n_elements, F &&fun) {
    if (n_elements <= 0) {
        return;
    }
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, n_blocks_linear(n_elements)) * sycl::range<3>(1, 1, N_THREADS_LINEAR),
                          sycl::range<3>(1, 1, N_THREADS_LINEAR)),
        [=](sycl::nd_item<3> item_ct1) { parallel_for_kernel<F>(n_elements, fun, item_ct1); });
}

template <typename F> inline void parallel_for_gpu(dpct::queue_ptr stream, size_t n_elements, F &&fun) {
    parallel_for_gpu(0, stream, n_elements, std::forward<F>(fun));
}

template <typename F> inline void parallel_for_gpu(size_t n_elements, F &&fun) {
    parallel_for_gpu(&dpct::get_default_queue(), n_elements, std::forward<F>(fun));
}

template <typename F>
void parallel_for_aos_kernel(const size_t n_elements, const uint32_t n_dims, F fun, const sycl::nd_item<3> &item_ct1) {
    const size_t dim = item_ct1.get_local_id(2);
    const size_t elem = item_ct1.get_local_id(1) + item_ct1.get_group(2) * item_ct1.get_local_range(1);
    if (dim >= n_dims) return;
    if (elem >= n_elements) return;

    fun(elem, dim);
}

template <typename F>
inline void parallel_for_gpu_aos(uint32_t shmem_size, dpct::queue_ptr stream, size_t n_elements, uint32_t n_dims,
                                 F &&fun) {
    if (n_elements <= 0 || n_dims <= 0) {
        return;
    }

    const sycl::range<3> threads = {1, div_round_up(N_THREADS_LINEAR, n_dims), n_dims};
    const size_t n_threads = threads[2] * threads[1];
    const sycl::range<3> blocks = {1, 1, (uint32_t)div_round_up(n_elements * n_dims, n_threads)};

    /*
    DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    stream->parallel_for(sycl::nd_range<3>(blocks * threads, threads), [=](sycl::nd_item<3> item_ct1) {
        parallel_for_aos_kernel(n_elements, n_dims, fun, item_ct1);
    });
}

template <typename F>
inline void parallel_for_gpu_aos(dpct::queue_ptr stream, size_t n_elements, uint32_t n_dims, F &&fun) {
    parallel_for_gpu_aos(0, stream, n_elements, n_dims, std::forward<F>(fun));
}

template <typename F> inline void parallel_for_gpu_aos(size_t n_elements, uint32_t n_dims, F &&fun) {
    parallel_for_gpu_aos(nullptr, n_elements, n_dims, std::forward<F>(fun));
}

template <typename F>
void parallel_for_soa_kernel(const size_t n_elements, const uint32_t n_dims, F fun, const sycl::nd_item<3> &item_ct1) {
    const size_t elem = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range(2);
    const size_t dim = item_ct1.get_group(1);
    if (elem >= n_elements) return;
    if (dim >= n_dims) return;

    fun(elem, dim);
}

template <typename F>
inline void parallel_for_gpu_soa(uint32_t shmem_size, dpct::queue_ptr stream, size_t n_elements, uint32_t n_dims,
                                 F &&fun) {
    if (n_elements <= 0 || n_dims <= 0) {
        return;
    }

    const sycl::range<3> blocks = {1, n_dims, n_blocks_linear(n_elements)};

    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, n_blocks_linear(n_elements)) * sycl::range<3>(1, 1, N_THREADS_LINEAR),
                          sycl::range<3>(1, 1, N_THREADS_LINEAR)),
        [=](sycl::nd_item<3> item_ct1) { parallel_for_soa_kernel(n_elements, n_dims, fun, item_ct1); });
}

template <typename F>
inline void parallel_for_gpu_soa(dpct::queue_ptr stream, size_t n_elements, uint32_t n_dims, F &&fun) {
    parallel_for_gpu_soa(0, stream, n_elements, n_dims, std::forward<F>(fun));
}

template <typename F> inline void parallel_for_gpu_soa(size_t n_elements, uint32_t n_dims, F &&fun) {
    parallel_for_gpu_soa(nullptr, n_elements, n_dims, std::forward<F>(fun));
}
#endif
