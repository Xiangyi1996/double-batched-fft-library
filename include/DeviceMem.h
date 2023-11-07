#pragma once

#include <level_zero/ze_api.h>

#include <atomic>
#include <dpct/dpct.hpp>
#include <iostream>
#include <random>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>
#include <vector>

#include "common.h"
#include "common_host.h"
// #include <fmt/format.h>
using namespace sycl;

#define DEBUG_GUARD_SIZE 0

inline std::atomic<size_t> &total_n_bytes_allocated() {
    static std::atomic<size_t> s_total_n_bytes_allocated{0};
    return s_total_n_bytes_allocated;
}

// A templated class for managing device memory
template <typename T> class DeviceMem {
  private:
    T *m_data = nullptr;
    int m_size = 0;
    bool m_managed = false;

  public:
    // Default constructor
    DeviceMem();

    // Constructor with size and queue
    DeviceMem(int size, sycl::queue &q);

    DeviceMem(size_t size, bool managed = false) : m_managed{managed} { resize(size); }

    void check_guards() const {
#if DEBUG_GUARD_SIZE > 0
        if (!m_data) return;

        uint8_t buf[DEBUG_GUARD_SIZE];
        const uint8_t *rawptr{reinterpret_cast<const uint8_t *>(m_data)};

        sycl::queue &q{dpct::get_default_queue()};
        q.memcpy(buf, rawptr - DEBUG_GUARD_SIZE, DEBUG_GUARD_SIZE).wait();
        for (int i = 0; i < DEBUG_GUARD_SIZE; ++i) {
            if (buf[i] != 0xff) {
                printf("TRASH BEFORE BLOCK offset %d data %p, read 0x%02x expected "
                       "0xff!\n",
                       i, m_data, buf[i]);
                break;
            }
        }

        q.memcpy(buf, rawptr + m_size * sizeof(T), DEBUG_GUARD_SIZE).wait();
        for (int i = 0; i < DEBUG_GUARD_SIZE; ++i) {
            if (buf[i] != 0xfe) {
                printf("TRASH AFTER BLOCK offset %d data %p, read 0x%02x expected 0xfe!\n", i, m_data, buf[i]);
                break;
            }
        }
#endif
    }
    // Allocate from Darius' code
    void allocate2(int size, queue &q);

    // Allocate memory on the device
    void allocate(int size, queue &q);

    // Allocate memory on the device
    void allocate(int size);

    // Free memory on the device
    void free_mem(queue &q);
    void free_mem();

    // Copy data from host to device
    void copy_from_host(std::vector<T> &data, int n, queue q);

    // Copy data from device to host
    void copy_to_host(std::vector<T> &data, int n, queue q);

    // Copy data from host to device
    void copy_from_host(std::vector<T> &data, queue q);

    // Copy data from device to host
    void copy_to_host(std::vector<T> &data, queue q);

    /** @name Copy operations
     *  @{
     */
    /// Copy data of num_elements from the raw pointer on the host
    void copy_from_host(const T *host_data, const size_t num_elements) {
        dpct::get_default_queue().memcpy(data(), host_data, num_elements * sizeof(T));
    }

    /// Copies data from the raw host pointer to fill the entire array
    void copy_from_host(const T *data) { copy_from_host(data, m_size); }

    /// Copies num_elements of data from the raw host pointer after enlarging the
    /// array so that everything fits in
    void enlarge_and_copy_from_host(const T *data, const size_t num_elements) {
        enlarge(num_elements);
        copy_from_host(data, num_elements);
    }

    /// Copies num_elements from the host vector after enlarging the array so that
    /// everything fits in
    void enlarge_and_copy_from_host(const std::vector<T> &data, const size_t num_elements) {
        enlarge_and_copy_from_host(data.data(), num_elements);
    }

    /// Copies the entire host vector after enlarging the array so that everything
    /// fits in
    void enlarge_and_copy_from_host(const std::vector<T> &data) {
        enlarge_and_copy_from_host(data.data(), data.size());
    }

    /// Copies num_elements of data from the raw host pointer after resizing the
    /// array
    void resize_and_copy_from_host(const T *data, const size_t num_elements) {
        resize(num_elements);
        copy_from_host(data, num_elements);
    }

    /// Copies num_elements from the host vector after resizing the array
    void resize_and_copy_from_host(const std::vector<T> &data, const size_t num_elements) {
        resize_and_copy_from_host(data.data(), num_elements);
    }

    /// Copies the entire host vector after resizing the array
    void resize_and_copy_from_host(const std::vector<T> &data) { resize_and_copy_from_host(data.data(), data.size()); }

    /// Copies num_elements of data from the raw host pointer to the device. Fails
    /// if there is not enough space available.
    void copy_to_host(T *host_data, const size_t num_elements) const {
        if (num_elements > m_size) {
            throw std::runtime_error{
                tinydpcppnn::format("Trying to copy {} elements, but memory size is only {}.", num_elements, m_size)};
        }

        dpct::get_default_queue().memcpy(host_data, data(), num_elements * sizeof(T));
    }

    /// Copies num_elements from the device to a vector on the host
    void copy_to_host(std::vector<T> &data, const size_t num_elements) const {
        if (data.size() < num_elements) {
            throw std::runtime_error{tinydpcppnn::format("Trying to copy {} elements, but vector size is only {}.",
                                                         num_elements, data.size())};
        }

        copy_to_host(data.data(), num_elements);
    }

    /// Copies num_elements from the device to a raw pointer on the host
    void copy_to_host(T *data) const { copy_to_host(data, m_size); }

    /// Copies all elements from the device to a vector on the host
    void copy_to_host(std::vector<T> &data) const {
        if (data.size() < m_size) {
            throw std::runtime_error{
                tinydpcppnn::format("Trying to copy {} elements, but vector size is only {}", m_size, data.size())};
        }

        copy_to_host(data.data(), m_size);
    }

    /// Copies size elements from another device array to this one, automatically
    /// resizing it
    void copy_from_device(const DeviceMem<T> &other, const size_t size) {
        if (size == 0) {
            return;
        }

        if (m_size < size) {
            resize(size);
        }

        dpct::get_default_queue().memcpy(m_data, other.m_data, size * sizeof(T));
    }

    /// Copies data from another device array to this one, automatically resizing
    /// it
    void copy_from_device(const DeviceMem<T> &other) { copy_from_device(other, other.m_size); }

    // Created an (owned) copy of the data
    DeviceMem<T> copy(size_t size) const {
        DeviceMem<T> result{size};
        result.copy_from_device(*this);
        return result;
    }

    DeviceMem<T> copy() const { return copy(m_size); }

    // Get the raw data pointer
    T *data() const {
        check_guards();
        return m_data;
    }
    void set_values(int size, float *array, queue &q);
    // Set data at a specific index
    void set_data(int id, T value);

    /** @name Resizing/enlargement
     *  @{
     */
    /// Resizes the array to the exact new size, even if it is already larger
    void resize(const size_t size);

    /// Enlarges the array if its size is smaller
    void enlarge(const size_t size);
    /** @} */

    /** @name Memset
     *  @{
     */
    /// Sets the memory of the first num_elements to value
    void memset(const int value, const size_t num_elements, const size_t offset = 0);

    /// Sets the memory of the all elements to value
    void memset(const int value);

    T &at(size_t idx) const {
        if (!m_managed) {
            throw std::runtime_error{tinydpcppnn::format("DeviceMem::at() not permitted if not managed.")};
        }

        if (idx > m_size) {
            throw std::runtime_error{tinydpcppnn::format("DeviceMem out of bounds: idx={} size={}", idx, m_size)};
        }

        return m_data[idx];
    }

    T &operator[](size_t idx) const {
#ifdef DEBUG_BUFFER_OVERRUN
        if (idx > m_size) {
            printf("WARNING: buffer overrun of %p at idx %zu\n", idx);
        }
#endif
        return m_data[idx];
    }

    T &operator[](uint32_t idx) const {
#ifdef DEBUG_BUFFER_OVERRUN
        if (idx > m_size) {
            printf("WARNING: buffer overrun of %p at idx %u\n", idx);
        }
#endif
        return m_data[idx];
    }

    // Get the size of the memory allocation
    size_t get_num_elements() const { return m_size; }
    size_t size() const { return get_num_elements(); }

    // Get bytes of allocated memory size
    size_t get_bytes() const { return m_size * sizeof(T); }

    // Synonym: Get bytes of allocated memory size
    size_t bytes() const { return get_bytes(); }

    // Initialize memory with values drawn from a normal distribution
    void initialize_normal(double dev, DeviceMem<T> &transposed, int input_width, int width, int output_width,
                           int n_hidden, queue q);

    // Initialize memory with values drawn from a normal distribution
    void initialize_normal(double dev, queue q);

    // Initialize memory with values drawn from a uniform distribution
    void initialize_uniform(double scale, DeviceMem<T> &transposed, int input_width, int width, int output_width,
                            int n_hidden, queue q);

    // Transpose the memory content
    void make_transposed(DeviceMem<T> &transposed, int input_width, int width, int output_width, int n_hidden, queue q);

    // Initialize memory with values drawn from a uniform distribution
    void initialize_uniform(queue q, double scale = 1.0);

    // Initialize memory with values according to Xavier uniform initialization
    void initialize_xavier_unif(DeviceMem<T> &transposed, int input_width, int width, int output_width, int n_hidden,
                                queue q);

    // Initialize memory with values according to Xavier uniform initialization
    void initialize_xavier_unif(int input_width, int output_width, queue q);

    // Initialize memory with values according to Xavier normal initialization
    void inititialize_xavier_normal(DeviceMem<T> &transposed, int input_width, int width, int output_width,
                                    int n_hidden, queue q);

    // Initialize memory with values according to Xavier normal initialization
    void initialize_xavier_normal(int input_width, int output_width, queue q);

    // Initialize memory with constant values
    void initialize_constant(T constant, DeviceMem<T> &transposed, queue q);

    // Initialize memory with constant values
    void initialize_constant(T constant, queue q);

    // Initialize memory with values according to He normal initialization
    void intitialize_he_normal(DeviceMem<T> &transposed, int input_width, int width, int output_width, int n_hidden,
                               queue q);

    // Initialize memory with values according to He normal initialization
    void intitialize_he_normal(int input_width, queue q);

    void initialize_arange(queue q, int input_width, int net_width, int out_width, int hidden_matrices);
    void initialize_arange(queue q);

    void allocate_memory(size_t n_bytes);
};

struct Interval {
    // Inclusive start, exclusive end
    size_t start, end;

    bool operator<(const Interval &other) const {
        // This operator is used to sort non-overlapping intervals. Since intervals
        // may be empty, the second half of the following expression is required to
        // resolve ambiguity when `end` of adjacent empty intervals is equal.
        return end < other.end || (end == other.end && start < other.start);
    }

    bool overlaps(const Interval &other) const { return !intersect(other).empty(); }

    Interval intersect(const Interval &other) const { return {std::max(start, other.start), std::min(end, other.end)}; }

    bool valid() const { return end >= start; }

    bool empty() const { return end <= start; }

    size_t size() const { return end - start; }
};

class DeviceMemArena {
  public:
    DeviceMemArena() {
        static bool printed_warning = false;
        m_device_id = dpct::get_current_device_id();
        sycl::queue &q{(dpct::get_device(m_device_id).default_queue())};

        try {
            ze_result_t res;
            m_device = sycl::get_native<backend::ext_oneapi_level_zero>(q.get_device());
            m_context = sycl::get_native<backend::ext_oneapi_level_zero>(q.get_context());

            size_t total_memory{q.get_device().get_info<sycl::info::device::global_mem_size>()};

            size_t page_size;
            res = zeVirtualMemQueryPageSize(m_context, m_device, total_memory, &page_size);
            if (res != ZE_RESULT_SUCCESS) throw std::runtime_error{"DeviceMemArena: Could not query page-size."};

            m_max_size = previous_multiple(total_memory, page_size);

            // Align memory at least by a cache line.
            m_alignment = q.get_device().get_info<sycl::info::device::global_mem_cache_line_size>();

            m_free_intervals = {{0, m_max_size}};

            // Reserve an address range that would be sufficient for housing the
            // entire available GPU RAM (if nothing else was using the GPU). This is
            // unlikely to exhaust all available addresses (even if multiple
            // DeviceMemArenas are used simultaneously), while also ensuring that we
            // never exhaust the reserved address range without running out of
            // physical memory beforehand.
            res = zeVirtualMemReserve(m_context, nullptr, m_max_size, &m_base_address);
            if (res != ZE_RESULT_SUCCESS) {
                m_base_address = nullptr;
                throw std::runtime_error{"DeviceMemArena: Could not reserve address space."};
            }
        } catch (std::runtime_error &ex) {
            // Use regular memory as fallback
            m_fallback_memory = std::make_shared<DeviceMem<uint8_t>>();

            if (!printed_warning) {
                printed_warning = true;
                log_warning(ex.what());
                log_warning("DeviceMemArena: GPU {} does not support virtual memory. "
                            "Falling back to regular allocations, which will be larger and can "
                            "cause occasional stutter.",
                            m_device_id);
            }
        }
    }

    DeviceMemArena(DeviceMemArena &&other) = default;
    DeviceMemArena(const DeviceMemArena &other) = delete;
    DeviceMemArena &operator=(DeviceMemArena &&other) = delete;
    DeviceMemArena &operator=(const DeviceMemArena &other) = delete;

    ~DeviceMemArena() {
        if (in_use()) log_warning("Attempting to free memory arena while it is still in use.");

        try {
            // Make sure we're clearing the GPU memory arena on the correct device.
            dpct::get_device(m_device_id).default_queue().wait();

            if (m_base_address) {
                ze_result_t res;
                total_n_bytes_allocated() -= m_size;

                res = zeVirtualMemUnmap(m_context, m_base_address, m_size);
                if (res != ZE_RESULT_SUCCESS) throw std::runtime_error{"~DeviceMemArena: Could not unmap memory."};

                while (!m_handles.empty()) {
                    res = zePhysicalMemDestroy(m_context, m_handles.back());
                    if (res != ZE_RESULT_SUCCESS)
                        throw std::runtime_error{"~DeviceMemArena: Could not destroy physical memory "
                                                 "allocation."};
                    m_handles.pop_back();
                }

                res = zeVirtualMemFree(m_context, m_base_address, m_max_size);
                if (res != ZE_RESULT_SUCCESS)
                    throw std::runtime_error{"~DeviceMemArena: Could not destroy address reservation."};
            }
        } catch (const std::runtime_error &error) {
            log_warning("Could not free memory arena: {}", error.what());
        }
    }

    uint8_t *data() {
        return m_fallback_memory ? m_fallback_memory->data() : reinterpret_cast<uint8_t *>(m_base_address);
    }

    std::shared_ptr<DeviceMem<uint8_t>> backing_memory() { return m_fallback_memory; }

    // Finds the smallest interval of free memory in the DeviceMemArena that's
    // large enough to hold the requested number of bytes. Then allocates
    // that memory.
    size_t allocate(size_t n_bytes) {
        // Permitting zero-sized allocations is error prone
        if (n_bytes == 0) n_bytes = m_alignment;

        // Align allocations with the nearest cache line (at least the granularity
        // of the memory allocations)
        n_bytes = next_multiple(n_bytes, m_alignment);

        Interval *best_candidate = &m_free_intervals.back();
        for (auto &f : m_free_intervals)
            if (f.size() >= n_bytes && f.size() < best_candidate->size()) best_candidate = &f;

        size_t start = best_candidate->start;

        // Note: the += operator can turn `best_candidate` into an empty interval,
        // which is fine because it will be absorbed into adjacent free intervals in
        // later calls to `merge_adjacent_intervals`.
        m_allocated_intervals[start] = best_candidate->start += n_bytes;

        enlarge(size());

        return start;
    }

    void free(size_t start) {
        if (m_allocated_intervals.count(start) == 0)
            throw std::runtime_error{"Attempted to free arena memory that was not allocated."};

        Interval interval = {start, m_allocated_intervals[start]};
        m_allocated_intervals.erase(start);

        m_free_intervals.insert(std::upper_bound(std::begin(m_free_intervals), std::end(m_free_intervals), interval),
                                interval);

        merge_adjacent_intervals();
    }

    void enlarge(size_t n_bytes) {
        ze_result_t res;

        if (n_bytes <= m_size) return;

        if (dpct::get_current_device_id() != m_device_id)
            throw std::runtime_error{tinydpcppnn::format("Attempted to use a DeviceMemArena of device {} from the "
                                                         "wrong device {}.",
                                                         m_device_id, dpct::get_current_device_id())};

        log_debug("DeviceMemArena: enlarging from {} to {}", bytes_to_string(m_size), bytes_to_string(n_bytes));

        sycl::queue &q{(dpct::get_device(m_device_id).default_queue())};

        if (m_fallback_memory) {
            static const double GROWTH_FACTOR = 1.5;

            q.wait();

            size_t page_size;
            size_t tmp = size_t(double(n_bytes) * GROWTH_FACTOR);
            res = zeVirtualMemQueryPageSize(m_context, m_device, tmp, &page_size);
            if (res != ZE_RESULT_SUCCESS)
                throw std::runtime_error{"DeviceMemArena::enlarge: Could not query page-size."};

            m_size = next_multiple(tmp, page_size);
            m_fallback_memory = std::make_shared<DeviceMem<uint8_t>>(m_fallback_memory->copy(m_size));

            q.wait();

            return;
        }

        // Compute the actual amount of memory to reserve (consider alignment, etc.)
        size_t n_bytes_to_allocate = n_bytes - m_size;

        size_t page_size;
        res = zeVirtualMemQueryPageSize(m_context, m_device, n_bytes_to_allocate, &page_size);
        if (res != ZE_RESULT_SUCCESS) throw std::runtime_error{"DeviceMemArena::enlarge: Could not query page-size."};

        n_bytes_to_allocate = next_multiple(n_bytes_to_allocate, page_size);

        // Perform the actual physical allocation.
        ze_physical_mem_desc_t pmemDesc = {
            ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC,
            nullptr,            // unused
            0,                  // flags
            n_bytes_to_allocate // size
        };
        m_handles.emplace_back();
        res = zePhysicalMemCreate(m_context, m_device, &pmemDesc, &m_handles.back());
        if (res != ZE_RESULT_SUCCESS)
            throw std::runtime_error{"DeviceMemArena::enlarge: Could not allocate physical memory."};

        // Map into virtual address space
        res = zeVirtualMemMap(m_context, reinterpret_cast<uint8_t *>(m_base_address) + m_size, n_bytes_to_allocate,
                              m_handles.back(), 0, ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE);
        if (res != ZE_RESULT_SUCCESS)
            throw std::runtime_error{tinydpcppnn::format("DeviceMemArena::enlarge: Could not map memory into "
                                                         "virtual address space. Retval={}",
                                                         res)};

        m_size += n_bytes_to_allocate;
        total_n_bytes_allocated() += n_bytes_to_allocate;

        q.wait();
    }

    size_t size() const { return m_free_intervals.back().start; }

    bool in_use() const { return m_free_intervals.size() != 1 || m_free_intervals.front().size() != m_max_size; }

    class Allocation {
      public:
        Allocation() = default;
        Allocation(dpct::queue_ptr stream, size_t offset, const std::shared_ptr<DeviceMemArena> &workspace)
            : m_stream{stream}, m_data{workspace->data() + offset}, m_offset{offset}, m_workspace{workspace},
              m_backing_memory{workspace->backing_memory()} {}

        ~Allocation() {
            if (m_workspace) {
                m_workspace->free(m_offset);
            }
        }

        Allocation(const Allocation &other) = delete;

        Allocation &operator=(Allocation &&other) {
            std::swap(m_stream, other.m_stream);
            std::swap(m_data, other.m_data);
            std::swap(m_offset, other.m_offset);
            std::swap(m_workspace, other.m_workspace);
            std::swap(m_backing_memory, other.m_backing_memory);
            return *this;
        }

        Allocation(Allocation &&other) { *this = std::move(other); }

        uint8_t *data() { return m_data; }

        const uint8_t *data() const { return m_data; }

        dpct::queue_ptr stream() const { return m_stream; }

      private:
        dpct::queue_ptr m_stream = &dpct::get_default_queue();
        uint8_t *m_data = nullptr;
        size_t m_offset = 0;
        std::shared_ptr<DeviceMemArena> m_workspace = nullptr;

        // Backing DeviceMem (if backed by a DeviceMem). Ensures that
        // the backing memory is only freed once all allocations that
        // use it were destroyed.
        std::shared_ptr<DeviceMem<uint8_t>> m_backing_memory = nullptr;
    };

  private:
    void merge_adjacent_intervals() {
        size_t j = 0;
        for (size_t i = 1; i < m_free_intervals.size(); ++i) {
            Interval &prev = m_free_intervals[j];
            Interval &cur = m_free_intervals[i];

            if (prev.end == cur.start) {
                prev.end = cur.end;
            } else {
                ++j;
                m_free_intervals[j] = m_free_intervals[i];
            }
        }
        m_free_intervals.resize(j + 1);
    }

    std::vector<Interval> m_free_intervals;
    std::unordered_map<size_t, size_t> m_allocated_intervals;

    size_t m_device_id;
    ze_device_handle_t m_device;
    ze_context_handle_t m_context;
    void *m_base_address{nullptr};
    size_t m_size = 0;

    std::vector<ze_physical_mem_handle_t> m_handles;

    // Used when virtual memory isn't supported.
    // Requires more storage + memcpy, but is more portable.
    std::shared_ptr<DeviceMem<uint8_t>> m_fallback_memory = nullptr;

    size_t m_alignment;
    size_t m_max_size;
};

inline std::unordered_map<dpct::queue_ptr, std::shared_ptr<DeviceMemArena>> &stream_gpu_memory_arenas() {
    static auto *stream_gpu_memory_arenas = new std::unordered_map<dpct::queue_ptr, std::shared_ptr<DeviceMemArena>>{};
    return *stream_gpu_memory_arenas;
}

inline std::unordered_map<int, std::shared_ptr<DeviceMemArena>> &global_gpu_memory_arenas() {
    static auto *global_gpu_memory_arenas = new std::unordered_map<int, std::shared_ptr<DeviceMemArena>>{};
    return *global_gpu_memory_arenas;
}

inline DeviceMemArena::Allocation allocate_workspace(dpct::queue_ptr stream, size_t n_bytes) {
    if (n_bytes == 0) {
        // Return a null allocation if no bytes were requested.
        return {};
    }

    auto &arena = stream ? stream_gpu_memory_arenas()[stream] : global_gpu_memory_arenas()[get_device()];
    if (!arena) {
        arena = std::make_shared<DeviceMemArena>();
    }
    return DeviceMemArena::Allocation{stream, arena->allocate(n_bytes), arena};
}