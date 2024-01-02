#pragma once

#include <level_zero/ze_api.h>

#include <atomic>
#include <dpct/dpct.hpp>
#include <fstream>
#include <iostream>
#include <random>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>
#include <vector>

#include "common.h"
#include "common_host.h"

using namespace sycl;

// A templated class for managing device memory
template <typename T> class DeviceMem {
  private:
    size_t m_size = 0;
    const sycl::queue &m_q;
    T *m_data = nullptr;

  public:
    // Default constructor
    DeviceMem() = delete;

    /**
     * @brief Constructor for the DeviceMem class
     *
     * @param size               Size of the memory to allocate in elements.
     * @param queue              SYCL queue associated with the object.
     */
    DeviceMem(const size_t size, const sycl::queue &q) : m_size(size), m_q(q) {
        assert(m_size >= 0);
        m_data = sycl::malloc_device<T>(m_size, m_q);
    }

    ~DeviceMem() { sycl::free(m_data, m_q); }

    // Copy data from host to device
    sycl::event copy_from_host(const std::vector<T> &data) {
        assert(data.size() == m_size);
        return m_q.memcpy(m_data, data.data(), get_bytes());
    }

    // Copy data from device to host
    sycl::event copy_to_host(std::vector<T> &data) const {
        assert(data.size() == m_size);
        return m_q.memcpy(data.data(), m_data, get_bytes());
    }

    /// Copies size elements from another device array to this one, automatically
    /// resizing it
    sycl::event copy_from_device(const DeviceMem<T> &other) {
        assert(other.m_size == m_size);

        return m_q.memcpy(m_data, other.m_data, get_bytes());
    }

    // Get the raw data pointer
    T const *data() const { return m_data; }

    /// Sets the memory of the all elements to value
    sycl::event fill(const T &value) { return m_q.fill(m_data, value, size()); }

    // Get the size of the memory allocation
    size_t size() const { return m_size; }

    // Get bytes of allocated memory size
    size_t get_bytes() const { return m_size * sizeof(T); }

    static void save_to_file(const DeviceMem<T> &vec, std::string filename) {
        // Open the file for writing
        std::ofstream file;
        file.open(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
        }

        std::vector<T> host(vec.size());
        vec.copy_to_host(host);

        // Write each value of the weights matrices to the file
        for (int i = 0; i < host.size(); i++) {
            file << (float)host[i] << "\n";
        }

        // Close the file
        file.close();
        return;
    }
};

/// TODO: this can be a sub struct of the Arena
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

            m_max_size = tinydpcppnn::math::previous_multiple(total_memory, page_size);

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
        n_bytes = tinydpcppnn::math::next_multiple(n_bytes, m_alignment);

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

            m_size = tinydpcppnn::math::next_multiple(tmp, page_size);
            m_fallback_memory = std::make_shared<DeviceMem<uint8_t>>(m_fallback_memory->copy(m_size));

            q.wait();

            return;
        }

        // Compute the actual amount of memory to reserve (consider alignment, etc.)
        size_t n_bytes_to_allocate = n_bytes - m_size;

        size_t page_size;
        res = zeVirtualMemQueryPageSize(m_context, m_device, n_bytes_to_allocate, &page_size);
        if (res != ZE_RESULT_SUCCESS) throw std::runtime_error{"DeviceMemArena::enlarge: Could not query page-size."};

        n_bytes_to_allocate = tinydpcppnn::math::next_multiple(n_bytes_to_allocate, page_size);

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