#pragma once

#include <atomic>
#include <dpct/dpct.hpp>
#include <iostream>
#include <random>
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
template <typename T>
class DeviceMem {
 private:
  T *m_data = nullptr;
  int m_size = 0;
  bool m_managed = false;

 public:
  // Default constructor
  DeviceMem();

  // Constructor with size and queue
  DeviceMem(int size, sycl::queue &q);

  DeviceMem(size_t size, bool managed = false) : m_managed{managed} {
    resize(size);
  }

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

  // Get the raw data pointer
  T *data() const { return m_data; }

  // Set data at a specific index
  void set_data(int id, T value);

  /** @name Resizing/enlargement
   *  @{
   */
  /// Resizes the array to the exact new size, even if it is already larger
  void resize(size_t size);

  // Get the size of the memory allocation
  int size() const { return m_size; }

  // Get bytes of allocated memory size
  size_t get_bytes() const { return m_size * sizeof(T); }

  // Synonym: Get bytes of allocated memory size
  size_t bytes() const { return get_bytes(); }

  // Initialize memory with values drawn from a normal distribution
  void initialize_normal(double dev, DeviceMem<T> &transposed, int input_width,
                         int width, int output_width, int n_hidden, queue q);

  // Initialize memory with values drawn from a normal distribution
  void initialize_normal(double dev, queue q);

  // Initialize memory with values drawn from a uniform distribution
  void initialize_uniform(double scale, DeviceMem<T> &transposed,
                          int input_width, int width, int output_width,
                          int n_hidden, queue q);

  // Transpose the memory content
  void make_transposed(DeviceMem<T> &transposed, int input_width, int width,
                       int output_width, int n_hidden, queue q);

  // Initialize memory with values drawn from a uniform distribution
  void initialize_uniform(queue q, double scale = 1.0);

  // Initialize memory with values according to Xavier uniform initialization
  void initialize_xavier_unif(DeviceMem<T> &transposed, int input_width,
                              int width, int output_width, int n_hidden,
                              queue q);

  // Initialize memory with values according to Xavier uniform initialization
  void initialize_xavier_unif(int input_width, int output_width, queue q);

  // Initialize memory with values according to Xavier normal initialization
  void inititialize_xavier_normal(DeviceMem<T> &transposed, int input_width,
                                  int width, int output_width, int n_hidden,
                                  queue q);

  // Initialize memory with values according to Xavier normal initialization
  void initialize_xavier_normal(int input_width, int output_width, queue q);

  // Initialize memory with constant values
  void initialize_constant(T constant, DeviceMem<T> &transposed, queue q);

  // Initialize memory with constant values
  void initialize_constant(T constant, queue q);

  // Initialize memory with values according to He normal initialization
  void intitialize_he_normal(DeviceMem<T> &transposed, int input_width,
                             int width, int output_width, int n_hidden,
                             queue q);

  // Initialize memory with values according to He normal initialization
  void intitialize_he_normal(int input_width, queue q);

  void initialize_arange(queue q, int input_width, int net_width, int out_width,
                         int hidden_matrices);
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

  bool overlaps(const Interval &other) const {
    return !intersect(other).empty();
  }

  Interval intersect(const Interval &other) const {
    return {std::max(start, other.start), std::min(end, other.end)};
  }

  bool valid() const { return end >= start; }

  bool empty() const { return end <= start; }

  size_t size() const { return end - start; }
};

class DeviceMemArena {
 public:
  DeviceMemArena() {}
  ~DeviceMemArena() {}
  uint8_t *data() { return m_data; };
  uint8_t *m_data;

  std::shared_ptr<DeviceMem<uint8_t>> backing_memory() {
    return m_fallback_memory;
  }
  std::shared_ptr<DeviceMem<uint8_t>> m_fallback_memory;
  void free(size_t start) {
    // if (m_allocated_intervals.count(start) == 0) {
    // 	throw std::runtime_error{"Attempted to free arena memory that was not
    // allocated."};
    // }

    // Interval interval = {start, m_allocated_intervals[start]};
    // m_allocated_intervals.erase(start);

    // m_free_intervals.insert(
    // 	std::upper_bound(std::begin(m_free_intervals),
    // std::end(m_free_intervals), interval), 	interval
    // );

    // merge_adjacent_intervals();
  }

  class Allocation {
   public:
    Allocation() = default;
    Allocation(dpct::queue_ptr stream, size_t offset,
               const std::shared_ptr<DeviceMemArena> &workspace)
        : m_stream{stream},
          m_data{workspace->data() + offset},
          m_offset{offset},
          m_workspace{workspace},
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

  // class DeviceMemArena {
  // public:
  // 	DeviceMemArena() {
  // 		m_device = cuda_device();

  // 		// Align memory at least by a cache line (128 bytes).
  // 		m_alignment = (size_t)128;
  // 		m_max_size = previous_multiple(cuda_memory_info().total,
  // cuda_memory_granularity());

  // 		m_free_intervals = {{0, m_max_size}};

  // 		// Reserve an address range that would be sufficient for housing
  // the entire
  // 		// available GPU RAM (if nothing else was using the GPU). This
  // is unlikely
  // 		// to exhaust all available addresses (even if multiple
  // DeviceMemArenas are
  // 		// used simultaneously), while also ensuring that we never
  // exhaust the
  // 		// reserved address range without running out of physical memory
  // beforehand.
  //                 /*
  //                 DPCT1007:70: Migration of cuMemAddressReserve is not
  //                 supported.
  //                 */
  //                 if (cuda_supports_virtual_memory() &&
  //                     cuMemAddressReserve(&m_base_address, m_max_size, 0, 0,
  //                     0) ==
  //                         0) {
  //                         return;
  // 		}

  // 		// Use regular memory as fallback
  // 		m_fallback_memory = std::make_shared<DeviceMem<uint8_t>>();

  // 		static bool printed_warning = false;
  // 		if (!printed_warning) {
  // 			printed_warning = true;
  // 			log_warning(
  // 				"DeviceMemArena: GPU {} does not support virtual
  // memory. " 				"Falling back to regular allocations,
  // which will be larger and can cause occasional stutter.",
  // m_device
  // 			);
  // 		}
  // 	}

  // 	DeviceMemArena(DeviceMemArena&& other) = default;
  // 	DeviceMemArena(const DeviceMemArena& other) = delete;
  // 	DeviceMemArena& operator=(DeviceMemArena&& other) = delete;
  // 	DeviceMemArena& operator=(const DeviceMemArena& other) = delete;

  // 	~DeviceMemArena() {
  // 		if (in_use()) {
  // 			log_warning("Attempting to free memory arena while it is
  // still in use.");
  // 		}

  // 		// try {
  // 		// 	// Make sure we're clearing the GPU memory arena on the
  // correct device.
  // 		// 	int previous_device = cuda_device();
  // 		// 	set_cuda_device(m_device);
  // 		// 	ScopeGuard revert_device = {[&]() {
  // set_cuda_device(previous_device); }};

  //     //                     /*
  //     //                     DPCT1003:71: Migrated API does not return error
  //     code.
  //     //                     (*, 0) is inserted. You may need to rewrite this
  //     code.
  //     //                     */
  //     //
  //     CUDA_CHECK_THROW((dpct::get_current_device().queues_wait_and_throw(),
  //     0));

  //     //                     if (m_base_address) {
  // 		// 		total_n_bytes_allocated() -= m_size;

  //     //                             /*
  //     //                             DPCT1007:74: Migration of cuMemUnmap  is
  //     not
  //     //                             supported.
  //     //                             */
  //     // CU_CHECK_THROW(cuMemUnmap(m_base_address, m_size));

  //     //                             for (const auto& handle : m_handles) {
  //     //                                     /*
  //     //                                     DPCT1007:76: Migration of
  //     cuMemRelease
  //     //                                     is not supported.
  //     //                                     */
  //     // CU_CHECK_THROW(cuMemRelease(handle));
  //     //                             }

  //     //                             /*
  //     //                             DPCT1007:77: Migration of
  //     cuMemAddressFree is
  //     //                             not supported.
  //     //                             */
  //     // CU_CHECK_THROW(cuMemAddressFree(m_base_address, m_max_size));
  //     //                     }
  // 		// } catch (const std::runtime_error& error) {
  // 		// 	// Don't need to report on memory-free problems when the
  // driver is shutting down.
  // 		// 	if (std::string{error.what()}.find("driver shutting
  // down") == std::string::npos) {
  // 		// 		log_warning("Could not free memory arena: {}",
  // error.what());
  // 		// 	}
  // 		// }
  // 	}

  // 	uint8_t* data() {
  // 		return m_fallback_memory ? m_fallback_memory->data() :
  // (uint8_t*)m_base_address;
  // 	}

  // 	std::shared_ptr<DeviceMem<uint8_t>> backing_memory() {
  // 		return m_fallback_memory;
  // 	}

  // 	// Finds the smallest interval of free memory in the DeviceMemArena
  // that's
  // 	// large enough to hold the requested number of bytes. Then allocates
  // 	// that memory.
  size_t allocate(size_t n_bytes) {
    // Permitting zero-sized allocations is error prone
    if (n_bytes == 0) {
      n_bytes = m_alignment;
    }

    // Align allocations with the nearest cache line (at least the granularity
    // of the memory allocations)
    n_bytes = next_multiple(n_bytes, m_alignment);

    Interval *best_candidate = &m_free_intervals.back();
    for (auto &f : m_free_intervals) {
      if (f.size() >= n_bytes && f.size() < best_candidate->size()) {
        best_candidate = &f;
      }
    }

    size_t start = best_candidate->start;

    // Note: the += operator can turn `best_candidate` into an empty interval,
    // which is fine because it will be absorbed into adjacent free intervals in
    // later calls to `merge_adjacent_intervals`.
    m_allocated_intervals[start] = best_candidate->start += n_bytes;

    enlarge(size());

    return start;
  }

  // 	void free(size_t start) {
  // 		if (m_allocated_intervals.count(start) == 0) {
  // 			throw std::runtime_error{"Attempted to free arena memory
  // that was not allocated."};
  // 		}

  // 		Interval interval = {start, m_allocated_intervals[start]};
  // 		m_allocated_intervals.erase(start);

  // 		m_free_intervals.insert(
  // 			std::upper_bound(std::begin(m_free_intervals),
  // std::end(m_free_intervals), interval), 			interval
  // 		);

  // 		merge_adjacent_intervals();
  // 	}

  void enlarge(size_t n_bytes) try { return; } catch (...) {
    return;
  };
  // dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // if (n_bytes <= m_size)
  // {
  // 	return;
  // }

  // 		if (cuda_device() != m_device) {
  // 			throw std::runtime_error{fmt::format("Attempted to use a
  // DeviceMemArena of device {} from the wrong device {}.", m_device,
  // cuda_device())};
  // 		}

  // 		log_debug("DeviceMemArena: enlarging from {} to {}",
  // bytes_to_string(m_size), bytes_to_string(n_bytes));

  // 		if (m_fallback_memory) {
  // 			static const double GROWTH_FACTOR = 1.5;

  //                         /*
  //                         DPCT1003:78: Migrated API does not return error
  //                         code.
  //                         (*, 0) is inserted. You may need to rewrite this
  //                         code.
  //                         */
  //                         CUDA_CHECK_THROW((dpct::get_current_device().queues_wait_and_throw(),
  //                         0));

  //                         m_size = next_multiple((size_t)(n_bytes *
  //                         GROWTH_FACTOR), cuda_memory_granularity());
  // 			m_fallback_memory =
  // std::make_shared<DeviceMem<uint8_t>>(m_fallback_memory->copy(m_size));

  //                         /*
  //                         DPCT1003:79: Migrated API does not return error
  //                         code.
  //                         (*, 0) is inserted. You may need to rewrite this
  //                         code.
  //                         */
  //                         CUDA_CHECK_THROW((dpct::get_current_device().queues_wait_and_throw(),
  //                         0));

  //                         return;
  // 		}

  // 		size_t n_bytes_to_allocate = n_bytes - m_size;
  // 		n_bytes_to_allocate = next_multiple(n_bytes_to_allocate,
  // cuda_memory_granularity());

  // 		CUmemAllocationProp prop = {};
  // 		prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  // 		prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  // 		prop.location.id = m_device;

  // 		// m_handles.emplace_back();
  //     //             /*
  //     //             DPCT1007:80: Migration of cuMemCreate is not supported.
  //     //             */
  //     //             CU_CHECK_THROW(cuMemCreate(&m_handles.back(),
  //     n_bytes_to_allocate, &prop, 0));

  //                 CUmemAccessDesc access_desc = {};
  // 		access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  // 		access_desc.location.id = prop.location.id;
  // 		access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  //                 // /*
  //                 // DPCT1007:81: Migration of cuMemMap is not supported.
  //                 // */
  //                 // CU_CHECK_THROW(cuMemMap(m_base_address + m_size,
  //                 //                         n_bytes_to_allocate, 0,
  //                 //                         m_handles.back(), 0));
  //                 // /*
  //                 // DPCT1007:82: Migration of cuMemSetAccess is not
  //                 supported.
  //                 // */
  //                 // CU_CHECK_THROW(cuMemSetAccess(m_base_address + m_size,
  //                 //                               n_bytes_to_allocate,
  //                 &access_desc,
  //                 //                               1));
  //                 m_size += n_bytes_to_allocate;

  // 		total_n_bytes_allocated() += n_bytes_to_allocate;

  // 		// Need to synchronize the device to make sure memory is
  // available to all streams. 		if (current_capture()) {
  // 			current_capture()->schedule_synchronize();
  // 		} else {
  //                         /*
  //                         DPCT1003:83: Migrated API does not return error
  //                         code.
  //                         (*, 0) is inserted. You may need to rewrite this
  //                         code.
  //                         */
  //                         CUDA_CHECK_THROW((dpct::get_current_device().queues_wait_and_throw(),
  //                         0));
  //                 }
  //         }
  //         catch (sycl::exception const &exc) {
  //           std::cerr << exc.what() << "Exception caught at file:" <<
  //           __FILE__
  //                     << ", line:" << __LINE__ << std::endl;
  //           std::exit(1);
  //         }

  size_t size() const { return m_free_intervals.back().start; }

  // 	bool in_use() const {
  // 		return m_free_intervals.size() != 1 ||
  // m_free_intervals.front().size() != m_max_size;
  // 	}

  // 	class Allocation {
  // 	public:
  // 		Allocation() = default;
  //                 Allocation(dpct::queue_ptr stream, size_t offset,
  //                            const std::shared_ptr<DeviceMemArena>
  //                            &workspace)
  //                     : m_stream{stream}, m_data{workspace->data() + offset},
  //                       m_offset{offset}, m_workspace{workspace},
  //                       m_backing_memory{workspace->backing_memory()}
  //                 {}

  // 		~Allocation() {
  // 			if (m_workspace) {
  // 				m_workspace->free(m_offset);
  // 			}
  // 		}

  // 		Allocation(const Allocation& other) = delete;

  // 		Allocation& operator=(Allocation&& other) {
  // 			std::swap(m_stream, other.m_stream);
  // 			std::swap(m_data, other.m_data);
  // 			std::swap(m_offset, other.m_offset);
  // 			std::swap(m_workspace, other.m_workspace);
  // 			std::swap(m_backing_memory, other.m_backing_memory);
  // 			return *this;
  // 		}

  // 		Allocation(Allocation&& other) {
  // 			*this = std::move(other);
  // 		}

  // 		uint8_t* data() {
  // 			return m_data;
  // 		}

  // 		const uint8_t* data() const {
  // 			return m_data;
  // 		}

  //                 dpct::queue_ptr stream() const {
  //                         return m_stream;
  // 		}

  // 	private:
  //                 dpct::queue_ptr m_stream = &dpct::get_default_queue();
  //                 uint8_t* m_data = nullptr;
  // 		size_t m_offset = 0;
  // 		std::shared_ptr<DeviceMemArena> m_workspace = nullptr;

  // 		// Backing DeviceMem (if backed by a DeviceMem). Ensures that
  // 		// the backing memory is only freed once all allocations that
  // 		// use it were destroyed.
  // 		std::shared_ptr<DeviceMem<uint8_t>> m_backing_memory = nullptr;
  // 	};

  // private:
  // 	void merge_adjacent_intervals() {
  // 		size_t j = 0;
  // 		for (size_t i = 1; i < m_free_intervals.size(); ++i) {
  // 			Interval& prev = m_free_intervals[j];
  // 			Interval& cur = m_free_intervals[i];

  // 			if (prev.end == cur.start) {
  // 				prev.end = cur.end;
  // 			} else {
  // 				++j;
  // 				m_free_intervals[j] = m_free_intervals[i];
  // 			}
  // 		}
  // 		m_free_intervals.resize(j+1);
  // 	}

  std::vector<Interval> m_free_intervals;
  std::unordered_map<size_t, size_t> m_allocated_intervals;

  // 	int m_device = 0;
  //         dpct::device_ptr m_base_address = {};
  //         size_t m_size = 0;

  // 	std::vector<CUmemGenericAllocationHandle> m_handles;

  // 	// Used then virtual memory isn't supported.
  // 	// Requires more storage + memcpy, but is more portable.
  // 	std::shared_ptr<DeviceMem<uint8_t>> m_fallback_memory = nullptr;
  size_t m_alignment;
  size_t m_max_size;
  // };
};

inline std::unordered_map<dpct::queue_ptr, std::shared_ptr<DeviceMemArena>> &
stream_gpu_memory_arenas() {
  static auto *stream_gpu_memory_arenas =
      new std::unordered_map<dpct::queue_ptr,
                             std::shared_ptr<DeviceMemArena>>{};
  return *stream_gpu_memory_arenas;
}

inline std::unordered_map<int, std::shared_ptr<DeviceMemArena>> &
global_gpu_memory_arenas() {
  static auto *global_gpu_memory_arenas =
      new std::unordered_map<int, std::shared_ptr<DeviceMemArena>>{};
  return *global_gpu_memory_arenas;
}

inline DeviceMemArena::Allocation allocate_workspace(dpct::queue_ptr stream,
                                                     size_t n_bytes) {
  if (n_bytes == 0) {
    // Return a null allocation if no bytes were requested.
    return {};
  }

  auto &arena = stream ? stream_gpu_memory_arenas()[stream]
                       : global_gpu_memory_arenas()[cuda_device()];
  if (!arena) {
    arena = std::make_shared<DeviceMemArena>();
  }
  return DeviceMemArena::Allocation{stream, arena->allocate(n_bytes), arena};
}