#include "DeviceMem.h"

#include "common_host.h"

#ifdef __SYCL_DEVICE_ONLY__

#define CONSTANT __attribute__((opencl_constant))

#else

#define CONSTANT

#endif
void get_float(float value, int &integer_val, int &fractional_val) {
  // careful with the code. Leading zeros not shown in fractional_val. This is
  // only to debug whether it's zero or non-zero. only for 4 decimals after
  // comma

  // Extract the integer part
  int integerPart = static_cast<int>(value);

  // Extract the fractional part as an integer
  int fractionalPart =
      //   static_cast<int>(std::fabs((value - static_cast<float>(integerPart))
      //   *
      static_cast<int>(((value - static_cast<float>(integerPart)) *
                        1000000));  // Adjust the multiplier as needed
  integer_val = integerPart;
  fractional_val = fractionalPart;
}
// Using namespace and alias for bfloat16
using namespace sycl;
using bf16 = sycl::ext::oneapi::bfloat16;

/**
 * Constructor for the SwiftNetMLP class.
 *
 */
template <typename T>
DeviceMem<T>::DeviceMem() {}

/**
 * Constructor for the SwiftNetMLP class.
 *
 * @param size               Size of the memory to allocate.
 * @param queue              SYCL queue associated with the object.
 */
template <typename T>
DeviceMem<T>::DeviceMem(int size, sycl::queue &q) {
  if (m_size != 0 || size <= 0) {
    return;
  }
  m_size = size;
  m_data = malloc_device<T>(size, q);
}
/**
 * Allocate memory for a DeviceMem object.
 *
 * @param size               Size of the memory to allocate.
 * @param queue              SYCL queue associated with the object.
 */
template <typename T>
void DeviceMem<T>::allocate2(int size, queue &q) {
  if (m_size != 0 || size <= 0) {
    return;
  }
  m_size = size;
  m_data = malloc_device<T>(size, q);
}

/**
 * Allocate memory for a DeviceMem object.
 *
 * @param size               Size of the memory to allocate.
 * @param queue              SYCL queue associated with the object.
 */
template <typename T>
void DeviceMem<T>::allocate(int n_bytes, queue &q) {
  if (m_size != 0 || n_bytes <= 0) {
    return;
  }
  // m_size = size;
  // m_data = malloc_device<T>(size, q);

  // if (n_bytes == 0) return;

  log_debug("DeviceMem: allocating {}.", bytes_to_string(n_bytes));

  uint8_t *rawptr{
      m_managed
          ? sycl::malloc_shared<uint8_t>(n_bytes + DEBUG_GUARD_SIZE * 2, q)
          : sycl::malloc_device<uint8_t>(n_bytes + DEBUG_GUARD_SIZE * 2, q)};
  if (!rawptr)
    throw std::runtime_error{"Allocation failed in line __LINE__ in __FILE__."};

#if DEBUG_GUARD_SIZE > 0
  q.memset(rawptr, 0xFF, DEBUG_GUARD_SIZE);
  q.memset(rawptr + n_bytes + DEBUG_GUARD_SIZE, 0xFE, DEBUG_GUARD_SIZE);
#endif

  rawptr += DEBUG_GUARD_SIZE;
  m_data = reinterpret_cast<T *>(rawptr);
  total_n_bytes_allocated() += n_bytes;
}

/**
 * Allocate memory for a DeviceMem object.
 *
 * @param size               Size of the memory to allocate.
 * @param queue              SYCL queue associated with the object.
 */
template <typename T>
void DeviceMem<T>::allocate(int n_bytes) {
  if (m_size != 0 || n_bytes <= 0) {
    return;
  }
  // m_size = size;
  // m_data = malloc_device<T>(size, q);

  // if (n_bytes == 0) return;

  log_debug("DeviceMem: allocating {}.", bytes_to_string(n_bytes));

  sycl::queue &q{dpct::get_default_queue()};
  uint8_t *rawptr{
      m_managed
          ? sycl::malloc_shared<uint8_t>(n_bytes + DEBUG_GUARD_SIZE * 2, q)
          : sycl::malloc_device<uint8_t>(n_bytes + DEBUG_GUARD_SIZE * 2, q)};
  if (!rawptr)
    throw std::runtime_error{"Allocation failed in line __LINE__ in __FILE__."};

#if DEBUG_GUARD_SIZE > 0
  q.memset(rawptr, 0xFF, DEBUG_GUARD_SIZE);
  q.memset(rawptr + n_bytes + DEBUG_GUARD_SIZE, 0xFE, DEBUG_GUARD_SIZE);
#endif

  rawptr += DEBUG_GUARD_SIZE;
  m_data = reinterpret_cast<T *>(rawptr);
  total_n_bytes_allocated() += n_bytes;
}

/**
 * Free memory for a DeviceMem object.
 *
 * @param queue              SYCL queue associated with the object.
 */
template <typename T>
void DeviceMem<T>::free_mem(queue &q) {
  if (!m_data) return;

  uint8_t *rawptr = reinterpret_cast<uint8_t *>(m_data);
  rawptr -= DEBUG_GUARD_SIZE;

  sycl::free(rawptr, q);

  total_n_bytes_allocated() -= get_bytes();

  m_data = nullptr;
  m_size = 0;
}

/**
 * Free memory for a DeviceMem object.
 *
 */
template <typename T>
void DeviceMem<T>::free_mem() {
  if (!m_data) return;

  uint8_t *rawptr = reinterpret_cast<uint8_t *>(m_data);
  rawptr -= DEBUG_GUARD_SIZE;

  sycl::free(rawptr, dpct::get_default_queue());

  total_n_bytes_allocated() -= get_bytes();

  m_data = nullptr;
  m_size = 0;
}

/**
 * Copy data from host to DeviceMem object.
 *
 * @param data               Array to copy the data from.
 * @param n                  Size of the data to copy.
 * @param queue              SYCL queue associated with the object.
 */
template <typename T>
void DeviceMem<T>::copy_from_host(std::vector<T> &data, int n, queue q) {
  q.memcpy(m_data, data.data(), n * sizeof(T)).wait();
}

/**
 * Copy data from DeviceMem object to host.
 *
 * @param data               Array to copy the data to.
 * @param n                  Size of the data to copy.
 * @param queue              SYCL queue associated with the object.
 */
template <typename T>
void DeviceMem<T>::copy_to_host(std::vector<T> &data, int n, queue q) {
  q.memcpy(data.data(), m_data, n * sizeof(T)).wait();
}

/**
 * Copy data from host to DeviceMem object. The size copied is the size of the
 * DeviceMem object.
 *
 * @param data               Array to copy the data from.
 * @param queue              SYCL queue associated with the object.
 */
template <typename T>
void DeviceMem<T>::copy_from_host(std::vector<T> &data, queue q) {
  copy_from_host(data, m_size, q);
}

/**
 * Copy data from DeviceMem object to host. The size copied is the size of the
 * DeviceMem object.
 *
 * @param data               Array to copy the data to.
 * @param n                  Size of the data to copy.
 * @param queue              SYCL queue associated with the object.
 */
template <typename T>
void DeviceMem<T>::copy_to_host(std::vector<T> &data, queue q) {
  copy_to_host(data, m_size, q);
}

/**
 * Set data to a specific id of the DeviceMem object.
 *
 * @param id                 Index to set the data.
 * @param value              Value to set.
 */
template <typename T>
void DeviceMem<T>::set_values(int size, float *array, queue &q) {
  auto local_m_data = m_data;
  q.parallel_for<>(range<1>(size), [=](id<1> idx) {
     local_m_data[idx] = (bf16)(array[idx]);
   }).wait();
}

/**
 * Set data to a specific id of the DeviceMem object.
 *
 * @param id                 Index to set the data.
 * @param value              Value to set.
 */
template <typename T>
void DeviceMem<T>::set_data(int id, T value) {
  m_data[id] = value;
}

/**
 * Initialize the device memory with values drawn from a normal distribution and
 * generate the transposed version.
 *
 * This function initializes the device memory with random values drawn from a
 * normal distribution with the specified standard deviation. It also generates
 * the transposed version of the weight matrices and stores them in the provided
 * transposed memory.
 *
 * @param dev            The standard deviation of the normal distribution.
 * @param transposed     The device memory to store the transposed weight
 * matrices.
 * @param input_width    The width of the input layer.
 * @param width          The width of the hidden layer weight matrices.
 * @param output_width   The width of the output layer.
 * @param n_hidden       The number of hidden layers in the network.
 * @param q              The SYCL queue for parallel computation.
 */
template <typename T>
void DeviceMem<T>::initialize_normal(double dev, DeviceMem<T> &transposed,
                                     int input_width, int width,
                                     int output_width, int n_hidden, queue q) {
  auto p = m_data;
  std::default_random_engine gen;
  std::normal_distribution<double> distrib(0.0, dev);
  T rnd;
  std::vector<T> data(m_size);
  std::vector<T> dataT(m_size);
  for (int i = 0; i < input_width; i++) {
    for (int j = 0; j < width; j++) {
      rnd = (T)distrib(gen);
      data[toPackedLayoutCoord(i * width + j, input_width, width)] = rnd;
      dataT[toPackedLayoutCoord(j * width + i, width, input_width)] = rnd;
    }
  }
  for (int k = 0; k < n_hidden; k++) {
    for (int i = 0; i < width; i++) {
      for (int j = 0; j < width; j++) {
        rnd = (T)distrib(gen);
        data[input_width * width + k * width * width +
             toPackedLayoutCoord(i * width + j, width, width)] = rnd;
        dataT[input_width * width + k * width * width +
              toPackedLayoutCoord(j * width + i, width, width)] = rnd;
      }
    }
  }
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < output_width; j++) {
      rnd = (T)distrib(gen);
      data[input_width * width + n_hidden * width * width +
           toPackedLayoutCoord(i * output_width + j, width, output_width)] =
          rnd;
      dataT[input_width * width + n_hidden * width * width +
            toPackedLayoutCoord(j * output_width + i, output_width, width)] =
          rnd;
    }
  }
  buffer<T, 1> buf(data.data(), data.size());
  buffer<T, 1> bufT(dataT.data(), dataT.size());
  q.submit([&](handler &h) {
     auto acc = buf.get_access(h);
     auto accT = bufT.get_access(h);
     h.parallel_for(m_size, [=](id<1> idx) {
       p[idx] = acc[idx];
       transposed.data()[idx] = accT[idx];
     });
   }).wait();
}

/**
 * Initialize the device memory with values drawn from a normal distribution.
 *
 * This function initializes the device memory with random values drawn from a
 * normal distribution with the specified standard deviation.
 *
 * @param dev   The standard deviation of the normal distribution.
 * @param q     The SYCL queue for parallel computation.
 */
template <typename T>
void DeviceMem<T>::initialize_normal(double dev, queue q) {
  std::default_random_engine gen;
  std::normal_distribution<double> distrib(0.0, dev);
  std::vector<T> data(m_size);
  for (int i = 0; i < m_size; i++) {
    data[i] = (T)distrib(gen);
    // std:dat:cout << "data: " << data[i] << std::endl;
  }
  q.memcpy(m_data, data.data(), m_size * sizeof(T)).wait();
}

/**
 * Initialize the device memory with values drawn from a uniform distribution.
 *
 * This function initializes the device memory with random values drawn from a
 * uniform distribution within the specified scale range.
 *
 * @param scale         The range for generating uniform random values.
 * @param transposed    The transposed version of the initialized data.
 * @param input_width   The width of the input.
 * @param width         The width of the layer.
 * @param output_width  The width of the output.
 * @param n_hidden      The number of hidden layers.
 * @param q             The SYCL queue for parallel computation.
 */
template <typename T>
void DeviceMem<T>::initialize_uniform(double scale, DeviceMem<T> &transposed,
                                      int input_width, int width,
                                      int output_width, int n_hidden, queue q) {
  //   std::cout << "WRONG" << std::endl;
  auto p = m_data;
  std::default_random_engine gen;
  std::uniform_real_distribution<double> distrib(0.0, scale);

  T rnd;

  std::vector<T> data(m_size);
  std::vector<T> dataT(m_size);
  for (int i = 0; i < input_width; i++) {
    for (int j = 0; j < width; j++) {
      rnd = (T)distrib(gen);
      data[toPackedLayoutCoord(i * width + j, input_width, width)] = rnd;
      dataT[toPackedLayoutCoord(j * width + i, width, input_width)] = rnd;

      //   std::cout << "data: "
      //             << toPackedLayoutCoord(i * width + j, input_width, width)
      //             << std::endl;
      //   std::cout << "dataT: "
      //             << toPackedLayoutCoord(j * width + i, width, input_width)
      //             << std::endl;
    }
  }
  //   std::cout << "+++++++++" << std::endl;
  for (int k = 0; k < n_hidden; k++) {
    for (int i = 0; i < width; i++) {
      for (int j = 0; j < width; j++) {
        rnd = (T)distrib(gen);
        data[input_width * width + k * width * width +
             toPackedLayoutCoord(i * width + j, width, width)] = rnd;
        dataT[input_width * width + k * width * width +
              toPackedLayoutCoord(j * width + i, width, width)] = rnd;

        // std::cout << "data: "
        //   << input_width * width + k * width * width +
        //  toPackedLayoutCoord(i * width + j, width, width)
        //   << std::endl;
        // std::cout << "dataT: "
        //           << input_width * width + k * width * width +
        //                  toPackedLayoutCoord(j * width + i, width, width)
        //           << std::endl;
      }
    }
  }
  //   std::cout << "+++++++++" << std::endl;

  for (int i = 0; i < width; i++) {
    for (int j = 0; j < output_width; j++) {
      rnd = (T)distrib(gen);
      data[input_width * width + n_hidden * width * width +
           toPackedLayoutCoord(i * output_width + j, width, output_width)] =
          rnd;
      dataT[input_width * width + n_hidden * width * width +
            toPackedLayoutCoord(j * width + i, output_width, width)] = rnd;
      //   std::cout << "data: "
      //             << input_width * width + n_hidden * width * width +
      //                    toPackedLayoutCoord(i * output_width + j, width,
      //                                        output_width)
      //             << std::endl;
      //   std::cout << "dataT: "
      //             << input_width * width + n_hidden * width * width +
      //                    toPackedLayoutCoord(j * width + i, output_width,
      //                    width)
      //             << std::endl;
    }
  }
  buffer<T, 1> buf(data.data(), data.size());

  //   std::cout << "Crashes here1" << std::endl;
  buffer<T, 1> bufT(dataT.data(), dataT.size());
  //   std::cout << "Crashes here2" << std::endl;
  q.submit([&](handler &h) {
     auto acc = buf.get_access(h);
     auto accT = bufT.get_access(h);
     h.parallel_for(m_size, [=](id<1> idx) {
       p[idx] = acc[idx];
       transposed.data()[idx] = accT[idx];
     });
   }).wait();
  //   std::cout << "NO ISSUE2" << std::endl;
}

/**
 * Create the transposed version of the data and store it in the provided
 * memory.
 *
 * This function calculates the transposed version of the data and stores it in
 * the provided `transposed` memory. It takes into account the layout of input,
 * hidden, and output layers and performs the transposition accordingly.
 *
 * @param transposed    The memory to store the transposed data.
 * @param input_width   The width of the input.
 * @param width         The width of the layer.
 * @param output_width  The width of the output.
 * @param n_hidden      The number of hidden layers.
 * @param q             The SYCL queue for parallel computation.
 */
template <typename T>
void DeviceMem<T>::make_transposed(DeviceMem<T> &transposed, int input_width,
                                   int width, int output_width, int n_hidden,
                                   queue q) {
  auto p = m_data;

  q.parallel_for<>(
       range<1>(input_width * width + n_hidden * width * width +
                width * output_width),
       [=](id<1> idx) {
         int i = 0;
         int j = 0;
         int mat_num = 0;
         int mat_offset = 0;

         if (idx < input_width * width) {
           i = idx / input_width;
           j = idx % input_width;
           transposed
               .data()[toPackedLayoutCoord(j * width + i, input_width, width)] =
               p[toPackedLayoutCoord(i * input_width + j, width, input_width)];
           //   int b_first;
           //   int b_second;
           //   static const CONSTANT char FMT[] =
           //       "Input Writing from %d to %d: %d.%d,\n";
           //   get_float(p[toPackedLayoutCoord(i * width + j, input_width,
           //   width)],
           //             b_first, b_second);
           //   sycl::ext::oneapi::experimental::printf(
           //       FMT, toPackedLayoutCoord(i * width + j, input_width, width),
           //       toPackedLayoutCoord(j * width + i, width, input_width),
           //       b_first, b_second);

         } else if (idx < input_width * width + n_hidden * width * width) {
           mat_num = (idx - input_width * width) / (width * width);
           mat_offset = (idx - input_width * width) % (width * width);
           i = mat_offset / width;
           j = mat_offset % width;

           //   static const CONSTANT char FMT[] =
           //       "Writing from %d, idx: %d i: %d, j: %d, mat offset: %d,
           //       packed: "
           //       "%d\n";
           //   sycl::ext::oneapi::experimental::printf(
           //       FMT, input_width * width + mat_num * width * width,
           //       int(idx), i, j, mat_offset, toPackedLayoutCoord(j * width +
           //       i, width, width));
           transposed.data()[input_width * width + mat_num * width * width +
                             toPackedLayoutCoord(j * width + i, width, width)] =
               p[input_width * width + mat_num * width * width +
                 toPackedLayoutCoord(i * width + j, width, width)];
         } else {
           mat_offset = (idx - input_width * width - n_hidden * width * width) %
                        (width * output_width);
           i = mat_offset / output_width;
           j = mat_offset % output_width;
           transposed.data()[input_width * width + n_hidden * width * width +
                             toPackedLayoutCoord(j * width + i, output_width,
                                                 width)] =
               p[input_width * width + n_hidden * width * width +
                 toPackedLayoutCoord(i * output_width + j, width,
                                     output_width)];

           //   int b_first;
           //   int b_second;
           //   static const CONSTANT char FMT[] =
           //       "Out Writing from %d to %d  i: %d, j: %d\n";
           //   get_float(
           //       transposed.data()[input_width * width + n_hidden * width *
           //       width +
           //                         toPackedLayoutCoord(i * output_width + j,
           //                         width,
           //                                             output_width)],
           //       b_first, b_second);
           //   sycl::ext::oneapi::experimental::printf(
           //       FMT,
           //       toPackedLayoutCoord(i * output_width + j, width,
           //       output_width), toPackedLayoutCoord(j * width + i,
           //       output_width, width), i, j);
         }
       })
      .wait();
}

/**
 * Initialize the device memory with values sampled from a uniform distribution.
 *
 * This function generates random values sampled from a uniform distribution
 * within the specified scale and initializes the device memory with those
 * values.
 *
 * @param q       The SYCL queue for memory operations.
 * @param scale   The scale of the uniform distribution.
 */
template <typename T>
void DeviceMem<T>::initialize_uniform(queue q, double scale) {
  std::default_random_engine gen;
  std::uniform_real_distribution<double> distrib(0.0, scale);
  std::vector<T> data(m_size);

  for (int i = 0; i < m_size; i++) {
    data[i] = (T)distrib(gen);
  }
  q.memcpy(m_data, data.data(), m_size * sizeof(T)).wait();
}
/**
 * Initialize the device memory with values sampled from a Xavier uniform
 * distribution.
 *
 * This function generates random values sampled from a Xavier uniform
 * distribution and initializes both the device memory and its transposed
 * version with those values.
 *
 * @param transposed    The memory to store the transposed data.
 * @param input_width   The width of the input.
 * @param width         The width of the layer.
 * @param output_width  The width of the output.
 * @param n_hidden      The number of hidden layers.
 * @param q             The SYCL queue for memory operations.
 */
template <typename T>
void DeviceMem<T>::initialize_xavier_unif(DeviceMem<T> &transposed,
                                          int input_width, int width,
                                          int output_width, int n_hidden,
                                          queue q) {
  double x = sqrt(6.0 / ((double)(input_width + output_width)));
  initialize_uniform(x, transposed, input_width, width, output_width, n_hidden,
                     q);
}
/**
 * Initialize the device memory with values sampled from a Xavier uniform
 * distribution.
 *
 * This function generates random values sampled from a Xavier uniform
 * distribution and initializes the device memory with those values.
 *
 * @param input_width   The width of the input.
 * @param output_width  The width of the output.
 * @param q             The SYCL queue for memory operations.
 */
template <typename T>
void DeviceMem<T>::initialize_xavier_unif(int input_width, int output_width,
                                          queue q) {
  double x = sqrt(6.0 / ((double)(input_width + output_width)));
  initialize_uniform(q, x);
}
/**
 * Initialize the device memory with values sampled from a Xavier normal
 * distribution.
 *
 * This function generates random values sampled from a Xavier normal
 * distribution and initializes both the device memory and its transposed
 * version with those values.
 *
 * @param transposed    The memory to store the transposed data.
 * @param input_width   The width of the input.
 * @param width         The width of the layer.
 * @param output_width  The width of the output.
 * @param n_hidden      The number of hidden layers.
 * @param q             The SYCL queue for memory operations.
 */
template <typename T>
void DeviceMem<T>::inititialize_xavier_normal(DeviceMem<T> &transposed,
                                              int input_width, int width,
                                              int output_width, int n_hidden,
                                              queue q) {
  double dev = sqrt(2.0 / ((double)(input_width + output_width)));
  initialize_normal(dev, transposed, input_width, width, output_width, n_hidden,
                    q);
}

/**
 * Initialize the device memory with values sampled from a Xavier normal
 * distribution.
 *
 * This function generates random values sampled from a Xavier normal
 * distribution and initializes the device memory with those values.
 *
 * @param input_width   The width of the input.
 * @param output_width  The width of the output.
 * @param q             The SYCL queue for memory operations.
 */
template <typename T>
void DeviceMem<T>::initialize_xavier_normal(int input_width, int output_width,
                                            queue q) {
  double dev = sqrt(2.0 / ((double)(input_width + output_width)));
  initialize_normal(dev, q);
}

/**
 * Initialize the device memory with a constant value and its transposed
 * version.
 *
 * This function initializes both the device memory and its transposed version
 * with a constant value.
 *
 * @param constant      The constant value to be filled in the memory.
 * @param transposed    The memory to store the transposed data.
 * @param q             The SYCL queue for memory operations.
 */
template <typename T>
void DeviceMem<T>::initialize_constant(T constant, DeviceMem<T> &transposed,
                                       queue q) {
  std::vector<T> data(m_size, constant);
  q.memcpy(m_data, data.data(), m_size * sizeof(T)).wait();
  q.memcpy(transposed.data(), data.data(), m_size * constant * sizeof(T))
      .wait();
}

template <typename T>
void DeviceMem<T>::initialize_arange(queue q) {
  std::vector<T> data(m_size);

  // Repeat the col_vector and perform the operations
  for (int i = 0; i < data.size(); i++) {
    data[i] = static_cast<T>((i - m_size / 2)) / static_cast<T>(m_size / 2);
    // std::cout << "Writing at idx: " << i << ": " << data[i] << ", ("
    //           << static_cast<T>((i - m_size / 2)) / static_cast<T>(m_size /
    //           2)
    //           << "), m_size: " << m_size << std::endl;
  }
  q.memcpy(m_data, data.data(), m_size * sizeof(T)).wait();
}
template <typename T>
void DeviceMem<T>::initialize_arange(queue q, int input_width, int net_width,
                                     int out_width, int hidden_matrices) {
  std::vector<T> data(m_size);

  // input
  //  Create a 1D vector that goes from 1 to input_width
  std::vector<T> col_vector;
  for (int i = 1; i <= input_width; ++i) {
    col_vector.push_back(static_cast<T>(i) - 32);
  }

  // Repeat the col_vector and perform the operations
  for (int i = 0; i < net_width; i++) {
    for (int j = 0; j < input_width; j++) {
      data[j * input_width + i] = col_vector[j] * 0.01;
    }
  }
  // middle layers
  // Create a 1D vector that goes from 1 to input_width
  std::vector<T> col_vector_mid;
  for (int i = 1; i <= net_width; ++i) {
    col_vector_mid.push_back(static_cast<T>(i) - 32);
  }
  for (int k = 0; k < hidden_matrices; ++k) {
    // Repeat the col_vector and perform the operations
    for (int i = 0; i < net_width; ++i) {
      for (int j = 0; j < net_width; ++j) {
        // data[net_width * input_width + k * net_width * net_width +
        //      i * net_width + j] = col_vector_mid[j] * 0.001;
        data[net_width * input_width + k * net_width * net_width +
             j * net_width + i] = col_vector_mid[j] * 0.01;
      }
    }
  }

  // output

  // Create a 1D vector that goes from 1 to input_width
  std::vector<T> col_vector_out;
  for (int i = 1; i <= out_width; ++i) {
    col_vector_out.push_back(static_cast<T>(i) - 32);
  }
  // Repeat the col_vector and perform the operations
  for (int i = 0; i < net_width; i++) {
    for (int j = 0; j < out_width; j++) {
      //   data[net_width * input_width + hidden_matrices * net_width *
      //   net_width +
      //        j * out_width + i] = col_vector_out[j] * 0.001;
      data[net_width * input_width + hidden_matrices * net_width * net_width +
           i * out_width + j] = col_vector_out[j] * 0.01;
    }
  }

  q.memcpy(m_data, data.data(), m_size * sizeof(T)).wait();
}
/**
 * Initialize the device memory with a constant value.
 *
 * This function initializes the device memory with a constant value.
 *
 * @param constant      The constant value to be filled in the memory.
 * @param q             The SYCL queue for memory operations.
 */
template <typename T>
void DeviceMem<T>::initialize_constant(T constant, queue q) {
  auto p = m_data;
  q.parallel_for<>(range<1>(m_size), [=](id<1> idx) {
     p[idx] = (T)constant;
   }).wait();
}

/**
 * Initialize the device memory with values sampled from a He normal
 * distribution.
 *
 * This function generates random values sampled from a He normal distribution
 * and initializes both the device memory and its transposed version with those
 * values.
 *
 * @param transposed    The memory to store the transposed data.
 * @param input_width   The width of the input.
 * @param width         The width of the layer.
 * @param output_width  The width of the output.
 * @param n_hidden      The number of hidden layers.
 * @param q             The SYCL queue for memory operations.
 */
template <typename T>
void DeviceMem<T>::intitialize_he_normal(DeviceMem<T> &transposed,
                                         int input_width, int width,
                                         int output_width, int n_hidden,
                                         queue q) {
  double dev = sqrt(2.0 / width);
  initialize_normal(dev, transposed, input_width, width, output_width, n_hidden,
                    q);
}

/**
 * Initialize the device memory with values sampled from a He normal
 * distribution.
 *
 * This function generates random values sampled from a He normal distribution
 * and initializes the device memory with those values.
 *
 * @param input_width   The width of the input.
 * @param q             The SYCL queue for memory operations.
 */
template <typename T>
void DeviceMem<T>::intitialize_he_normal(int input_width, queue q) {
  double dev = sqrt(2.0 / input_width);
  initialize_normal(dev, q);
}

template <typename T>
void DeviceMem<T>::allocate_memory(size_t n_bytes) try {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  if (n_bytes == 0) {
    return;
  }

  // fmt::print("GPUMemory: allocating {}.", bytes_to_string(n_bytes));

  uint8_t *rawptr = nullptr;
  if (m_managed) {
    /*
    DPCT1003:63: Migrated API does not return error code.
    (*, 0) is inserted. You may need to rewrite this code.
    */
    /*
    DPCT1064:110: Migrated cudaMallocManaged call is used in
    a macro definition and is not valid for all macro uses.
    Adjust the code.
    */
    CUDA_CHECK_THROW(
        (rawptr = (uint8_t *)sycl::malloc_shared(n_bytes + DEBUG_GUARD_SIZE * 2,
                                                 dpct::get_default_queue()),
         0));
  } else {
    /*
    DPCT1003:64: Migrated API does not return error code.
    (*, 0) is inserted. You may need to rewrite this code.
    */
    /*
    DPCT1064:111: Migrated cudaMalloc call is used in a
    macro definition and is not valid for all macro uses.
    Adjust the code.
    */
    CUDA_CHECK_THROW(
        (rawptr = (uint8_t *)sycl::malloc_device(n_bytes + DEBUG_GUARD_SIZE * 2,
                                                 dpct::get_default_queue()),
         0));
  }
#if DEBUG_GUARD_SIZE > 0
  CUDA_CHECK_THROW(cudaMemset(rawptr, 0xff, DEBUG_GUARD_SIZE));
  CUDA_CHECK_THROW(
      cudaMemset(rawptr + n_bytes + DEBUG_GUARD_SIZE, 0xfe, DEBUG_GUARD_SIZE));
#endif
  if (rawptr) rawptr += DEBUG_GUARD_SIZE;
  m_data = (T *)(rawptr);
  total_n_bytes_allocated() += n_bytes;
} catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

template <typename T>
void DeviceMem<T>::resize(const size_t size) {
  if (size == m_size) return;
  free_mem();
  allocate_memory(size * sizeof(T));
  m_size = size;
}

template <typename T>
void DeviceMem<T>::enlarge(const size_t size) {
  if (size > m_size) resize(size);
}

/** @name Memset
 *  @{
 */
/// Sets the memory of the first num_elements to value
template <typename T>
void DeviceMem<T>::memset(const int value, const size_t num_elements,
                          const size_t offset) {
  if (num_elements + offset > m_size) {
    throw std::runtime_error{
        fmt::format("Could not set memory: Number of elements {}+{} larger "
                    "than allocated memory {}.",
                    num_elements, offset, m_size)};
  }

  dpct::get_default_queue().memset(m_data + offset, value,
                                   num_elements * sizeof(T));
}

/// Sets the memory of the all elements to value
template <typename T>
void DeviceMem<T>::memset(const int value) {
  memset(value, m_size);
}
/** @} */

template class DeviceMem<float>;
template class DeviceMem<bf16>;
template class DeviceMem<uint8_t>;