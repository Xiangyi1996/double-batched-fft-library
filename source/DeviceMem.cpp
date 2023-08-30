#include "DeviceMem.h"

// Using namespace and alias for bfloat16
using namespace sycl;
using bf16 = sycl::ext::oneapi::bfloat16;

/**
 * Constructor for the SwiftNetMLP class.
 *
 */
template<typename T>
DeviceMem<T>::DeviceMem() {}

/**
 * Constructor for the SwiftNetMLP class.
 *
 * @param size               Size of the memory to allocate.
 * @param queue              SYCL queue associated with the object.
 */
template<typename T>
DeviceMem<T>::DeviceMem(int size, queue q) {
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
template<typename T>
void DeviceMem<T>::allocate(int size, queue q) {
	if (m_size != 0 || size <= 0) {
		return;
	}
	m_size = size;
	m_data = malloc_device<T>(size, q);
}

/**
 * Free memory for a DeviceMem object.
 *
 * @param queue              SYCL queue associated with the object.
 */
template<typename T>
void DeviceMem<T>::free_mem(queue q) {
	m_size = 0;
	free(m_data, q);
}

/**
 * Free memory for a DeviceMem object.
 *
 */
template<typename T>
void DeviceMem<T>::free_mem() {
	m_size = 0;
	free(m_data);
}

/**
 * Copy data from host to DeviceMem object.
 *
 * @param data               Array to copy the data from.
 * @param n                  Size of the data to copy.
 * @param queue              SYCL queue associated with the object.
 */
template<typename T>
void DeviceMem<T>::copy_from_host(std::vector<T>& data, int n, queue q) {
	q.memcpy(m_data, data.data(), n * sizeof(T));
	q.wait();
}

/**
 * Copy data from DeviceMem object to host.
 *
 * @param data               Array to copy the data to.
 * @param n                  Size of the data to copy.
 * @param queue              SYCL queue associated with the object.
 */
template<typename T>
void DeviceMem<T>::copy_to_host(std::vector<T>& data, int n, queue q) {
	q.memcpy(data.data(), m_data, n * sizeof(T));
	q.wait();
}

/**
 * Copy data from host to DeviceMem object. The size copied is the size of the DeviceMem object.
 *
 * @param data               Array to copy the data from.
 * @param queue              SYCL queue associated with the object.
 */
template<typename T>
void DeviceMem<T>::copy_from_host(std::vector<T>& data, queue q) {
	copy_from_host(data, m_size, q);
}

/**
 * Copy data from DeviceMem object to host. The size copied is the size of the DeviceMem object.
 *
 * @param data               Array to copy the data to.
 * @param n                  Size of the data to copy.
 * @param queue              SYCL queue associated with the object.
 */
template<typename T>
void DeviceMem<T>::copy_to_host(std::vector<T>& data, queue q) {
	copy_to_host(data, m_size, q);
}

/**
 * Set data to a specific id of the DeviceMem object.
 *
 * @param id                 Index to set the data.
 * @param value              Value to set.
 */
template<typename T>
void DeviceMem<T>::set_data(int id, T value) {
	m_data[id] = value;
}

/**
 * Initialize the device memory with values drawn from a normal distribution and generate the transposed version.
 *
 * This function initializes the device memory with random values drawn from a normal distribution
 * with the specified standard deviation. It also generates the transposed version of the weight matrices
 * and stores them in the provided transposed memory.
 *
 * @param dev            The standard deviation of the normal distribution.
 * @param transposed     The device memory to store the transposed weight matrices.
 * @param input_width    The width of the input layer.
 * @param width          The width of the hidden layer weight matrices.
 * @param output_width   The width of the output layer.
 * @param n_hidden       The number of hidden layers in the network.
 * @param q              The SYCL queue for parallel computation.
 */
template<typename T>
void DeviceMem<T>::initialize_normal(double dev, DeviceMem<T>& transposed, int input_width, int width, int output_width, int n_hidden, queue q) {
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
				data[input_width * width + k * width * width + toPackedLayoutCoord(i * width + j, width, width)] = rnd;
				dataT[input_width * width + k * width * width + toPackedLayoutCoord(j * width + i, width, width)] = rnd;
			}
		}
	}
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < output_width; j++) {
			rnd = (T)distrib(gen);
			data[input_width * width + n_hidden * width * width + toPackedLayoutCoord(i * output_width + j, width, output_width)] = rnd;
			dataT[input_width * width + n_hidden * width * width + toPackedLayoutCoord(j * output_width + i, output_width, width)] = rnd;
		}
	}
	buffer<T, 1> buf(data.data(), data.size());
	buffer<T, 1> bufT(dataT.data(), dataT.size());
	q.submit([&](handler& h) {
		auto acc = buf.get_access(h);
		auto accT = bufT.get_access(h);
		h.parallel_for(m_size, [=](id<1> idx) {
			p[idx] = acc[idx];
			transposed.data()[idx] = accT[idx];
			});
		});
}

/**
 * Initialize the device memory with values drawn from a normal distribution.
 *
 * This function initializes the device memory with random values drawn from a normal distribution
 * with the specified standard deviation.
 *
 * @param dev   The standard deviation of the normal distribution.
 * @param q     The SYCL queue for parallel computation.
 */
template<typename T>
void DeviceMem<T>::initialize_normal(double dev, queue q) {
	std::default_random_engine gen;
	std::normal_distribution<double> distrib(0.0, dev);
	std::vector<T> data(m_size);
	for (int i = 0; i < m_size; i++) {
		data[i] = (T)distrib(gen);
	}
	q.memcpy(m_data, data.data(), m_size * sizeof(T));
}

/**
 * Initialize the device memory with values drawn from a uniform distribution.
 *
 * This function initializes the device memory with random values drawn from a uniform distribution
 * within the specified scale range.
 *
 * @param scale         The range for generating uniform random values.
 * @param transposed    The transposed version of the initialized data.
 * @param input_width   The width of the input.
 * @param width         The width of the layer.
 * @param output_width  The width of the output.
 * @param n_hidden      The number of hidden layers.
 * @param q             The SYCL queue for parallel computation.
 */
template<typename T>
void DeviceMem<T>::initialize_uniform(double scale, DeviceMem<T>& transposed, int input_width, int width, int output_width, int n_hidden, queue q) {
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
		}
	}

	for (int k = 0; k < n_hidden; k++) {
		for (int i = 0; i < width; i++) {
			for (int j = 0; j < width; j++) {
				rnd = (T)distrib(gen);
				data[input_width * width + k * width * width + toPackedLayoutCoord(i * width + j, width, width)] = rnd;
				dataT[input_width * width + k * width * width + toPackedLayoutCoord(j * width + i, width, width)] = rnd;
			}
		}
	}

	for (int i = 0; i < width; i++) {
		for (int j = 0; j < output_width; j++) {
			rnd = (T)distrib(gen);
			data[input_width * width + n_hidden * width * width + toPackedLayoutCoord(i * output_width + j, width, output_width)] = rnd;
			dataT[input_width * width + n_hidden * width * width + toPackedLayoutCoord(j * width + i, output_width, width)] = rnd;
		}
	}
	buffer<T, 1> buf(data.data(), data.size());
	buffer<T, 1> bufT(dataT.data(), dataT.size());
	q.submit([&](handler& h) {
		auto acc = buf.get_access(h);
		auto accT = bufT.get_access(h);
		h.parallel_for(m_size, [=](id<1> idx) {
			p[idx] = acc[idx];
			transposed.data()[idx] = accT[idx];
			});
		});
}

/**
 * Create the transposed version of the data and store it in the provided memory.
 *
 * This function calculates the transposed version of the data and stores it in the provided
 * `transposed` memory. It takes into account the layout of input, hidden, and output layers
 * and performs the transposition accordingly.
 *
 * @param transposed    The memory to store the transposed data.
 * @param input_width   The width of the input.
 * @param width         The width of the layer.
 * @param output_width  The width of the output.
 * @param n_hidden      The number of hidden layers.
 * @param q             The SYCL queue for parallel computation.
 */
template<typename T>
void DeviceMem<T>::make_transposed(DeviceMem<T>& transposed, int input_width, int width, int output_width, int n_hidden, queue q) {
	auto p = m_data;

	q.parallel_for<>(range<1>(input_width * width + n_hidden * width * width + width * output_width), [=](id<1> idx) {
		int i = 0;
		int j = 0;
		int mat_num = 0;
		int mat_offset = 0;

		if (idx < input_width * width) {
			i = idx / input_width;
			j = idx % input_width;
			transposed.data()[toPackedLayoutCoord(j * width + i, width, input_width)] = p[toPackedLayoutCoord(i * width + j, input_width, width)];
		}

		else if (idx < input_width * width + n_hidden * width * width) {
			mat_num = idx / (width * width);
			mat_offset = (idx - input_width * width) % (width * width);
			i = mat_offset / input_width;
			j = mat_offset % input_width;
			transposed.data()[input_width * width + mat_num * width * width + toPackedLayoutCoord(j * width + i, width, width)] = p[input_width * width + mat_num * width * width + toPackedLayoutCoord(i * width + j, width, width)];
		}

		else {
			mat_offset = (idx - input_width * width - n_hidden * width * width) % (width * output_width);
			i = mat_offset / input_width;
			j = mat_offset % input_width;
			transposed.data()[input_width * width + n_hidden * width * width + toPackedLayoutCoord(j * width + i, output_width, width)] = p[input_width * width + n_hidden * width * width + toPackedLayoutCoord(i * width + j, width, output_width)];

		}
		});
}

/**
 * Initialize the device memory with values sampled from a uniform distribution.
 *
 * This function generates random values sampled from a uniform distribution within
 * the specified scale and initializes the device memory with those values.
 *
 * @param q       The SYCL queue for memory operations.
 * @param scale   The scale of the uniform distribution.
 */
template<typename T>
void DeviceMem<T>::initialize_uniform(queue q, double scale) {
	std::default_random_engine gen;
	std::uniform_real_distribution<double> distrib(0.0, scale);
	std::vector<T> data(m_size);
	for (int i = 0; i < m_size; i++) {
		data[i] = (T)distrib(gen);
	}
	q.memcpy(m_data, data.data(), m_size * sizeof(T));
}
/**
 * Initialize the device memory with values sampled from a Xavier uniform distribution.
 *
 * This function generates random values sampled from a Xavier uniform distribution and
 * initializes both the device memory and its transposed version with those values.
 *
 * @param transposed    The memory to store the transposed data.
 * @param input_width   The width of the input.
 * @param width         The width of the layer.
 * @param output_width  The width of the output.
 * @param n_hidden      The number of hidden layers.
 * @param q             The SYCL queue for memory operations.
 */
template<typename T>
void DeviceMem<T>::initialize_xavier_unif(DeviceMem<T>& transposed, int input_width, int width, int output_width, int n_hidden, queue q) {
	double x = sqrt(6.0 / ((double)(input_width + output_width)));
	initialize_uniform(x, transposed, input_width, width, output_width, n_hidden, q);
}
/**
 * Initialize the device memory with values sampled from a Xavier uniform distribution.
 *
 * This function generates random values sampled from a Xavier uniform distribution and
 * initializes the device memory with those values.
 *
 * @param input_width   The width of the input.
 * @param output_width  The width of the output.
 * @param q             The SYCL queue for memory operations.
 */
template<typename T>
void DeviceMem<T>::initialize_xavier_unif(int input_width, int output_width, queue q) {
	double x = sqrt(6.0 / ((double)(input_width + output_width)));
	initialize_uniform(q, x);
}
/**
 * Initialize the device memory with values sampled from a Xavier normal distribution.
 *
 * This function generates random values sampled from a Xavier normal distribution and
 * initializes both the device memory and its transposed version with those values.
 *
 * @param transposed    The memory to store the transposed data.
 * @param input_width   The width of the input.
 * @param width         The width of the layer.
 * @param output_width  The width of the output.
 * @param n_hidden      The number of hidden layers.
 * @param q             The SYCL queue for memory operations.
 */
template<typename T>
void DeviceMem<T>::inititialize_xavier_normal(DeviceMem<T>& transposed, int input_width, int width, int output_width, int n_hidden, queue q) {
	double dev = sqrt(2.0 / ((double)(input_width + output_width)));
	initialize_normal(dev, transposed, input_width, width, output_width, n_hidden, q);
}

/**
 * Initialize the device memory with values sampled from a Xavier normal distribution.
 *
 * This function generates random values sampled from a Xavier normal distribution and
 * initializes the device memory with those values.
 *
 * @param input_width   The width of the input.
 * @param output_width  The width of the output.
 * @param q             The SYCL queue for memory operations.
 */
template<typename T>
void DeviceMem<T>::initialize_xavier_normal(int input_width, int output_width, queue q) {
	double dev = sqrt(2.0 / ((double)(input_width + output_width)));
	initialize_normal(dev, q);
}

/**
 * Initialize the device memory with a constant value and its transposed version.
 *
 * This function initializes both the device memory and its transposed version with a
 * constant value.
 *
 * @param constant      The constant value to be filled in the memory.
 * @param transposed    The memory to store the transposed data.
 * @param q             The SYCL queue for memory operations.
 */
template<typename T>
void DeviceMem<T>::initialize_constant(T constant, DeviceMem<T>& transposed, queue q) {
	std::vector<T> data(m_size, constant);
	q.memcpy(m_data, data.data(), m_size * sizeof(T));
	q.memcpy(transposed.data(), data.data(), m_size * constant);
}

/**
 * Initialize the device memory with a constant value.
 *
 * This function initializes the device memory with a constant value.
 *
 * @param constant      The constant value to be filled in the memory.
 * @param q             The SYCL queue for memory operations.
 */
template<typename T>
void DeviceMem<T>::initialize_constant(T constant, queue q) {
	auto p = m_data;
	q.parallel_for<>(range<1>(m_size), [=](id<1> idx) {
		p[idx] = (T)constant;
		}).wait();
}

/**
 * Initialize the device memory with values sampled from a He normal distribution.
 *
 * This function generates random values sampled from a He normal distribution and
 * initializes both the device memory and its transposed version with those values.
 *
 * @param transposed    The memory to store the transposed data.
 * @param input_width   The width of the input.
 * @param width         The width of the layer.
 * @param output_width  The width of the output.
 * @param n_hidden      The number of hidden layers.
 * @param q             The SYCL queue for memory operations.
 */
template<typename T>
void DeviceMem<T>::intitialize_he_normal(DeviceMem<T>& transposed, int input_width, int width, int output_width, int n_hidden, queue q) {
	double dev = sqrt(2.0 / width);
	initialize_normal(dev, transposed, input_width, width, output_width, n_hidden, q);
}

/**
 * Initialize the device memory with values sampled from a He normal distribution.
 *
 * This function generates random values sampled from a He normal distribution and
 * initializes the device memory with those values.
 *
 * @param input_width   The width of the input.
 * @param q             The SYCL queue for memory operations.
 */
template<typename T>
void DeviceMem<T>::intitialize_he_normal(int input_width, queue q) {
	double dev = sqrt(2.0 / input_width);
	initialize_normal(dev, q);
}

template class DeviceMem<float>;
template class DeviceMem<bf16>;
