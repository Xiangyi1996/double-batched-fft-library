#pragma once

#include <iostream>
#include <vector>
#include <random>

using namespace sycl;
template<typename T>
class DeviceMem {
private:
	T* m_data = nullptr;
	int m_size = 0;
public:
	DeviceMem() {}

	DeviceMem(int size, queue q) {
		if (m_size != 0 || size <= 0) {
			return;
		}
		m_size = size;
		m_data = malloc_shared<T>(size, q);
	}

	void allocate(int size, queue q) {
		if (m_size != 0 || size <= 0) {
			return;
		}
		m_size = size;
		m_data = malloc_shared<T>(size, q);
	}

	

	void allocate(int size) {
		if (m_size != 0 || size <= 0) {
			return;
		}
		m_size = size;
		m_data = malloc_shared<T>(size);
	}

	void free_mem(queue q) {
		m_size = 0;
		free(m_data, q);
	}
	void free_mem() {
		m_size = 0;
		free(m_data);
	}

	void copy_from_host(std::vector<T>& data, int n, queue q) {
		q.memcpy(m_data, data.data(), n * sizeof(T));
	}

	void copy_to_host(std::vector<T>& data, int n, queue q) {
		q.memcpy(data.data(), m_data, n * sizeof(T));
	}

	void copy_from_host(std::vector<T>& data, queue q) {
		copy_from_host(data, m_size, q);
	}

	void copy_to_host(std::vector<T>& data, queue q) {
		copy_to_host(data, m_size, q);
	}

	T* data() const {
		return m_data;
	}

	void set_data(int id, T value) {
		m_data[id] = value;
	}
	int size() const {
		return m_size;
	}
	// Update for the future : use oneMKL RNG for weights intialization
	
			//Initialziation
	template<bool transpose>
	void initialize_normal( double dev, DeviceMem<T>& transposed = nullptr, int input_width = 0, int width = 0, int output_width = 0, int n_hidden = 0) {
		
		std::default_random_engine gen;
		std::normal_distribution<double> distrib(0.0, dev);
		if (!transpose) {
			for (int i = 0; i < m_size; i++) {
				m_data[i] = (T)distrib(gen);
			}
		}
		if (transpose) {
			for (int i = 0; i < input_width; i++) {
				for (int j = 0; j < width; j++) {
					m_data[i * width + j] = (T)distrib(gen);
					transposed.data()[j * width + i] = m_data[i * width + j];
				}
			}
			for (int k = 0; k < n_hidden; k++) {
				for (int i = 0; i < width; i++) {
					for (int j = 0; j < width; j++) {
						m_data[input_width * width + k * width * width + i * width + j] = (T)distrib(gen);
						transposed.data()[input_width * width + k * width * width + j * width + i] = m_data[input_width * width + k * width * width + i * width + j];
						}
					}
				}
			for (int i = 0; i < width; i++) {
				for (int j = 0; j < output_width; j++) {
					m_data[input_width * width + n_hidden * width * width + i * width + j] = (T)distrib(gen);
					transposed.data()[input_width * width + n_hidden * width * width + j * width + i] = m_data[input_width * width + n_hidden * width * width + i * width + j];
					}
				}

			}
		}
		
	
	template<bool transpose>
	void initialize_uniform( double scale = 1.0, DeviceMem<T>&transposed = nullptr, int input_width = 0, int width = 0, int output_width = 0, int n_hidden = 0) {
		
		std::default_random_engine gen;
		std::uniform_real_distribution<double> distrib(0.0, scale);
		if (!transpose) {
			for (int i = 0; i < m_size; i++) {
				m_data[i] = (T)distrib(gen);
			}
		}
		if (transpose) {
			for (int i = 0; i < input_width; i++) {
				for (int j = 0; j < width; j++) {
					m_data[i * width + j] = (T)distrib(gen);
					transposed.data()[j * width + i] = m_data[i * width + j];
					
				}
			}
			for (int k = 0; k < n_hidden; k++) {
				for (int i = 0; i < width; i++) {
					for (int j = 0; j < width; j++) {
						m_data[input_width * width + k * width * width + i * width + j] = (T)distrib(gen);
						transposed.data()[input_width * width + k * width * width + j * width + i] = m_data[input_width * width + k * width * width + i * width + j];
					}
				}
			}
			for (int i = 0; i < width; i++) {
				for (int j = 0; j < output_width; j++) {
					m_data[input_width * width + n_hidden * width * width + i * width + j] = (T)distrib(gen);
					transposed.data()[input_width * width + n_hidden * width * width + j * width + i] = m_data[input_width * width + n_hidden * width * width + i * width + j];
				}
			}
		}
	}
	template<bool transpose>
	void initialize_xavier_unif( DeviceMem<T>&transposed = nullptr, int input_width = 0, int width = 0, int output_width = 0, int n_hidden = 0) {
		double x = sqrt(6.0 / ((double)(input_width + output_width)));
		initialize_uniform<transpose>( x, transposed, input_width, width, output_width, n_hidden);
	}

	template<bool transpose>
	void inititialize_xavier_normal( DeviceMem<T>&transposed = nullptr, int input_width = 0, int width = 0, int output_width = 0, int n_hidden = 0) {
		double dev = sqrt(2.0 / ((double)(input_width + output_width)));
		initialize_normal<transpose>( dev, transposed, input_width, width, output_width, n_hidden);
	}

	template<bool transpose>
	void initialize_constant(T constant,  DeviceMem<T>& transposed = nullptr, int input_width = 0, int width = 0, int output_width = 0, int n_hidden = 0) {
		for (int i = 0; i < m_size; i++) {
			m_data[i] = constant;
		}
		if (transpose) {
			for (int i = 0; i < m_size; i++) {
				transposed.data()[i] = constant;
			}
		}
	}

	template<bool transpose>
	void intitialize_he_normal(int intput_width,  DeviceMem<T>&transposed = nullptr, int input_width = 0, int width = 0, int output_width = 0, int n_hidden = 0) {
		double dev = sqrt(2.0 / width);
		initialize_normal<transpose>( dev, transposed, input_width, width, output_width, n_hidden);
	}
};
