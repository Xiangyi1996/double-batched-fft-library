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
	void initialize_normal( double dev, DeviceMem<T>& transposed , int input_width, int width , int output_width, int n_hidden ) {
		
		std::default_random_engine gen;
		std::normal_distribution<double> distrib(0.0, dev);
		
		
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
		
	void initialize_normal(double dev) {
		std::default_random_engine gen;
		std::normal_distribution<double> distrib(0.0, dev);
		for (int i = 0; i < m_size; i++) {
			m_data[i] = (T)distrib(gen);
		}
	}
	void initialize_uniform( double scale , DeviceMem<T>&transposed , int input_width , int width , int output_width , int n_hidden) {
		
		std::default_random_engine gen;
		std::uniform_real_distribution<double> distrib(0.0, scale);
		
		
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

	void make_transposed(DeviceMem<T>& transposed,int input_width,int width,int output_width,int n_hidden) {
		for (int i = 0; i < input_width; i++) {
			for (int j = 0; j < width; j++) {
				transposed.data()[j * width + i] = m_data[i * width + j];

			}
		}
		for (int k = 0; k < n_hidden; k++) {
			for (int i = 0; i < width; i++) {
				for (int j = 0; j < width; j++) {
					transposed.data()[input_width * width + k * width * width + j * width + i] = m_data[input_width * width + k * width * width + i * width + j];
				}
			}
		}
		for (int i = 0; i < width; i++) {
			for (int j = 0; j < output_width; j++) {
				transposed.data()[input_width * width + n_hidden * width * width + j * width + i] = m_data[input_width * width + n_hidden * width * width + i * width + j];
			}
		}
	}

	void initialize_uniform(double scale = 1.0) {
		std::default_random_engine gen;
		std::uniform_real_distribution<double> distrib(0.0, scale);
		for (int i = 0; i < m_size; i++) {
			m_data[i] = (T)distrib(gen);
		}
	}
	
	void initialize_xavier_unif( DeviceMem<T>&transposed , int input_width , int width , int output_width , int n_hidden) {
		double x = sqrt(6.0 / ((double)(input_width + output_width)));
		initialize_uniform( x, transposed, input_width, width, output_width, n_hidden);
	}

	void initialize_xavier_unif(int input_width, int output_width) {
		double x = sqrt(6.0 / ((double)(input_width + output_width)));
		initialize_uniform(x);
	}

	
	void inititialize_xavier_normal( DeviceMem<T>&transposed, int input_width , int width , int output_width , int n_hidden ) {
		double dev = sqrt(2.0 / ((double)(input_width + output_width)));
		initialize_normal( dev, transposed, input_width, width, output_width, n_hidden);
	}

	void initialize_xavier_normal(int input_width, int output_width) {
		double dev = sqrt(2.0 / ((double)(input_width + output_width)));
		initialize_normal(dev);
	}

	void inititialize_constant(T constant, DeviceMem<T>& transposed) {
		for (int i = 0; i < m_size; i++) {
			transposed.data()[i] = constant;
			m_data = constant;
		}
	}

	void initialize_constant(T constant) {
		for (int i = 0; i < m_size; i++) {
			m_data[i] = constant;
		}
	}

	
	void intitialize_he_normal( DeviceMem<T>&transposed , int input_width , int width , int output_width , int n_hidden ) {
		double dev = sqrt(2.0 / width);
		initialize_normal( dev, transposed, input_width, width, output_width, n_hidden);
	}
	void intitialize_he_normal(int input_width) {
		double dev = sqrt(2.0 / input_width);
		initialize_normal(dev);

	}


};
