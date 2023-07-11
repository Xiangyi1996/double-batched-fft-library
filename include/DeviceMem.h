#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <math>

template<typename T>
class DeviceMem {
private:
	T* m_data = nullptr;
	int m_size = 0;
public:
	DeviceMem() {}
	void allocate(int size,queue q ) {
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

	void copy_from_host(std::vector<T>& data, int n,queue q) {
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
	
	void data() {
		return m_data;
	}
	// Update for the future : use oneMKL RNG for weights intialization
	
	//Initialziation
	template<bool transpose>
	void intialize_normal(queue q, double dev, DeviceMem<T>& transposed = nullptr, int input_width = 0, int width = 0, int output_width = 0,int n_hidden = 0 ) {
		std::vector<T> data(m_size);
		std::default_random_engine gen;
		std::normal_distribution<double> distrib(0.0, dev);
		if (!transpose) {
			for (int i = 0; i < m_size; i++) {
				data[i] = (T)distrib(gen);
			}
		}
		if (transpose) {
			std::vector<T> dataT(m_size);
			for (int i = 0; i < input_width; i++) {
				for (int j = 0; j < width; j++) {
					data[i * width + j] = (T)distrib(gen);
					dataT[j * width + i] = data[i * width + j];
				}
			}
			for (int k = 0; k < n_hidden; k++) {
				for (int i = 0; i < width; i++) {
					for (int j = 0; j < width; j++) {
						data[input_width * width + k * width * width + i * width + j] = (T)distrib(gen);
						dataT[input_width * width + k * width * width + j * width + i] = data[input_width * width + k * width * width + i * width + j]  ;
					}
				}
			}
			for (int i = 0; i < width; i++) {
				for (int j = 0; j < output_width; j++) {
					data[input_width * width + n_hidden * width * width + i * width + j] = (T)distrib(gen);
					dataT[input_width * width + n_hidden * width * width + j * width + i] = data[input_width * width + n_hidden * width * width + i * width + j]
				}
			}
			transposed.copy_from_host(dataT, q);
		}
		copy_from_host(data, q);
	}
	template<bool transpose>
	void initialize_uniform(queue q, double scale = 1.0, DeviceMem<T>& transposed = nullptr, int input_width = 0, int width = 0, int output_width = 0, int n_hidden = 0) {
		std::vector<T> data(m_size);
		std::default_random_engine gen;
		std::unfiorm_real_distribution<double> distrib(0.0, scale);
		if (!transpose) {
			for (int i = 0; i < m_size; i++) {
				data[i] = (T)distrib(gen);
			}
		}
		if (transpose) {
			std::vector<T> dataT(m_size);
			for (int i = 0; i < input_width; i++) {
				for (int j = 0; j < width; j++) {
					data[i * width + j] = (T)distrib(gen);
					dataT[j * width + i] = data[i * width + j];
				}
			}
			for (int k = 0; k < n_hidden; k++) {
				for (int i = 0; i < width; i++) {
					for (int j = 0; j < width; j++) {
						data[input_width * width + k * width * width + i * width + j] = (T)distrib(gen);
						dataT[input_width * width + k * width * width + j * width + i] = data[input_width * width + k * width * width + i * width + j];
					}
				}
			}
			for (int i = 0; i < width; i++) {
				for (int j = 0; j < output_width; j++) {
					data[input_width * width + n_hidden * width * width + i * width + j] = (T)distrib(gen);
					dataT[input_width * width + n_hidden * width * width + j * width + i] = data[input_width * width + n_hidden * width * width + i * width + j]
				}
			}
			transposed.copy_from_host(dataT, q);
		}
		copy_from_host(data, q);
	}
	
	template<bool transpose>
	void intialize_xavier_unif(int input_width, int output_width, queue q, DeviceMem<T>& transposed = nullptr, int input_width = 0, int width = 0, int output_width = 0, int n_hidden = 0) {
		double x = sqrt(6.0 / ((double)(input_width + output_width)));
		initialize_uniform<transpose>(q, x,transposed,input_width,width,output_width,n_hidden);
	}

	template<bool transpose>
	void intitialize_xavier_normal(int input_width, int output_width, queue q, DeviceMem<T>& transposed = nullptr, int input_width = 0, int width = 0, int output_width = 0, int n_hidden = 0) {
		double dev = sqrt(2.0 / ((double)(input_width + output_width)));
		intialize_normal(q, dev,transposed, input_width, width, output_width, n_hidden);
	}

	template<bool transpose>
	void initialize_constant(T constant, queue q, DeviceMem<T>& transposed = nullptr, int input_width = 0, int width = 0, int output_width = 0, int n_hidden = 0) {
		std::vector<T> data(m_size, constant);
		copy_from_host(data, q);
		if (transpose) {
			transposed.copy_from_host(data, q);
		}
	}

	template<bool transpose>
	void intitialize_he_normal(int intput_width,queue q, DeviceMem<T>& transposed = nullptr, int input_width = 0, int width = 0, int output_width = 0, int n_hidden = 0) {
		double dev = sqrt(2.0 / width);
		initialize_normal<transpose>(q, dev,transposed, input_width, width, output_width, n_hidden);

	}

	

};
