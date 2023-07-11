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
		m_data = malloc_device<T>(size, q);
	}
	void allocate(int size) {
		if (m_size != 0 || size <= 0) {
			return;
		}
		m_size = size;
		m_data = malloc_device<T>(size);
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
	
	// Update for the future : use oneMKL RNG for weights intialization
	void intialize_xavier_unif(int input_width, int output_width, queue q) {
		double x = sqrt(6.0 / ((double)(input_width + output_width)));
		initialize_uniform(q, x);
	}
	
	void intitialize_xavier_normal(int input_width,int output_width,queue q) {
		double dev = sqrt(2.0 / ((double)(input_width + output_width)));
		intialize_normal(q, dev);
	}

	/*void intialize_he_unif() {

	}*/

	void intitialize_he_normal(int intput_width,queue q) {	
		double dev = sqrt(2.0 / width);
		initialize_normal(q, dev);

	}

	void intialize_normal(queue q, double dev) {
		std::vector<T> data(m_size);
		std::default_random_engine gen;
		std::normal_distribution<double> distrib(0.0, dev);
		for (int i = 0; i < m_size; i++) {
			data[i] = distrib(gen);
		}
		copy_from_host(data, q);
	}

	void initialize_uniform(queue q, double scale =1.0 ) {
		std::vector<T> data(m_size);
		std::default_random_engine gen;
		std::unfiorm_real_distribution<double> distrib(0.0, scale);
		for (int i = 0; i < m_size; i++) {
			data[i] = distrib(gen);
		}
		copy_from_host(data, q);
	}

};
