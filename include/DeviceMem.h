#pragma once

#include <iostream>
#include <vector>
#include <random>
#include "common.h"

using namespace sycl;
template<typename T>
class DeviceMem {
private:
    T* m_data = nullptr;
    int m_size = 0;
public:
    DeviceMem();

    DeviceMem(int size, queue q);

    void allocate(int size, queue q);

    void allocate(int size);

    void free_mem(queue q);
    void free_mem();
    void copy_from_host(std::vector<T>& data, int n, queue q);
    void copy_to_host(std::vector<T>& data, int n, queue q);

    void copy_from_host(std::vector<T>& data, queue q);

    void copy_to_host(std::vector<T>& data, queue q);

    T* data() const {
        return m_data;
    }

    void set_data(int id, T value);
    int size() const {
        return m_size;
    }
    void initialize_normal(double dev, DeviceMem<T>& transposed, int input_width, int width, int output_width, int n_hidden, queue q);

    void initialize_normal(double dev, queue q);

    void initialize_uniform(double scale, DeviceMem<T>& transposed, int input_width, int width, int output_width, int n_hidden, queue q);

    void make_transposed(DeviceMem<T>& transposed, int input_width, int width, int output_width, int n_hidden, queue q);

    void initialize_uniform(queue q, double scale = 1.0);

    void initialize_xavier_unif(DeviceMem<T>& transposed, int input_width, int width, int output_width, int n_hidden, queue q);

    void initialize_xavier_unif(int input_width, int output_width, queue q);


    void inititialize_xavier_normal(DeviceMem<T>& transposed, int input_width, int width, int output_width, int n_hidden, queue q);

    void initialize_xavier_normal(int input_width, int output_width, queue q);

    void initialize_constant(T constant, DeviceMem<T>& transposed, queue q);

    void initialize_constant(T constant, queue q);

    void intitialize_he_normal(DeviceMem<T>& transposed, int input_width, int width, int output_width, int n_hidden, queue q);

    void intitialize_he_normal(int input_width, queue q);
};
