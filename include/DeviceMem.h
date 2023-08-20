#pragma once

#include <iostream>
#include <random>
#include <vector>

#include "common.h"

using namespace sycl;

// A templated class for managing device memory
template <typename T>
class DeviceMem {
 private:
  T* m_data = nullptr;
  int m_size = 0;

 public:
  // Default constructor
  DeviceMem();

  // Constructor with size and queue
  DeviceMem(int size, queue q);

  // Allocate memory on the device
  void allocate(int size, queue q);

  // Allocate memory on the device
  void allocate(int size);

  // Free memory on the device
  void free_mem(queue q);
  void free_mem();

  // Copy data from host to device
  void copy_from_host(std::vector<T>& data, int n, queue q);

  // Copy data from device to host
  void copy_to_host(std::vector<T>& data, int n, queue q);

  // Copy data from host to device
  void copy_from_host(std::vector<T>& data, queue q);

  // Copy data from device to host
  void copy_to_host(std::vector<T>& data, queue q);

  // Get the raw data pointer
  T* data() const { return m_data; }

  // Set data at a specific index
  void set_data(int id, T value);

  // Get the size of the memory allocation
  int size() const { return m_size; }

  // Initialize memory with values drawn from a normal distribution
  void initialize_normal(double dev, DeviceMem<T>& transposed, int input_width,
                         int width, int output_width, int n_hidden, queue q);

  // Initialize memory with values drawn from a normal distribution
  void initialize_normal(double dev, queue q);

  // Initialize memory with values drawn from a uniform distribution
  void initialize_uniform(double scale, DeviceMem<T>& transposed,
                          int input_width, int width, int output_width,
                          int n_hidden, queue q);

  // Transpose the memory content
  void make_transposed(DeviceMem<T>& transposed, int input_width, int width,
                       int output_width, int n_hidden, queue q);

  // Initialize memory with values drawn from a uniform distribution
  void initialize_uniform(queue q, double scale = 1.0);

  // Initialize memory with values according to Xavier uniform initialization
  void initialize_xavier_unif(DeviceMem<T>& transposed, int input_width,
                              int width, int output_width, int n_hidden,
                              queue q);

  // Initialize memory with values according to Xavier uniform initialization
  void initialize_xavier_unif(int input_width, int output_width, queue q);

  // Initialize memory with values according to Xavier normal initialization
  void inititialize_xavier_normal(DeviceMem<T>& transposed, int input_width,
                                  int width, int output_width, int n_hidden,
                                  queue q);

  // Initialize memory with values according to Xavier normal initialization
  void initialize_xavier_normal(int input_width, int output_width, queue q);

  // Initialize memory with constant values
  void initialize_constant(T constant, DeviceMem<T>& transposed, queue q);

  // Initialize memory with constant values
  void initialize_constant(T constant, queue q);

  // Initialize memory with values according to He normal initialization
  void intitialize_he_normal(DeviceMem<T>& transposed, int input_width,
                             int width, int output_width, int n_hidden,
                             queue q);

  // Initialize memory with values according to He normal initialization
  void intitialize_he_normal(int input_width, queue q);

  void initialize_test_input(queue q);
};
