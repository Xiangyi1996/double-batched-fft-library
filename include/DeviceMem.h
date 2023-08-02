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
    DeviceMem() {}

    DeviceMem(int size, queue q) {
        if (m_size != 0 || size <= 0) {
            return;
        }
        m_size = size;
        m_data = malloc_device<T>(size, q);
    }

    void allocate(int size, queue q) {
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

    void initialize_normal(double dev, DeviceMem<T>& transposed, int input_width, int width, int output_width, int n_hidden, queue q) {
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
        //q.memcpy(m_data, data.data(), m_size * sizeof(T));
        //q.memcpy(transposed.data(), dataT.data(), m_size * sizeof(T));
    }

    void initialize_normal(double dev, queue q) {
        std::default_random_engine gen;
        std::normal_distribution<double> distrib(0.0, dev);
        std::vector<T> data(m_size);
        for (int i = 0; i < m_size; i++) {
            data[i] = (T)distrib(gen);
        }
        q.memcpy(m_data, data.data(), m_size * sizeof(T));
    }

    void initialize_uniform(double scale, DeviceMem<T>& transposed, int input_width, int width, int output_width, int n_hidden, queue q) {
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
        //q.memcpy(m_data, data.data(), m_size * sizeof(T));
        //q.memcpy(transposed.data(), dataT.data(), m_size * sizeof(T));
    }

    void make_transposed(DeviceMem<T>& transposed, int input_width, int width, int output_width, int n_hidden, queue q) {
        auto p = m_data;

        q.parallel_for<>(range<1>(input_width * width + n_hidden * width * width + width * output_width), [=](id<1> idx) {
            int i = 0;
            int j = 0;
            int mat_num = 0;
            int mat_offset = 0;

            if (idx < input_width * width) {
                i = idx / input_width;
                j = idx % input_width;
                transposed.data()[toPackedLayoutCoord(j * width + i, width, input_width)] = m_data[toPackedLayoutCoord(i * width + j, input_width, width)];
            }

            else if (idx < input_width * width + n_hidden * width * width) {
                mat_num = idx / (width * width);
                mat_offset = (idx - input_width * width) % (width * width);
                i = mat_offset / input_width;
                j = mat_offset % input_width;
                transposed.data()[input_width * width + mat_num * width * width + toPackedLayoutCoord(j * width + i, width, width)] = m_data[input_width * width + mat_num * width * width + toPackedLayoutCoord(i * width + j, width, width)];
            }

            else {
                mat_offset = (idx - input_width * width - n_hidden * width * width) % (width * output_width);
                i = mat_offset / input_width;
                j = mat_offset % input_width;
                transposed.data()[input_width * width + n_hidden * width * width + toPackedLayoutCoord(j * width + i, output_width, width)] = m_data[input_width * width + n_hidden * width * width + toPackedLayoutCoord(i * width + j, width, output_width)];

            }
            });
    }

    void initialize_uniform(queue q, double scale = 1.0) {
        std::default_random_engine gen;
        std::uniform_real_distribution<double> distrib(0.0, scale);
        std::vector<T> data(m_size);
        for (int i = 0; i < m_size; i++) {
            data[i] = (T)distrib(gen);
        }
        q.memcpy(m_data, data.data(), m_size * sizeof(T));
    }

    void initialize_xavier_unif(DeviceMem<T>& transposed, int input_width, int width, int output_width, int n_hidden, queue q) {
        double x = sqrt(6.0 / ((double)(input_width + output_width)));
        initialize_uniform(x, transposed, input_width, width, output_width, n_hidden, q);
    }

    void initialize_xavier_unif(int input_width, int output_width, queue q) {
        double x = sqrt(6.0 / ((double)(input_width + output_width)));
        initialize_uniform(x, q);
    }


    void inititialize_xavier_normal(DeviceMem<T>& transposed, int input_width, int width, int output_width, int n_hidden, queue q) {
        double dev = sqrt(2.0 / ((double)(input_width + output_width)));
        initialize_normal(dev, transposed, input_width, width, output_width, n_hidden, q);
    }

    void initialize_xavier_normal(int input_width, int output_width, queue q) {
        double dev = sqrt(2.0 / ((double)(input_width + output_width)));
        initialize_normal(dev, q);
    }

    void initialize_constant(T constant, DeviceMem<T>& transposed, queue q) {
        std::vector<T> data(m_size, constant);
        q.memcpy(m_data, data.data(), m_size * sizeof(T));
        q.memcpy(transposed.data(), data.data(), m_size * constant);
    }

    void initialize_constant(T constant, queue q) {
        auto p = m_data;
        q.parallel_for<>(range<1>(m_size), [=](id<1> idx) {
            p[idx] = (T)constant;
            }).wait();
    }

    void intitialize_he_normal(DeviceMem<T>& transposed, int input_width, int width, int output_width, int n_hidden, queue q) {
        double dev = sqrt(2.0 / width);
        initialize_normal(dev, transposed, input_width, width, output_width, n_hidden, q);
    }

    void intitialize_he_normal(int input_width, queue q) {
        double dev = sqrt(2.0 / input_width);
        initialize_normal(dev, q);
    }
};
