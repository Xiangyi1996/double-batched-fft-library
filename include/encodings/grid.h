/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TOR
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** @file   grid.h
 *  @author Thomas Müller, NVIDIA & Alex Evans, NVIDIA & Jianfei Guo, Shanghai
 * AI Lab
 *  @brief  Trainable hierarchy of N-D grids of floating point values.
 *          The grids can be backed by dense memory, tiled memory, or by hash
 * tables.
 */

#ifndef TINYNN_ENCODINGS_GRID_H
#define TINYNN_ENCODINGS_GRID_H

#include "DeviceMem.h"
#include "common.h"
#include "common_device.h"
#include "encoding.h"
#include "grid_interface.h"
#include "vec.h"

#include <stdexcept>
#include <stdint.h>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

#include "oneapi/mkl/rng.hpp"

using json = nlohmann::json;

template <typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL, HashType HASH_TYPE>
void kernel_grid(const uint32_t num_elements, const uint32_t num_grid_features, const GridOffsetTable offset_table,
                 const uint32_t base_resolution, const float log2_per_level_scale, float max_level,
                 const float *__restrict__ max_level_gpu, const InterpolationType interpolation_type,
                 const GridType grid_type, const T *__restrict__ grid, float *positions_in_data, uint32_t pin_stride_i,
                 uint32_t pin_stride_j, T *__restrict__ encoded_positions, float *__restrict__ dy_dx,
                 const sycl::nd_item<3> &item_ct1) {
    assert(grid != nullptr && "grid is nullptr, expected a valid pointer");

    MatrixView<const float> positions_in(positions_in_data, pin_stride_i, pin_stride_j);
    const uint32_t i = item_ct1.get_global_id(2);
    if (i >= num_elements) return;

    const uint32_t level = item_ct1.get_group(1); // <- the level is the same for all threads

    max_level = max_level_gpu ? ((max_level_gpu[i] * num_grid_features) / N_FEATURES_PER_LEVEL)
                              : ((max_level * num_grid_features) / N_FEATURES_PER_LEVEL);

    if (level >= max_level + 1e-3f) {
        if (encoded_positions) {
            for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f)
                encoded_positions[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = (T)0.0f;
        }

        // Gradient is zero for zeroed-out dimensions.
        if (dy_dx) {
            for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f)
                ((tnn::vec<N_POS_DIMS> *)dy_dx)[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = {0.0f};
        }

        return;
    }

    grid += offset_table.data[level] * N_FEATURES_PER_LEVEL;
    const uint32_t hashmap_size = offset_table.data[level + 1] - offset_table.data[level];

    const float scale = grid_scale(level, log2_per_level_scale, base_resolution);
    const uint32_t resolution = grid_resolution(scale);

    float pos[N_POS_DIMS];
    float pos_derivative[N_POS_DIMS];
    tnn::uvec<N_POS_DIMS> pos_grid;

    if (interpolation_type == InterpolationType::Nearest || interpolation_type == InterpolationType::Linear) {
        for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
            auto input = positions_in(dim, i);
            auto l_pos = &pos[dim];
            auto l_pos_derivative = &pos_derivative[dim];
            auto l_pos_grid = &pos_grid[dim];
            // scale

            *l_pos = sycl::fma(scale, (float)input, 0.5f);
            float tmp = sycl::floor(*l_pos);
            *l_pos_grid = (uint32_t)(int)tmp;
            *l_pos -= (float)tmp;
            *l_pos_derivative = identity_derivative(*l_pos);
            *l_pos = identity_fun(*l_pos);
            // pos_fract(positions_in(dim, i), &pos[dim], &pos_grid[dim], scale,  [](
            // args ) { identity(args); },  [](args) {df(args);}, [](args){d2f(args);}
            // )
        }
        //   pos_fract(positions_in(dim, i), &pos[dim], &pos_derivative[dim],
        //   &pos_grid[dim], scale, identity_fun, identity_derivative);
    } else {
        for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
            auto input = positions_in(dim, i);
            auto l_pos = &pos[dim];
            auto l_pos_derivative = &pos_derivative[dim];
            auto l_pos_grid = &pos_grid[dim];
            // scale

            *l_pos = sycl::fma(scale, (float)input, 0.5f);
            float tmp = sycl::floor(*l_pos);
            *l_pos_grid = (uint32_t)(int)tmp;
            *l_pos -= (float)tmp;
            *l_pos_derivative = smoothstep_derivative(*l_pos);
            *l_pos = smoothstep(*l_pos);
            // pos_fract(positions_in(dim, i), &pos[dim], &pos_grid[dim], scale,  [](
            // args ) { identity(args); },  [](args) {df(args);}, [](args){d2f(args);}
            // )
        }
        //   pos_fract(positions_in(dim, i), &pos[dim], &pos_derivative[dim],
        //   &pos_grid[dim], scale, smoothstep, smoothstep_derivative);
    }

    auto grid_val = [&](const tnn::uvec<N_POS_DIMS> &local_pos) {
        const uint32_t index =
            grid_index<N_POS_DIMS, HASH_TYPE>(grid_type, hashmap_size, resolution, local_pos) * N_FEATURES_PER_LEVEL;
        return *(tnn::tvec < T, N_FEATURES_PER_LEVEL,
                 PARAMS_ALIGNED ? sizeof(T) * N_FEATURES_PER_LEVEL : sizeof(T) > *)&grid[index];
    };

    if (interpolation_type == InterpolationType::Nearest) {
        auto result = grid_val(pos_grid);

        if (encoded_positions) {
            for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f)
                encoded_positions[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = result[f];
        }

        // Gradient is zero when there's no interpolation.
        if (dy_dx) {
            for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f)
                ((tnn::vec<N_POS_DIMS> *)dy_dx)[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = {0.0f};
        }

        return;
    }

    if (encoded_positions) {
        // N-linear interpolation
        tnn::tvec<T, N_FEATURES_PER_LEVEL, PARAMS_ALIGNED ? sizeof(T) * N_FEATURES_PER_LEVEL : sizeof(T)> result = {};

        for (uint32_t idx = 0; idx < (1 << N_POS_DIMS); ++idx) {
            float weight = 1;
            tnn::uvec<N_POS_DIMS> pos_grid_local;

            for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
                if ((idx & (1 << dim)) == 0) {
                    weight *= 1 - pos[dim];
                    pos_grid_local[dim] = pos_grid[dim];
                } else {
                    weight *= pos[dim];
                    pos_grid_local[dim] = pos_grid[dim] + 1;
                }
            }

            result = fma((T)weight, grid_val(pos_grid_local), result);
        }

        for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f)
            encoded_positions[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = result[f];
    }

    // Gradient
    if (dy_dx) {
        tnn::vec<N_POS_DIMS> grads[N_FEATURES_PER_LEVEL] = {0.0f};

        for (uint32_t grad_dim = 0; grad_dim < N_POS_DIMS; ++grad_dim) {
            for (uint32_t idx = 0; idx < (1 << (N_POS_DIMS - 1)); ++idx) {
                float weight = scale;
                tnn::uvec<N_POS_DIMS> pos_grid_local;

                for (uint32_t non_grad_dim = 0; non_grad_dim < N_POS_DIMS - 1; ++non_grad_dim) {
                    const uint32_t dim = non_grad_dim >= grad_dim ? (non_grad_dim + 1) : non_grad_dim;

                    if ((idx & (1 << non_grad_dim)) == 0) {
                        weight *= 1 - pos[dim];
                        pos_grid_local[dim] = pos_grid[dim];
                    } else {
                        weight *= pos[dim];
                        pos_grid_local[dim] = pos_grid[dim] + 1;
                    }
                }

                pos_grid_local[grad_dim] = pos_grid[grad_dim];
                auto val_left = grid_val(pos_grid_local);
                pos_grid_local[grad_dim] = pos_grid[grad_dim] + 1;
                auto val_right = grid_val(pos_grid_local);

                for (uint32_t feature = 0; feature < N_FEATURES_PER_LEVEL; ++feature)
                    grads[feature][grad_dim] +=
                        weight * ((float)val_right[feature] - (float)val_left[feature]) * pos_derivative[grad_dim];
            }
        }

        for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f)
            ((tnn::vec<N_POS_DIMS> *)dy_dx)[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = grads[f];
    }
}

template <typename T, uint32_t N_POS_DIMS = 3, uint32_t N_FEATURES_PER_LEVEL = 2,
          HashType HASH_TYPE = HashType::CoherentPrime>
class GridEncodingTemplated : public GridEncoding<T> {
  public:
    /////////////////////////////////////////////////////////////////////////////////
    // TODO: SYCL does not support atomic operations on half-precision data-types.
    // //
    /////////////////////////////////////////////////////////////////////////////////
    using grad_t = float;

    // #if TCNN_MIN_GPU_ARCH >= 62 || TCNN_MIN_GPU_ARCH == 60
    //     // The GPUs that we tested this on do not have an efficient 1D fp16
    //     // atomicAdd feature. Thus, we accumulate gradients at fp32 if we're
    //     // forced to use 1D atomicAdds. As soon as 2D or higher is possible,
    //     // we can make use the efficient atomicAdd(half2) function.
    //     using grad_t = std::conditional_t<N_FEATURES_PER_LEVEL == 1, float, T>;
    // #else
    //     // atomicAdd(__half2) is only supported with compute capability 60 and
    //     above.
    //     // Since atomicAdd(__half) is relatively slow / doesn't exist for low
    //     compute
    //     // capabilities, accumulate in fp32 instead.
    //     using grad_t = float;
    // #endif

    GridEncodingTemplated(uint32_t n_features, uint32_t log2_hashmap_size, uint32_t base_resolution,
                          float per_level_scale, bool stochastic_interpolation, InterpolationType interpolation_type,
                          GridType grid_type)
        : m_n_features{n_features}, m_log2_hashmap_size{log2_hashmap_size}, m_base_resolution{base_resolution},
          m_per_level_scale{per_level_scale}, m_stochastic_interpolation{stochastic_interpolation},
          m_interpolation_type{interpolation_type}, m_grid_type{grid_type} {
        m_n_levels = tinydpcppnn::math::div_round_up(m_n_features, N_FEATURES_PER_LEVEL);
        uint32_t offset = 0;

        if (m_n_levels > MAX_N_LEVELS) {
            // throw std::runtime_error{fmt::format("GridEncoding: m_n_levels={} must
            // be at most MAX_N_LEVELS={}", m_n_levels, MAX_N_LEVELS)};
            throw std::runtime_error{"GridEncoding: m_n_levels={} must be at most MAX_N_LEVELS={}"}; //,
                                                                                                     // m_n_levels,
                                                                                                     // MAX_N_LEVELS)};
        }

        for (uint32_t i = 0; i < m_n_levels; ++i) {
            // Compute number of dense params required for the given level
            const uint32_t resolution = grid_resolution(grid_scale(i, std::log2(per_level_scale), base_resolution));

            uint32_t max_params = std::numeric_limits<uint32_t>::max() / 2;
            uint32_t params_in_level =
                std::pow((float)resolution, N_POS_DIMS) > (float)max_params ? max_params : powi(resolution, N_POS_DIMS);

            // Make sure memory accesses will be aligned
            params_in_level = tinydpcppnn::math::next_multiple(params_in_level, 8u);

            if (grid_type == GridType::Dense) {
            } // No-op
            else if (grid_type == GridType::Tiled) {
                // If tiled grid needs fewer params than dense, then use fewer and tile.
                params_in_level = std::min(params_in_level, powi(base_resolution, N_POS_DIMS));
            } else if (grid_type == GridType::Hash) {
                // If hash table needs fewer params than dense, then use fewer and rely
                // on the hash.
                params_in_level = std::min(params_in_level, (1u << log2_hashmap_size));
            } else {
                throw std::runtime_error{"GridEncoding: invalid grid type {}"}; //, to_string(grid_type))};
                // throw std::runtime_error{fmt::format("GridEncoding: invalid grid type
                // {}", to_string(grid_type))};
            }

            m_offset_table.data[i] = offset;
            offset += params_in_level;

            log_debug("GridEncoding at level {}: resolution={} params_in_level={}", i, resolution, params_in_level);
        }

        m_offset_table.data[m_n_levels] = offset;
        m_offset_table.size = m_n_levels + 1;

        this->m_n_params = m_offset_table.data[m_n_levels] * N_FEATURES_PER_LEVEL;
        // std::cout << "this->m_n_params: " << this->m_n_params << ", "
        //           << "m_offset_table.data[m_n_levels] : " << m_offset_table.data[m_n_levels]
        //           << ", N_FEATURES_PER_LEVEL : " << N_FEATURES_PER_LEVEL << std::endl;

        m_n_output_dims = m_n_features;

        if (n_features % N_FEATURES_PER_LEVEL != 0) {
            throw std::runtime_error{"GridEncoding: n_features={} must be a multiple of "
                                     "N_FEATURES_PER_LEVEL={}"}; //, n_features, N_FEATURES_PER_LEVEL)};
            // throw std::runtime_error{fmt::format("GridEncoding: n_features={} must
            // be a multiple of N_FEATURES_PER_LEVEL={}", n_features,
            // N_FEATURES_PER_LEVEL)};
        }
    }

    std::unique_ptr<Context> forward_impl(sycl::queue *const q, const DeviceMatrix<float> &input,
                                          DeviceMatrix<T> *output = nullptr, bool use_inference_params = false,
                                          bool prepare_input_gradients = false) override {

        if (padded_output_width() == 0) throw std::invalid_argument("Can't have width == 0");
        if (use_inference_params) throw std::invalid_argument("Can't use inference params.");
        if (prepare_input_gradients) throw std::invalid_argument("Can't prepare input gradients");
        if (!output) return nullptr;
        if (!q) throw std::invalid_argument("Invalid queue ptr");
        if (output->n() != padded_output_width())
            throw std::invalid_argument("Dimension mismatch grid encoding forw_impl");

        const uint32_t batch_size = input.m();

        // zero the padded values, i.e., the last
        {
            auto out = output->data() + m_n_output_dims;
            const size_t bytes_to_zero = m_n_to_pad * sizeof(T);
            const int stride = padded_output_width();

            for (int iter = 0; iter < batch_size; iter++) {
                q->memset(out + iter * stride, 0, bytes_to_zero);
            }

            q->wait();
        }

        // Idea: each block only takes care of _one_ hash level (but may iterate
        // over multiple input elements). This way, only one level of the hashmap
        // needs to fit into caches at a time (and it reused for consecutive
        // elements) until it is time to process the next level.

        static constexpr uint32_t N_THREADS_HASHGRID = 512;
        const sycl::range<3> blocks_hashgrid = {1, m_n_levels,
                                                tinydpcppnn::math::div_round_up(batch_size, N_THREADS_HASHGRID)};

        DeviceMem<T> encoded_positions_soa(output->size(), *q);

        // TODO: Fix the following
        q->submit([&](sycl::handler &cgh) {
            uint32_t pin_stride_i = input.view().stride_i;
            uint32_t pin_stride_j = input.view().stride_j;

            cgh.parallel_for(sycl::nd_range<3>(blocks_hashgrid * sycl::range<3>(1, 1, N_THREADS_HASHGRID),
                                               sycl::range<3>(1, 1, N_THREADS_HASHGRID)),
                             [=](sycl::nd_item<3> item) {
                                 kernel_grid<T, N_POS_DIMS, N_FEATURES_PER_LEVEL, HASH_TYPE>(
                                     batch_size, m_n_features, m_offset_table, m_base_resolution,
                                     std::log2(m_per_level_scale), this->m_max_level, this->m_max_level_gpu,
                                     m_interpolation_type, m_grid_type, this->params(), input.data(), pin_stride_i,
                                     pin_stride_j, encoded_positions_soa.data(), nullptr, item);
                             });
        });
        q->wait();

        // Transpose result (was stored row major due to coalescing)
        // TODO: translate this
        const sycl::range<3> threads_transpose = {1, 8, m_n_levels * N_FEATURES_PER_LEVEL};
        const uint32_t blocks_transpose = tinydpcppnn::math::div_round_up(batch_size, (uint32_t)threads_transpose[1]);
        q->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, blocks_transpose) * threads_transpose, threads_transpose),
                [=](sycl::nd_item<3> item) {
                    transpose_encoded_position<T>(batch_size, encoded_positions_soa.data(), output->pitched_ptr(),
                                                  item);
                });
        });
        q->wait();

        return nullptr;
    }

    void backward_impl(sycl::queue *const q, const Context &ctx, const DeviceMatrix<float> &input,
                       const DeviceMatrix<T> &output, const DeviceMatrix<T> &dL_doutput,
                       DeviceMatrix<float> *dL_dinput = nullptr, bool use_inference_params = false,
                       GradientMode param_gradients_mode = GradientMode::Overwrite) override {

        throw std::logic_error("Not yet implemented");
    }

    void backward_backward_input_impl(sycl::queue *const q, const Context &ctx, const DeviceMatrix<float> &input,
                                      const DeviceMatrix<float> &dL_ddLdinput, const DeviceMatrix<T> &dL_doutput,
                                      DeviceMatrix<T> *dL_ddLdoutput = nullptr,
                                      DeviceMatrix<float> *dL_dinput = nullptr, bool use_inference_params = false,
                                      GradientMode param_gradients_mode = GradientMode::Overwrite) // TODO: override
    {
        throw std::logic_error("Not yet implemented");
    }

    uint32_t input_width() const override { return N_POS_DIMS; }

    uint32_t padded_output_width() const override { return m_n_output_dims + m_n_to_pad; }

    uint32_t output_width() const override { return padded_output_width(); }

    void set_padded_output_width(uint32_t padded_output_width) override {
        if (padded_output_width < m_n_output_dims) throw std::invalid_argument("Invalid padding.");
        m_n_to_pad = padded_output_width - m_n_output_dims;
    }

    void set_params_impl(T *params, T *inference_params,
                         T *gradients) // TODO: override
    {}

    void initialize_params(float *params_full_precision, float scale = 1) override {
        // // Initialize the hashgrid from the GPU, because the number of parameters
        // can be quite large. generate_random_uniform<float>(rnd, this->n_params(),
        // params_full_precision, -1e-4f * scale, 1e-4f * scale);

        constexpr std::uint64_t seed = 777;
        oneapi::mkl::rng::philox4x32x10 engine(dpct::get_default_queue(), seed);
        oneapi::mkl::rng::uniform<float> distribution(-1e-4f * scale, 1e-4f * scale);
        oneapi::mkl::rng::generate(distribution, engine, this->n_params(), params_full_precision).wait();
    }

    size_t level_n_params(uint32_t level) const override {
        return level_params_offset(level + 1) - level_params_offset(level);
    }

    size_t level_params_offset(uint32_t level) const override {
        if (level >= m_offset_table.size) throw std::runtime_error{"Out of bounds params offset request."};
        return m_offset_table.data[level];
    }

    const GridOffsetTable &grid_offset_table() const override { return m_offset_table; }

    std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const // TODO: override
    {
        // Even though we have parameters, they can't really be considered a
        // "layer". So we return an empty array here.
        return {};
    }

    uint32_t n_pos_dims() const override { return N_POS_DIMS; }

    uint32_t n_features_per_level() const override { return N_FEATURES_PER_LEVEL; }

    json hyperparams() const // TODO: override
    {
        json result = {
            {"otype", "Grid"},
            {"type", to_string(m_grid_type)},
            {"n_levels", m_n_levels},
            {"n_features_per_level", N_FEATURES_PER_LEVEL},
            {"base_resolution", m_base_resolution},
            {"per_level_scale", m_per_level_scale},
            {"interpolation", to_string(m_interpolation_type)},
            {"hash", to_string(HASH_TYPE)},
        };

        if (m_grid_type == GridType::Hash) {
            result["log2_hashmap_size"] = m_log2_hashmap_size;
        }

        return result;
    }

  private:
    struct ForwardContext : public Context {
        DeviceMatrix<float, MatrixLayout::RowMajor> positions;
        DeviceMatrix<float, MatrixLayout::RowMajor> dy_dx;
    };

    uint32_t m_n_features;
    uint32_t m_n_levels;
    GridOffsetTable m_offset_table;
    uint32_t m_log2_hashmap_size;
    uint32_t m_base_resolution;

    uint32_t m_n_dims_to_pass_through;

    // derived sizes
    uint32_t m_n_output_dims;
    uint32_t m_n_to_pad = 0;

    float m_per_level_scale;

    // uint32_t this->m_n_params;
    bool m_stochastic_interpolation;
    InterpolationType m_interpolation_type;
    GridType m_grid_type;
};

template <typename T, uint32_t N_FEATURES_PER_LEVEL, HashType HASH_TYPE>
std::shared_ptr<GridEncoding<T>> create_grid_encoding_templated_2(uint32_t n_dims_to_encode, const json &encoding) {
    const uint32_t log2_hashmap_size = encoding.value("log2_hashmap_size", 19u);
    const std::string encoding_type = encoding.value("otype", "Grid");
    const std::string default_type = equals_case_insensitive(encoding_type, "TiledGrid")
                                         ? "Tiled"
                                         : (equals_case_insensitive(encoding_type, "DenseGrid") ? "Dense" : "Hash");

    uint32_t n_features;
    if (encoding.contains("n_features") || encoding.contains("n_grid_features")) {
        n_features = encoding.contains("n_features") ? encoding["n_features"] : encoding["n_grid_features"];
        if (encoding.contains("n_levels")) {
            throw std::runtime_error{"GridEncoding: may not specify n_features and n_levels "
                                     "simultaneously (one determines the other)"};
        }
    } else {
        n_features = N_FEATURES_PER_LEVEL * encoding.value("n_levels", 16u);
    }

    const uint32_t n_levels = n_features / N_FEATURES_PER_LEVEL;
    const GridType grid_type = string_to_grid_type(encoding.value("type", default_type));
    const uint32_t base_resolution = encoding.value("base_resolution", 16u);

#define TCNN_GRID_PARAMS                                                                                               \
    n_features, log2_hashmap_size, base_resolution,                                                                    \
        encoding.value("per_level_scale", grid_type == GridType::Dense                                                 \
                                              ? std::exp(std::log(256.0f / (float)base_resolution) / (n_levels - 1))   \
                                              : 2.0f),                                                                 \
        encoding.value("stochastic_interpolation", false),                                                             \
        string_to_interpolation_type(encoding.value("interpolation", "Linear")), grid_type

    // If higher-dimensional hash encodings are desired, corresponding switch
    // cases can be added
    switch (n_dims_to_encode) {
    // case 1: return new GridEncodingTemplated<T, 1, N_FEATURES_PER_LEVEL,
    // HASH_TYPE>{ TCNN_GRID_PARAMS };
    case 2:
        return std::make_shared<GridEncodingTemplated<T, 2, N_FEATURES_PER_LEVEL, HASH_TYPE>>(TCNN_GRID_PARAMS);
    case 3:
        return std::make_shared<GridEncodingTemplated<T, 3, N_FEATURES_PER_LEVEL, HASH_TYPE>>(TCNN_GRID_PARAMS);
    case 4:
        return std::make_shared<GridEncodingTemplated<T, 4, N_FEATURES_PER_LEVEL, HASH_TYPE>>(TCNN_GRID_PARAMS);
    // case 5: return new GridEncodingTemplated<T, 5, N_FEATURES_PER_LEVEL,
    // HASH_TYPE>{ TCNN_GRID_PARAMS }; case 6: return new
    // GridEncodingTemplated<T, 6, N_FEATURES_PER_LEVEL, HASH_TYPE>{
    // TCNN_GRID_PARAMS }; case 7: return new GridEncodingTemplated<T, 7,
    // N_FEATURES_PER_LEVEL, HASH_TYPE>{ TCNN_GRID_PARAMS };
    default:
        throw std::runtime_error{"GridEncoding: number of input dims must be 2 or 3."};
    }
#undef TCNN_GRID_PARAMS
}

template <typename T, HashType HASH_TYPE>
std::shared_ptr<GridEncoding<T>> create_grid_encoding_templated_1(uint32_t n_dims_to_encode, const json &encoding) {
    const uint32_t n_features_per_level = encoding.value("n_features_per_level", 2u);
    switch (n_features_per_level) {
    case 1:
        return create_grid_encoding_templated_2<T, 1, HASH_TYPE>(n_dims_to_encode, encoding);
    case 2:
        return create_grid_encoding_templated_2<T, 2, HASH_TYPE>(n_dims_to_encode, encoding);
    case 4:
        return create_grid_encoding_templated_2<T, 4, HASH_TYPE>(n_dims_to_encode, encoding);
    case 8:
        return create_grid_encoding_templated_2<T, 8, HASH_TYPE>(n_dims_to_encode, encoding);
    default:
        throw std::runtime_error{"GridEncoding: n_features_per_level must be 1, 2, 4, or 8."};
    }
}

template <typename T>
std::shared_ptr<GridEncoding<T>> create_grid_encoding(uint32_t n_dims_to_encode, const json &encoding) {
    const HashType hash_type = string_to_hash_type(encoding.value("hash", "CoherentPrime"));
    switch (hash_type) {
    case HashType::Prime:
        return create_grid_encoding_templated_1<T, HashType::Prime>(n_dims_to_encode, encoding);
    case HashType::CoherentPrime:
        return create_grid_encoding_templated_1<T, HashType::CoherentPrime>(n_dims_to_encode, encoding);
    case HashType::ReversedPrime:
        return create_grid_encoding_templated_1<T, HashType::ReversedPrime>(n_dims_to_encode, encoding);
    case HashType::Rng:
        return create_grid_encoding_templated_1<T, HashType::Rng>(n_dims_to_encode, encoding);
    default:
        throw std::runtime_error{"GridEncoding: invalid hash type."};
    }
}

#endif // Include guard.