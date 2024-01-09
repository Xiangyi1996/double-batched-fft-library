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

#include "common_device.h"
#include <DeviceMem.h>
#include <common.h>
#include <encoding.h>
#include <grid_interface.h>

#include <sycl/sycl.hpp>
// #include <multi_stream.h>
// #include <random.h>

#include <stdint.h>
#include <vec.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "oneapi/mkl/rng.hpp"

using json = nlohmann::json;

// template <typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL,
// HashType HASH_TYPE> void kernel_grid(
//     const uint32_t num_elements,
//     const uint32_t num_grid_features,
//     const GridOffsetTable offset_table,
//     const uint32_t base_resolution,
//     const float log2_per_level_scale,
//     float max_level,
//     const float* __restrict__ max_level_gpu,
//     const InterpolationType interpolation_type,
//     const GridType grid_type,
//     const T* __restrict__ grid,
//     MatrixView<const float> positions_in,
//     T* __restrict__ encoded_positions,
//     float* __restrict__ dy_dx,
//     const sycl::nd_item<3> &item_ct1 )
// {
//     const uint32_t i = item_ct1.get_global_id(2);
//     if (i >= num_elements) return;

//     const uint32_t level = item_ct1.get_group(1); // <- the level is the same
//     for all threads

//     max_level = max_level_gpu ?
//                     ( (max_level_gpu[i] * num_grid_features) /
//                     N_FEATURES_PER_LEVEL ) : ( (max_level        *
//                     num_grid_features) / N_FEATURES_PER_LEVEL ) ;

//     if ( level >= max_level + 1e-3f )
//     {
//         if (encoded_positions)
//         {
//             for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f)
//                 encoded_positions[i + (level * N_FEATURES_PER_LEVEL + f) *
//                 num_elements] = (T)0.0f;
//         }

//         // Gradient is zero for zeroed-out dimensions.
//         if (dy_dx)
//         {
//             for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f)
//                 ((tnn::vec<N_POS_DIMS>*)dy_dx)[i + (level *
//                 N_FEATURES_PER_LEVEL + f) * num_elements] = {0.0f};
//         }

//         return;
//     }

//     grid += offset_table.data[level] * N_FEATURES_PER_LEVEL;
//     const uint32_t hashmap_size = offset_table.data[level + 1] -
//     offset_table.data[level];

//     const float scale = grid_scale(level, log2_per_level_scale,
//     base_resolution); const uint32_t resolution = grid_resolution(scale);

//     float pos[N_POS_DIMS];
//     float pos_derivative[N_POS_DIMS];
//     tnn::uvec<N_POS_DIMS> pos_grid;

//     if (interpolation_type == InterpolationType::Nearest ||
//     interpolation_type == InterpolationType::Linear)
//     {
//         for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim)
//         {
//             auto input = positions_in(dim, i);
//             auto l_pos = &pos[dim];
//             auto l_pos_derivative = &pos_derivative[dim];
//             auto l_pos_grid = &pos_grid[dim];
//             // scale

//             *l_pos = sycl::fma(scale, (float)input, 0.5f);
//             float tmp = sycl::floor(*l_pos);
//             *l_pos_grid = (uint32_t)(int)tmp;
//             *l_pos -= (float)tmp;
//             *l_pos_derivative = identity_derivative(*l_pos);
//             *l_pos = identity_fun(*l_pos);
//             //pos_fract(positions_in(dim, i), &pos[dim], &pos_grid[dim],
//             scale,  []( args ) { identity(args); },  [](args) {df(args);},
//             [](args){d2f(args);} )
//         }
//          //   pos_fract(positions_in(dim, i), &pos[dim],
//          &pos_derivative[dim], &pos_grid[dim], scale, identity_fun,
//          identity_derivative);
//     }
//     else
//     {
//         for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim)
//         {
//             auto input = positions_in(dim, i);
//             auto l_pos = &pos[dim];
//             auto l_pos_derivative = &pos_derivative[dim];
//             auto l_pos_grid = &pos_grid[dim];
//             // scale

//             *l_pos = sycl::fma(scale, (float)input, 0.5f);
//             float tmp = sycl::floor(*l_pos);
//             *l_pos_grid = (uint32_t)(int)tmp;
//             *l_pos -= (float)tmp;
//             *l_pos_derivative = smoothstep_derivative(*l_pos);
//             *l_pos = smoothstep(*l_pos);
//             //pos_fract(positions_in(dim, i), &pos[dim], &pos_grid[dim],
//             scale,  []( args ) { identity(args); },  [](args) {df(args);},
//             [](args){d2f(args);} )
//         }
//          //   pos_fract(positions_in(dim, i), &pos[dim],
//          &pos_derivative[dim], &pos_grid[dim], scale, smoothstep,
//          smoothstep_derivative);
//     }

//     auto grid_val = [&](const tnn::uvec<N_POS_DIMS>& local_pos)
//     {
//         const uint32_t index = grid_index<N_POS_DIMS, HASH_TYPE>(grid_type,
//         hashmap_size, resolution, local_pos) * N_FEATURES_PER_LEVEL; return
//         *(tnn::tvec<T, N_FEATURES_PER_LEVEL, PARAMS_ALIGNED ? sizeof(T) *
//         N_FEATURES_PER_LEVEL : sizeof(T)>*)&grid[index];
//     };

//     if (interpolation_type == InterpolationType::Nearest)
//     {
//         auto result = grid_val(pos_grid);

//         if (encoded_positions)
//         {
//             for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f)
//                 encoded_positions[i + (level * N_FEATURES_PER_LEVEL + f) *
//                 num_elements] = result[f];
//         }

//         // Gradient is zero when there's no interpolation.
//         if (dy_dx)
//         {
//             for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f)
//                 ((tnn::vec<N_POS_DIMS>*)dy_dx)[i + (level *
//                 N_FEATURES_PER_LEVEL + f) * num_elements] = {0.0f};
//         }

//         return;
//     }

//     if (encoded_positions)
//     {
//         // N-linear interpolation
//         tnn::tvec<T, N_FEATURES_PER_LEVEL, PARAMS_ALIGNED ? sizeof(T) *
//         N_FEATURES_PER_LEVEL : sizeof(T)> result = {};

//         for ( uint32_t idx = 0; idx < (1 << N_POS_DIMS); ++idx )
//         {
//             float weight = 1;
//             tnn::uvec<N_POS_DIMS> pos_grid_local;

//             for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim)
//             {
//                 if ((idx & (1<<dim)) == 0)
//                 {
//                     weight *= 1 - pos[dim];
//                     pos_grid_local[dim] = pos_grid[dim];
//                 }
//                 else
//                 {
//                     weight *= pos[dim];
//                     pos_grid_local[dim] = pos_grid[dim] + 1;
//                 }
//             }

//             result = fma((T)weight, grid_val(pos_grid_local), result);
//         }

//         for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f)
//             encoded_positions[i + (level * N_FEATURES_PER_LEVEL + f) *
//             num_elements] = result[f];
//     }

//     // Gradient
//     if (dy_dx)
//     {
//         tnn::vec<N_POS_DIMS> grads[N_FEATURES_PER_LEVEL] = {0.0f};

//         for (uint32_t grad_dim = 0; grad_dim < N_POS_DIMS; ++grad_dim)
//         {

//             for (uint32_t idx = 0; idx < (1 << (N_POS_DIMS-1)); ++idx)
//             {
//                 float weight = scale;
//                 tnn::uvec<N_POS_DIMS> pos_grid_local;

//                 for (uint32_t non_grad_dim = 0; non_grad_dim < N_POS_DIMS-1;
//                 ++non_grad_dim)
//                 {
//                     const uint32_t dim = non_grad_dim >= grad_dim ?
//                     (non_grad_dim+1) : non_grad_dim;

//                     if ((idx & (1<<non_grad_dim)) == 0)
//                     {
//                         weight *= 1 - pos[dim];
//                         pos_grid_local[dim] = pos_grid[dim];
//                     }
//                     else
//                     {
//                         weight *= pos[dim];
//                         pos_grid_local[dim] = pos_grid[dim] + 1;
//                     }
//                 }

//                 pos_grid_local[grad_dim] = pos_grid[grad_dim];
//                 auto val_left = grid_val(pos_grid_local);
//                 pos_grid_local[grad_dim] = pos_grid[grad_dim] + 1;
//                 auto val_right = grid_val(pos_grid_local);

//                 for (uint32_t feature = 0; feature < N_FEATURES_PER_LEVEL;
//                 ++feature)
//                     grads[feature][grad_dim] += weight *
//                     ((float)val_right[feature] - (float)val_left[feature]) *
//                     pos_derivative[grad_dim];
//             }
//         }

//         for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f)
//             ((tnn::vec<N_POS_DIMS>*)dy_dx)[i + (level * N_FEATURES_PER_LEVEL
//             + f) * num_elements] = grads[f];
//     }
// }

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

/*
template <typename T, typename GRAD_T, uint32_t N_POS_DIMS, uint32_t
N_FEATURES_PER_LEVEL, uint32_t N_FEATURES_PER_THREAD, HashType HASH_TYPE> void
kernel_grid_backward( const uint32_t num_elements, const uint32_t
num_grid_features, const GridOffsetTable offset_table, const uint32_t
base_resolution, const float log2_per_level_scale, float max_level, const float*
__restrict__ max_level_gpu, const bool stochastic_interpolation, const
InterpolationType interpolation_type, const GridType grid_type, GRAD_T*
__restrict__ grid_gradient, MatrixView<const float> positions_in, const T*
__restrict__ dL_dy, const sycl::nd_item<3> &item_ct1)
{
    const uint32_t i = ( item_ct1.get_global_id(2) * N_FEATURES_PER_THREAD ) /
N_FEATURES_PER_LEVEL; if (i >= num_elements) return;

    const uint32_t level   = item_ct1.get_group(1); // <- the level is the same
for all threads. const uint32_t feature = item_ct1.get_global_id(2) *
N_FEATURES_PER_THREAD - i*N_FEATURES_PER_LEVEL;

    max_level = max_level_gpu ?
                    ( (max_level_gpu[i] * num_grid_features) /
N_FEATURES_PER_LEVEL ) : ( (max_level        * num_grid_features) /
N_FEATURES_PER_LEVEL ) ;

    if (level >= max_level + 1e-3f) return;

    grid_gradient += offset_table.data[level] * N_FEATURES_PER_LEVEL;
    const uint32_t hashmap_size = offset_table.data[level + 1] -
offset_table.data[level];

    const float scale = grid_scale(level, log2_per_level_scale,
base_resolution); const uint32_t resolution = grid_resolution(scale);

    auto add_grid_gradient =
    [&](const tnn::uvec<N_POS_DIMS> &local_pos, const tnn::tvec<GRAD_T,
N_FEATURES_PER_THREAD> &grad, const float weight )
    {
        uint32_t index = grid_index<N_POS_DIMS, HASH_TYPE>(grid_type,
hashmap_size, resolution, local_pos) * N_FEATURES_PER_LEVEL + feature;

        // Original
        // atomic_add_gmem(grid_gradient + index, (GRAD_T)weight * grad);
        // TODO: Consider optimisations for half-precision, once they are
supported by SYCL. for ( size_t i = 0; i < N_FEATURES_PER_THREAD; ++i )
        {
            sycl::atomic_ref<GRAD_T,sycl::memory_order::relaxed,sycl::memory_scope::device>
atomic( grid_gradient + index + i ); atomic.fetch_add( ((GRAD_T)weight) *
grad[i] );
        }
    };

    float pos[N_POS_DIMS];
    tnn::uvec<N_POS_DIMS> pos_grid;

    if (interpolation_type == InterpolationType::Nearest || interpolation_type
== InterpolationType::Linear)
    {
        for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
            pos_fract(input,pos,pos_derivative,pos_2nd_derivative,pos_grid,scale,
[]( args ) { f(args); },  [](args) {df(args);},  [](args){d2f(args);} )
        }
        //pos_fract(positions_in(dim, i), &pos[dim], &pos_grid[dim], scale,
identity_fun);
    }
    else
    {
        for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim)
            pos_fract(positions_in(dim, i), &pos[dim], &pos_grid[dim], scale,
smoothstep);
    }

    tnn::tvec<T, N_FEATURES_PER_THREAD> grad;

    for (uint32_t f = 0; f < N_FEATURES_PER_THREAD; ++f)
        grad[f] = dL_dy[i + (level * N_FEATURES_PER_LEVEL + feature + f) *
num_elements];

    if (interpolation_type == InterpolationType::Nearest)
    {
        add_grid_gradient(pos_grid, grad, 1.0f);
        return;
    }

    if (stochastic_interpolation)
    {
        float sample = random_val(1337, i + level * num_elements);
        tnn::uvec<N_POS_DIMS> pos_grid_local;

        for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim)
            pos_grid_local[dim] = (sample>=pos[dim]) ? (pos_grid[dim]) :
(pos_grid[dim]+1);

        add_grid_gradient(pos_grid_local, grad, 1.0f);
        return;
    }

    // N-linear interpolation

    for (uint32_t idx = 0; idx < (1 << N_POS_DIMS); ++idx)
    {
        float weight = 1;
        tnn::uvec<N_POS_DIMS> pos_grid_local;

        for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
            if ((idx & (1<<dim)) == 0) {
                weight *= 1 - pos[dim];
                pos_grid_local[dim] = pos_grid[dim];
            } else {
                weight *= pos[dim];
                pos_grid_local[dim] = pos_grid[dim] + 1;
            }
        }

        add_grid_gradient(pos_grid_local, grad, weight);
    }
}
*/

template <typename T, typename GRAD_T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL,
          uint32_t N_FEATURES_PER_THREAD, HashType HASH_TYPE>
void kernel_grid_backward(const uint32_t num_elements, const uint32_t num_grid_features,
                          const GridOffsetTable offset_table, const uint32_t base_resolution,
                          const float log2_per_level_scale, float max_level, const float *__restrict__ max_level_gpu,
                          const bool stochastic_interpolation, const InterpolationType interpolation_type,
                          const GridType grid_type, GRAD_T *__restrict__ grid_gradient, float *positions_in_data,
                          uint32_t pin_stride_i, uint32_t pin_stride_j,
                          // MatrixView<const float> positions_in,
                          const T *__restrict__ dL_dy, const sycl::nd_item<3> &item_ct1) {
    MatrixView<const float> positions_in(positions_in_data, pin_stride_i, pin_stride_j);
    const uint32_t i = (item_ct1.get_global_id(2) * N_FEATURES_PER_THREAD) / N_FEATURES_PER_LEVEL;
    if (i >= num_elements) return;

    const uint32_t level = item_ct1.get_group(1); // <- the level is the same for all threads.
    const uint32_t feature = item_ct1.get_global_id(2) * N_FEATURES_PER_THREAD - i * N_FEATURES_PER_LEVEL;

    max_level = max_level_gpu ? ((max_level_gpu[i] * num_grid_features) / N_FEATURES_PER_LEVEL)
                              : ((max_level * num_grid_features) / N_FEATURES_PER_LEVEL);

    if (level >= max_level + 1e-3f) return;

    grid_gradient += offset_table.data[level] * N_FEATURES_PER_LEVEL;
    const uint32_t hashmap_size = offset_table.data[level + 1] - offset_table.data[level];

    const float scale = grid_scale(level, log2_per_level_scale, base_resolution);
    const uint32_t resolution = grid_resolution(scale);

    auto add_grid_gradient = [&](const tnn::uvec<N_POS_DIMS> &local_pos,
                                 const tnn::tvec<GRAD_T, N_FEATURES_PER_THREAD> &grad, const float weight) {
        uint32_t index =
            grid_index<N_POS_DIMS, HASH_TYPE>(grid_type, hashmap_size, resolution, local_pos) * N_FEATURES_PER_LEVEL +
            feature;

        // Original
        // atomic_add_gmem(grid_gradient + index, (GRAD_T)weight * grad);
        // TODO: Consider optimisations for half-precision, once they are
        // supported by SYCL.
        for (size_t i = 0; i < N_FEATURES_PER_THREAD; ++i) {
            int tmp = 0;
            // sycl::atomic_ref<GRAD_T,sycl::memory_order::relaxed,sycl::memory_scope::device>
            // atomic( *grid_gradient /*+ index + i*/ ); atomic.fetch_add(
            // ((GRAD_T)weight) * grad[i] );
        }
    };

    float pos[N_POS_DIMS];
    tnn::uvec<N_POS_DIMS> pos_grid;

    if (interpolation_type == InterpolationType::Nearest || interpolation_type == InterpolationType::Linear) {
        for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
            auto input = positions_in(dim, i);
            auto l_pos = &pos[dim];
            auto l_pos_grid = &pos_grid[dim];
            // scale

            *l_pos = sycl::fma(scale, (float)input, 0.5f);
            float tmp = sycl::floor(*l_pos);
            *l_pos_grid = (uint32_t)(int)tmp;
            *l_pos -= tmp;
            *l_pos = identity_fun(*l_pos);
            // pos_fract(positions_in(dim, i), &pos[dim], &pos_grid[dim], scale,  [](
            // args ) { identity(args); },  [](args) {df(args);}, [](args){d2f(args);}
            // )
        }
        // pos_fract(positions_in(dim, i), &pos[dim], &pos_grid[dim], scale,
        // identity_fun);

    } else {
        for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
            auto input = positions_in(dim, i);
            auto l_pos = &pos[dim];
            auto l_pos_grid = &pos_grid[dim];
            // scale

            *l_pos = sycl::fma(scale, (float)input, 0.5f);
            float tmp = sycl::floor(*l_pos);
            *l_pos_grid = (uint32_t)(int)tmp;
            *l_pos -= tmp;
            *l_pos = smoothstep(*l_pos);
            // pos_fract(positions_in(dim, i), &pos[dim], &pos_grid[dim], scale,  [](
            // args ) { identity(args); },  [](args) {df(args);}, [](args){d2f(args);}
            // )
        }
        // pos_fract(positions_in(dim, i), &pos[dim], &pos_grid[dim], scale,
        // smoothstep);
    }

    tnn::tvec<T, N_FEATURES_PER_THREAD> grad;

    for (uint32_t f = 0; f < N_FEATURES_PER_THREAD; ++f)
        grad[f] = dL_dy[i + (level * N_FEATURES_PER_LEVEL + feature + f) * num_elements];

    if (interpolation_type == InterpolationType::Nearest) {
        add_grid_gradient(pos_grid, grad, 1.0f);
        return;
    }

    if (stochastic_interpolation) {
        float sample = random_val(1337, i + level * num_elements);
        tnn::uvec<N_POS_DIMS> pos_grid_local;

        for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim)
            pos_grid_local[dim] = (sample >= pos[dim]) ? (pos_grid[dim]) : (pos_grid[dim] + 1);

        add_grid_gradient(pos_grid_local, grad, 1.0f);
        return;
    }

    // N-linear interpolation

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

        add_grid_gradient(pos_grid_local, grad, weight);
    }
}

template <typename T, uint32_t N_POS_DIMS>
void kernel_grid_backward_input(const uint32_t num_elements, const uint32_t num_grid_features, const T *dL_dy_rm,
                                const float *__restrict__ dy_dx, MatrixView<float> dL_dx,
                                const sycl::nd_item<3> &item_ct1) {
    const uint32_t i = item_ct1.get_global_id(2);
    if (i >= num_elements) return;

    tnn::vec<N_POS_DIMS> result = {0.0f};

    for (int k = 0; k < num_grid_features; ++k) {
        float dL_dy_local = (float)dL_dy_rm[i + k * num_elements];
        auto dy_dx_local = ((tnn::vec<N_POS_DIMS> *)dy_dx)[i + k * num_elements];

        for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim)
            result[dim] += dL_dy_local * dy_dx_local[dim];
    }

    for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim)
        dL_dx(dim, i) = result[dim];
}

template <typename T, typename GRAD_T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL,
          uint32_t N_FEATURES_PER_THREAD, HashType HASH_TYPE>
void kernel_grid_backward_input_backward_grid(const uint32_t num_elements, const uint32_t num_grid_features,
                                              const GridOffsetTable offset_table, const uint32_t base_resolution,
                                              const float log2_per_level_scale, float max_level,
                                              const float *__restrict__ max_level_gpu,
                                              // const bool stochastic_interpolation, // TODO: is this needed?
                                              const InterpolationType interpolation_type, const GridType grid_type,
                                              // inputs
                                              MatrixView<const float> dL_ddLdx, MatrixView<const float> positions_in,
                                              const T *__restrict__ dL_dy,
                                              // outputs
                                              GRAD_T *__restrict__ grid_gradient, const sycl::nd_item<3> &item_ct1) {
    const uint32_t i = (item_ct1.get_global_id(2) * N_FEATURES_PER_THREAD) / N_FEATURES_PER_LEVEL;
    if (i >= num_elements) return;

    const uint32_t level = item_ct1.get_group(1); // <- the level is the same for all threads.
    const uint32_t feature = item_ct1.get_global_id(2) * N_FEATURES_PER_THREAD - i * N_FEATURES_PER_LEVEL;

    max_level = max_level_gpu ? ((max_level_gpu[i] * num_grid_features) / N_FEATURES_PER_LEVEL)
                              : ((max_level * num_grid_features) / N_FEATURES_PER_LEVEL);

    if (level >= max_level + 1e-3f) return;

    grid_gradient += offset_table.data[level] * N_FEATURES_PER_LEVEL;
    const uint32_t hashmap_size = offset_table.data[level + 1] - offset_table.data[level];

    const float scale = grid_scale(level, log2_per_level_scale, base_resolution);
    const uint32_t resolution = grid_resolution(scale);

    auto add_grid_gradient = [&](const tnn::uvec<N_POS_DIMS> &local_pos,
                                 const tnn::tvec<GRAD_T, N_FEATURES_PER_THREAD> &grad, const float weight) {
        const uint32_t index =
            grid_index<N_POS_DIMS, HASH_TYPE>(grid_type, hashmap_size, resolution, local_pos) * N_FEATURES_PER_LEVEL +
            feature;
        // Original
        // atomic_add_gmem(grid_gradient + index, (GRAD_T)weight * grad);
        // TODO: Consider optimisations for half-precision, once they are
        // supported by SYCL.
        for (size_t i = 0; i < N_FEATURES_PER_THREAD; ++i) {
            sycl::atomic_ref<GRAD_T, sycl::memory_order::relaxed, sycl::memory_scope::device> atomic(grid_gradient +
                                                                                                     index + i);
            atomic.fetch_add(((GRAD_T)weight) * grad[i]);
        }
    };

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
            *l_pos -= tmp;
            *l_pos_derivative = identity_derivative(*l_pos);
            *l_pos = identity_fun(*l_pos);
            // pos_fract(positions_in(dim, i), &pos[dim], &pos_grid[dim], scale,  [](
            // args ) { identity(args); },  [](args) {df(args);}, [](args){d2f(args);}
            // )
        }
        // pos_fract(positions_in(dim, i), &pos[dim], &pos_derivative[dim],
        // &pos_grid[dim], scale, identity_fun, identity_derivative);
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
            *l_pos -= tmp;
            *l_pos_derivative = smoothstep_derivative(*l_pos);
            *l_pos = smoothstep(*l_pos);
            // pos_fract(positions_in(dim, i), &pos[dim], &pos_grid[dim], scale,  [](
            // args ) { identity(args); },  [](args) {df(args);}, [](args){d2f(args);}
            // )
        }
        // pos_fract(positions_in(dim, i), &pos[dim], &pos_derivative[dim],
        // &pos_grid[dim], scale, smoothstep, smoothstep_derivative);
    }

    tnn::tvec<T, N_FEATURES_PER_THREAD> grad;

    for (uint32_t f = 0; f < N_FEATURES_PER_THREAD; ++f)
        grad[f] = dL_dy[i + (level * N_FEATURES_PER_LEVEL + feature + f) * num_elements];

    // d(dydx)_dgrid is zero when there's no interpolation.
    if (interpolation_type == InterpolationType::Nearest) return;

    // for N-linear interpolation
    for (uint32_t grad_dim = 0; grad_dim < N_POS_DIMS; ++grad_dim) {
        float grad_in = scale * dL_ddLdx(grad_dim, i) * pos_derivative[grad_dim];

        for (uint32_t idx = 0; idx < (1 << (N_POS_DIMS - 1)); ++idx) {
            float weight = grad_in;
            tnn::uvec<N_POS_DIMS> pos_grid_local;

            for (uint32_t non_grad_dim = 0; non_grad_dim < N_POS_DIMS - 1; ++non_grad_dim) {
                const uint32_t dim = non_grad_dim >= grad_dim ? (non_grad_dim + 1) : non_grad_dim;

                if ((idx & 1 << non_grad_dim) == 0) {
                    weight *= 1 - pos[dim];
                    pos_grid_local[dim] = pos_grid[dim];
                } else {
                    weight *= pos[dim];
                    pos_grid_local[dim] = pos_grid[dim] + 1;
                }
            }

            // left
            pos_grid_local[grad_dim] = pos_grid[grad_dim];
            add_grid_gradient(pos_grid_local, grad, -weight);
            // right
            pos_grid_local[grad_dim] = pos_grid[grad_dim] + 1;
            add_grid_gradient(pos_grid_local, grad, weight);
        }
    }
}

template <typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL, uint32_t N_FEATURES_PER_THREAD,
          HashType HASH_TYPE>
void kernel_grid_backward_input_backward_input(const uint32_t num_elements, const uint32_t num_grid_features,
                                               const GridOffsetTable offset_table, const uint32_t base_resolution,
                                               const float log2_per_level_scale, float max_level,
                                               const float *__restrict__ max_level_gpu,
                                               const InterpolationType interpolation_type, const GridType grid_type,
                                               // inputs
                                               MatrixView<const float> dL_ddLdx, MatrixView<const float> positions_in,
                                               const T *__restrict__ dL_dy, const T *__restrict__ grid,
                                               // outputs
                                               MatrixView<float> dL_dx, const sycl::nd_item<3> &item_ct1) {
    const uint32_t i = (item_ct1.get_global_id(2) * N_FEATURES_PER_THREAD) / N_FEATURES_PER_LEVEL;
    if (i >= num_elements) return;

    const uint32_t level = item_ct1.get_group(1); // <- the level is the same for all threads.
    const uint32_t feature = item_ct1.get_global_id(2) * N_FEATURES_PER_THREAD - i * N_FEATURES_PER_LEVEL;

    max_level = max_level_gpu ? ((max_level_gpu[i] * num_grid_features) / N_FEATURES_PER_LEVEL)
                              : ((max_level * num_grid_features) / N_FEATURES_PER_LEVEL);

    if (level >= max_level + 1e-3f) return;

    grid += offset_table.data[level] * N_FEATURES_PER_LEVEL;
    const uint32_t hashmap_size = offset_table.data[level + 1] - offset_table.data[level];

    const float scale = grid_scale(level, log2_per_level_scale, base_resolution);
    const uint32_t resolution = grid_resolution(scale);

    float pos[N_POS_DIMS];
    float pos_derivative[N_POS_DIMS];
    float pos_2nd_derivative[N_POS_DIMS];
    tnn::uvec<N_POS_DIMS> pos_grid;

    if (interpolation_type == InterpolationType::Nearest || interpolation_type == InterpolationType::Linear) {
        for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
            auto input = positions_in(dim, i);
            auto l_pos = &pos[dim];
            auto l_pos_derivative = &pos_derivative[dim];
            auto l_pos_2nd_derivative = &pos_2nd_derivative[dim];
            auto l_pos_grid = &pos_grid[dim];
            // scale

            *l_pos = sycl::fma(scale, (float)input, 0.5f);
            float tmp = sycl::floor(*l_pos);
            *l_pos_grid = (uint32_t)(int)tmp;
            *l_pos -= (float)tmp;
            *l_pos_2nd_derivative = identity_2nd_derivative(*l_pos);
            *l_pos_derivative = identity_derivative(*l_pos);
            *l_pos = identity_fun(*l_pos);
            // pos_fract(positions_in(dim, i), &pos[dim], &pos_grid[dim], scale,  [](
            // args ) { identity(args); },  [](args) {df(args);}, [](args){d2f(args);}
            // )
        }
        // pos_fract(positions_in(dim, i), &pos[dim], &pos_derivative[dim],
        // &pos_2nd_derivative[dim], &pos_grid[dim], scale, identity_fun,
        // identity_derivative, identity_2nd_derivative);
    } else {
        for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
            auto input = positions_in(dim, i);
            auto l_pos = &pos[dim];
            auto l_pos_derivative = &pos_derivative[dim];
            auto l_pos_2nd_derivative = &pos_2nd_derivative[dim];
            auto l_pos_grid = &pos_grid[dim];
            // scale

            *l_pos = sycl::fma(scale, (float)input, 0.5f);
            float tmp = sycl::floor(*l_pos);
            *l_pos_grid = (uint32_t)(int)tmp;
            *l_pos -= (float)tmp;
            *l_pos_2nd_derivative = smoothstep_2nd_derivative(*l_pos);
            *l_pos_derivative = smoothstep_derivative(*l_pos);
            *l_pos = smoothstep(*l_pos);
            // pos_fract(positions_in(dim, i), &pos[dim], &pos_grid[dim], scale,  [](
            // args ) { identity(args); },  [](args) {df(args);}, [](args){d2f(args);}
            // )
        }
        //   pos_fract(positions_in(dim, i), &pos[dim], &pos_derivative[dim],
        //   &pos_2nd_derivative[dim], &pos_grid[dim], scale, smoothstep,
        //   smoothstep_derivative, smoothstep_2nd_derivative);
    }

    tnn::tvec<T, N_FEATURES_PER_THREAD> grad;

    for (uint32_t f = 0; f < N_FEATURES_PER_THREAD; ++f)
        grad[f] = dL_dy[i + (level * N_FEATURES_PER_LEVEL + feature + f) * num_elements];

    // d(dydx)_dx is zero when there's no interpolation
    if (interpolation_type == InterpolationType::Nearest) return;

    // for N-linear interpolation
    auto calc_dLdx = [&](const tnn::uvec<N_POS_DIMS> &local_pos, const float weight) {
        const uint32_t index =
            grid_index<N_POS_DIMS, HASH_TYPE>(grid_type, hashmap_size, resolution, local_pos) * N_FEATURES_PER_LEVEL +
            feature;
        float dL_dx_dim = 0;

        for (uint32_t f = 0; f < N_FEATURES_PER_THREAD; ++f)
            dL_dx_dim += (float)grid[index + f] * (float)grad[f] * weight;

        return dL_dx_dim;
    };

    tnn::tvec<float, N_POS_DIMS> grad_in_diag;
    tnn::tvec<float, N_POS_DIMS> grad_in_other;

    for (uint32_t grad_dim = 0; grad_dim < N_POS_DIMS; ++grad_dim) {
        // from diagonal part of Hessian
        grad_in_diag[grad_dim] = scale * scale * dL_ddLdx(grad_dim, i) * pos_2nd_derivative[grad_dim];
        // from other part of Hessian
        grad_in_other[grad_dim] =
            scale * scale * dL_ddLdx(grad_dim, i) * pos_derivative[grad_dim]; // will do " *
                                                                              // pos_derivative[real_other_grad_dim]
                                                                              // " later
    }

    static constexpr bool dimension_greater_than_1 = (N_POS_DIMS > 1);

    for (uint32_t grad_dim = 0; grad_dim < N_POS_DIMS; ++grad_dim) {
        float grad_out = 0;

        for (uint32_t idx = 0; idx < (1 << (N_POS_DIMS - 1)); ++idx) {
            // from diagonal part of Hessian; d(doutput_d[grad_dim])_d[grad_dim]
            // NOTE: LinearInterpolations' diagonal part is 0.
            if (interpolation_type == InterpolationType::Smoothstep) {
                float weight_2nd_diag = grad_in_diag[grad_dim];
                tnn::uvec<N_POS_DIMS> pos_grid_local;

                for (uint32_t non_grad_dim = 0; non_grad_dim < N_POS_DIMS - 1; ++non_grad_dim) {
                    const uint32_t dim = non_grad_dim >= grad_dim ? (non_grad_dim + 1) : non_grad_dim;
                    // real non_grad_dim
                    if ((idx & 1 << non_grad_dim) == 0) {
                        weight_2nd_diag *= 1 - pos[dim];
                        pos_grid_local[dim] = pos_grid[dim];
                    } else {
                        weight_2nd_diag *= pos[dim];
                        pos_grid_local[dim] = pos_grid[dim] + 1;
                    }
                }

                // left
                pos_grid_local[grad_dim] = pos_grid[grad_dim];
                grad_out += calc_dLdx(pos_grid_local, -weight_2nd_diag);
                // right
                pos_grid_local[grad_dim] = pos_grid[grad_dim] + 1;
                grad_out += calc_dLdx(pos_grid_local, weight_2nd_diag);
            }

            // from other part of Hessian;
            // d(doutput_d[real_other_grad_dim])_d[grad_dim]
            if (dimension_greater_than_1) {
                for (uint32_t other_grad_dim = 0; other_grad_dim < N_POS_DIMS - 1; ++other_grad_dim) {
                    const uint32_t real_other_grad_dim =
                        other_grad_dim >= grad_dim ? (other_grad_dim + 1) : other_grad_dim;
                    float weight_2nd_other = grad_in_other[real_other_grad_dim] * pos_derivative[grad_dim];
                    tnn::uvec<N_POS_DIMS> pos_grid_local;

                    for (uint32_t non_grad_dim = 0; non_grad_dim < N_POS_DIMS - 1; ++non_grad_dim) {
                        // real non_grad_dim
                        const uint32_t dim = non_grad_dim >= real_other_grad_dim ? (non_grad_dim + 1) : non_grad_dim;
                        if ((idx & 1 << non_grad_dim) == 0) {
                            if (dim != grad_dim)
                                weight_2nd_other *= 1 - pos[dim];
                            else
                                weight_2nd_other *= -1;
                            pos_grid_local[dim] = pos_grid[dim];
                        } else {
                            if (dim != grad_dim) weight_2nd_other *= pos[dim];
                            pos_grid_local[dim] = pos_grid[dim] + 1;
                        }
                    }

                    // left
                    pos_grid_local[real_other_grad_dim] = pos_grid[real_other_grad_dim];
                    grad_out += calc_dLdx(pos_grid_local, -weight_2nd_other);
                    // right
                    pos_grid_local[real_other_grad_dim] = pos_grid[real_other_grad_dim] + 1;
                    grad_out += calc_dLdx(pos_grid_local, weight_2nd_other);
                }
            }
        }

        // Original
        // tnn::atomic_add_gmem_float((float*)&dL_dx(grad_dim, i), grad_out);

        float *target = &(dL_dx(grad_dim, i));
        float val = grad_out;

        sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device> atomic_target(*target);
        atomic_target.fetch_add(val);
    }
}

template <typename T, uint32_t N_POS_DIMS>
void kernel_grid_backward_input_backward_dLdoutput(const uint32_t num_elements, const uint32_t num_grid_features,
                                                   // inputs
                                                   MatrixView<const float> dL_ddLdx, const float *__restrict__ dy_dx,
                                                   const T *dL_dy_rm,
                                                   // ouputs
                                                   MatrixView<T> dL_ddLdy, const sycl::nd_item<3> &item_ct1) {
    const uint32_t i = item_ct1.get_global_id(2);
    if (i >= num_elements) return;

    for (uint32_t k = 0; k < num_grid_features; ++k) {
        auto dy_dx_local = ((tnn::vec<N_POS_DIMS> *)dy_dx)[i + k * num_elements];
        float result = 0;

        for (uint32_t grad_dim = 0; grad_dim < N_POS_DIMS; ++grad_dim)
            result += dy_dx_local[grad_dim] * dL_ddLdx(grad_dim, i);

        dL_ddLdy(k, i) = (T)result;
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

        auto forward = std::make_unique<Context>(); // alternative for now. remove later
        /*
        auto forward = std::make_unique<ForwardContext>();
        const uint32_t num_elements = input.n();
        if ((!output && !prepare_input_gradients) || padded_output_width() == 0 || num_elements == 0) return forward;

        // Take care of padding on the auxiliary stream
        if (output && m_n_to_pad > 0) {
            if (output->layout() == MatrixLayout::AoS) {
                auto n_output_dims = m_n_output_dims;
                auto out = output->pitched_ptr();
                uint32_t n_dims = m_n_to_pad;

                if (n_dims <= 0 || num_elements <= 0) return forward;

                const size_t n_threads = n_dims * tinydpcppnn::math::div_round_up(N_THREADS_LINEAR, n_dims);
                const sycl::range<3> threads = {1, tinydpcppnn::math::div_round_up(N_THREADS_LINEAR, n_dims), n_dims};
                const sycl::range<3> blocks = {
                    1, 1, tinydpcppnn::math::div_round_up((size_t)(num_elements * n_dims), n_threads)};
                const sycl::nd_range<3> ndrange{blocks * threads, threads};

                q->parallel_for(ndrange, [=](nd_item<3> it) {
                     const size_t dim = it.get_local_id(2);
                     const size_t elem = it.get_local_id(1) + it.get_group(2) * it.get_local_range(1);
                     if (dim >= n_dims || elem >= num_elements) return;

                     out(elem)[n_output_dims + dim] = 0;
                 }).wait();
            } else {
                auto out = output->data() + num_elements * m_n_output_dims;
                size_t size = num_elements * m_n_to_pad;

                q->parallel_for(sycl::range<1>(size), [=](size_t i) { out[i] = 0; }).wait();
            }
        }
        q->wait();

        // Idea: each block only takes care of _one_ hash level (but may iterate
        // over multiple input elements). This way, only one level of the hashmap
        // needs to fit into caches at a time (and it reused for consecutive
        // elements) until it is time to process the next level.

        static constexpr uint32_t N_THREADS_HASHGRID = 512;
        const sycl::range<3> blocks_hashgrid = {1, m_n_levels,
                                                tinydpcppnn::math::div_round_up(num_elements, N_THREADS_HASHGRID)};

        T *encoded_positions_soa = output ? output->data() : nullptr;

        DeviceMemArena::Allocation workspace;
        if (output && output->layout() == MatrixLayout::AoS) {
            workspace = allocate_workspace(q, num_elements * m_n_features * sizeof(T));

            encoded_positions_soa = (T *)workspace.data();
        }

        if (prepare_input_gradients) {
            forward->dy_dx = DeviceMatrix<float, MatrixLayout::RowMajor>{N_POS_DIMS * m_n_features, input.n(), *q};
        }
        auto use_inference_params2 = use_inference_params ? this->inference_params() : this->params();
        // std::cout << "use_inference_params: " << use_inference_params << ", pointer: " << use_inference_params2
        //           << std::endl;
        q->submit([&](sycl::handler &cgh) {
            auto m_n_features_ct1 = m_n_features;
            auto m_offset_table_ct2 = m_offset_table;
            auto m_base_resolution_ct3 = m_base_resolution;
            auto std_log2_m_per_level_scale_ct4 = std::log2(m_per_level_scale);
            auto this_m_max_level_ct5 = this->m_max_level;
            auto this_m_max_level_gpu_ct6 = this->m_max_level_gpu;
            auto m_interpolation_type_ct7 = m_interpolation_type;
            auto m_grid_type_ct8 = m_grid_type;
            auto use_inference_params_this_inference_params_this_params_ct9 =
                use_inference_params ? this->inference_params() : this->params();
            auto forward_positions_data_forward_positions_view_input_view_ct10 =
                forward->positions.data() ? forward->positions.view() : input.view();
            float *positions_in_data = forward->positions.data() ? forward->positions.data() : input.data();
            uint32_t pin_stride_i =
                forward->positions.data() ? forward->positions.view().stride_i : input.view().stride_i;
            uint32_t pin_stride_j =
                forward->positions.data() ? forward->positions.view().stride_j : input.view().stride_j;

            auto forward_dy_dx_data_ct12 = forward->dy_dx.data();

            cgh.parallel_for(sycl::nd_range<3>(blocks_hashgrid * sycl::range<3>(1, 1, N_THREADS_HASHGRID),
                                               sycl::range<3>(1, 1, N_THREADS_HASHGRID)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 kernel_grid<T, N_POS_DIMS, N_FEATURES_PER_LEVEL, HASH_TYPE>(
                                     num_elements, m_n_features_ct1, m_offset_table_ct2, m_base_resolution_ct3,
                                     std_log2_m_per_level_scale_ct4, this_m_max_level_ct5, this_m_max_level_gpu_ct6,
                                     m_interpolation_type_ct7, m_grid_type_ct8,
                                     use_inference_params_this_inference_params_this_params_ct9, positions_in_data,
                                     pin_stride_i, pin_stride_j,
                                     // forward_positions_data_forward_positions_view_input_view_ct10,
                                     encoded_positions_soa, forward_dy_dx_data_ct12, item_ct1);
                             });
        });
        q->wait();
        if (output && output->layout() == MatrixLayout::AoS) {
            // Transpose result (was stored row major due to coalescing)
            const sycl::range<3> threads_transpose = {1, 8, m_n_levels * N_FEATURES_PER_LEVEL};
            const uint32_t blocks_transpose =
                tinydpcppnn::math::div_round_up(num_elements, (uint32_t)threads_transpose[1]);
            q->submit([&](sycl::handler &cgh) {
                auto output_pitched_ptr_ct2 = output->pitched_ptr();
                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, blocks_transpose) * threads_transpose, threads_transpose),
                    [=](sycl::nd_item<3> item_ct1) {
                        transpose_encoded_position<T>(num_elements, encoded_positions_soa, output_pitched_ptr_ct2,
                                                      item_ct1);
                    });
            });
            q->wait();
        }
        q->wait();
        */
        return forward;
    }

    void backward_impl(sycl::queue *const q, const Context &ctx, const DeviceMatrix<float> &input,
                       const DeviceMatrix<T> &output, const DeviceMatrix<T> &dL_doutput,
                       DeviceMatrix<float> *dL_dinput = nullptr, bool use_inference_params = false,
                       GradientMode param_gradients_mode = GradientMode::Overwrite) override {

        /*
        const uint32_t num_elements = input.n();
        if ((!dL_dinput && param_gradients_mode == GradientMode::Ignore) || num_elements == 0) return;

        const auto &forward = dynamic_cast<const ForwardContext &>(ctx);

        const T *dL_dy_rm = dL_doutput.data();

        DeviceMemArena::Allocation workspace;
        if (dL_doutput.layout() == MatrixLayout::ColumnMajor) {
            workspace = allocate_workspace(q, num_elements * m_n_features * sizeof(T));

            // Transpose dL_dy. Use the buffer previously occupied by the encoded
            // positions
            const sycl::range<3> threads_transpose = {1, 8, m_n_levels * N_FEATURES_PER_LEVEL};
            const uint32_t blocks_transpose =
                tinydpcppnn::math::div_round_up(num_elements, (uint32_t)threads_transpose[1]);
            q->submit([&](sycl::handler &cgh) {
                auto workspace_data_ct1 = (T *)workspace.data();
                auto dL_doutput_pitched_ptr_ct2 = dL_doutput.pitched_ptr();

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, blocks_transpose) * threads_transpose, threads_transpose),
                    [=](sycl::nd_item<3> item_ct1) {
                        transpose_gradients<T>(num_elements, workspace_data_ct1, dL_doutput_pitched_ptr_ct2, item_ct1);
                    });
            });

            dL_dy_rm = (const T *)workspace.data();
        }

        if (param_gradients_mode != GradientMode::Ignore) {
            // We accumulate gradients with grad_t precision, which, for performance
            // reasons, is not always T. If not, accumulate in a temporary buffer and
            // cast later.
            grad_t *grid_gradient;
            DeviceMemArena::Allocation grid_gradient_tmp;

            if (!std::is_same<grad_t, T>::value) {
                grid_gradient_tmp = allocate_workspace(q, this->m_n_params * sizeof(grad_t));
                grid_gradient = (grad_t *)grid_gradient_tmp.data();
            } else {
                grid_gradient = (grad_t *)this->gradients();
            }

            if (param_gradients_mode == GradientMode::Overwrite) {
                q->memset(grid_gradient, 0, this->n_params() * sizeof(grad_t));
            }

            static constexpr uint32_t N_THREADS_HASHGRID = 256;
            static constexpr uint32_t N_FEATURES_PER_THREAD = std::min(2u, N_FEATURES_PER_LEVEL);

            const sycl::range<3> blocks_hashgrid = {
                1, m_n_levels,
                tinydpcppnn::math::div_round_up(num_elements * N_FEATURES_PER_LEVEL / N_FEATURES_PER_THREAD,
                                                N_THREADS_HASHGRID)};

            q->submit([&](sycl::handler &cgh) {
                auto m_n_features_ct1 = m_n_features;
                auto m_offset_table_ct2 = m_offset_table;
                auto m_base_resolution_ct3 = m_base_resolution;
                auto std_log2_m_per_level_scale_ct4 = std::log2(m_per_level_scale);
                auto this_m_max_level_ct5 = this->m_max_level;
                auto this_m_max_level_gpu_ct6 = this->m_max_level_gpu;
                auto m_stochastic_interpolation_ct7 = m_stochastic_interpolation;
                auto m_interpolation_type_ct8 = m_interpolation_type;
                auto m_grid_type_ct9 = m_grid_type;
                auto forward_positions_data_forward_positions_view_input_view_ct11 =
                    forward.positions.data() ? forward.positions.view() : input.view();
                float *positions_in_data = forward.positions.data() ? forward.positions.data() : input.data();
                uint32_t pin_stride_i =
                    forward.positions.data() ? forward.positions.view().stride_i : input.view().stride_i;
                uint32_t pin_stride_j =
                    forward.positions.data() ? forward.positions.view().stride_j : input.view().stride_j;

                cgh.parallel_for(sycl::nd_range<3>(blocks_hashgrid * sycl::range<3>(1, 1, N_THREADS_HASHGRID),
                                                   sycl::range<3>(1, 1, N_THREADS_HASHGRID)),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     kernel_grid_backward<T, grad_t, N_POS_DIMS, N_FEATURES_PER_LEVEL,
                                                          N_FEATURES_PER_THREAD, HASH_TYPE>(
                                         num_elements, m_n_features_ct1, m_offset_table_ct2, m_base_resolution_ct3,
                                         std_log2_m_per_level_scale_ct4, this_m_max_level_ct5, this_m_max_level_gpu_ct6,
                                         m_stochastic_interpolation_ct7, m_interpolation_type_ct8, m_grid_type_ct9,
                                         grid_gradient, positions_in_data, pin_stride_i, pin_stride_j,
                                         // forward_positions_data_forward_positions_view_input_view_ct11,
                                         // // MatrixView
                                         dL_dy_rm, item_ct1);
                                 });
            });

            if constexpr (!std::is_same<grad_t, T>::value) {
                auto grad = this->gradients();
                auto grad_tmp = grid_gradient;

                q->parallel_for(sycl::range<1>(this->n_params()), [&](size_t i) { grad[i] = (T)grad_tmp[i]; });
            }
        }

        if (!dL_dinput) return;

        linear_kernel(kernel_grid_backward_input<T, N_POS_DIMS>, 0, q, num_elements, m_n_features, dL_dy_rm,
                      forward.dy_dx.data(), dL_dinput->view());
                      */
    }

    void backward_backward_input_impl(sycl::queue *const q, const Context &ctx, const DeviceMatrix<float> &input,
                                      const DeviceMatrix<float> &dL_ddLdinput, const DeviceMatrix<T> &dL_doutput,
                                      DeviceMatrix<T> *dL_ddLdoutput = nullptr,
                                      DeviceMatrix<float> *dL_dinput = nullptr, bool use_inference_params = false,
                                      GradientMode param_gradients_mode = GradientMode::Overwrite) // TODO: override
    {
        /*
        const uint32_t num_elements = input.n();
        if ((!dL_ddLdoutput && param_gradients_mode == GradientMode::Ignore) || padded_output_width() == 0 ||
            num_elements == 0)
            return;

        const auto &forward = dynamic_cast<const ForwardContext &>(ctx);

        const T *dL_dy_rm = dL_doutput.data();

        DeviceMemArena::Allocation workspace;
        if (dL_doutput.layout() == MatrixLayout::ColumnMajor) {
            workspace = allocate_workspace(q, num_elements * m_n_features * sizeof(T));

            // Transpose dL_dy. Use the buffer previously occupied by the encoded
            // positions
            const sycl::range<3> threads_transpose = {1, 8, m_n_levels * N_FEATURES_PER_LEVEL};
            const uint32_t blocks_transpose =
                tinydpcppnn::math::div_round_up(num_elements, (uint32_t)threads_transpose[1]);
            q->submit([&](sycl::handler &cgh) {
                auto workspace_data_ct1 = (T *)workspace.data();
                auto dL_doutput_pitched_ptr_ct2 = dL_doutput.pitched_ptr();

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, blocks_transpose) * threads_transpose, threads_transpose),
                    [=](sycl::nd_item<3> item_ct1) {
                        transpose_gradients<T>(num_elements, workspace_data_ct1, dL_doutput_pitched_ptr_ct2, item_ct1);
                    });
            });

            dL_dy_rm = (const T *)workspace.data();
        }

        if (param_gradients_mode != GradientMode::Ignore) {
            // We accumulate gradients with grad_t precision, which, for performance
            // reasons, is not always T. If not, accumulate in a temporary buffer and
            // cast later.
            grad_t *grid_gradient;
            DeviceMemArena::Allocation grid_gradient_tmp;

            if (!std::is_same<grad_t, T>::value) {
                grid_gradient_tmp = allocate_workspace(q, this->m_n_params * sizeof(grad_t));
                grid_gradient = (grad_t *)grid_gradient_tmp.data();
            } else {
                grid_gradient = (grad_t *)this->gradients();
            }

            if (param_gradients_mode == GradientMode::Overwrite) {
                q->memset(grid_gradient, 0, this->n_params() * sizeof(grad_t));
            }

            static constexpr uint32_t N_THREADS_HASHGRID = 256;
            static constexpr uint32_t N_FEATURES_PER_THREAD = std::min(2u, N_FEATURES_PER_LEVEL);

            const sycl::range<3> blocks_hashgrid = {
                1, m_n_levels,
                tinydpcppnn::math::div_round_up(num_elements * N_FEATURES_PER_LEVEL / N_FEATURES_PER_THREAD,
                                                N_THREADS_HASHGRID)};

            // from dL_d(dL_dx) to dL_dgrid
            q->submit([&](sycl::handler &cgh) {
                auto m_n_features_ct1 = m_n_features;
                auto m_offset_table_ct2 = m_offset_table;
                auto m_base_resolution_ct3 = m_base_resolution;
                auto std_log2_m_per_level_scale_ct4 = std::log2(m_per_level_scale);
                auto this_m_max_level_ct5 = this->m_max_level;
                auto this_m_max_level_gpu_ct6 = this->m_max_level_gpu;
                auto m_interpolation_type_ct7 = m_interpolation_type;
                auto m_grid_type_ct8 = m_grid_type;
                auto dL_ddLdinput_view_ct9 = dL_ddLdinput.view();
                auto forward_positions_data_forward_positions_view_input_view_ct10 =
                    forward.positions.data() ? forward.positions.view() : input.view();

                cgh.parallel_for(
                    sycl::nd_range<3>(blocks_hashgrid * sycl::range<3>(1, 1, N_THREADS_HASHGRID),
                                      sycl::range<3>(1, 1, N_THREADS_HASHGRID)),
                    [=](sycl::nd_item<3> item_ct1) {
                        kernel_grid_backward_input_backward_grid<T, grad_t, N_POS_DIMS, N_FEATURES_PER_LEVEL,
                                                                 N_FEATURES_PER_THREAD, HASH_TYPE>(
                            num_elements, m_n_features_ct1, m_offset_table_ct2, m_base_resolution_ct3,
                            std_log2_m_per_level_scale_ct4, this_m_max_level_ct5, this_m_max_level_gpu_ct6,
                            m_interpolation_type_ct7, m_grid_type_ct8, dL_ddLdinput_view_ct9,
                            forward_positions_data_forward_positions_view_input_view_ct10, dL_dy_rm, grid_gradient,
                            item_ct1);
                    });
            });

            if constexpr (!std::is_same<grad_t, T>::value) {
                auto grad = this->gradients();
                auto grad_tmp = grid_gradient;
                q->parallel_for(sycl::range<1>(this->n_params()), [&](size_t i) { grad[i] = (T)grad_tmp[i]; });
            }
        }

        if (dL_ddLdoutput) {
            // from dL_d(dL_dx) to dL_doutput
            linear_kernel(kernel_grid_backward_input_backward_dLdoutput<T, N_POS_DIMS>, 0, q, num_elements,
                          m_n_features,
                          // inputs
                          dL_ddLdinput.view(), forward.dy_dx.data(), dL_dy_rm,
                          // outputs
                          dL_ddLdoutput->view());
        }

        if (dL_dinput) {
            static constexpr uint32_t N_THREADS_HASHGRID = 256;
            static constexpr uint32_t N_FEATURES_PER_THREAD = std::min(2u, N_FEATURES_PER_LEVEL);

            const sycl::range<3> blocks_hashgrid = {
                1, m_n_levels,
                tinydpcppnn::math::div_round_up(num_elements * N_FEATURES_PER_LEVEL / N_FEATURES_PER_THREAD,
                                                N_THREADS_HASHGRID)};

            // from dL_d(dL_dx) to dL_dx
            q->submit([&](sycl::handler &cgh) {
                auto m_n_features_ct1 = m_n_features;
                auto m_offset_table_ct2 = m_offset_table;
                auto m_base_resolution_ct3 = m_base_resolution;
                auto std_log2_m_per_level_scale_ct4 = std::log2(m_per_level_scale);
                auto this_m_max_level_ct5 = this->m_max_level;
                auto this_m_max_level_gpu_ct6 = this->m_max_level_gpu;
                auto m_interpolation_type_ct7 = m_interpolation_type;
                auto m_grid_type_ct8 = m_grid_type;
                auto dL_ddLdinput_view_ct9 = dL_ddLdinput.view();
                auto forward_positions_data_forward_positions_view_input_view_ct10 =
                    forward.positions.data() ? forward.positions.view() : input.view();
                auto use_inference_params_this_inference_params_this_params_ct12 =
                    use_inference_params ? this->inference_params() : this->params();
                auto dL_dinput_view_ct13 = dL_dinput->view();

                cgh.parallel_for(sycl::nd_range<3>(blocks_hashgrid * sycl::range<3>(1, 1, N_THREADS_HASHGRID),
                                                   sycl::range<3>(1, 1, N_THREADS_HASHGRID)),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     kernel_grid_backward_input_backward_input<T, N_POS_DIMS, N_FEATURES_PER_LEVEL,
                                                                               N_FEATURES_PER_THREAD, HASH_TYPE>(
                                         num_elements, m_n_features_ct1, m_offset_table_ct2, m_base_resolution_ct3,
                                         std_log2_m_per_level_scale_ct4, this_m_max_level_ct5, this_m_max_level_gpu_ct6,
                                         m_interpolation_type_ct7, m_grid_type_ct8, dL_ddLdinput_view_ct9,
                                         forward_positions_data_forward_positions_view_input_view_ct10, dL_dy_rm,
                                         use_inference_params_this_inference_params_this_params_ct12,
                                         dL_dinput_view_ct13, item_ct1);
                                 });
            });
        }
        */
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