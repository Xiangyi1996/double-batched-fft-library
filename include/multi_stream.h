/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** @file   multi_stream.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Helper class for parallelizing workload across multiple streams.
 */

#pragma once

#include <common.h>
#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>

#include <chrono>
#include <stack>

namespace tnn {

void free_multi_streams(dpct::queue_ptr parent_stream);

// Synchronization helpers
struct StreamAndEvent {
  public:
    StreamAndEvent() try {
        /*
        DPCT1003:112: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CUDA_CHECK_THROW((m_stream = dpct::get_current_device().create_queue(), 0));
        /*
        DPCT1003:113: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CUDA_CHECK_THROW((m_event = new sycl::event(), 0));
    } catch (sycl::exception const &exc) {
        std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
        std::exit(1);
    }

    ~StreamAndEvent() {
        if (m_stream) {
            free_multi_streams(m_stream);
            // TODO: free_gpu_memory_arena(m_stream);
            dpct::get_current_device().destroy_queue(m_stream);
        }

        if (m_event) {
            dpct::destroy_event(m_event);
        }
    }

    // Only allow moving of these guys. No copying.
    StreamAndEvent &operator=(const StreamAndEvent &) = delete;
    StreamAndEvent(const StreamAndEvent &) = delete;
    StreamAndEvent &operator=(StreamAndEvent &&other) {
        std::swap(m_stream, other.m_stream);
        std::swap(m_event, other.m_event);
        return *this;
    }

    StreamAndEvent(StreamAndEvent &&other) { *this = std::move(other); }

    void wait_for(dpct::event_ptr event) try {
        /*
        DPCT1003:114: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CUDA_CHECK_THROW((m_stream->ext_oneapi_submit_barrier({*event}), 0));
    } catch (sycl::exception const &exc) {
        std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
        std::exit(1);
    }

    void wait_for(dpct::queue_ptr stream) try {
        /*
        DPCT1012:115: Detected kernel execution time measurement pattern
        and generated an initial code for time measurements in SYCL. You
        can change the way time is measured depending on your goals.
        */
        /*
        DPCT1024:116: The original code returned the error code that was
        further consumed by the program logic. This original code was
        replaced with 0. You may need to rewrite the program logic
        consuming the error code.
        */
        m_event_ct1 = std::chrono::steady_clock::now();
        CUDA_CHECK_THROW((*m_event = stream->ext_oneapi_submit_barrier(), 0));
        wait_for(m_event);
    } catch (sycl::exception const &exc) {
        std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
        std::exit(1);
    }

    void signal(dpct::queue_ptr stream) try {
        /*
        DPCT1012:117: Detected kernel execution time measurement pattern
        and generated an initial code for time measurements in SYCL. You
        can change the way time is measured depending on your goals.
        */
        /*
        DPCT1024:118: The original code returned the error code that was
        further consumed by the program logic. This original code was
        replaced with 0. You may need to rewrite the program logic
        consuming the error code.
        */
        m_event_ct1 = std::chrono::steady_clock::now();
        CUDA_CHECK_THROW((*m_event = m_stream->ext_oneapi_submit_barrier(), 0));
        /*
        DPCT1003:119: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CUDA_CHECK_THROW((stream->ext_oneapi_submit_barrier({*m_event}), 0));
    } catch (sycl::exception const &exc) {
        std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
        std::exit(1);
    }

    dpct::queue_ptr get() { return m_stream; }

  private:
    dpct::queue_ptr m_stream = {};
    dpct::event_ptr m_event = {};
    std::chrono::time_point<std::chrono::steady_clock> m_event_ct1;
};

struct MultiStream {
  public:
    MultiStream() try {
        /*
        DPCT1003:120: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CUDA_CHECK_THROW((m_event = new sycl::event(), 0));
    } catch (sycl::exception const &exc) {
        std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
        std::exit(1);
    }

    ~MultiStream() { dpct::destroy_event(m_event); }

    MultiStream &operator=(const MultiStream &) = delete;
    MultiStream(const MultiStream &) = delete;
    MultiStream &operator=(MultiStream &&) = delete;
    MultiStream(MultiStream &&) = delete;

    void signal(dpct::queue_ptr outer_stream) {
        for (size_t i = 0; i < m_n_streams; ++i) {
            m_streams[i].signal(outer_stream);
        }
    }

    void wait_for(dpct::queue_ptr stream) try {
        if (m_n_streams == 0) {
            return;
        }

        /*
        DPCT1012:121: Detected kernel execution time measurement pattern
        and generated an initial code for time measurements in SYCL. You
        can change the way time is measured depending on your goals.
        */
        /*
        DPCT1024:122: The original code returned the error code that was
        further consumed by the program logic. This original code was
        replaced with 0. You may need to rewrite the program logic
        consuming the error code.
        */
        m_event_ct1 = std::chrono::steady_clock::now();
        CUDA_CHECK_THROW((*m_event = stream->ext_oneapi_submit_barrier(), 0));
        for (size_t i = 0; i < m_n_streams; ++i) {
            m_streams[i].wait_for(m_event);
        }
    } catch (sycl::exception const &exc) {
        std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
        std::exit(1);
    }

    void resize(size_t n_streams) {
        if (n_streams > m_streams.size()) {
            m_streams.resize(n_streams);
        }
        m_n_streams = n_streams;
    }

    dpct::queue_ptr get(size_t idx) {
        if (idx >= m_n_streams) {
            throw std::runtime_error{"MultiStream: invalid stream index requested: {}/{}"}; //, idx, m_n_streams)};
        }
        return m_streams.at(idx).get();
    }

  private:
    std::vector<StreamAndEvent> m_streams;
    // May be less than m_streams.size()!
    // The user may only need to sync fewer than that.
    size_t m_n_streams = 0;
    dpct::event_ptr m_event;
    std::chrono::time_point<std::chrono::steady_clock> m_event_ct1;
};

inline std::unordered_map<dpct::queue_ptr, std::stack<std::shared_ptr<MultiStream>>> &stream_multi_streams() {
    static auto *stream_multi_streams =
        new std::unordered_map<dpct::queue_ptr, std::stack<std::shared_ptr<MultiStream>>>{};
    return *stream_multi_streams;
}

inline std::unordered_map<int, std::stack<std::shared_ptr<MultiStream>>> &global_multi_streams() {
    static auto *global_multi_streams = new std::unordered_map<int, std::stack<std::shared_ptr<MultiStream>>>{};
    return *global_multi_streams;
}

inline std::stack<std::shared_ptr<MultiStream>> &get_multi_stream_stack(dpct::queue_ptr parent_stream) {
    return parent_stream ? stream_multi_streams()[parent_stream] : global_multi_streams()[cuda_device()];
}

inline void free_multi_streams(dpct::queue_ptr parent_stream) {
    CHECK_THROW(parent_stream);

    // Copy the multi stream shared_ptr's into a separate variable,
    // such that their destruction happens after unordered_map::erase(...)
    // is already finished. This alleviates potential non-reentrancy problems.
    auto multi_streams = stream_multi_streams()[parent_stream];
    stream_multi_streams().erase(parent_stream);
}

inline std::shared_ptr<MultiStream> reserve_multi_stream(dpct::queue_ptr parent_stream, size_t n_streams) {
    auto &stack = get_multi_stream_stack(parent_stream);
    if (stack.empty()) {
        stack.push(std::make_shared<MultiStream>());
    }
    auto result = stack.top();
    stack.pop();

    result->resize(n_streams);
    return result;
}

inline void return_multi_stream(dpct::queue_ptr parent_stream, std::shared_ptr<MultiStream> multi_stream) {
    if (parent_stream ? (stream_multi_streams().count(parent_stream) == 0)
                      : (global_multi_streams().count(cuda_device()) == 0)) {
        throw std::runtime_error{"Attempted to return multi stream to the wrong parent stream."};
    }

    auto &stack = get_multi_stream_stack(parent_stream);
    stack.push(multi_stream);
}

// RAII wrapper around MultiStream
struct SyncedMultiStream {
  public:
    SyncedMultiStream() = default;
    SyncedMultiStream(dpct::queue_ptr stream, size_t n_streams) : m_main_stream{stream}, m_n_streams{n_streams} {
        if (m_n_streams == 0) {
            throw std::runtime_error{"SyncedMultiStream: must request at least one stream"};
        } else if (m_n_streams == 1) {
            return;
        }

        m_multi_stream = reserve_multi_stream(m_main_stream, m_n_streams - 1);
        m_multi_stream->wait_for(m_main_stream);
    }

    ~SyncedMultiStream() {
        if (m_multi_stream) {
            m_multi_stream->signal(m_main_stream);
            return_multi_stream(m_main_stream, m_multi_stream);
        }
    }

    // Only allow moving of these guys. No copying.
    SyncedMultiStream &operator=(const SyncedMultiStream &other) = delete;
    SyncedMultiStream(const SyncedMultiStream &) = delete;

    SyncedMultiStream &operator=(SyncedMultiStream &&other) {
        std::swap(m_multi_stream, other.m_multi_stream);
        std::swap(m_main_stream, other.m_main_stream);
        std::swap(m_n_streams, other.m_n_streams);
        return *this;
    }

    SyncedMultiStream(SyncedMultiStream &&other) { *this = std::move(other); }

    dpct::queue_ptr get(size_t idx) {
        if (m_n_streams == 0) {
            throw std::runtime_error{"SyncedMultiStream: must have at least one stream"};
        }

        if (idx == 0) {
            return m_main_stream;
        } else {
            if (!m_multi_stream) {
                throw std::runtime_error{"SyncedMultiStream: invalid multistream"};
            }

            return m_multi_stream->get(idx - 1);
        }
    }

  private:
    std::shared_ptr<MultiStream> m_multi_stream = nullptr;
    dpct::queue_ptr m_main_stream = &dpct::get_default_queue();
    size_t m_n_streams = 0;
};

} // namespace tnn
