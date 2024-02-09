/**
 * @file kernel_esimd.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Esimd implementation of the forward, backward and inference kernels class.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <algorithm>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>
#include <vector>

#include "DeviceMatrix.h"
#include "common.h"
#include "oneapi/mkl.hpp"

namespace sycl::ext::intel::esimd::xmx {
template <int SystolicDepth, int RepeatCount, typename T, typename CT, typename BT, typename AT,
          dpas_argument_type BPrecision = detail::dpas_precision_from_type<BT>(),
          dpas_argument_type APrecision = detail::dpas_precision_from_type<AT>(), int N, int N_orig, int BN, int AN,
          int AN_orig>
__ESIMD_NS::simd<T, N> dpas(__ESIMD_NS::simd_view<simd<CT, N_orig>, region1d_t<CT, N, 1>> C, __ESIMD_NS::simd<BT, BN> B,
                            __ESIMD_NS::simd_view<simd<AT, AN_orig>, region1d_t<AT, AN, 1>> A) {
    (void)detail::verify_parameters_and_deduce_exec_size<SystolicDepth, RepeatCount, T, CT, BT, AT, BPrecision,
                                                         APrecision, BN, AN>();

    using MsgT = int;
    constexpr int ANCasted = AN * sizeof(AT) / sizeof(MsgT);
    constexpr int BNCasted = BN * sizeof(BT) / sizeof(MsgT);
    __ESIMD_NS::simd<MsgT, ANCasted> ACasted = A.template bit_cast_view<MsgT>();
    __ESIMD_NS::simd<MsgT, BNCasted> BCasted = B.template bit_cast_view<MsgT>();
    using CRawT = typename __ESIMD_NS::simd<CT, N>::raw_element_type;
    using RawT = typename __ESIMD_NS::simd<T, N>::raw_element_type;
    return __esimd_dpas2<BPrecision, APrecision, SystolicDepth, RepeatCount, RawT, CRawT, MsgT, MsgT, N, BNCasted,
                         ANCasted>(C.data(), BCasted.data(), ACasted.data());
}
}; // namespace sycl::ext::intel::esimd::xmx

namespace tinydpcppnn {
namespace kernels {
namespace esimd {

using namespace sycl::ext::intel::esimd;
using sycl::ext::intel::experimental::esimd::cache_hint;
using namespace sycl::ext::intel::experimental::esimd;
using bf16 = sycl::ext::oneapi::bfloat16;

/**
 * @brief Struct to decide type for accumulation (CType) for XMX at compile time,
 * depending on the given type T.
 *
 * Currently returns CType == T, except when T == bf16, then CType == float.
 *
 * @tparam T
 */
template <typename T> struct XMXCType {
    typedef T CType;
};
template <> struct XMXCType<bf16> {
    typedef float CType;
};
template <> struct XMXCType<sycl::half> {
#if TARGET_DEVICE == 0
    typedef sycl::half CType;
#elif TARGET_DEVICE == 1
    typedef float CType;
#endif
};

/**
 * @brief Struct which gives us the value to use for TN in the dpas instruction
 * Depending on the device.
 *
 */
struct XMXTn {
#if TARGET_DEVICE == 0
    static constexpr int TN = 16;
#elif TARGET_DEVICE == 1
    static constexpr int TN = 8;
#endif
};

/**
 * @brief Struct to give us the maximum number of bytes in a send instruction,
 * depending on the device
 *
 */
struct XMXMaxSendBytes {
#if TARGET_DEVICE == 0
    static constexpr int MaxBytes = 512;
#elif TARGET_DEVICE == 1
    static constexpr int MaxBytes = 256;
#endif
};

/**
 * @brief
 *
 * @tparam T type for the computations. Everything that is supported by xmx::dpas
 * should work fine.
 * @tparam INPUT_WIDTH The width of the input layer of the network. In general
 * it should be a multiple of TK, right now it is equal to WIDTH.
 * @tparam WIDTH Denotes the width of every hidden layer, may be 16, 32, 64, 128.
 * @tparam OUTPUT_WIDTH The width of the output layer, currently equal to WIDTH. Later a multiple of TN.
 * @tparam activation Activation function. Currently either none or ReLU.
 * @tparam output_activation Activation for the output layer. Currently None.
 * @tparam TN Device dependent, whatever is supported by the chosen device. 8 for DG2, 16 for PVC.
 */
template <typename T, int INPUT_WIDTH, int WIDTH, int OUTPUT_WIDTH, Activation activation, Activation output_activation>
class EsimdKernels {

    using Tc = typename XMXCType<T>::CType;
    static constexpr int TN = XMXTn::TN;

  public:
    static std::vector<sycl::event> forward_impl(sycl::queue &q, const DeviceMatricesView<T> &weights,
                                                 const DeviceMatrixView<T> &input,
                                                 DeviceMatricesView<T> intermediate_output, const int n_hidden_layers,
                                                 const std::vector<sycl::event> &deps) {
        return forward_impl_general<false>(q, weights, input, intermediate_output, n_hidden_layers, deps);
    }

    static std::vector<sycl::event> backward_impl(sycl::queue &q, const DeviceMatricesView<T> &weights,
                                                  const DeviceMatrixView<T> &input, DeviceMatricesView<T> output,
                                                  DeviceMatricesView<T> intermediate_backward,
                                                  const DeviceMatricesView<T> &intermediate_forward,
                                                  const int n_hidden_layers, const std::vector<sycl::event> &deps) {

        // make sure there is no remainder and no out of bounds accesses
        static_assert(WIDTH % TN == 0);
        // only works for input_width == width == output_width
        static_assert(INPUT_WIDTH == WIDTH);
        static_assert(OUTPUT_WIDTH == WIDTH);
        const size_t M = input.m();

        constexpr int SG_SIZE = TN;
        // this may be adjusted in the future in dpendence of M
        constexpr size_t TM = 8;
        assert(M % TM == 0);
        int ITEMS_IN_WG = std::min<int>(M / TM, 64);
        /// TODO: say we use M/TM = 65. Then this results in WG=1 SG and too many slm load of B.
        /// Better: Use max size WGs and return those which are larger than M/TM. But
        /// requires special care for the loading of B
        while (M / TM % ITEMS_IN_WG != 0) {
            ITEMS_IN_WG--;
        }
        if (ITEMS_IN_WG <= 0) throw std::logic_error("Number of SGS per WG cannot be less than 1");

        assert(M % TM == 0);
        // TK depends on the datatype T
        constexpr size_t TK = 8 * std::min<size_t>(8, 32 / (8 * sizeof(T)));

        auto e = q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(deps);

            cgh.parallel_for(sycl::nd_range<1>(M / TM, ITEMS_IN_WG), [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const size_t loc_row_offset = item.get_global_linear_id() * TM;

                simd<T, TM * WIDTH> As;
                loadRow<TM, TK, cache_hint::uncached, cache_hint::uncached>(input.GetPointer(loc_row_offset, 0), As);

                // store backward activated input to the last intermediate output
                // note that output_activation == ReLU does not need any work since that means
                // forward >= 0
                if constexpr (output_activation != Activation::None && output_activation != Activation::ReLU) {
                    // applyBackwardActivation<output_activation, TM, WIDTH>(
                    //     sg, A_sg_start, forward_loc + layer_offset_A + M * WIDTH, A_sg_start);
                }

                // store activated in intermediate output
                storeRow<TM, TK, cache_hint::uncached, cache_hint::uncached>(
                    As, intermediate_backward.GetElementPointer(n_hidden_layers, loc_row_offset, 0));
                simd<Tc, TM * WIDTH> Cs;
                // we are also doing output->last hidden layer
                for (int layer = n_hidden_layers; layer > 0; layer--) {
                    Cs = static_cast<Tc>(0);

                    MAD<TM, TK>(As, weights.GetMatrixPointer(layer), Cs);

                    applyBackwardActivation<activation, TM, TK>(
                        intermediate_forward.GetElementPointer(layer, loc_row_offset, 0), Cs, As);

                    storeRow<TM, TK, cache_hint::uncached, cache_hint::uncached>(
                        As, intermediate_backward.GetElementPointer(layer - 1, loc_row_offset, 0));
                }
            });
        });

        // NOTE: MKL gemm_batch is slower. Rescale with 1/M for stability.
        std::vector<sycl::event> events(n_hidden_layers + 1);
        if constexpr (std::is_same<T, sycl::ext::oneapi::bfloat16>::value) { // need to cast to onemkls bf16 type.
            for (int iter = 0; iter < n_hidden_layers + 1; iter++) {
                events[iter] = oneapi::mkl::blas::row_major::gemm(
                    q, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, WIDTH, WIDTH, M,
                    (float)(1.0 / M),
                    reinterpret_cast<const oneapi::mkl::bfloat16 *>(intermediate_forward.GetMatrixPointer(iter)), WIDTH,
                    reinterpret_cast<oneapi::mkl::bfloat16 *>(intermediate_backward.GetMatrixPointer(iter)), WIDTH, 0,
                    reinterpret_cast<oneapi::mkl::bfloat16 *>(output.GetMatrixPointer(iter)), WIDTH, {e});
            }
        } else if constexpr (!std::is_same<T, sycl::ext::oneapi::bfloat16>::value) {
            for (int iter = 0; iter < n_hidden_layers + 1; iter++) {
                events[iter] = oneapi::mkl::blas::row_major::gemm(
                    q, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, WIDTH, WIDTH, M, 1.0 / M,
                    intermediate_forward.GetMatrixPointer(iter), WIDTH, intermediate_backward.GetMatrixPointer(iter),
                    WIDTH, 0, output.GetMatrixPointer(iter), WIDTH, {e});
            }
        }
        return events;
    }

    static std::vector<sycl::event> inference_impl(sycl::queue &q, const DeviceMatricesView<T> &weights,
                                                   const DeviceMatrixView<T> &input,
                                                   DeviceMatricesView<T> intermediate_output, const int n_hidden_layers,
                                                   const std::vector<sycl::event> &deps) {
        return forward_impl_general<true>(q, weights, input, intermediate_output, n_hidden_layers, deps);
    }

    /*************the following functions are only public for testing purposes*******************/

    // in register everything is in block major format with blocks of size TMxTK
    template <int TM, int TK, cache_hint L1, cache_hint L3, int TMWIDTH>
    SYCL_ESIMD_FUNCTION static void storeRow(simd<T, TMWIDTH> &src, T *const dest) {

        static_assert(TM == 1 || TM == 2 || TM == 4 || TM == 8);
        static_assert(WIDTH % TK == 0);
        static_assert(TMWIDTH == TM * WIDTH);
        static_assert(sizeof(T) <= 4);

        constexpr int rows_per_load = std::min<int>(XMXMaxSendBytes::MaxBytes / (WIDTH * sizeof(T)), TM);
        static_assert(rows_per_load > 0);
        auto src_2d = src.template bit_cast_view<T, TMWIDTH / TK, TK>(); // block major

#pragma unroll
        for (int row = 0; row < TM; row += rows_per_load) {
            simd<T, WIDTH * rows_per_load> tmp;
#pragma unroll
            for (int locrowiter = 0; locrowiter < rows_per_load; locrowiter++) {
                tmp.template select<WIDTH, 1>(locrowiter * WIDTH) =
                    src_2d.template select<WIDTH / TK, TM, TK, 1>(row + locrowiter, 0);
            }
            lsc_block_store<T, rows_per_load * WIDTH, lsc_data_size::default_size, L1, L3>(dest + row * WIDTH, tmp,
                                                                                           overaligned_tag<8>());
        }
    }

    // in register everything is in block major format with blocks of size TMxTK
    template <int TM, int TK, cache_hint L1, cache_hint L3, int TMWIDTH>
    SYCL_ESIMD_FUNCTION static void loadRow(T const *const src, simd<T, TMWIDTH> &dest) {
        static_assert(TM == 1 || TM == 2 || TM == 4 || TM == 8);
        static_assert(WIDTH % TK == 0);
        static_assert(TMWIDTH == TM * WIDTH);
        static_assert(sizeof(T) <= 4);
        constexpr int elems_per_pos = 4 / sizeof(T);
        constexpr int blocks_per_load = TK * elems_per_pos > WIDTH ? 1 : elems_per_pos;
        constexpr int nloads = WIDTH / (TK * blocks_per_load);
        static_assert(nloads > 0);

// DG2 does not have 2d send instructions
#if TARGET_DEVICE == 0
        auto dest_int = dest.template bit_cast_view<int32_t>();
#pragma unroll
        for (int load_iter = 0; load_iter < nloads; load_iter++) {
            dest_int.template select<TM * TK / elems_per_pos * blocks_per_load, 1>(TM * TK / elems_per_pos *
                                                                                   blocks_per_load * load_iter) =
                lsc_load_2d<int32_t, TK / elems_per_pos, TM, blocks_per_load, false, false, L1, L3>(
                    reinterpret_cast<int32_t const *>(src), WIDTH * sizeof(T) - 1, TM - 1, WIDTH * sizeof(T) - 1,
                    load_iter * TK, 0);
        }
#elif TARGET_DEVICE == 1
        for (int blockiter = 0; blockiter < WIDTH / TK; blockiter++) {
            for (int rowiter = 0; rowiter < TM; rowiter++) {
                dest.template select<TK, 1>(blockiter * TM * TK + rowiter * TK) =
                    lsc_block_load<T, TK, lsc_data_size::default_size, L1, L3>(src + blockiter * TK + rowiter * WIDTH);
            }
        }
#endif
    }

    // we are assuming a block major layout and vnni'd B
    template <int TM, int TK, int TMWIDTH>
    SYCL_ESIMD_FUNCTION static void MAD(simd<T, TMWIDTH> &As, T const *const __restrict__ B, simd<Tc, TMWIDTH> &Cs) {
        static_assert(TM >= 1 && TM <= 8);
        static_assert(TN == 16 || TN == 8);
        static_assert(TMWIDTH % TM == 0);
        static_assert(TMWIDTH / TM == WIDTH);
        static_assert(WIDTH % TK == 0 && WIDTH % TN == 0);
        static_assert(sizeof(T) <= 4 && sizeof(Tc) <= 4);
        constexpr int vnni_factor = std::max<int>(1, 4 / sizeof(T));

#pragma collapse 2 unroll
        for (int iterA = 0; iterA < WIDTH; iterA += TK) {
            for (int iterB = 0; iterB < WIDTH; iterB += TN) {
                simd<T, TK * TN> BlockB;
                auto BlockB_float = BlockB.template bit_cast_view<float>();
#if TARGET_DEVICE == 0

                BlockB_float =
                    lsc_load_2d<float, TN, TK / vnni_factor, 1, false, false, cache_hint::cached, cache_hint::cached>(
                        reinterpret_cast<float const *>(B), vnni_factor * WIDTH * sizeof(T) - 1,
                        WIDTH / vnni_factor - 1, vnni_factor * WIDTH * sizeof(T) - 1, iterB, iterA / vnni_factor);
#elif TARGET_DEVICE == 1
                for (int rowiter = 0; rowiter < TK / vnni_factor; rowiter++) {
                    // BlockB_float[rowiter] = rowiter;
                    // BlockB.template select<vnni_factor * TN, 1>(rowiter * vnni_factor * TN) =
                    //     lsc_block_load<T, vnni_factor * TN>(B + vnni_factor *iterB + iterA * WIDTH + rowiter *
                    //     vnni_factor * WIDTH);
                    BlockB_float.template select<TN, 1>(rowiter * TN) =
                        lsc_block_load<float, TN, lsc_data_size::default_size, cache_hint::cached, cache_hint::cached>(
                            reinterpret_cast<float const *>(B) + iterB + iterA / vnni_factor * WIDTH + rowiter * WIDTH);
                }
#endif
                Cs.template select<TM * TN, 1>(iterB * TM) = xmx::dpas<8, TM, Tc>(
                    Cs.template select<TM * TN, 1>(iterB * TM), BlockB, As.template select<TM * TK, 1>(iterA * TM));
            }
        }
    }

    template <Activation act, int TM, int TK, int N, typename Tsrc, typename Tdest>
    SYCL_ESIMD_FUNCTION static void applyActivation(simd<Tsrc, N> &Src, simd<Tdest, N> &Dest) {
        static_assert(TM >= 1 && TM <= 8);
        static_assert(TN == 16 || TN == 8);
        static_assert(TK == 8 || TK == 16 || TK == 32 || TK == 64);

        if constexpr (act == Activation::None)
            reBlock<TM, TK>(convert<Tdest, Tsrc>(Src), Dest);
        else if constexpr (act == Activation::ReLU)
            reBlock<TM, TK>(max<Tdest>(convert<Tdest, Tsrc>(Src), simd<Tdest, N>(static_cast<Tdest>(0))), Dest);
    }

    template <Activation act, int TM, int TK, int N, typename Tdec, typename Tsrc, typename Tdest>
    SYCL_ESIMD_FUNCTION static void applyBackwardActivation(Tdec const *const Dec, simd<Tsrc, N> &Src,
                                                            simd<Tdest, N> &Dest) {
        static_assert(TM >= 1 && TM <= 8);
        static_assert(TN == 16 || TN == 8);
        static_assert(TK == 8 || TK == 16 || TK == 32 || TK == 64);

        if constexpr (act == Activation::None)
            reBlock<TM, TK>(convert<Tdest, Tsrc>(Src), Dest);
        else if constexpr (act == Activation::ReLU) {
            simd<Tdec, N> loc_dec;
            loadRow<TM, TN, cache_hint::uncached, cache_hint::uncached>(Dec, loc_dec);
            simd_mask<N> m = loc_dec <= simd<Tdec, N>(0);
            Src.merge(simd<Tsrc, N>(0), m); // ATTENTION: this changes Src.
            reBlock<TM, TK>(convert<Tdest, Tsrc>(Src), Dest);
        }
    }

    // TK == 8, 16, 32, 64; TN == 8 (DG2), 16 (PVC)
    // Src contains of blocks sized TM*TN
    // Dest of blocks sized TM*TK
    template <int TM, int TK, int N> SYCL_ESIMD_FUNCTION static void reBlock(simd<T, N> Src, simd<T, N> &Dest) {
        static_assert(TK == TN || TK == 2 * TN || TK == 4 * TN || TK == 8 * TN || TK == TN / 2);

        if constexpr (TK == TN) {
            Dest = Src;
        } else if constexpr (TK == 2 * TN || TK == 4 * TN || TK == 8 * TN) {
            constexpr int FAC = TK / TN;
            auto Dest_2d = Dest.template bit_cast_view<T, N / TK, TK>(); // block major 2d layout
            for (int iter = 0; iter < WIDTH / TN; iter++) {              // iterate over the blocks in SRC
                Dest_2d.template select<TM, 1, TN, 1>((iter / FAC) * TM, (iter % FAC) * TN) =
                    Src.template select<TM * TN, 1>(iter * TM * TN);
            }
        } else if constexpr (TK == TN / 2) {
            auto Src_2d = Src.template bit_cast_view<T, N / TN, TN>(); // block major 2d layout
            for (int iter = 0; iter < WIDTH / TK; iter++) {            // iterate over the blocks in SRC
                Dest.template select<TM * TK, 1>(iter * TM * TK) =
                    Src_2d.template select<TM, 1, TK, 1>((iter / 2) * TM, (iter % 2) * TK);
            }
        }
    }

  private:
    template <bool INFERENCE>
    static std::vector<sycl::event>
    forward_impl_general(sycl::queue &q, const DeviceMatricesView<T> &weights, const DeviceMatrixView<T> &input,
                         DeviceMatricesView<T> intermediate_output, const int n_hidden_layers,
                         const std::vector<sycl::event> &deps) {

        // throw std::logic_error("General function should not be called.");
        const size_t M = input.m();
        static_assert(INPUT_WIDTH == WIDTH);
        static_assert(OUTPUT_WIDTH == WIDTH);
        static_assert(WIDTH % TN == 0);

        constexpr int TM = 8;
        // make sure there is no remainder and no out of bounds accesses
        // this may be adjusted in the future
        assert(M % TM == 0);
        // TK depends on the datatype T
        constexpr int TK = 8 * std::min<int>(8, 32 / (8 * sizeof(T)));

        // TODO: 64 depends on the device. It is different for non-PVC hardware
        int ITEMS_IN_WG = std::min<int>(M / TM, 64);
        while (M / TM % ITEMS_IN_WG != 0) {
            ITEMS_IN_WG--;
        }
        if (ITEMS_IN_WG <= 0) throw std::logic_error("Number of SGS per WG cannot be less than 1");

        // One Block Row has TM rows an N columns.
        auto e = q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(deps);

            cgh.parallel_for(sycl::nd_range<1>(M / TM, ITEMS_IN_WG), [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const size_t loc_row_offset = item.get_global_linear_id() * TM;

                // we store blocks contiguously
                simd<T, TM * WIDTH> As;
                loadRow<TM, TK, cache_hint::uncached, cache_hint::uncached>(input.GetPointer(loc_row_offset, 0), As);

                // if not inference activate and store in intermediate output
                if constexpr (!INFERENCE) {
                    simd<T, TM * WIDTH> tmpA;
                    applyActivation<activation, TM, TK>(As, tmpA);
                    storeRow<TM, TK, cache_hint::uncached, cache_hint::uncached>(
                        tmpA, intermediate_output.GetElementPointer(0, loc_row_offset, 0));
                }

                simd<Tc, TM * WIDTH> Cs;
                for (int layer = 0; layer < n_hidden_layers; layer++) {
                    // reset result matrices
                    Cs = static_cast<Tc>(0);

                    MAD<TM, TK>(As, weights.GetMatrixPointer(layer), Cs);

                    // activate and save
                    applyActivation<activation, TM, TK>(Cs, As);

                    if constexpr (!INFERENCE)
                        storeRow<TM, TK, cache_hint::uncached, cache_hint::uncached>(
                            As, intermediate_output.GetElementPointer(
                                    layer + 1, loc_row_offset, 0) /*+ (layer + 1) * M * WIDTH + layer_offset_A*/);
                }

                // generate output, i.e. last GEMM
                Cs = static_cast<Tc>(0);

                MAD<TM, TK>(As, weights.GetMatrixPointer(n_hidden_layers), Cs);

                // activate
                applyActivation<output_activation, TM, TK>(Cs, As);

                // save to HBM
                if constexpr (!INFERENCE)
                    storeRow<TM, TK, cache_hint::uncached, cache_hint::write_back>(
                        As, intermediate_output.GetElementPointer(
                                n_hidden_layers + 1, loc_row_offset,
                                0) /*+ (n_hidden_layers + 1) * M * WIDTH + layer_offset_A*/);
                else if constexpr (INFERENCE) // storing at the beginning since no intermediate results
                    storeRow<TM, TK, cache_hint::uncached, cache_hint::write_back>(
                        As, intermediate_output.GetElementPointer(0, loc_row_offset, 0));
            });
        });

        return {e};
    }
};

} // namespace esimd
} // namespace kernels
} // namespace tinydpcppnn
