// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

// This file lists all the inference, forward, backward and fused (forward+backw)
// functions we have.
//
// In general, there should always be one 'general' implementation which
// ignores performance and then specialized implementations which are optimized
// for their use case.

// The netweork forward_impl, inference_impl, backward_impl functions will then
// decide at runtime which one to choose. May do an abstraction around this?
// The netweok *_impl functions may also have template specializations to
// make the choice quicker.

#pragma once

#include <algorithm>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>
#include <vector>

#include "DeviceMatrix.h"
#include "common.h"
#include "common_kernel.h"
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

template <typename T> struct XMXCType {
    typedef T CType;
};
template <> struct XMXCType<bf16> {
    typedef float CType;
};
template <> struct XMXCType<sycl::half> {
    typedef sycl::half CType;
};

template <typename T, int INPUT_WIDTH, int WIDTH, int OUTPUT_WIDTH, Activation activation, Activation output_activation,
          size_t TN>
class EsimdKernels : public Kernels<T> {

    using Tc = typename XMXCType<T>::CType;

  public:
    std::vector<sycl::event> forward_impl(sycl::queue &q, const DeviceMatricesView<T> &weights,
                                          const DeviceMatrixView<T> &input, DeviceMatricesView<T> intermediate_output,
                                          const int n_hidden_layers, const std::vector<sycl::event> &deps) override {
        return forward_impl_general<false>(q, weights, input, intermediate_output, n_hidden_layers, deps);
    }

    std::vector<sycl::event> backward_impl(sycl::queue &q, const DeviceMatricesView<T> &weights,
                                           const DeviceMatrixView<T> &input, DeviceMatricesView<T> output,
                                           DeviceMatricesView<T> intermediate_backward,
                                           const DeviceMatricesView<T> &intermediate_forward, const int n_hidden_layers,
                                           const std::vector<sycl::event> &deps) override {

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

                    // TODO: Apply correct backward activation
                    applyActivation<Activation::None, TM, TK>(Cs, As);

                    storeRow<TM, TK, cache_hint::uncached, cache_hint::uncached>(
                        As, intermediate_backward.GetElementPointer(layer - 1, loc_row_offset, 0));
                }
            });
        });

        // NOTE: MKL gemm_batch is slower.
        std::vector<sycl::event> events(n_hidden_layers + 1);
        if constexpr (std::is_same<T, sycl::ext::oneapi::bfloat16>::value) { // need to cast to onemkls bf16 type.
            for (int iter = 0; iter < n_hidden_layers + 1; iter++) {
                events[iter] = oneapi::mkl::blas::row_major::gemm(
                    q, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, WIDTH, WIDTH, M, 1.0f,
                    reinterpret_cast<const oneapi::mkl::bfloat16 *>(intermediate_forward.GetMatrixPointer(iter)), WIDTH,
                    reinterpret_cast<oneapi::mkl::bfloat16 *>(intermediate_backward.GetMatrixPointer(iter)), WIDTH,
                    1.0f, reinterpret_cast<oneapi::mkl::bfloat16 *>(output.GetMatrixPointer(iter)), WIDTH, {e});
            }
        } else if constexpr (!std::is_same<T, sycl::ext::oneapi::bfloat16>::value) {
            for (int iter = 0; iter < n_hidden_layers + 1; iter++) {
                events[iter] = oneapi::mkl::blas::row_major::gemm(
                    q, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, WIDTH, WIDTH, M, 1.0,
                    intermediate_forward.GetMatrixPointer(iter), WIDTH, intermediate_backward.GetMatrixPointer(iter),
                    WIDTH, 1.0, output.GetMatrixPointer(iter), WIDTH, {e});
            }
        }
        return events;
    }

    std::vector<sycl::event> inference_impl(sycl::queue &q, const DeviceMatricesView<T> &weights,
                                            const DeviceMatrixView<T> &input, DeviceMatricesView<T> intermediate_output,
                                            const int n_hidden_layers, const std::vector<sycl::event> &deps) override {
        return forward_impl_general<true>(q, weights, input, intermediate_output, n_hidden_layers, deps);
    }

    /*************the following functions are only public for testing purposes*******************/

    // in register everything is in block major format with blocks of size TMxTK
    template <int TM, int TK, cache_hint L1, cache_hint L3, int TMWIDTH>
    SYCL_ESIMD_FUNCTION static void storeRow(simd<T, TMWIDTH> &src, T *const dest) {

        static_assert(TM >= 1 && TM <= 8);
        static_assert(WIDTH % TK == 0);
        static_assert(TMWIDTH == TM * WIDTH);
        static_assert(sizeof(T) <= 4);

        constexpr int rows_per_load = std::min<int>(512 / (WIDTH * sizeof(T)), TM);
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
        static_assert(TM >= 1 && TM <= 8);
        static_assert(WIDTH % TK == 0);
        static_assert(TMWIDTH == TM * WIDTH);
        static_assert(sizeof(T) <= 4);
        constexpr int elems_per_pos = 4 / sizeof(T);
        constexpr int blocks_per_load = TK * elems_per_pos > WIDTH ? 1 : elems_per_pos;
        constexpr int nloads = WIDTH / (TK * blocks_per_load);
        static_assert(nloads > 0);

        auto dest_int = dest.template bit_cast_view<int32_t>();
#pragma unroll
        for (int load_iter = 0; load_iter < nloads; load_iter++) {
            dest_int.template select<TM * TK / elems_per_pos * blocks_per_load, 1>(TM * TK / elems_per_pos *
                                                                                   blocks_per_load * load_iter) =
                lsc_load_2d<int32_t, TK / elems_per_pos, TM, blocks_per_load, false, false, L1, L3>(
                    reinterpret_cast<int32_t const *>(src), WIDTH * sizeof(T) - 1, TM - 1, WIDTH * sizeof(T) - 1,
                    load_iter * TK, 0);
        }
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
        for (int iterA = 0; iterA < TMWIDTH; iterA += TM * TK) {
            for (int iterB = 0; iterB < WIDTH; iterB += TN) {
                simd<T, TK * TN> BlockB;
                auto BlockB_float = BlockB.template bit_cast_view<float>();
                BlockB_float =
                    lsc_load_2d<float, TN, TK / vnni_factor, 1, false, false, cache_hint::cached, cache_hint::cached>(
                        reinterpret_cast<float const *>(B), vnni_factor * WIDTH * sizeof(T) - 1,
                        WIDTH / vnni_factor - 1, vnni_factor * WIDTH * sizeof(T) - 1, iterB, iterA / TM / vnni_factor);

                Cs.template select<TM * TN, 1>(iterB * TM) = xmx::dpas<8, TM, Tc>(
                    Cs.template select<TM * TN, 1>(iterB * TM), BlockB, As.template select<TM * TK, 1>(iterA));
            }
        }
    }

    template <Activation act, int TM, int TK, int N, typename Tsrc, typename Tdest>
    SYCL_ESIMD_FUNCTION static void applyActivation(simd<Tsrc, N> &Src, simd<Tdest, N> &Dest) {
        static_assert(TM >= 1 && TM <= 8);
        static_assert(TN == 16 || TN == 8);
        static_assert(TK == TN); // otherwise we would need to reshuffle due to block major format

        if constexpr (act == Activation::None)
            Dest = convert<Tdest, Tsrc>(Src);
        else if constexpr (act == Activation::ReLU)
            Dest = max<Tdest>(convert<Tdest, Tsrc>(Src), simd<Tdest, N>(static_cast<Tdest>(0)));
    }

  private:
    template <bool INFERENCE>
    std::vector<sycl::event> forward_impl_general(sycl::queue &q, const DeviceMatricesView<T> &weights,
                                                  const DeviceMatrixView<T> &input,
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

template <typename T, int WIDTH, int TN, int INPUT_WIDTH, int OUTPUT_WIDTH, Activation ACT>
std::unique_ptr<Kernels<T>> createKernels_helper3(Activation out_act) {
    switch (out_act) {
    case Activation::None:
        return std::make_unique<EsimdKernels<T, INPUT_WIDTH, WIDTH, OUTPUT_WIDTH, ACT, Activation::None, TN>>();
        break;
    default:
        throw std::invalid_argument("Invalid output activation");
    }
}

template <typename T, int WIDTH, int TN, int INPUT_WIDTH, int OUTPUT_WIDTH>
std::unique_ptr<Kernels<T>> createKernels_helper2(Activation act, Activation out_act) {
    switch (act) {
    case Activation::ReLU:
        return createKernels_helper3<T, WIDTH, TN, INPUT_WIDTH, OUTPUT_WIDTH, Activation::ReLU>(out_act);
        break;
    case Activation::None:
        return createKernels_helper3<T, WIDTH, TN, INPUT_WIDTH, OUTPUT_WIDTH, Activation::None>(out_act);
        break;
    default:
        throw std::invalid_argument("Invalid activation");
    }
}

template <typename T, int WIDTH, int TN, int INPUT_WIDTH>
std::unique_ptr<Kernels<T>> createKernels_helper1(const int output_width, Activation act, Activation out_act) {

    return createKernels_helper2<T, WIDTH, TN, INPUT_WIDTH, WIDTH>(act, out_act);
    // switch (output_width) {
    // case 16:
    //     return createKernels_helper2<T, Tc, WIDTH, TN, INPUT_WIDTH, 16>(act, out_act);
    //     break;
    // case 32:
    //     return createKernels_helper2<T, Tc, WIDTH, TN, INPUT_WIDTH, 32>(act, out_act);
    //     break;
    // case 64:
    //     return createKernels_helper2<T, Tc, WIDTH, TN, INPUT_WIDTH, 64>(act, out_act);
    //     break;
    // case 128:
    //     return createKernels_helper2<T, Tc, WIDTH, TN, INPUT_WIDTH, 128>(act, out_act);
    //     break;
    // default:
    //     throw std::invalid_argument("Invalid output_width");
    // }
}

template <typename T, int WIDTH, int TN>
std::unique_ptr<Kernels<T>> createKernels(const int input_width, const int output_width, Activation act,
                                          Activation out_act) {
    // temporarily use this
    return createKernels_helper1<T, WIDTH, TN, WIDTH>(output_width, act, out_act);

    // switch (input_width) {
    // case 16:
    //     return createKernels_helper1<T, Tc, WIDTH, TN, 16>(output_width, act, out_act);
    //     break;
    // case 32:
    //     return createKernels_helper1<T, Tc, WIDTH, TN, 32>(output_width, act, out_act);
    //     break;
    // case 64:
    //     return createKernels_helper1<T, Tc, WIDTH, TN, 64>(output_width, act, out_act);
    //     break;
    // case 128:
    //     return createKernels_helper1<T, Tc, WIDTH, TN, 128>(output_width, act, out_act);
    //     break;
    // default:
    //     throw std::invalid_argument("Invalid input_width");
    // }
}

} // namespace esimd
} // namespace kernels
} // namespace tinydpcppnn
