#pragma once
#include "common.h" //Activation enum
#include <CL/sycl.hpp>

using namespace sycl;

template <typename T, typename resT> void elt_activation(Activation activation, T &elt, resT &res) {
    switch (activation) {
    case Activation::ReLU:
        if (elt < (T)0.0f) {
            res = (resT)0.0f;
        } else {
            res = (resT)elt;
        }
        return;
        break;
    case Activation::LeakyReLU:
        if (elt >= 0) {
            res = (resT)elt;
        } else {
            res = (resT)0.01f * (resT)elt;
        }
        return;
        break;
    case Activation::Exponential:
        res = (resT)exp(elt);
        return;
        break;
    case Activation::Sine:
        res = (resT)((elt)-floor(elt / (2 * PI)) * 2 * PI);
        res = (resT)sinf(res);
        return;
        break;
    case Activation::Sigmoid:
        res = (resT)(1.0f / (1.0f + expf(-elt)));
        return;
        break;
    case Activation::None:
        res = elt;
        return;
        break;
        /*case Activation::Tanh:
                res = (resT)(((expf(elt) - expf(-elt))) / (expf(elt) +
           expf(-elt))); return; break;*/
    default:
        return;
        break;
    }
}

template <typename T> T elt_activation_ret(Activation activation, T &elt) {
    float q = ((float)elt / (2 * PI));

    switch (activation) {
    case Activation::ReLU:
        if (elt < (T)0.0f) {
            return (T)0.0f;
        }
        return elt;
        break;

    case Activation::LeakyReLU:
        if (elt >= 0) {
            return (T)elt;
        } else {
            return (T)0.01f * (float)elt;
        }
        break;

    case Activation::Exponential:
        return (T)std::exp((float)elt);

    case Activation::Sigmoid:
        return (T)(1.0f / (1.0f + expf((float)-elt)));
        break;
    case Activation::None:
        return elt;
        break;
    case Activation::Tanh:
        return (T)(tanhf((float)elt));
        break;
    default:
        return elt;
        break;
    }
}

template <typename outT, typename fwdT> void elt_activation_bwd(Activation activation, outT &elt, fwdT fwd) {
    switch (activation) {
    case Activation::ReLU:
        if (fwd < (fwdT)0.0f) {
            elt = (outT)0.0f;
        }
        return;
        break;

    case Activation::LeakyReLU:
        if (fwd >= 0) {
            elt = (outT)elt;
        } else {
            elt = (outT)(0.01f * (float)elt);
        }
        return;
        break;

    case Activation::Exponential:
        elt = elt * fwd;
        return;
        break;

    case Activation::Sine:
        // not supported
        return;
        break;
    case Activation::Sigmoid:
        elt = elt * (outT)(fwd * (1.0f - (float)fwd));
        return;
        break;
    case Activation::None:
        return;
        break;
    case Activation::Tanh:
        elt = elt * (outT)(1.0f - ((float)fwd * (float)fwd));
        return;
        break;
    default:
        return;
        break;
    }
}

template <typename outT, typename fwdT, typename resT>
void elt_activation_bwd(Activation activation, outT &elt, fwdT fwd, resT &res) {
    switch (activation) {
    case Activation::ReLU:
        if (fwd < (fwdT)0.0f) {
            res = (resT)0.0f;
        } else {
            res = (resT)elt;
        }
        return;
        break;

    case Activation::LeakyReLU:
        if (fwd >= 0) {
            res = (resT)elt;
        } else {
            res = (outT)(0.01f * (float)elt);
        }
        return;
        break;

    case Activation::Exponential:
        res = (resT)(elt * fwd);
        return;
        break;
    case Activation::Sine:
        // not supported
        return;
        break;
    case Activation::Sigmoid:
        res = (resT)elt * (resT)(fwd * (1.0f - (float)fwd));
        return;
        break;
    case Activation::None:
        res = elt;
        return;
        break;
    case Activation::Tanh:
        res = (resT)elt * (resT)(1.0f - ((float)fwd * (float)fwd));
        return;
        break;
    default:
        return;
        break;
    }
}
