#pragma once
#include <CL/sycl.hpp>
#include <cmath>
using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
enum class Activation {
    ReLU,
    LeakyReLU,
    Exponential,
    Sine,
    Sigmoid,
    Squareplus,
    Softplus,
    Tanh,
    None,
};

template<typename T, typename joint_matrix_t>
void matrix_activation(ext::oneapi::sub_group sg, Activation activation, joint_matrix_t& frag) {
    auto data = get_wi_data(sg, frag);
    switch (activation) {
    case Activation::ReLU:
        for (int i = 0; i < data.length(); i++) {
            if (data[i] >= 0) {
                data[i] = data[i];
            }
            else {
                data[i] = (T)0.0f;
            }
        }
        return;
    case Activation::LeakyReLU:
        for (int i = 0; i < data.length(); i++) {
            if (data[i] >= 0) {
                data[i] = (T)data[i];
            }
            else {
                data[i] = (T)0.01f * (float)data[i];
            }
        }
        return;
    case Activation::Exponential:
        for (int i = 0; i < data.length(); i++) {
            data[i] = (T)expf((float)data[i]);
        }
        return;
    case Activation::Sine:
        for (int i = 0; i < data.length(); i++) {
            data[i] = (T)sinf((float)data[i]);
        }
        return;
    case Activation::Sigmoid:
        for (int i = 0; i < data.length(); i++) {
            data[i] = (T)(1.0f / (1.0f + expf((float)-data[i])));
        }
        return;
    case Activation::None:
        return;
    case Activation::Tanh:
        for (int i = 0; i < data.length(); i++) {
            data[i] = (T)(tanhf((float)data[i]));
        }
    default:
        return;

    }
}


//template<typename T,typename joint_matrix_t,typename forward_matrix_t>
//void matrix_activation_backward(ext::oneapi::sub_group sg, Activation activation, joint_matrix_t & frag, forward_matrix_t & forward) {
//      auto data = get_wi_data(sg, frag);
//      auto fwd_data = get_wi_data(sg, forward);
//      switch (activation) {
//      case Activation::ReLU:
//              for (int i = 0; i < data.length(); i++) {
//                      data[i] = data[i] * relu(fwd_data[i]);
//              }
//              return;
//      case Activation::LeakyReLU:
//              for (int i = 0; i < data.length(); i++) {
//                      if (fwd_data[i] > 0) {
//                              data[i] = data[i];
//                      }
//                      else {
//                              data[i] = data[i] * (T)0.01f;
//                      }
//              }
//              return;
//      case Activation::Exponential:
//              for (int i = 0; i < data.length(); i++) {
//                      data[i] = data[i] * fwd_data[i];
//              }
//              return;
//      case Activation::Sine:
//
//              return;
//      case Activation::Sigmoid:
//              for (int i = 0; i < data.length(); i++) {
//                      data[i] = data[i] * (T)(fwd_data[i] * (T)(1.0f - (float)fwd_data[i]));
//              }
//              return;
//      case Activation::None:
//              return;
//      case Activation::Tanh:
//              for (int i = 0; i < data.length(); i++) {
//                      data[i] = data[i] * (T)(1.0f - ((float)fwd_data[i] * (float)fwd_data[i]));
//              }
//      default:
//              return;
//
//      }
//}