#pragma once
#include <CL/sycl.hpp>

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

static constexpr float PI = 3.14159265358979323846f;

template<typename T>
void elt_activation(Activation activation, T& elt) {
	float q = ((float)elt / (2 * PI));
	switch (activation) {
	case Activation::ReLU:
		if (elt < (T)0.0f) {
			elt = (T)0.0f;
		}
		return;
	case Activation::LeakyReLU:
		if (elt >= 0) {
			elt = (T)elt;
		}
		else {
			elt = (T)0.01f * (float)elt;
		}
		return;
	case Activation::Exponential:
		elt = (T)exp((float)elt);

		return;
	case Activation::Sine:

		elt = (T)((elt)-floor(q) * 2 * PI);
		elt = (T)sinf((float)elt);

		return;
	case Activation::Sigmoid:

		elt = (T)(1.0f / (1.0f + expf((float)-elt)));
		return;
	case Activation::None:
		return;
	case Activation::Tanh:
		elt = (T)(tanhf((float)elt));
		return;
	default:
		return;

	}
}
template<typename T>
T elt_activation_ret(Activation activation, T& elt) {
	float q = ((float)elt / (2 * PI));
	switch (activation) {
	case Activation::ReLU:
		if (elt < (T)0.0f) {
			return (T)0.0f;
		}
		return elt;
	case Activation::LeakyReLU:
		if (elt >= 0) {
			return (T)elt;
		}
		else {
			return (T)0.01f * (float)elt;
		}
		
	case Activation::Exponential:
		return (T)exp((float)elt);

	
			
	case Activation::Sigmoid:

		return (T)(1.0f / (1.0f + expf((float)-elt)));
		
	case Activation::None:
		return elt;
	case Activation::Tanh:
		return (T)(tanhf((float)elt));
		
	default:
		return elt;

	}
}

template<typename outT, typename fwdT>
void elt_activation_bwd(Activation activation, outT& elt, fwdT fwd) {
	switch (activation) {
	case Activation::ReLU:
		if (fwd < (fwdT)0.0f) {
			elt = (outT)0.0f;
		}
		return;

	case Activation::LeakyReLU:
		if (fwd >= 0) {
			elt = (outT)elt;
		}
		else {
			elt = (outT)(0.01f * (float)elt);
		}
		return;

	case Activation::Exponential:
		elt = elt * fwd;
		return;

	case Activation::Sine:
		//not supported
		return;

	case Activation::Sigmoid:
		elt = elt * (outT)(fwd * (1.0f - (float)fwd));
		return;

	case Activation::None:
		return;

	case Activation::Tanh:
		elt = elt * (outT)(1.0f - ((float)fwd * (float)fwd));
		return;

	default:
		return;

	}
}
template<typename outT, typename fwdT, typename resT>
void elt_activation_bwd(Activation activation, outT& elt, fwdT fwd, resT& res) {
	switch (activation) {
	case Activation::ReLU:
		if (fwd < (fwdT)0.0f) {
			res = (resT)0.0f;
			return;

	case Activation::LeakyReLU:
		if (fwd >= 0) {
			res = (resT)elt;
		}
		else {
			res = (outT)(0.01f * (float)elt);
		}
		return;

	case Activation::Exponential:
		res = (resT)(elt * fwd);
		return;

	case Activation::Sine:
		//not supported
		return;

	case Activation::Sigmoid:
		res = (resT)elt * (resT)(fwd * (1.0f - (float)fwd));
		return;

	case Activation::None:
		res = elt;
		return;

	case Activation::Tanh:
		res = (resT)elt * (resT)(1.0f - ((float)fwd * (float)fwd));
		return;

	default:
		return;
		}
	}
}
template<typename T,int SG_SZ>
void matrix_activation(nd_item<1> it, Activation activation, device_ptr<T> out, int stride, stream outs) {
	int id = it.get_local_id() %SG_SZ;

	for (int i = 0; i < 8; i++) {
		elt_activation<T>(activation, out[i * stride + id]);
	}
	return;
}

template<typename outT, typename fwdT, typename resT, int SG_SZ>
void matrix_activation_backward(nd_item<1> it, Activation activation, device_ptr<outT> out, device_ptr<fwdT> fwd, resT* res, int stride) {
	int id = it.get_local_id() % SG_SZ;

	for (int i = 0; i < 8; i++) {

		elt_activation_bwd<outT, fwdT, resT>(activation, out[i * stride + id], fwd[i * stride + id], res[i * stride + id]);

	}
}
