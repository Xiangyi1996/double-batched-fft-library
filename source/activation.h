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

//template<typename T,typename joint_matrix_t>
//void matrix_activation(ext::oneapi::sub_group& sg, Activation activation, joint_matrix_t& frag) {
//	auto data = get_wi_data(sg, frag);
//	switch (activation) {
//	case Activation::ReLU:
//		for (int i = 0; i < data.length(); i++) {
//			if (data[i] >= 0) {
//				data[i] = data[i];
//			}
//			else {
//				data[i] = (T)0.0f;
//			}
//		}
//		return;
//	case Activation::LeakyReLU:
//		for (int i = 0; i < data.length(); i++) {
//			if (data[i] >= 0) {
//				data[i] = (T)data[i];
//			}
//			else {
//				data[i] = (T)0.01f*(float)data[i];
//			}
//		}
//		return;
//	case Activation::Exponential:
//		for (int i = 0; i < data.length(); i++) {
//			data[i] = (T)exp((float)data[i]);
//		}
//		return;
//	case Activation::Sine:
//		for (int i = 0; i < data.length(); i++) {
//			data[i] = (T)sinf((float)data[i]);
//		}
//		return;
//	case Activation::Sigmoid:
//		for (int i = 0; i < data.length(); i++) {
//			data[i] = (T)(1.0f / (1.0f + expf((float)-data[i])));
//		}
//		return;
//	case Activation::None:
//		return;
//	case Activation::Tanh:
//		for (int i = 0; i < data.length(); i++) {
//			data[i] = (T)(tanhf((float)data[i]));
//		}
//	default:
//		return;
//
//	}
//}


template<typename T>
void elt_activation(Activation activation, T& elt) {
	float q = ((float)elt / (2 * PI));
	switch (activation) {
	case Activation::ReLU:
		if (elt < (T)0.0f) {
			elt = (T)0.0f;
		
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
}
template<typename outT, typename fwdT>
void elt_activation_bwd(Activation activation, outT& elt, fwdT fwd) {
	switch (activation) {
	case Activation::ReLU:
		if (fwd < (fwdT)0.0f) {
			elt = (outT)0.0f;

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

		elt = elt * (outT)(fwd *  (1.0f - (float)fwd));
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
}
template<typename outT, typename fwdT,typename resT>
void elt_activation_bwd(Activation activation, outT& elt, fwdT fwd,resT& res) {
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
		res =(resT)(elt * fwd);

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
		res =(resT) elt * (resT)(1.0f - ((float)fwd * (float)fwd));
		return;
	default:
		return;

		}
	}
}
template<typename T>
void matrix_activation(nd_item<1> it, Activation activation, device_ptr<T> out,int stride,stream outs) {
	
	int id = it.get_local_id();
	
	for (int i = 0; i < 8; i++) {
		elt_activation<T>(activation, out[i * stride + id]);
			
	}
	return;
	
}

template<typename outT, typename fwdT,typename resT,int SG_SZ>
void matrix_activation_backward(nd_item<1> it, Activation activation, device_ptr<outT> out, device_ptr<fwdT> fwd,resT* res, int stride) {
	int id = it.get_local_id();
	for (int i = 0; i < 8; i++) {
		
		elt_activation_bwd<outT, fwdT, resT>(activation, out[i * stride + id], fwd[i * stride + id], res[i * stride + id]);
		
		if (SG_SZ == 8) {
			elt_activation_bwd<outT, fwdT, resT>(activation, out[i * stride + 8 + id], fwd[i * stride + 8 + id], res[i * stride + 8 + id]);
		}
			
		
		
	}
 }

//void matrix_activation(Activation activation, float* out) {
//	int length = 64 * 64;
//	switch (activation) {
//	case Activation::ReLU:
//		for (int i = 0; i < length; i++) {
//			if (data[i] >= 0) {
//				data[i] = data[i];
//			}
//			else {
//				data[i] = (T)0.0f;
//			}
//		}
//		return;
//	case Activation::LeakyReLU:
//		for (int i = 0; i < length; i++) {
//			if (data[i] >= 0) {
//				data[i] = (T)data[i];
//			}
//			else {
//				data[i] = (T)0.01f * (float)data[i];
//			}
//		}
//		return;
//	case Activation::Exponential:
//		for (int i = 0; i < length; i++) {
//			data[i] = (T)expf((float)data[i]);
//		}
//		return;
//	case Activation::Sine:
//		for (int i = 0; i < length; i++) {
//			data[i] = (T)sinf((float)data[i]);
//		}
//		return;
//	case Activation::Sigmoid:
//		for (int i = 0; i < length; i++) {
//			data[i] = (T)(1.0f / (1.0f + expf((float)-data[i])));
//		}
//		return;
//	case Activation::None:
//		return;
//	case Activation::Tanh:
//		for (int i = 0; i < length; i++) {
//			data[i] = (T)(tanhf((float)data[i]));
//		}
//	default:
//		return;
//
//	}
//}
//void matrix_activation_backward(Activation activation, float* out, float* forward) {
//	switch (activation) {
//	case Activation::ReLU:
//		
//			for (int t = 0; t < result.num_elements; t++) {
//				out[t] = out[t] * (forward[t] > (T)0.0f);
//			}
//		return;
//	case Activation::LeakyReLU:
//		
//			for (int t = 0; t < result.num_elements; t++) {
//				out[t] = out[t] * (forward[t] > (T)0.0f ? 1.0f : 0.01f);
//			}
//		return;
//	case Activation::Exponential:
//		
//			for (int t = 0; t < result.num_elements; t++) {
//				out[t] = out[t] * forward[t];
//			}
//		return;
//	case Activation::Sine:
//		// Sine requires stored pre-activations, which we don't have. We only
//		// write out the post-activations.
//		// assert(false); // Commented out due to isolated strange side-effects on Windows
//		return;
//	case Activation::Sigmoid:
//		
//			for (int t = 0; t < result.num_elements; t++) {
//				out[t] = out[t] * (forward[t] * (1.0f - (float)forward[t]));
//			}
//		return;
//	case Activation::Tanh:
//		
//			for (int t = 0; t < result.num_elements; t++) {
//				out[t] = out[t] * (1.0f - (forward[t] * forward[t]));
//			}
//		return;
//	case Activation::None: result = frag; return;
//	default:
//		// Unsupported activation
//		// assert(false); // Commented out due to isolated strange side-effects on Windows
//		return;
//	}
//}

//template<typename T,typename joint_matrix_t,typename forward_matrix_t>
//void matrix_activation_backward(ext::oneapi::sub_group sg, Activation activation, joint_matrix_t & frag, forward_matrix_t & forward) {
//	auto data = get_wi_data(sg, frag);
//	auto fwd_data = get_wi_data(sg, forward);
//	switch (activation) {
//	case Activation::ReLU:
//		for (int i = 0; i < data.length(); i++) {
//			data[i] = data[i] * relu(fwd_data[i]);
//		}
//		return;
//	case Activation::LeakyReLU:
//		for (int i = 0; i < data.length(); i++) {
//			if (fwd_data[i] > 0) {
//				data[i] = data[i];
//			}
//			else {
//				data[i] = data[i] * (T)0.01f;
//			}
//		}
//		return;
//	case Activation::Exponential:
//		for (int i = 0; i < data.length(); i++) {
//			data[i] = data[i] * fwd_data[i];
//		}
//		return;
//	case Activation::Sine:
//
//		return;
//	case Activation::Sigmoid:
//		for (int i = 0; i < data.length(); i++) {
//			data[i] = data[i] * (T)(fwd_data[i] * (T)(1.0f - (float)fwd_data[i]));
//		}
//		return;
//	case Activation::None:
//		return;
//	case Activation::Tanh:
//		for (int i = 0; i < data.length(); i++) {
//			data[i] = data[i] * (T)(1.0f - ((float)fwd_data[i] * (float)fwd_data[i]));
//		}
//	default:
//		return;
//
//	}
//}
