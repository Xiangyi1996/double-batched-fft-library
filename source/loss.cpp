
#include "loss.h"

//#include "L1.h"
#include "L2.h"
//#include "RelativeL1.h"
//#include "RelativeL2.h"
//#include "cross_entropy.h"


Loss* create_loss(const json& loss) {
	std::string loss_type = loss.value("otype", "RelativeL2");

	if (isequalstring(loss_type, "L2")) {
		return new L2Loss();
	}
	//else if (isequalstring(loss_type, "RelativeL2")) {
	//	return new RelativeL2Loss();
	//}
	//else if (isequalstring(loss_type, "L1")) {
	//	return new L1Loss();
	//}
	//else if (isequalstring(loss_type, "RelativeL1")) {
	//	return new RelativeL1Loss();
	//}
	//else if (isequalstring(loss_type, "CrossEntropy")) {
	//	return new CrossEntropyLoss<T>();
	//}
	else {
		throw std::runtime_error{"Invalid loss type: "};
	}
}


