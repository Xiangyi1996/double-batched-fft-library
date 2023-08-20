#include <iostream>

// This file is only to test whether CMakeLists works

#include "tnn_api.h"

int main() {
  tnn::SwiftNetModule snm(1, 64, 64, 128, 2, Activation::None,
                          Activation::None);
  //   std::cout << "HI" << std::endl;
}