#pragma once

#include<iostream>
#include<vector>
#include <CL/sycl.hpp>



extern SYCL_EXTERNAL int toPackedLayoutCoord(int idx, int rows, int cols);
// Row et cols correspondent au nombre de rows et cols de la matrice d'origine pas celle en packec layout ( qui est donc cols*2 rows/2)
extern SYCL_EXTERNAL int fromPackedLayoutCoord(int idx, int rows, int cols);

extern SYCL_EXTERNAL bool isequalstring(const std::string& str1, const std::string& str2);

