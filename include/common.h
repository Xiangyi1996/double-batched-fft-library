#pragma once
#include<iostream>
#include<vector>
#include <CL/sycl.hpp>



int toPackedLayoutCoord(int idx, int cols, int rows) {
	int i = idx / cols;
	int j = idx % cols;
	if (i % 2 == 0) {
		return i * cols + 2 * j;
	}
	else {
		return (i-1) * cols +2*j + 1
	}
}
// Row et cols correspondent au nombre de rows et cols de la matrice d'origine pas celle en packec layout ( qui est donc cols*2 rows/2)
int fromPackedLayoutCoord(int idx, int cols, int rows) {
	int i = idx / (cols * 2);
	int j = idx % (cols * 2);
	if (j % 2 == 0) {
		return (i * 2) * cols + j / 2;
	}
	else {
		return (i * 2 + 1) * cols + (j - 1) / 2;
	}
}
