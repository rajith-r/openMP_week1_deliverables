#include <iostream>

void matmul_serial(const float* A, const float* B, float* C, int M, int K, int N);
void prefix_sum_serial(const float* input, float* output, int size);