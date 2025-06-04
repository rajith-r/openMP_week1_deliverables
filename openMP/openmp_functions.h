#pragma once	
#include <vector>

void map(int n, int m, int num_threads);
void matmul_omp(const float* A, const float* B, float* C, int M, int K, int N);
void matmul_omp_simd(const float* A, const float* B, float* C, int M, int K, int N);
void prefix_sum_omp(const std::vector<float>& input, std::vector<float>& output, int N);