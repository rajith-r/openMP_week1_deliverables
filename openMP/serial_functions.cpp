#include <iostream>

void matmul_serial(const float* A, const float* B, float* C, int M, int K, int N) {
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			C[i * N + j] = 0.0f;
			for (int k = 0; k < K; ++k) {
				C[i * N + j] += A[i * K + k] * B[k * N + j];
			}
		}
	}
}


void prefix_sum_serial(const float* input, float* output, int size) {
	if (size <= 0) return;
	float sum = 0;
	for (int i = 0; i < size; ++i) {
		sum += input[i];
		output[i] = sum;
	}
}