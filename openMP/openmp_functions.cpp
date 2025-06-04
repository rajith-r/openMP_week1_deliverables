#include <omp.h>
#include <iostream>
#include <vector>
#include "openmp_functions.h"
using namespace std;
// A: M x K, B: K x N, C: M x N
void matmul_omp(const float* A, const float* B, float* C, int M, int K, int N) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void matmul_omp_simd(const float* A, const float* B, float* C, int M, int K, int N) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0;
			// Use SIMD to vectorize the inner loop
            #pragma omp simd reduction(+:sum)   
            for (int k = 0; k < K; k += 4) {
                sum += A[i * K + k] * B[k * N + j];
                if (k + 1 < K) sum += A[i * K + k + 1] * B[(k + 1) * N + j];
                if (k + 2 < K) sum += A[i * K + k + 2] * B[(k + 2) * N + j];
                if (k + 3 < K) sum += A[i * K + k + 3] * B[(k + 3) * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void prefix_sum_omp(const std::vector<float>& input, std::vector<float>& output,int N) {
    output.resize(N);
    std::vector<float> temp(N);
	int num_threads = std::min(N, omp_get_max_threads()); // Use at most N threads or max available threads
    omp_set_num_threads(num_threads);
	std::vector<float> partial_sums(num_threads, 0.0f);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int chunk_size = (N + num_threads - 1) / num_threads; // Calculate chunk size for each thread
        int start = tid * chunk_size;
        int end = std::min(start + chunk_size, N);

        // Compute local prefix sum
        float sum = 0;
        for (int i = start; i < end; ++i) {
            sum += input[i];
            temp[i] = sum;
        }
        partial_sums[tid] = sum; 
       
        #pragma omp barrier

        #pragma omp single
        {
            for (int i = 1; i < num_threads; ++i) {
                partial_sums[i] += partial_sums[i - 1];
            }
        }

        #pragma omp barrier
		// Write the final prefix sum to output
        for (int i = start; i < end; ++i) {
            if (tid > 0) {
                output[i] = temp[i] + partial_sums[tid - 1]; // Add the previous thread's sum
            } else {
                output[i] = temp[i]; // First thread just copies its local sum
            }
		}
    }
}


//prefix sum with blelloch scan algorithm

    

