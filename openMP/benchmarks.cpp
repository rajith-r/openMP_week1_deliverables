#include "benchmarks.h"
#include "openmp_functions.h" // For matmul_omp, etc.
#include "serial_functions.h" // For matmul_serial
#include <vector>
#include <chrono>
#include <iostream>
#include <omp.h>

using namespace std;
using namespace std::chrono;

void benchmark_serial_matmul(int M, int K, int N) {
    vector<float> A(M * K);
    vector<float> B(K * N);
    for (float& a : A) a = static_cast<float>(rand()) / RAND_MAX;
    for (float& b : B) b = static_cast<float>(rand()) / RAND_MAX;
    vector<float> C(M * N, 0.0f);
    auto start = high_resolution_clock::now();
    matmul_serial(A.data(), B.data(), C.data(), M, K, N);
    auto end = high_resolution_clock::now();
    cout << "matmul_serial: "
        << duration_cast<milliseconds>(end - start).count() << " ms\n";
}

void benchmark_serial_prefix_sum(int size) {
    vector<float> input(size);
    srand(42);
    for (int i = 0; i < size; ++i) {
        input[i] = static_cast<float>(rand()) / RAND_MAX;  // [0,1] random float
    }
    vector<float> output(size, 0.0f);
    auto start = high_resolution_clock::now();
    prefix_sum_serial(input.data(), output.data(), size);
    auto end = high_resolution_clock::now();
    cout << "prefix_sum_serial (" << size << "): "
        << duration_cast<milliseconds>(end - start).count() << " ms\n";
}

void benchmark_matmul(int M, int K, int N) {
    vector<float> A(M * K);
    vector<float> B(K * N);
    for (float& a : A) a = static_cast<float>(rand()) / RAND_MAX;
    for (float& b : B) b = static_cast<float>(rand()) / RAND_MAX;
    vector<float> C(M * N, 0.0f);

    for (int threads = 4; threads <= 8; ++threads) {
        omp_set_num_threads(threads);
        cout << "\n--- Threads: " << threads << " ---\n";

        auto start = high_resolution_clock::now();
        matmul_omp(A.data(), B.data(), C.data(), M, K, N);
        auto end = high_resolution_clock::now();
        cout << "matmul_omp: "
            << duration_cast<milliseconds>(end - start).count() << " ms\n";

        fill(C.begin(), C.end(), 0.0f);

        start = high_resolution_clock::now();
        matmul_omp_simd(A.data(), B.data(), C.data(), M, K, N);
        end = high_resolution_clock::now();
        cout << "matmul_omp_simd: "
            << duration_cast<milliseconds>(end - start).count() << " ms\n";
    }
}

void benchmark_prefix_sum(int size) {
    vector<float> input(size);
	srand(42);
    for (int i = 0; i < size; ++i) {
        input[i] = static_cast<float>(rand()) / RAND_MAX;  // [0,1] random float
    }
    vector<float> output;

    for (int threads = 4; threads <= 8; ++threads) {
        omp_set_num_threads(threads);
        output.clear();  // reset

        auto start = high_resolution_clock::now();
        prefix_sum_omp(input, output, size);
        auto end = high_resolution_clock::now();

		//cout << "\n serial last sum: " << output[size - 1] << "\n";
        cout << "prefix_sum_omp (" << size << ") with " << threads
            << " threads: " << duration_cast<milliseconds>(end - start).count() << " ms\n";
    }
}

//benchmark for matmul 128 128 128
//-- - Threads: 4 -- -
//matmul_omp : 2 ms
//matmul_omp_simd : 1 ms
//
//-- - Threads : 5 -- -
//matmul_omp : 2 ms
//matmul_omp_simd : 1 ms
//
//-- - Threads : 6 -- -
//matmul_omp : 2 ms
//matmul_omp_simd : 0 ms
//
//-- - Threads : 7 -- -
//matmul_omp : 2 ms
//matmul_omp_simd : 0 ms
//
//-- - Threads : 8 -- -
//matmul_omp : 3 ms
//matmul_omp_simd : 0 ms

//benchmark for prefix sum 
//prefix_sum_omp(1048576) with 4 threads: 6 ms
//prefix_sum_omp(1048576) with 5 threads : 7 ms
//prefix_sum_omp(1048576) with 6 threads : 5 ms
//prefix_sum_omp(1048576) with 7 threads : 5 ms
//prefix_sum_omp(1048576) with 8 threads : 6 ms


//512 512 512
//-- - Threads: 4 -- -
//matmul_omp : 94 ms
//matmul_omp_simd : 93 ms
//
//-- - Threads : 5 -- -
//matmul_omp : 80 ms
//matmul_omp_simd : 86 ms
//
//-- - Threads : 6 -- -
//matmul_omp : 78 ms
//matmul_omp_simd : 67 ms
//
//-- - Threads : 7 -- -
//matmul_omp : 64 ms
//matmul_omp_simd : 59 ms
//
//-- - Threads : 8 -- -
//matmul_omp : 53 ms
//matmul_omp_simd : 52 ms
//prefix_sum_omp(1048576) with 4 threads : 6 ms
//prefix_sum_omp(1048576) with 5 threads : 5 ms
//prefix_sum_omp(1048576) with 6 threads : 6 ms
//prefix_sum_omp(1048576) with 7 threads : 4 ms
//prefix_sum_omp(1048576) with 8 threads : 6 ms