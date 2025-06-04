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

        cout << "prefix_sum_omp (" << size << ") with " << threads
            << " threads: " << duration_cast<milliseconds>(end - start).count() << " ms\n";
    }
}

