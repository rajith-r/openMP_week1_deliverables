/******************************************************************************

Welcome to GDB Online.
GDB online is an online compiler and debugger tool for C, C++, Python, Java, PHP, Ruby, Perl,
C#, OCaml, VB, Swift, Pascal, Fortran, Haskell, Objective-C, Assembly, HTML, CSS, JS, SQLite, Prolog.
Code, Compile, Run and Debug online from anywhere in world.

*******************************************************************************/
#include <pthread.h>
#include <iostream>
#include <random>
#include <chrono>
using namespace std;
using namespace std::chrono;

struct MatMulArgs {
	const float* A;
	const float* B;
	float* C;
	int M, K, N;
	int i;
	int j;
};


struct MatulRowArgs {
	const float* A;
	const float* B;
	float* C;
	int M, K, N;
	int i; // Row index
};

struct prefixSumArgs {
	const float* input;
	float* output;
	int start;
	int end;
};

struct prefixSumAccArgs {
	float* output;
	float* partial_chunk_sums;
	int start;
	int end;
	int  chunk;
};

void fill_random(vector<float>& mat) {
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<> dis(0.0f, 1.0f);
	for (float& x : mat) x = dis(gen);
}

// Function to perform matrix multiplication using pthreads one element per thread inefficiently
void* pthread_matmul(void* arg) {
	MatMulArgs* args = static_cast<MatMulArgs*>(arg);
	const float* A = args->A;
	const float* B = args->B;
	float* C = args->C;
	int M = args->M, K = args->K, N = args->N;
	int i = args->i, j = args->j;

	float sum = 0.0f;
	for (int k = 0; k < K; ++k) {
		sum += A[i * K + k] * B[k * N + j];
	}
	C[i * N + j] = sum;

	return nullptr;
}

//Fybction to perform matrix multiplication using pthreads one row per thread

void* pthread_matmul_row(void* arg) {
	MatulRowArgs* args = static_cast<MatulRowArgs*>(arg);
	const float* A = args->A;
	const float* B = args->B;
	float* C = args->C;
	int M = args->M, K = args->K, N = args->N;
	int i = args->i; // Row index
	for (int j = 0; j < N; ++j) {
		float sum = 0.0f;
		for (int k = 0; k < K; ++k) {
			sum += A[i * K + k] * B[k * N + j];
		}
		C[i * N + j] = sum;
	}
	return nullptr;
}

void* pthread_prefix_sum(void* arg) {
	prefixSumArgs* args = static_cast<prefixSumArgs*>(arg);
	const float* input = args->input;
	float* output = args->output;
	int start = args->start;
	int end = args->end;
	if (start >= end) return nullptr; // No work to do
	cout << "Thread processing range: " << start << " to " << end << endl;
	// Calculate prefix sum for the given range
	float sum = 0.0f;
	for (int i = start; i < end; ++i) {
		sum += input[i];
		output[i] = sum;
	}
	// If this is not the first chunk add the last value of the previous segment
	//POSSIBLE RACE CONDITION
//     if (start > 0) {
//         float prev_sum = output[start - 1];
//         for (int i = start; i < end; ++i) {
//             output[i] += prev_sum;
//         }
// 	}
	//RETURN AND JOIN ALL SUMS INTO A PREFIX_SUM_ARRAY
	return nullptr;
}

void* pthread_prefix_sum_acc(void* arg) {
	prefixSumAccArgs* args = static_cast<prefixSumAccArgs*>(arg);
	float* output = args->output;
	float* partial_chunk_sums = args->partial_chunk_sums;
	int start = args->start;
	int end = args->end;
	int chunk = args->chunk;
	if (start >= end) return nullptr; // No work to do
	// Add the accumulated sum from the previous chunk to the current chunk
	float accumulated_sum = 0.0f;
	for (int i = start; i < end; ++i) {
		if (chunk > 0) {
			output[i] += partial_chunk_sums[chunk - 1];
		}
	}
	return nullptr;
}

int main() {
	const int M = 128, K = 128, N = 128;
	vector<float> A(M * K), B(K * N), C(M * N);
	fill_random(A);
	fill_random(B);
	pthread_t threads[M * N];
	MatMulArgs args[M * N];
	auto start = high_resolution_clock::now();
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			args[i * N + j] = { A.data(), B.data(), C.data(), M, K, N, i, j };
			pthread_create(&threads[i * N + j], nullptr, pthread_matmul, &args[i * N + j]);
		}
	}

	for (int i = 0; i < M * N; ++i) {
		pthread_join(threads[i], nullptr);
	}
	auto end = high_resolution_clock::now();

	cout << "Time (pthread): "
		<< duration_cast<milliseconds>(end - start).count() << " ms\n";


	// Reset C for the next multiplication
	fill(C.begin(), C.end(), 0.0f);

	pthread_t row_threads[M];
	MatulRowArgs row_args[M];
	start = high_resolution_clock::now();
	for (int i = 0; i < M; ++i) {
		row_args[i] = { A.data(), B.data(), C.data(), M, K, N, i };
		pthread_create(&row_threads[i], nullptr, pthread_matmul_row, &row_args[i]);
	}

	for (int i = 0; i < M; ++i) {
		pthread_join(row_threads[i], nullptr);
	}
	end = high_resolution_clock::now();

	cout << "Time (pthread row): "
		<< duration_cast<milliseconds>(end - start).count() << " ms\n";

	//prefix_sum
	int size = 10000000; // 1 million elements
	vector<float> input(size);

	fill_random(input);
	vector<float> output(size);
	const int min_threads = 4; // Minimum number of threads
	const int max_threads = 8; // Maximum number of threads
	for (int threads = min_threads; threads <= max_threads; threads++) {
		int chunk_size = size / threads;
		vector<float> partial_chunk_sums(chunk_size);
		prefixSumArgs args[threads];
		pthread_t thread_ids[threads];
		auto start = high_resolution_clock::now();
		for (int i = 0; i < threads; ++i) {
			args[i] = { input.data(), output.data(), i * chunk_size, (i + 1) * chunk_size };
			if (i == threads - 1) {
				args[i].end = size; // Last thread takes the remaining elements
			}
			pthread_create(&thread_ids[i], nullptr, pthread_prefix_sum, &args[i]);
		}
		for (int i = 0; i < threads; ++i) {
			pthread_join(thread_ids[i], nullptr);
		}


		for (int i = 0; i < threads; i++) {
			partial_chunk_sums[i] = output[((i + 1) * chunk_size) - 1];
			if (i > 0) {
				partial_chunk_sums[i] += partial_chunk_sums[i - 1];
			}
		}

		pthread_t acc_thread[threads];
		prefixSumAccArgs acc_args[threads];
		for (int i = 0; i < threads; ++i) {
			acc_args[i] = { output.data(), partial_chunk_sums.data(), i * chunk_size, (i + 1) * chunk_size,i };
			if (i == threads - 1) {
				acc_args[i].end = size; // Last thread takes the remaining elements
			}
			pthread_create(&acc_thread[i], nullptr, pthread_prefix_sum_acc, &acc_args[i]);
		}
		for (int i = 0; i < threads; ++i) {
			pthread_join(acc_thread[i], nullptr);  // Add this after launching all acc threads
		}
		auto end = high_resolution_clock::now();
		cout << "Time (pthread prefix sum with " << threads << " threads): "
			<< duration_cast<milliseconds>(end - start).count() << " ms\n";

		float expected_sum = accumulate(input.begin(), input.end(), 0.0f);
		cout << "Final output sum: " << output.back() << ", expected: " << expected_sum << endl;
	}
	return 0;
}




// metrics for 128x128x128 matrix multiplication using pthreads:
//Time(pthread) : 850 ms
//Time(pthread row) : 5 ms


//Rule of Thumb
//Use as many threads as CPU cores or logical units(maybe a few times more), not hundreds of thousands.