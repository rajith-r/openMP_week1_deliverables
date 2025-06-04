#include <iostream>
#include "openmp_functions.h"
#include "serial_functions.h"
#include "benchmarks.h"
#include <omp.h>
#include <vector>
int main() {
    const int M = 2, K = 3, N = 2;
    float A[M * K] = { 1, 2, 3,
                      4, 5, 6 };

    float B[K * N] = { 7, 8,
                      9, 10,
                      11, 12 };

    float C[M * N] = { 0 };

	// measure time for the serial matrix multiplication
	std::cout << "Performing matrix multiplication using serial function...\n";
	double start_time_serial = omp_get_wtime();
	matmul_serial(A, B, C, M, K, N);
	double end_time_serial = omp_get_wtime();
	std::cout << "Result matrix C after serial multiplication:\n";
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			std::cout << C[i * N + j] << " ";
		}
		std::cout << "\n";
	}
	std::cout << "Time taken for serial matrix multiplication: " << (end_time_serial - start_time_serial) << " seconds\n";

	//measure time for the matrix multiplication
	std::cout << "Performing matrix multiplication using OpenMP...\n";
	double start_time = omp_get_wtime();
    matmul_omp(A, B, C, M, K, N);
	double end_time = omp_get_wtime();
    std::cout << "Result matrix C:\n";
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << C[i * N + j] << " ";
        }
        std::cout << "\n";
    }
	std::cout << "Time taken for matrix multiplication: " << (end_time - start_time) << " seconds\n";

	//measure time for the matrix multiplication with SIMD
	std::cout << "Performing matrix multiplication using OpenMP with SIMD...\n";
	double start_time_simd = omp_get_wtime();
	matmul_omp_simd(A, B, C, M, K, N);
	double end_time_simd = omp_get_wtime();
	// Display the result matrix C after SIMD
	std::cout << "Result matrix C after SIMD:\n";
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << C[i * N + j] << " ";
        }
        std::cout << "\n";
	}
	std::cout << "Time taken for matrix multiplication with SIMD: " << (end_time_simd - start_time_simd) << " seconds\n";

	// huge input for performance testing
	const int M_large = 1000, K_large = 1000, N_large = 1000;
	float* A_large = new float[M_large * K_large];
	float* B_large = new float[K_large * N_large];
	float* C_large = new float[M_large * N_large];

	// Initialize matrices A and B with random values
	for (int i = 0; i < M_large * K_large; ++i) {
		A_large[i] = static_cast<float>(rand()) / RAND_MAX;
	}
	for (int i = 0; i < K_large * N_large; ++i) {
		B_large[i] = static_cast<float>(rand()) / RAND_MAX;
	}

	// Measure time for the large matrix multiplication using serial function
	std::cout << "Performing large matrix multiplication using serial function...\n";
	double start_time_large_serial = omp_get_wtime();
	matmul_serial(A_large, B_large, C_large, M_large, K_large, N_large);
	double end_time_large_serial = omp_get_wtime();
	std::cout << "Time taken for large matrix multiplication using serial function: " << (end_time_large_serial - start_time_large_serial) << " seconds\n";

	// Measure time for the large matrix multiplication
	std::cout << "Performing large matrix multiplication using OpenMP...\n";
	double start_time_large = omp_get_wtime();
	matmul_omp(A_large, B_large, C_large, M_large, K_large, N_large);
	double end_time_large = omp_get_wtime();
	std::cout << "Time taken for large matrix multiplication: " << (end_time_large - start_time_large) << " seconds\n";

	// Measure time for the large matrix multiplication with SIMD
	std::cout << "Performing large matrix multiplication using OpenMP with SIMD...\n";
	double start_time_large_simd = omp_get_wtime();
	matmul_omp_simd(A_large, B_large, C_large, M_large, K_large, N_large);
	double end_time_large_simd = omp_get_wtime();
	std::cout << "Time taken for large matrix multiplication with SIMD: " << (end_time_large_simd - start_time_large_simd) << " seconds\n";



	// Measure time for prefix sum using OpenMP
	std::vector<float> input = { 1, 2, 3, 4, 5,6,7,8,9,10 };
	std::vector<float> output;
	std::cout << "Performing prefix sum using OpenMP...\n";
	double start_time_prefix = omp_get_wtime();
	prefix_sum_omp(input, output, input.size());
	double end_time_prefix = omp_get_wtime();
	std::cout << "Result of prefix sum:\n";
	for (const auto& val : output) {
		std::cout << val << " ";
	}
	std::cout << "\n";
	std::cout << "Time taken for prefix sum using OpenMP: " << (end_time_prefix - start_time_prefix) << " seconds\n";
    
	
	benchmark_matmul(512, 512, 512);
	benchmark_prefix_sum(1 << 20); // 1 million elements
	return 0;
}

//#include <stdlib.h>
//#include <stdio.h>
//#include <omp.h>
//
//#ifndef N
//#define N 5
//#endif
//#ifndef FS
//#define FS 38
//#endif
//
//struct node {
//    int data;
//    int fibdata;
//    struct node* next;
//};
//
//int fib(int n) {
//    int x, y;
//    if (n < 2) {
//        return (n);
//    }
//    else {
//        x = fib(n - 1);
//        y = fib(n - 2);
//        return (x + y);
//    }
//}
//
//void processwork(struct node* p)
//{
//    int n;
//    n = p->data;
//    p->fibdata = fib(n);
//}
//
//struct node* init_list(struct node* p) {
//    int i;
//    struct node* head = NULL;
//    struct node* temp = NULL;
//
//    head = (struct node*)malloc(sizeof(struct node));
//    p = head;
//    p->data = FS;
//    p->fibdata = 0;
//    for (i = 0; i < N; i++) {
//        temp = (struct node*)malloc(sizeof(struct node));
//        p->next = temp;
//        p = temp;
//        p->data = FS + i + 1;
//        p->fibdata = i + 1;
//    }
//    p->next = NULL;
//    return head;
//}
//
//int main(int argc, char* argv[]) {
//    double start, end;
//    struct node* p = NULL;
//    struct node* temp = NULL;
//    struct node* head = NULL;
//
//    printf("Process linked list\n");
//    printf("  Each linked list node will be processed by function 'processwork()'\n");
//    printf("  Each ll node will compute %d fibonacci numbers beginning with %d\n", N, FS);
//
//    p = init_list(p);
//    head = p;
//
//    start = omp_get_wtime();
//    {
//        while (p != NULL) {
//            processwork(p);
//            p = p->next;
//        }
//    }
//
//    end = omp_get_wtime();
//    p = head;
//    while (p != NULL) {
//        printf("%d : %d\n", p->data, p->fibdata);
//        temp = p->next;
//        free(p);
//        p = temp;
//    }
//    free(p);
//
//    printf("Compute Time: %f seconds\n", end - start);
//
//    return 0;
//}
/*
**  PROGRAM: Mandelbrot area
**
**  PURPOSE: Program to compute the area of a  Mandelbrot set.
**           Correct answer should be around 1.510659.
**           WARNING: this program may contain errors
**
**  USAGE:   Program runs without input ... just run the executable
**
**  HISTORY: Written:  (Mark Bull, August 2011).
**           Changed "comples" to "d_comples" to avoid collsion with
**           math.h complex type (Tim Mattson, September 2011)
*/

//#include <stdio.h>
//#include <stdlib.h>
//#include <math.h>
//#include <omp.h>
//
//# define NPOINTS 1000
//# define MAXITER 1000
//
//void testpoint(struct d_complex  c);
//
//struct d_complex {
//    double r;
//    double i;
//};
//
//struct d_complex c;
//int numoutside = 0;
//
//int main() {
//    int i, j;
//    double area, error, eps = 1.0e-5;
//
//
//    //   Loop over grid of points in the complex plane which contains the Mandelbrot set,
//    //   testing each point to see whether it is inside or outside the set.
//
//#pragma omp parallel for default(none) private(c,j)
//    for (i = 0; i < NPOINTS; i++) {
//        for (j = 0; j < NPOINTS; j++) {
//            c.r = -2.0 + 2.5 * (double)(i) / (double)(NPOINTS)+eps;
//            c.i = 1.125 * (double)(j) / (double)(NPOINTS)+eps;
//            testpoint(c);
//        }
//    }
//
//    // Calculate area of set and error estimate and output the results
//
//    area = 2.0 * 2.5 * 1.125 * (double)(NPOINTS * NPOINTS - numoutside) / (double)(NPOINTS * NPOINTS);
//    error = area / (double)NPOINTS;
//
//    printf("Area of Mandlebrot set = %12.8f +/- %12.8f\n", area, error);
//    printf("Correct answer should be around 1.510659\n");
//
//}
//
//void testpoint(struct d_complex  c) {
//
//    // Does the iteration z=z*z+c, until |z| > 2 when point is known to be outside set
//    // If loop count reaches MAXITER, point is considered to be inside the set
//
//    struct d_complex z;
//    int iter;
//    double temp;
//
//    z = c;
//    for (iter = 0; iter < MAXITER; iter++) {
//        temp = (z.r * z.r) - (z.i * z.i) + c.r;
//        z.i = z.r * z.i * 2 + c.i;
//        z.r = temp;
//        if ((z.r * z.r + z.i * z.i) > 4.0) {
//            #pragma omp atomic
//            numoutside++;
//            break;
//        }
//    }
//}




//#include <iostream>
//#include <omp.h>
//#include <chrono>  // Include high-resolution timer
//static long num_steps = 100000;
//double step;
//int main()
//{
//	double x, pi, sum = 0.0;
//	step = 1.0 / (double)num_steps;
//	auto start = std::chrono::high_resolution_clock::now();
//	#pragma omp parallel for private(x) reduction(+:sum)
//	for (int i = 0; i < num_steps; i++)
//	{
//		x = (i + 0.5) * step;
//		sum += 4.0 / (1.0 + x * x);
//	}
//	pi = step * sum;
//	auto end = std::chrono::high_resolution_clock::now();
//	double duration = std::chrono::duration<double>(end - start).count();
//	std::cout << "PI: " << pi << std::endl;
//	std::cout << "Execution time: " << duration << " seconds" << std::endl;
//
//
//}
	// Parallelize the loop


// openMP.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

//#include <iostream>
//#include <omp.h>
//#include <chrono>  // Include high-resolution timer
//static long num_steps = 100000;
//double step;
//#define NUM_THREADS 2
//int main()
//{
//	double x, pi;
//	step = 1.0 / (double)num_steps;
//
//	omp_set_num_threads(NUM_THREADS);
//	// Start timer
//	auto start = std::chrono::high_resolution_clock::now();
//	pi = 0.0;
//#pragma omp parallel
//	{
//		int i;
//		double x;
//		double sum = 0.0;
//		int id = omp_get_thread_num();
//		int num_threads = omp_get_num_threads();
//		sum = 0.0;
//		for (i = id; i < num_steps; i += num_threads)
//		{
//			x = (i + 0.5) * step;
//			sum += 4.0 / (1.0 + x * x);
//		}
//		
//		#pragma omp critical
//		{
//			pi += sum * step;
//		}
//	}
//
//
//	// Stop timer
//	auto end = std::chrono::high_resolution_clock::now();
//	double duration = std::chrono::duration<double>(end - start).count();
//
//	std::cout << "PI: " << pi << std::endl;
//	std::cout << "Execution time: " << duration << " seconds" << std::endl;
//
//}
// openMP.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

//#include <iostream>
//#include <omp.h>
//#include <chrono>  // Include high-resolution timer
//static long num_steps = 100000;
//double step;
//#define padding 8
//#define NUM_THREADS 2
//int main()
//{
//	double x, pi,sum[NUM_THREADS][padding];
//	step = 1.0 / (double)num_steps;
//	
//	omp_set_num_threads(NUM_THREADS);
//	// Start timer
//	auto start = std::chrono::high_resolution_clock::now();
//#pragma omp parallel
//	{
//		int i;
//		double x;
//		int id = omp_get_thread_num();
//		int num_threads = omp_get_num_threads();
//		sum[id][0] = 0.0;
//		for (i = id; i < num_steps; i += num_threads)
//		{
//			x = (i + 0.5) * step;
//			sum[id][0] += 4.0 / (1.0 + x * x);
//		}
//
//	}
//	pi = 0.0;
//	for (int i = 0; i < NUM_THREADS; i++)
//	{
//		pi += sum[i][0] * step;
//	}
//
//	// Stop timer
//	auto end = std::chrono::high_resolution_clock::now();
//	double duration = std::chrono::duration<double>(end - start).count();
//
//	std::cout << "PI: " << pi << std::endl;
//	std::cout << "Execution time: " << duration << " seconds" << std::endl;
//
//
//	//for (i = 0; i < num_steps; i++)
//	//{
//	//	x = (i + 0.5) * step;
//	//	sum = sum + 4.0 / (1.0 + x * x);
//	//}
//	//pi = step * sum;
//}



// openMP.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

//#include <iostream>
//#include <omp.h>
//#include <chrono>  // Include high-resolution timer
//static long num_steps = 100000;
//double step;
//#define NUM_THREADS 2
//int main()
//{
//	double x, pi, sum[NUM_THREADS];
//	step = 1.0 / (double)num_steps;
//
//	omp_set_num_threads(NUM_THREADS);
//	// Start timer
//	auto start = std::chrono::high_resolution_clock::now();
//#pragma omp parallel
//	{
//		int i;
//		double x;
//		int id = omp_get_thread_num();
//		int num_threads = omp_get_num_threads();
//		sum[id] = 0.0;
//		for (i = id; i < num_steps; i += num_threads)
//		{
//			x = (i + 0.5) * step;
//			sum[id] += 4.0 / (1.0 + x * x);
//		}
//
//	}
//	pi = 0.0;
//	for (int i = 0; i < NUM_THREADS; i++)
//	{
//		pi += sum[i] * step;
//	}
//
//	// Stop timer
//	auto end = std::chrono::high_resolution_clock::now();
//	double duration = std::chrono::duration<double>(end - start).count();
//
//	std::cout << "PI: " << pi << std::endl;
//	std::cout << "Execution time: " << duration << " seconds" << std::endl;
//
//
//	//for (i = 0; i < num_steps; i++)
//	//{
//	//	x = (i + 0.5) * step;
//	//	sum = sum + 4.0 / (1.0 + x * x);
//	//}
//	//pi = step * sum;
//}