#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>
#include <algorithm>
#include <random>
#include <iostream>

using namespace std::chrono;

// GPU co-rank function for parallel merge
__device__ void co_rank(const int* A, const int* B, int k, const int N, const int M, int* i_out, int* j_out) {
    int low = max(0, k-M);
    int high = min(k, N);
    
    while (low <= high) {
        int i = (low + high) / 2;
        int j = k - i;
        
        if (j < 0) {
            high = i - 1;
            continue;
        }
        if (j > M) {
            low = i + 1;
            continue;
        }
        if (i > 0 && j < M && A[i-1] > B[j]) {
            high = i - 1;
        }
        else if (j > 0 && i < N && B[j-1] > A[i]) {
            low = i + 1;
        }
        else {
            *i_out = i;
            *j_out = j;
            return;
        }
    }
}

// GPU kernel for parallel merge
__global__ void parallel_merge(const int* A, const int* B, int* C, const int N, const int M) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < N + M) {
        int i, j;
        co_rank(A, B, tid, N, M, &i, &j);
        
        if (j >= M || (i < N && A[i] <= B[j])) {
            C[tid] = A[i];
        } else {
            C[tid] = B[j];
        }
    }
}

// CPU sequential merge implementation
void sequential_merge(const int* A, const int* B, int* C, int N, int M) {
    int i = 0, j = 0, k = 0;
    
    while (i < N && j < M) {
        if (A[i] <= B[j]) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }
    
    // Copy remaining elements
    while (i < N) {
        C[k++] = A[i++];
    }
    
    while (j < M) {
        C[k++] = B[j++];
    }
}

// CPU implementation of co-rank for validation
void cpu_co_rank(const int* A, const int* B, int k, int N, int M, int* i_out, int* j_out) {
    int low = std::max(0, k-M);
    int high = std::min(k, N);
    
    while (low <= high) {
        int i = (low + high) / 2;
        int j = k - i;
        
        if (j < 0) {
            high = i - 1;
            continue;
        }
        if (j > M) {
            low = i + 1;
            continue;
        }
        if (i > 0 && j < M && A[i-1] > B[j]) {
            high = i - 1;
        }
        else if (j > 0 && i < N && B[j-1] > A[i]) {
            low = i + 1;
        }
        else {
            *i_out = i;
            *j_out = j;
            return;
        }
    }
}

// CPU parallel merge (for validation)
void cpu_parallel_merge(const int* A, const int* B, int* C, int N, int M) {
    for (int k = 0; k < N + M; k++) {
        int i, j;
        cpu_co_rank(A, B, k, N, M, &i, &j);
        
        if (j >= M || (i < N && A[i] <= B[j])) {
            C[k] = A[i];
        } else {
            C[k] = B[j];
        }
    }
}

// Function to check if an array is sorted
bool is_sorted(const int* arr, int n) {
    for (int i = 1; i < n; i++) {
        if (arr[i] < arr[i-1]) {
            return false;
        }
    }
    return true;
}

// Function to verify merge results
bool verify_merge(const int* A, const int* B, const int* C, int N, int M) {
    // Check if C is sorted
    if (!is_sorted(C, N+M)) {
        std::cout << "Error: Merged array is not sorted" << std::endl;
        return false;
    }
    
    // Check if all elements are present
    int* temp_A = new int[N];
    int* temp_B = new int[M];
    int* temp_C = new int[N+M];
    
    std::copy(A, A+N, temp_A);
    std::copy(B, B+M, temp_B);
    std::copy(C, C+N+M, temp_C);
    
    std::sort(temp_A, temp_A+N);
    std::sort(temp_B, temp_B+M);
    
    int* expected = new int[N+M];
    std::merge(temp_A, temp_A+N, temp_B, temp_B+M, expected);
    
    for (int i = 0; i < N+M; i++) {
        if (temp_C[i] != expected[i]) {
            std::cout << "Error: Mismatch at position " << i << ". Expected " 
                      << expected[i] << ", got " << temp_C[i] << std::endl;
            
            delete[] temp_A;
            delete[] temp_B;
            delete[] temp_C;
            delete[] expected;
            return false;
        }
    }
    
    delete[] temp_A;
    delete[] temp_B;
    delete[] temp_C;
    delete[] expected;
    return true;
}

// Function to generate sorted random arrays
void generate_sorted_arrays(int* A, int* B, int N, int M) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 1000000);
    
    for (int i = 0; i < N; i++) {
        A[i] = dis(gen);
    }
    
    for (int i = 0; i < M; i++) {
        B[i] = dis(gen);
    }
    
    std::sort(A, A+N);
    std::sort(B, B+M);
}

// Benchmark function
void run_benchmark(int N, int M) {
    std::cout << "\n===== Arrays: A[" << N << "], B[" << M << "] =====" << std::endl;
    
    // Allocate host memory
    int *A = new int[N];
    int *B = new int[M];
    int *C_seq = new int[N+M];
    int *C_par_cpu = new int[N+M];
    int *C_par_gpu = new int[N+M];
    
    // Generate sorted random arrays
    generate_sorted_arrays(A, B, N, M);
    
    if (N <= 10 && M <= 10) {
        std::cout << "Array A: ";
        for (int i = 0; i < N; i++) {
            std::cout << A[i] << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Array B: ";
        for (int i = 0; i < M; i++) {
            std::cout << B[i] << " ";
        }
        std::cout << std::endl;
    }
    
    // CPU Sequential Merge Timing
    auto seq_start = high_resolution_clock::now();
    sequential_merge(A, B, C_seq, N, M);
    auto seq_end = high_resolution_clock::now();
    auto seq_duration = duration_cast<microseconds>(seq_end - seq_start);
    
    // CPU Parallel Merge Timing
    auto par_cpu_start = high_resolution_clock::now();
    cpu_parallel_merge(A, B, C_par_cpu, N, M);
    auto par_cpu_end = high_resolution_clock::now();
    auto par_cpu_duration = duration_cast<microseconds>(par_cpu_end - par_cpu_start);
    
    // Verify CPU parallel merge against sequential merge
    bool cpu_par_correct = true;
    for (int i = 0; i < N+M; i++) {
        if (C_seq[i] != C_par_cpu[i]) {
            cpu_par_correct = false;
            break;
        }
    }
    
    // GPU Merge
    int *d_A, *d_B, *d_C;
    
    // Allocate device memory
    cudaMalloc(&d_A, N * sizeof(int));
    cudaMalloc(&d_B, M * sizeof(int));
    cudaMalloc(&d_C, (N+M) * sizeof(int));
    
    // Copy data to device
    cudaMemcpy(d_A, A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, M * sizeof(int), cudaMemcpyHostToDevice);
    
    // Set up execution configuration
    int blockSize = 256;
    int gridSize = ((N+M) + blockSize - 1) / blockSize;
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record start time
    cudaEventRecord(start);
    
    // Launch kernel
    parallel_merge<<<gridSize, blockSize>>>(d_A, d_B, d_C, N, M);
    
    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float gpu_milliseconds = 0;
    cudaEventElapsedTime(&gpu_milliseconds, start, stop);
    
    // Copy result back to host
    cudaMemcpy(C_par_gpu, d_C, (N+M) * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Verify GPU results
    bool gpu_correct = verify_merge(A, B, C_par_gpu, N, M);
    
    if (N <= 10 && M <= 10) {
        std::cout << "Sequential Merge: ";
        for (int i = 0; i < N+M; i++) {
            std::cout << C_seq[i] << " ";
        }
        std::cout << std::endl;
        
        std::cout << "GPU Parallel Merge: ";
        for (int i = 0; i < N+M; i++) {
            std::cout << C_par_gpu[i] << " ";
        }
        std::cout << std::endl;
    }
    
    // Print performance results
    std::cout << "Performance Results:" << std::endl;
    std::cout << "CPU Sequential Time:   " << seq_duration.count() << " microseconds" << std::endl;
    std::cout << "CPU Parallel Time:     " << par_cpu_duration.count() << " microseconds" << std::endl;
    std::cout << "GPU Time:              " << gpu_milliseconds * 1000 << " microseconds" << std::endl;
    
    // Calculate speedups
    float cpu_speedup = (float)seq_duration.count() / par_cpu_duration.count();
    float gpu_speedup = (float)seq_duration.count() / (gpu_milliseconds * 1000);
    
    std::cout << "CPU Parallel Speedup:  " << cpu_speedup << "x" << std::endl;
    std::cout << "GPU Speedup:           " << gpu_speedup << "x" << std::endl;
    std::cout << "CPU Parallel Correct:  " << (cpu_par_correct ? "Yes" : "No") << std::endl;
    std::cout << "GPU Results Correct:   " << (gpu_correct ? "Yes" : "No") << std::endl;
    
    // Clean up
    delete[] A;
    delete[] B;
    delete[] C_seq;
    delete[] C_par_cpu;
    delete[] C_par_gpu;
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    // Test with various array sizes
    int sizes[][2] = {
        {5, 5},          // Tiny arrays (original example)
        {100, 100},      // Small arrays
        {10000, 10000},  // Medium arrays
        {100000, 100000},// Large arrays
        {1000000, 1000000}, // Very large arrays
        {1000000, 1000}  // Asymmetric case
    };
    
    int num_tests = sizeof(sizes) / sizeof(sizes[0]);
    
    for (int i = 0; i < num_tests; i++) {
        run_benchmark(sizes[i][0], sizes[i][1]);
    }
    
    return 0;
}