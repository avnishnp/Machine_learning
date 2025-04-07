#include <iostream>
#include <chrono>
#include <cmath>

using namespace std::chrono;

// GPU kernel for vector-matrix multiplication
__global__ void vectorMatrixMult(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += A[i*N+j] * B[j];
        }
        C[i] = sum;
    }
}

// CPU implementation for comparison
void vectorMatrixMultCPU(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += A[i*N+j] * B[j];
        }
        C[i] = sum;
    }
}

// Verify results
bool verifyResults(const float* C_ref, const float* C, int N) {
    for (int i = 0; i < N; i++) {
        if (fabs(C_ref[i] - C[i]) > 1e-5) {
            printf("Mismatch at position %d: CPU=%f, GPU=%f\n", i, C_ref[i], C[i]);
            return false;
        }
    }
    return true;
}

// Print arrays (for small arrays)
void printMatrix(const float* M, int rows, int cols, const char* name) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", M[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void printVector(const float* V, int N, const char* name) {
    printf("%s: ", name);
    for (int i = 0; i < N; i++) {
        printf("%.2f ", V[i]);
    }
    printf("\n\n");
}

int main() {
    // Test with various sizes
    int sizes[] = {10, 100, 1000, 10000, 100000};
    
    for (int s = 0; s < 5; s++) {
        int N = sizes[s];
        printf("\n===== Matrix Size: %d x %d, Vector Size: %d =====\n", N, N, N);
        
        // Allocate host memory
        float *A = (float *)malloc(N*N*sizeof(float));
        float *B = (float *)malloc(N*sizeof(float));
        float *C_cpu = (float *)malloc(N*sizeof(float));
        float *C_gpu = (float *)malloc(N*sizeof(float));
        
        // Initialize input data
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i * N + j] = 1.0f;  // Simple initialization for verification
            }
            B[i] = 2.0f;
            C_cpu[i] = 0.0f;
            C_gpu[i] = 0.0f;
        }
        
        // CPU implementation timing
        auto cpu_start = high_resolution_clock::now();
        vectorMatrixMultCPU(A, B, C_cpu, N);
        auto cpu_end = high_resolution_clock::now();
        auto cpu_duration = duration_cast<microseconds>(cpu_end - cpu_start);
        
        // Allocate device memory
        float *d_a, *d_b, *d_c;
        cudaMalloc(&d_a, N*N*sizeof(float));
        cudaMalloc(&d_b, N*sizeof(float));
        cudaMalloc(&d_c, N*sizeof(float));
        
        // Copy data from host to device
        auto h2d_start = high_resolution_clock::now();
        cudaMemcpy(d_a, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, B, N*sizeof(float), cudaMemcpyHostToDevice);
        auto h2d_end = high_resolution_clock::now();
        auto h2d_duration = duration_cast<microseconds>(h2d_end - h2d_start);
        
        // GPU kernel timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        int blockSize = 256;
        int gridSize = (N + blockSize - 1) / blockSize;
        
        cudaEventRecord(start);
        vectorMatrixMult<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float kernel_ms = 0;
        cudaEventElapsedTime(&kernel_ms, start, stop);
        
        // Copy results back
        auto d2h_start = high_resolution_clock::now();
        cudaMemcpy(C_gpu, d_c, N*sizeof(float), cudaMemcpyDeviceToHost);
        auto d2h_end = high_resolution_clock::now();
        auto d2h_duration = duration_cast<microseconds>(d2h_end - d2h_start);
        
        // Verify results
        bool correct = verifyResults(C_cpu, C_gpu, N);
        
        // Print small matrices (only for N=10)
        if (N == 10) {
            printMatrix(A, N, N, "Matrix A");
            printVector(B, N, "Vector B");
            printVector(C_cpu, N, "Result C (CPU)");
            printVector(C_gpu, N, "Result C (GPU)");
        }
        
        // Calculate total GPU time including transfers
        auto total_gpu_time = h2d_duration.count() + (kernel_ms * 1000) + d2h_duration.count();
        
        // Print performance results
        printf("Performance Results:\n");
        printf("CPU time:                  %ld microseconds\n", cpu_duration.count());
        printf("GPU Memory Transfer Time:  %ld microseconds (H2D: %ld, D2H: %ld)\n", 
               h2d_duration.count() + d2h_duration.count(),
               h2d_duration.count(), d2h_duration.count());
        printf("GPU Kernel Time:           %.3f microseconds\n", kernel_ms * 1000);
        printf("GPU Total Time:            %.3f microseconds\n", total_gpu_time);
        
        // Compute speedup
        float kernel_speedup = cpu_duration.count() / (kernel_ms * 1000);
        float total_speedup = cpu_duration.count() / (float)total_gpu_time;
        
        printf("Speedup (kernel-only):     %.2fx\n", kernel_speedup);
        printf("Speedup (with transfers):  %.2fx\n", total_speedup);
        printf("Results match: %s\n", correct ? "Yes" : "No");
        
        // Free memory
        free(A);
        free(B);
        free(C_cpu);
        free(C_gpu);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        
        // Destroy CUDA events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    return 0;
}