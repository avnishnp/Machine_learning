#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

using namespace std::chrono;
// mask means the kernel
#define Mask_width 5  

// CUDA constant memory for the mask
__constant__ float M[Mask_width];

// GPU kernel for 1D convolution without tiling
__global__ void oned_convolution_kernel(const float* A, float* C, int n) {
    int threadId = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadId;
    if (i < n) {
        float result = 0.0f;
        for (int k = -Mask_width/2; k <= Mask_width/2; k++) {
            if (i + k >= 0 && i + k < n) {
                result += A[i + k] * M[k + Mask_width/2];
            }
        }
        C[i] = result;
    }
}

// CPU implementation for 1D convolution
void oned_convolution_cpu(const float* A, const float* mask, float* C, int n) {
    for (int i = 0; i < n; i++) {
        float result = 0.0f;
        for (int k = -Mask_width/2; k <= Mask_width/2; k++) {
            if (i + k >= 0 && i + k < n) {
                result += A[i + k] * mask[k + Mask_width/2];
            }
        }
        C[i] = result;
    }
}

// Host function to check for CUDA errors
void checkCudaError(const char* message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << message << " - CUDA Error: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Verify results between CPU and GPU implementations
bool verifyResults(const float* A, const float* B, int size, float tolerance = 1e-5) {
    for (int i = 0; i < size; i++) {
        if (std::abs(A[i] - B[i]) > tolerance) {
            printf("Mismatch at position %d: CPU=%f, GPU=%f\n", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

// Run benchmark for a specific array size
void runBenchmark(int n) {
    printf("\n===== Array Size: %d =====\n", n);
    
    // Allocate host memory
    float *A = new float[n];
    float *C_cpu = new float[n];
    float *C_gpu = new float[n];
    float mask[Mask_width];
    
    // Initialize data
    for (int i = 0; i < Mask_width; i++) {
        mask[i] = i;  // Simple mask values
    }
    
    for (int i = 0; i < n; i++) {
        A[i] = i % 100;  // Use modulo to keep values manageable
    }
    
    // CPU implementation timing
    auto cpu_start = high_resolution_clock::now();
    oned_convolution_cpu(A, mask, C_cpu, n);
    auto cpu_end = high_resolution_clock::now();
    auto cpu_duration = duration_cast<microseconds>(cpu_end - cpu_start);
    
    // Allocate device memory
    float *d_a, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));
    
    // Copy data to device
    auto h2d_start = high_resolution_clock::now();
    cudaMemcpy(d_a, A, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(M, mask, Mask_width * sizeof(float));
    auto h2d_end = high_resolution_clock::now();
    auto h2d_duration = duration_cast<microseconds>(h2d_end - h2d_start);
    
    // Launch kernel with timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    cudaEventRecord(start);
    oned_convolution_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_c, n);
    cudaEventRecord(stop);
    
    // Check for kernel launch errors
    checkCudaError("Kernel launch failed");
    
    cudaEventSynchronize(stop);
    float kernel_ms = 0;
    cudaEventElapsedTime(&kernel_ms, start, stop);
    
    // Copy results back to host
    auto d2h_start = high_resolution_clock::now();
    cudaMemcpy(C_gpu, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
    auto d2h_end = high_resolution_clock::now();
    auto d2h_duration = duration_cast<microseconds>(d2h_end - d2h_start);
    
    // Verify results
    bool results_match = verifyResults(C_cpu, C_gpu, n);
    
    // Print input/output for small arrays
    if (n <= 20) {
        printf("Input A: ");
        for (int i = 0; i < n; i++) {
            printf("%.2f ", A[i]);
        }
        printf("\n\nMask: ");
        for (int i = 0; i < Mask_width; i++) {
            printf("%.2f ", mask[i]);
        }
        printf("\n\nCPU Result: ");
        for (int i = 0; i < n; i++) {
            printf("%.2f ", C_cpu[i]);
        }
        printf("\n\nGPU Result: ");
        for (int i = 0; i < n; i++) {
            printf("%.2f ", C_gpu[i]);
        }
        printf("\n");
    }
    
    // Calculate total GPU time including transfers
    auto total_gpu_time = h2d_duration.count() + (kernel_ms * 1000) + d2h_duration.count();
    
    // Print performance results
    printf("\nPerformance Results:\n");
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
    printf("Results match: %s\n", results_match ? "Yes" : "No");
    
    // Free memory
    delete[] A;
    delete[] C_cpu;
    delete[] C_gpu;
    cudaFree(d_a);
    cudaFree(d_c);
    
    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    // Test with various array sizes
    int test_sizes[] = {10, 100, 1000, 10000, 100000, 1000000, 10000000};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    for (int i = 0; i < num_tests; i++) {
        runBenchmark(test_sizes[i]);
    }
    
    return 0;
}