#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>  // Added the vector header
#include <utility> // Added for std::pair
#include <cuda_runtime.h>

using namespace std::chrono;

// Fixed GPU kernel for Layer Normalization
__global__ void LayerNorm(const float* A, float* B, int rows, int cols) {
    // Calculate row index
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        // Allocate shared memory for each thread block
        extern __shared__ float shared[];
        
        // Each block handles one row, and we need to ensure proper indexing
        float* my_row_data = &shared[threadIdx.x * cols]; // Each thread gets its own row section
        
        // Copy row data to shared memory
        for (int col = 0; col < cols; col++) {
            my_row_data[col] = A[row * cols + col];
        }
        
        // No need for __syncthreads() here as each thread works on its own row data
        
        // Compute mean
        float mean = 0.0f;
        for (int col = 0; col < cols; col++) {
            mean += my_row_data[col];
        }
        mean /= cols;
        
        // Compute variance
        float variance = 0.0f;
        for (int col = 0; col < cols; col++) {
            float diff = my_row_data[col] - mean;
            variance += diff * diff;
        }
        variance /= cols;
        float stddev = sqrtf(variance + 1e-7);
        
        // Normalize
        for (int col = 0; col < cols; col++) {
            B[row * cols + col] = (my_row_data[col] - mean) / stddev;
        }
    }
}

// CPU implementation for Layer Normalization
void LayerNormCPU(const float* A, float* B, int rows, int cols) {
    for (int row = 0; row < rows; row++) {
        // Compute mean
        float mean = 0.0f;
        for (int col = 0; col < cols; col++) {
            mean += A[row * cols + col];
        }
        mean /= cols;
        
        // Compute variance
        float variance = 0.0f;
        for (int col = 0; col < cols; col++) {
            float diff = A[row * cols + col] - mean;
            variance += diff * diff;
        }
        variance /= cols;
        float stddev = sqrt(variance + 1e-7);
        
        // Normalize
        for (int col = 0; col < cols; col++) {
            B[row * cols + col] = (A[row * cols + col] - mean) / stddev;
        }
    }
}

// Verify results between CPU and GPU implementations
bool verifyResults(const float* A, const float* B, int size, float tolerance = 1e-4) {
    for (int i = 0; i < size; i++) {
        if (fabs(A[i] - B[i]) > tolerance) {
            printf("Mismatch at position %d: CPU=%f, GPU=%f\n", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

// Print a small matrix
void printMatrix(const float* M, int rows, int cols, const char* name) {
    printf("%s:\n", name);
    for (int i = 0; i < rows && i < 10; i++) { // Print at most 10 rows
        for (int j = 0; j < cols && j < 10; j++) { // Print at most 10 columns
            printf("%.2f ", M[i * cols + j]);
        }
        printf("%s", (cols > 10) ? "...\n" : "\n");
    }
    if (rows > 10) printf("...\n");
    printf("\n");
}

// Test different problem sizes
void runBenchmark(int rows, int cols) {
    printf("\n===== Matrix Size: %d x %d =====\n", rows, cols);
    
    // Allocate host memory
    float *A = (float*)malloc(rows * cols * sizeof(float));
    float *B_cpu = (float*)malloc(rows * cols * sizeof(float));
    float *B_gpu = (float*)malloc(rows * cols * sizeof(float));
    
    // Initialize input matrix with random values
    srand(42); // Fixed seed for reproducibility
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            A[i * cols + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
    
    // CPU implementation timing
    auto cpu_start = high_resolution_clock::now();
    LayerNormCPU(A, B_cpu, rows, cols);
    auto cpu_end = high_resolution_clock::now();
    auto cpu_duration = duration_cast<microseconds>(cpu_end - cpu_start);
    
    // Allocate device memory
    float *d_a, *d_b;
    cudaMalloc(&d_a, rows * cols * sizeof(float));
    cudaMalloc(&d_b, rows * cols * sizeof(float));
    
    // Copy data from host to device
    auto h2d_start = high_resolution_clock::now();
    cudaMemcpy(d_a, A, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    auto h2d_end = high_resolution_clock::now();
    auto h2d_duration = duration_cast<microseconds>(h2d_end - h2d_start);
    
    // GPU kernel timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Configure kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
    
    // We need shared memory per block equal to: rows per block * cols * sizeof(float)
    // This is a conservative estimate - each thread needs cols*sizeof(float) shared memory
    size_t shared_memory_size = threadsPerBlock * cols * sizeof(float);
    
    // Check if we're requesting too much shared memory
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    size_t max_shared = prop.sharedMemPerBlock;
    
    if (shared_memory_size > max_shared) {
        printf("WARNING: Requested shared memory (%zu bytes) exceeds device limit (%zu bytes).\n", 
               shared_memory_size, max_shared);
        printf("Reducing threads per block to fit in shared memory.\n");
        
        // Adjust threads per block to fit within shared memory limits
        threadsPerBlock = (max_shared / (cols * sizeof(float)));
        threadsPerBlock = (threadsPerBlock > 0) ? threadsPerBlock : 1;
        blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
        shared_memory_size = threadsPerBlock * cols * sizeof(float);
        
        printf("Adjusted to %d threads per block, %d blocks in grid.\n", 
               threadsPerBlock, blocksPerGrid);
    }
    
    // Launch kernel
    cudaEventRecord(start);
    LayerNorm<<<blocksPerGrid, threadsPerBlock, shared_memory_size>>>(d_a, d_b, rows, cols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
    
    float kernel_ms = 0;
    cudaEventElapsedTime(&kernel_ms, start, stop);
    
    // Copy results back to host
    auto d2h_start = high_resolution_clock::now();
    cudaMemcpy(B_gpu, d_b, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    auto d2h_end = high_resolution_clock::now();
    auto d2h_duration = duration_cast<microseconds>(d2h_end - d2h_start);
    
    // Verify results
    bool results_match = verifyResults(B_cpu, B_gpu, rows * cols);
    
    // Print small matrices for visual verification (only for small sizes)
    if (rows <= 10 && cols <= 10) {
        printMatrix(A, rows, cols, "Input Matrix A");
        printMatrix(B_cpu, rows, cols, "CPU Result");
        printMatrix(B_gpu, rows, cols, "GPU Result");
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
    printf("Results match: %s\n", results_match ? "Yes" : "No");
    
    // Free memory
    free(A);
    free(B_cpu);
    free(B_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    
    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    // Test with various matrix sizes
    int test_sizes[][2] = {
        {10, 10},         // Small square matrix
        {100, 100},       // Medium square matrix
        {1000, 1000},     // Large square matrix
        {10000, 100},     // Many rows, fewer columns
        {100, 10000}      // Fewer rows, many columns
    };
    
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    for (int i = 0; i < num_tests; i++) {
        runBenchmark(test_sizes[i][0], test_sizes[i][1]);
    }
    
    return 0;
}