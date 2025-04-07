#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

using namespace std::chrono;
// mask mean the 2d kernel
#define MASK_WIDTH 5
#define MASK_HEIGHT 5
#define MASK_RADIUS_X (MASK_WIDTH/2)
#define MASK_RADIUS_Y (MASK_HEIGHT/2)

// CUDA constant memory for the 2D mask
__constant__ float M[MASK_HEIGHT][MASK_WIDTH];

// GPU kernel for 2D convolution without tiling
__global__ void conv2d_kernel(const float* A, float* C, int width, int height, int pitch) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        float result = 0.0f;
        
        // Apply the mask centered at (row, col)
        for (int i = -MASK_RADIUS_Y; i <= MASK_RADIUS_Y; i++) {
            for (int j = -MASK_RADIUS_X; j <= MASK_RADIUS_X; j++) {
                // Check if the position is within bounds
                if (row + i >= 0 && row + i < height && col + j >= 0 && col + j < width) {
                    result += A[(row + i) * pitch + (col + j)] * 
                              M[i + MASK_RADIUS_Y][j + MASK_RADIUS_X];
                }
            }
        }
        
        C[row * pitch + col] = result;
    }
}

// CPU implementation for 2D convolution
void conv2d_cpu(const float* A, const float mask[MASK_HEIGHT][MASK_WIDTH], 
                float* C, int width, int height, int pitch) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            float result = 0.0f;
            
            // Apply the mask centered at (row, col)
            for (int i = -MASK_RADIUS_Y; i <= MASK_RADIUS_Y; i++) {
                for (int j = -MASK_RADIUS_X; j <= MASK_RADIUS_X; j++) {
                    // Check if the position is within bounds
                    if (row + i >= 0 && row + i < height && col + j >= 0 && col + j < width) {
                        result += A[(row + i) * pitch + (col + j)] * 
                                  mask[i + MASK_RADIUS_Y][j + MASK_RADIUS_X];
                    }
                }
            }
            
            C[row * pitch + col] = result;
        }
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
            printf("Mismatch at position %d: CPU=%f, GPU=%f, diff=%f\n", 
                   i, A[i], B[i], std::abs(A[i] - B[i]));
            return false;
        }
    }
    return true;
}

// Print a small matrix
void printMatrix(const float* M, int width, int height, int pitch, const char* name) {
    printf("%s:\n", name);
    for (int i = 0; i < height && i < 10; i++) {
        for (int j = 0; j < width && j < 10; j++) {
            printf("%.2f ", M[i * pitch + j]);
        }
        printf("%s", (width > 10) ? "...\n" : "\n");
    }
    if (height > 10) printf("...\n");
    printf("\n");
}

// Run benchmark for a specific matrix size
void runBenchmark(int width, int height) {
    printf("\n===== Matrix Size: %d x %d =====\n", width, height);
    
    // For simplicity, we use a pitch equal to width
    int pitch = width;
    int size = pitch * height;
    
    // Allocate host memory
    float *A = new float[size];
    float *C_cpu = new float[size];
    float *C_gpu = new float[size];
    float mask[MASK_HEIGHT][MASK_WIDTH];
    
    // Initialize mask with simple pattern
    for (int i = 0; i < MASK_HEIGHT; i++) {
        for (int j = 0; j < MASK_WIDTH; j++) {
            mask[i][j] = (i * MASK_WIDTH + j) / (float)(MASK_HEIGHT * MASK_WIDTH);
        }
    }
    
    // Initialize input matrix
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            A[i * pitch + j] = (i * width + j) % 100;  // Modulo to keep values manageable
        }
    }
    
    // CPU implementation timing
    auto cpu_start = high_resolution_clock::now();
    conv2d_cpu(A, mask, C_cpu, width, height, pitch);
    auto cpu_end = high_resolution_clock::now();
    auto cpu_duration = duration_cast<microseconds>(cpu_end - cpu_start);
    
    // Allocate device memory
    float *d_a, *d_c;
    cudaMalloc(&d_a, size * sizeof(float));
    cudaMalloc(&d_c, size * sizeof(float));
    
    // Copy data to device
    auto h2d_start = high_resolution_clock::now();
    cudaMemcpy(d_a, A, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(M, mask, MASK_HEIGHT * MASK_WIDTH * sizeof(float));
    auto h2d_end = high_resolution_clock::now();
    auto h2d_duration = duration_cast<microseconds>(h2d_end - h2d_start);
    
    // Configure kernel execution
    dim3 blockDim(16, 16);  // 16x16 threads per block
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, 
                 (height + blockDim.y - 1) / blockDim.y);
    
    // Launch kernel with timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    conv2d_kernel<<<gridDim, blockDim>>>(d_a, d_c, width, height, pitch);
    cudaEventRecord(stop);
    
    // Check for kernel launch errors
    checkCudaError("Kernel launch failed");
    
    cudaEventSynchronize(stop);
    float kernel_ms = 0;
    cudaEventElapsedTime(&kernel_ms, start, stop);
    
    // Copy results back to host
    auto d2h_start = high_resolution_clock::now();
    cudaMemcpy(C_gpu, d_c, size * sizeof(float), cudaMemcpyDeviceToHost);
    auto d2h_end = high_resolution_clock::now();
    auto d2h_duration = duration_cast<microseconds>(d2h_end - d2h_start);
    
    // Verify results
    bool results_match = verifyResults(C_cpu, C_gpu, size);
    
    // Print matrices for small sizes
    if (width <= 10 && height <= 10) {
        printMatrix(A, width, height, pitch, "Input Matrix A");
        printf("Mask:\n");
        for (int i = 0; i < MASK_HEIGHT; i++) {
            for (int j = 0; j < MASK_WIDTH; j++) {
                printf("%.2f ", mask[i][j]);
            }
            printf("\n");
        }
        printf("\n");
        printMatrix(C_cpu, width, height, pitch, "CPU Result");
        printMatrix(C_gpu, width, height, pitch, "GPU Result");
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
    
    // Compute GFLOPS (approximate computation)
    // Each output element requires MASK_WIDTH * MASK_HEIGHT multiplications and additions
    double operations = (double)width * height * MASK_WIDTH * MASK_HEIGHT * 2; // mult and add
    double cpu_gflops = operations / (cpu_duration.count() * 1000); // Giga FLOPS
    double gpu_gflops = operations / (kernel_ms * 1000000); // Giga FLOPS
    
    printf("Approximate GFLOPS - CPU:  %.3f\n", cpu_gflops);
    printf("Approximate GFLOPS - GPU:  %.3f\n", gpu_gflops);
    
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
    // Test with various matrix sizes
    int test_sizes[][2] = {
        {10, 10},         // Tiny matrix
        {100, 100},       // Small matrix
        {512, 512},       // Medium matrix
        {1024, 1024},     // Large matrix
        {2048, 2048},     // Very large matrix
        {4096, 4096}      // Huge matrix (if memory allows)
    };
    
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    for (int i = 0; i < num_tests; i++) {
        runBenchmark(test_sizes[i][0], test_sizes[i][1]);
    }
    
    return 0;
}