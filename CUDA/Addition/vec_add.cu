#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

// GPU kernel for vector addition
__global__ void vectorAdd(const int* A, const int* B, int* C, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        C[tid] = A[tid] + B[tid];
    }
}

// CPU function for vector addition
void vectorAddCPU(const int* A, const int* B, int* C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

void runTest(int N) {
    cout << "Vector size: " << N << endl;
    
    // Allocate and initialize host memory
    vector<int> A(N), B(N), C_cpu(N), C_gpu(N);
    for (int i = 0; i < N; i++) {
        A[i] = i + 1;
        B[i] = i + 2;
    }
    
    // ======== CPU TIMING (COMPUTATION ONLY) =========
    auto cpu_start = high_resolution_clock::now();
    
    vectorAddCPU(A.data(), B.data(), C_cpu.data(), N);
    
    auto cpu_end = high_resolution_clock::now();
    auto cpu_duration = duration_cast<microseconds>(cpu_end - cpu_start);
    
    // ======== GPU TIMING (SEPARATE MEASUREMENTS) =========
    
    // Allocate device memory
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(int));
    cudaMalloc(&d_B, N * sizeof(int));
    cudaMalloc(&d_C, N * sizeof(int));
    
    // Measure memory transfer: Host to Device
    auto h2d_start = high_resolution_clock::now();
    
    cudaMemcpy(d_A, A.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    
    auto h2d_end = high_resolution_clock::now();
    auto h2d_duration = duration_cast<microseconds>(h2d_end - h2d_start);
    
    // Measure kernel execution only (using CUDA events for accuracy)
    cudaEvent_t kernel_start, kernel_stop;
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    cudaEventRecord(kernel_start);
    
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    cudaEventRecord(kernel_stop);
    cudaEventSynchronize(kernel_stop);
    
    float kernel_duration_ms = 0;
    cudaEventElapsedTime(&kernel_duration_ms, kernel_start, kernel_stop);
    
    // Measure memory transfer: Device to Host
    auto d2h_start = high_resolution_clock::now();
    
    cudaMemcpy(C_gpu.data(), d_C, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    auto d2h_end = high_resolution_clock::now();
    auto d2h_duration = duration_cast<microseconds>(d2h_end - d2h_start);
    
    // Calculate total GPU time
    auto total_gpu_duration_us = h2d_duration.count() + 
                               (kernel_duration_ms * 1000) + 
                               d2h_duration.count();
    
    // Verify results
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (C_cpu[i] != C_gpu[i]) {
            correct = false;
            cout << "Results differ at index " << i << ": CPU=" << C_cpu[i] 
                 << ", GPU=" << C_gpu[i] << endl;
            break;
        }
    }
    
    if (correct) {
        cout << "Results match! Both implementations produced identical outputs." << endl;
    }
    
    // Print timing results
    cout << "CPU computation time: " << cpu_duration.count() << " microseconds" << endl;
    cout << "GPU timings (microseconds):" << endl;
    cout << "  Host to Device transfer: " << h2d_duration.count() << endl;
    cout << "  Kernel execution:        " << (kernel_duration_ms * 1000) << endl;
    cout << "  Device to Host transfer: " << d2h_duration.count() << endl;
    cout << "  Total GPU time:          " << total_gpu_duration_us << endl;
    
    // Calculate speedup
    double speedup = (double)cpu_duration.count() / (kernel_duration_ms * 1000);
    double speedup_with_transfers = (double)cpu_duration.count() / total_gpu_duration_us;
    
    cout << "Speedup (kernel only): " << speedup << "x" << endl;
    cout << "Speedup (with transfers): " << speedup_with_transfers << "x" << endl;
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // Destroy CUDA events
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    
    cout << endl;
}

int main() {
    // Test with different vector sizes
    vector<int> sizes = {100, 1000, 100000, 1000000, 10000000};
    
    for (int size : sizes) {
        runTest(size);
    }
    
    return 0;
}