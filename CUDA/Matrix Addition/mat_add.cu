#include <iostream>
#include <chrono>
#include <cmath>

using namespace std::chrono;

// GPU kernel - Row-based approach (one thread per row)
__global__ void MatrixAdd_C(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        for(int j = 0; j < N; j++) {
            C[i*N+j] = A[i*N+j] + B[i*N+j];
        }
    }
}

// GPU kernel - Element-based approach (one thread per element)
__global__ void MatrixAdd_B(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N) { // Fixed condition: should be < N, not >= N 
        C[i*N+j] = A[i*N+j] + B[i*N+j];
    }
}

// GPU kernel - Column-based approach (one thread per column)
__global__ void MatrixAdd_D(const float* A, const float* B, float* C, int N) {
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j < N) {
        for(int i = 0; i < N; i++) {
            C[i*N+j] = A[i*N+j] + B[i*N+j];
        }
    }
}

// CPU implementation for comparison
void MatrixAdd_CPU(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i*N+j] = A[i*N+j] + B[i*N+j];
        }
    }
}

// Verify results
bool VerifyResults(const float* C_ref, const float* C, int N) {
    for (int i = 0; i < N*N; i++) {
        if (fabs(C_ref[i] - C[i]) > 1e-5) {
            return false;
        }
    }
    return true;
}

// Utility to print a matrix (for small matrices)
void PrintMatrix(const float* M, int N, const char* name) {
    printf("%s:\n", name);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", M[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    // Test with various matrix sizes
    int sizes[] = {10, 100, 1000, 2000};
    
    for (int s = 0; s < 4; s++) {
        int N = sizes[s];
        printf("\n===== Matrix Size: %d x %d =====\n", N, N);
        
        // Allocate host memory
        float *A = (float *)malloc(N*N*sizeof(float));
        float *B = (float *)malloc(N*N*sizeof(float));
        float *C_cpu = (float *)malloc(N*N*sizeof(float));
        float *C_c = (float *)malloc(N*N*sizeof(float));
        float *C_b = (float *)malloc(N*N*sizeof(float));
        float *C_d = (float *)malloc(N*N*sizeof(float));
        
        // Initialize input matrices
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i * N + j] = 1.0f;
                B[i * N + j] = 2.0f;
            }
        }
        
        // CPU implementation timing
        auto cpu_start = high_resolution_clock::now();
        MatrixAdd_CPU(A, B, C_cpu, N);
        auto cpu_end = high_resolution_clock::now();
        auto cpu_duration = duration_cast<microseconds>(cpu_end - cpu_start);
        
        // Allocate device memory
        float *d_a, *d_b, *d_c;
        cudaMalloc(&d_a, N*N*sizeof(float));
        cudaMalloc(&d_b, N*N*sizeof(float));
        cudaMalloc(&d_c, N*N*sizeof(float));
        
        // Copy data to device
        auto h2d_start = high_resolution_clock::now();
        cudaMemcpy(d_a, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, B, N*N*sizeof(float), cudaMemcpyHostToDevice);
        auto h2d_end = high_resolution_clock::now();
        auto h2d_duration = duration_cast<microseconds>(h2d_end - h2d_start);
        
        // Kernel C timing (row-based)
        cudaEvent_t start_c, stop_c;
        cudaEventCreate(&start_c);
        cudaEventCreate(&stop_c);
        
        dim3 blockC(256);
        dim3 gridC((N + blockC.x - 1) / blockC.x);
        
        cudaEventRecord(start_c);
        MatrixAdd_C<<<gridC, blockC>>>(d_a, d_b, d_c, N);
        cudaEventRecord(stop_c);
        cudaEventSynchronize(stop_c);
        
        float kernel_c_ms = 0;
        cudaEventElapsedTime(&kernel_c_ms, start_c, stop_c);
        
        cudaMemcpy(C_c, d_c, N*N*sizeof(float), cudaMemcpyDeviceToHost);
        bool c_correct = VerifyResults(C_cpu, C_c, N);
        
        // Kernel B timing (element-based)
        cudaEvent_t start_b, stop_b;
        cudaEventCreate(&start_b);
        cudaEventCreate(&stop_b);
        
        dim3 blockB(32, 16);
        dim3 gridB((N + blockB.x - 1) / blockB.x, (N + blockB.y - 1) / blockB.y);
        
        cudaEventRecord(start_b);
        MatrixAdd_B<<<gridB, blockB>>>(d_a, d_b, d_c, N);
        cudaEventRecord(stop_b);
        cudaEventSynchronize(stop_b);
        
        float kernel_b_ms = 0;
        cudaEventElapsedTime(&kernel_b_ms, start_b, stop_b);
        
        cudaMemcpy(C_b, d_c, N*N*sizeof(float), cudaMemcpyDeviceToHost);
        bool b_correct = VerifyResults(C_cpu, C_b, N);
        
        // Kernel D timing (column-based)
        cudaEvent_t start_d, stop_d;
        cudaEventCreate(&start_d);
        cudaEventCreate(&stop_d);
        
        dim3 blockD(1, 256);
        dim3 gridD(1, (N + blockD.y - 1) / blockD.y);
        
        cudaEventRecord(start_d);
        MatrixAdd_D<<<gridD, blockD>>>(d_a, d_b, d_c, N);
        cudaEventRecord(stop_d);
        cudaEventSynchronize(stop_d);
        
        float kernel_d_ms = 0;
        cudaEventElapsedTime(&kernel_d_ms, start_d, stop_d);
        
        auto d2h_start = high_resolution_clock::now();
        cudaMemcpy(C_d, d_c, N*N*sizeof(float), cudaMemcpyDeviceToHost);
        auto d2h_end = high_resolution_clock::now();
        auto d2h_duration = duration_cast<microseconds>(d2h_end - d2h_start);
        
        bool d_correct = VerifyResults(C_cpu, C_d, N);
        
        // Print small matrices (only for N=10)
        if (N == 10) {
            PrintMatrix(A, N, "Matrix A");
            PrintMatrix(B, N, "Matrix B");
            PrintMatrix(C_cpu, N, "Result (CPU)");
        }
        
        // Print performance results
        printf("Performance Results:\n");
        printf("CPU time: %ld microseconds\n", cpu_duration.count());
        printf("GPU Memory Transfer: Host->Device: %ld microseconds, Device->Host: %ld microseconds\n", 
               h2d_duration.count(), d2h_duration.count());
        printf("MatrixAdd_C (Row-based)    - Kernel time: %.3f microseconds, Correctness: %s\n", 
               kernel_c_ms * 1000, c_correct ? "PASS" : "FAIL");
        printf("MatrixAdd_B (Element-based) - Kernel time: %.3f microseconds, Correctness: %s\n", 
               kernel_b_ms * 1000, b_correct ? "PASS" : "FAIL");
        printf("MatrixAdd_D (Column-based)  - Kernel time: %.3f microseconds, Correctness: %s\n", 
               kernel_d_ms * 1000, d_correct ? "PASS" : "FAIL");
        
        // Speedup calculations (kernel-only)
        printf("\nSpeedup (kernel-only, compared to CPU):\n");
        printf("MatrixAdd_C (Row-based):     %.2fx\n", cpu_duration.count() / (kernel_c_ms * 1000));
        printf("MatrixAdd_B (Element-based): %.2fx\n", cpu_duration.count() / (kernel_b_ms * 1000));
        printf("MatrixAdd_D (Column-based):  %.2fx\n", cpu_duration.count() / (kernel_d_ms * 1000));
        
        // Free memory
        free(A);
        free(B);
        free(C_cpu);
        free(C_c);
        free(C_b);
        free(C_d);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        
        // Destroy CUDA events
        cudaEventDestroy(start_c);
        cudaEventDestroy(stop_c);
        cudaEventDestroy(start_b);
        cudaEventDestroy(stop_b);
        cudaEventDestroy(start_d);
        cudaEventDestroy(stop_d);
    }
    
    return 0;
}