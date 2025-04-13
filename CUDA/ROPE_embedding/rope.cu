#include <iostream>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

using namespace std::chrono;

/*
CUDA runs one thread per pair for each token in each sequence of the batch.
RoPE does a rotation per 2D pair of embedding dimensions.
batch_size = number of sequences in a batch

seq_len = number of tokens per sequence

dim / 2 = number of 2D pairs in each token embedding

So total_threads = batch_size × seq_len × (dim / 2)
*/

// ROPE (Rotary Position Embedding) implementation
// GPU kernel for applying ROPE embeddings
__global__ void applyRopeEmbeddings(float* vectors, int batch_size, int seq_len, int dim, float base = 10000.0f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / (seq_len * dim / 2);
    int remainder = idx % (seq_len * dim / 2);
    int pos = remainder / (dim / 2);
    int dim_idx = remainder % (dim / 2);
    
    if (batch_idx >= batch_size || pos >= seq_len || dim_idx >= dim / 2)
        return;
    
    // Calculate sin and cos values for the rotational embedding
    float freq = 1.0f / powf(base, (2.0f * dim_idx) / dim);
    float sin_val = sinf(pos * freq);
    float cos_val = cosf(pos * freq);
    
    // Calculate input indices
    int input_idx1 = batch_idx * seq_len * dim + pos * dim + dim_idx * 2;
    int input_idx2 = input_idx1 + 1;
    
    // Apply rotation
    float x1 = vectors[input_idx1];
    float x2 = vectors[input_idx2];
    
    vectors[input_idx1] = x1 * cos_val - x2 * sin_val;
    vectors[input_idx2] = x1 * sin_val + x2 * cos_val;
}

// CPU implementation for comparison
void applyRopeEmbeddingsCPU(float* vectors, int batch_size, int seq_len, int dim, float base = 10000.0f) {
    for (int b = 0; b < batch_size; b++) {
        for (int p = 0; p < seq_len; p++) {
            for (int d = 0; d < dim / 2; d++) {
                float freq = 1.0f / powf(base, (2.0f * d) / dim);
                float sin_val = sinf(p * freq);
                float cos_val = cosf(p * freq);
                
                int idx1 = b * seq_len * dim + p * dim + d * 2;
                int idx2 = idx1 + 1;
                
                float x1 = vectors[idx1];
                float x2 = vectors[idx2];
                
                vectors[idx1] = x1 * cos_val - x2 * sin_val;
                vectors[idx2] = x1 * sin_val + x2 * cos_val;
            }
        }
    }
}

// Verify results
bool verifyResults(const float* result1, const float* result2, int total_size) {
    for (int i = 0; i < total_size; i++) {
        if (fabs(result1[i] - result2[i]) > 1e-4) {
            printf("Mismatch at position %d: CPU=%f, GPU=%f\n", i, result1[i], result2[i]);
            return false;
        }
    }
    return true;
}

// Print sample of embedding vectors (for small arrays)
void printEmbeddingSample(const float* vectors, int batch_size, int seq_len, int dim, const char* name) {
    printf("%s (sample):\n", name);
    // Print first 3 positions of the first batch
    for (int p = 0; p < std::min(3, seq_len); p++) {
        printf("Position %d: ", p);
        for (int d = 0; d < std::min(8, dim); d++) {
            printf("%.4f ", vectors[0 * seq_len * dim + p * dim + d]);
        }
        printf("...\n");
    }
    printf("\n");
}

int main() {
    // Test with various sizes
    // Format: {batch_size, sequence_length, embedding_dimension}
    int configs[][3] = {
        {1, 128, 64},     // Small config
        {4, 512, 128},    // Medium config
        {16, 1024, 256},  // Large config
        {32, 2048, 512},  // XLarge config
        {64, 4096, 1024}  // XXLarge config
    };
    
    for (int c = 0; c < 5; c++) {
        int batch_size = configs[c][0];
        int seq_len = configs[c][1];
        int dim = configs[c][2];
        
        // Ensure dim is even (ROPE works on pairs of values)
        if (dim % 2 != 0) {
            printf("Dimension must be even for ROPE. Adjusting %d to %d\n", dim, dim+1);
            dim += 1;
        }
        
        int total_size = batch_size * seq_len * dim;
        
        printf("\n===== Config: Batch=%d, Sequence Length=%d, Embedding Dim=%d =====\n", 
               batch_size, seq_len, dim);
        
        // Allocate host memory
        float *h_vectors_cpu = (float *)malloc(total_size * sizeof(float));
        float *h_vectors_gpu = (float *)malloc(total_size * sizeof(float));
        
        // Initialize input data with random values
        for (int i = 0; i < total_size; i++) {
            h_vectors_cpu[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;  // Values between -1 and 1
            h_vectors_gpu[i] = h_vectors_cpu[i];  // Copy same values for GPU computation
        }
        
        // CPU implementation timing
        auto cpu_start = high_resolution_clock::now();
        applyRopeEmbeddingsCPU(h_vectors_cpu, batch_size, seq_len, dim);
        auto cpu_end = high_resolution_clock::now();
        auto cpu_duration = duration_cast<microseconds>(cpu_end - cpu_start);
        
        // Allocate device memory
        float *d_vectors;
        cudaMalloc(&d_vectors, total_size * sizeof(float));
        
        // Copy data from host to device
        auto h2d_start = high_resolution_clock::now();
        cudaMemcpy(d_vectors, h_vectors_gpu, total_size * sizeof(float), cudaMemcpyHostToDevice);
        auto h2d_end = high_resolution_clock::now();
        auto h2d_duration = duration_cast<microseconds>(h2d_end - h2d_start);
        
        // GPU kernel timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // Calculate grid and block dimensions
        int threads_per_block = 256;
        int total_threads = batch_size * seq_len * dim / 2;  // Each thread handles a pair of values
        int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
        
        cudaEventRecord(start);
        applyRopeEmbeddings<<<blocks, threads_per_block>>>(d_vectors, batch_size, seq_len, dim);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float kernel_ms = 0;
        cudaEventElapsedTime(&kernel_ms, start, stop);
        
        // Copy results back
        auto d2h_start = high_resolution_clock::now();
        cudaMemcpy(h_vectors_gpu, d_vectors, total_size * sizeof(float), cudaMemcpyDeviceToHost);
        auto d2h_end = high_resolution_clock::now();
        auto d2h_duration = duration_cast<microseconds>(d2h_end - d2h_start);
        
        // Verify results
        bool correct = verifyResults(h_vectors_cpu, h_vectors_gpu, total_size);
        
        // Print sample of embeddings (only for small config)
        if (c == 0) {
            printEmbeddingSample(h_vectors_cpu, batch_size, seq_len, dim, "ROPE Embeddings (CPU)");
            printEmbeddingSample(h_vectors_gpu, batch_size, seq_len, dim, "ROPE Embeddings (GPU)");
        }
        
        // Calculate total GPU time including transfers
        auto total_gpu_time = h2d_duration.count() + (kernel_ms * 1000) + d2h_duration.count();
        
        // Calculate throughput
        double cpu_throughput = (double)(batch_size * seq_len * dim) / cpu_duration.count();  // Elements per microsecond
        double gpu_throughput = (double)(batch_size * seq_len * dim) / (kernel_ms * 1000);    // Elements per microsecond
        
        // Print performance results
        printf("Performance Results:\n");
        printf("CPU time:                  %ld microseconds (%.2f M elements/sec)\n", 
               cpu_duration.count(), cpu_throughput);
        printf("GPU Memory Transfer Time:  %ld microseconds (H2D: %ld, D2H: %ld)\n", 
               h2d_duration.count() + d2h_duration.count(),
               h2d_duration.count(), d2h_duration.count());
        printf("GPU Kernel Time:           %.3f microseconds (%.2f M elements/sec)\n", 
               kernel_ms * 1000, gpu_throughput);
        printf("GPU Total Time:            %.3f microseconds\n", total_gpu_time);
        
        // Compute speedup
        float kernel_speedup = cpu_duration.count() / (kernel_ms * 1000);
        float total_speedup = cpu_duration.count() / (float)total_gpu_time;
        
        printf("Speedup (kernel-only):     %.2fx\n", kernel_speedup);
        printf("Speedup (with transfers):  %.2fx\n", total_speedup);
        printf("Results match: %s\n", correct ? "Yes" : "No");
        
        // Free memory
        free(h_vectors_cpu);
        free(h_vectors_gpu);
        cudaFree(d_vectors);
        
        // Destroy CUDA events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    return 0;
}