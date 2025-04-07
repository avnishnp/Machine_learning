#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <cmath>
#include <curand.h>
#include <fstream>
#include <chrono>

using namespace std::chrono;

// GPU kernel for attention mechanism forward pass
__global__
void forward_kernel(const float* query_matrix_device_pointer, const float* key_matrix_device_pointer, const float* value_matrix_device_pointer, const int sequence_length, const int embedding_dimension,
                    const int total_columns_in_blocks, const int total_rows_in_blocks, const int block_size_columns, const int block_size_rows, const float softmax_scale,
                    float* sum_matrix_device_pointer, float *max_matrix_device_pointer, float* output_matrix_device_pointer) {
    int thread_index_x = threadIdx.x;
    int block_index_x = blockIdx.x; 
    int block_index_y = blockIdx.y;  // batch and head index

    // Offset into query_matrix_device_pointer,key_matrix_device_pointer,value_matrix_device_pointer,output_matrix_device_pointer,sum_matrix_device_pointer,max_matrix_device_pointer - different for each batch and head
    int qkv_offset = (block_index_x * gridDim.y * sequence_length * embedding_dimension) + (block_index_y * sequence_length * embedding_dimension);  // gridDim.y = num_heads
    int lm_offset = (block_index_x * gridDim.y * sequence_length) + (block_index_y * sequence_length);  // offset for sum_matrix_device_pointer and max_matrix_device_pointer

    // Define SRAM for Q,K,V,S
    extern __shared__ float shared_memory[];
    int tile_size = block_size_columns * embedding_dimension;  // size of query_matrix_tile, key_matrix_tile, value_matrix_tile
    float* query_matrix_tile = shared_memory;
    float* key_matrix_tile = &shared_memory[tile_size];
    float* value_matrix_tile = &shared_memory[tile_size * 2];
    float* score_matrix_tile = &shared_memory[tile_size * 3];
    float eps=1e-10;
    for (int column_block_index = 0; column_block_index < total_columns_in_blocks; column_block_index++) {

        // Load key_matrix_tile, value_matrix_tile to SRAM
        for (int embedding_index = 0; embedding_index < embedding_dimension; embedding_index++) {
            key_matrix_tile[(thread_index_x * embedding_dimension) + embedding_index] = key_matrix_device_pointer[qkv_offset + (tile_size * column_block_index) + (thread_index_x * embedding_dimension) + embedding_index];
            value_matrix_tile[(thread_index_x * embedding_dimension) + embedding_index] = value_matrix_device_pointer[qkv_offset + (tile_size * column_block_index) + (thread_index_x * embedding_dimension) + embedding_index];
        }
        __syncthreads();  

        for (int row_block_index = 0; row_block_index < total_rows_in_blocks; row_block_index++)  {
            
            for (int embedding_index = 0; embedding_index < embedding_dimension; embedding_index++) {
                query_matrix_tile[(thread_index_x * embedding_dimension) + embedding_index] = query_matrix_device_pointer[qkv_offset + (tile_size * row_block_index) + (thread_index_x * embedding_dimension) + embedding_index];
            }
            float row_max_previous = max_matrix_device_pointer[lm_offset + (block_size_rows * row_block_index) + thread_index_x];
            float row_sum_previous = sum_matrix_device_pointer[lm_offset + (block_size_rows * row_block_index) + thread_index_x];

            
            float row_max = -INFINITY;
            for (int column_index_inner = 0; column_index_inner < block_size_columns; column_index_inner++) {
                float sum = 0;
                for (int embedding_index = 0; embedding_index < embedding_dimension; embedding_index++) {
                    sum += query_matrix_tile[(thread_index_x * embedding_dimension) + embedding_index] * key_matrix_tile[(column_index_inner * embedding_dimension) + embedding_index];
                }
                sum *= softmax_scale;
                score_matrix_tile[(block_size_columns * thread_index_x) + column_index_inner] = sum;

                if (sum > row_max)
                    row_max = sum;
            }

            // probability_matrix_tile = exp(score_matrix_tile - row_max), row_sum = rowsum(probability_matrix_tile)
            float row_sum = 0;
            for (int column_index_inner = 0; column_index_inner < block_size_columns; column_index_inner++) {
                score_matrix_tile[(block_size_columns * thread_index_x) + column_index_inner] = __expf(score_matrix_tile[(block_size_columns * thread_index_x) + column_index_inner] - row_max);
                row_sum += score_matrix_tile[(block_size_columns * thread_index_x) + column_index_inner];
            }

            float row_max_new = max(row_max_previous, row_max);
            float row_sum_new = (__expf(row_max_previous - row_max_new) * row_sum_previous) + (__expf(row_max - row_max_new) * row_sum);


            // Write output_matrix_device_pointer, sum_matrix_device_pointer, max_matrix_device_pointer to HBM
            for (int embedding_index = 0; embedding_index < embedding_dimension; embedding_index++) {
                float probability_times_value = 0;  // Pij * Vj
                for (int column_index_inner = 0; column_index_inner < block_size_columns; column_index_inner++) {
                    probability_times_value += score_matrix_tile[(block_size_columns * thread_index_x) + column_index_inner] * value_matrix_tile[(column_index_inner * embedding_dimension) + embedding_index]+eps;
                }
                output_matrix_device_pointer[qkv_offset + (tile_size * row_block_index) + (thread_index_x * embedding_dimension) + embedding_index] = (1 / (eps+row_sum_new)) \
                    * ((row_sum_previous * __expf(row_max_previous - row_max_new) * output_matrix_device_pointer[qkv_offset + (tile_size * row_block_index) + (thread_index_x * embedding_dimension) + embedding_index]) \
                    + (__expf(row_max - row_max_new+eps) * probability_times_value));
            }
            max_matrix_device_pointer[lm_offset + (block_size_rows * row_block_index) + thread_index_x] = row_max_new;
            sum_matrix_device_pointer[lm_offset + (block_size_rows * row_block_index) + thread_index_x] = row_sum_new;
        }
        __syncthreads();  // otherwise, thread can use the wrong key_matrix_tile, value_matrix_tile in inner loop
    }
}

// CPU implementation of attention mechanism
void attention_cpu(const float* query, const float* key, const float* value, 
                float* output, float* sum_matrix, float* max_matrix,
                int batch_size, int num_heads, int sequence_length, int embedding_dimension) {
    
    float softmax_scale = 1.0f / sqrtf(embedding_dimension);
    float eps = 1e-10;
    
    // For each batch and head
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            // For each query position
            for (int i = 0; i < sequence_length; i++) {
                float row_max = -INFINITY;
                float row_sum = 0.0f;
                
                // Calculate scores and find max
                float* scores = new float[sequence_length];
                for (int j = 0; j < sequence_length; j++) {
                    float score = 0.0f;
                    // Dot product between query and key
                    for (int d = 0; d < embedding_dimension; d++) {
                        int q_idx = ((b * num_heads * sequence_length) + (h * sequence_length) + i) * embedding_dimension + d;
                        int k_idx = ((b * num_heads * sequence_length) + (h * sequence_length) + j) * embedding_dimension + d;
                        score += query[q_idx] * key[k_idx];
                    }
                    score *= softmax_scale;
                    scores[j] = score;
                    
                    if (score > row_max) row_max = score;
                }
                
                // Calculate softmax (exp(score - max) / sum(exp(score - max)))
                for (int j = 0; j < sequence_length; j++) {
                    scores[j] = expf(scores[j] - row_max);
                    row_sum += scores[j];
                }
                
                // Store max and sum
                int max_idx = (b * num_heads * sequence_length) + (h * sequence_length) + i;
                max_matrix[max_idx] = row_max;
                sum_matrix[max_idx] = row_sum;
                
                // For each dimension of the output
                for (int d = 0; d < embedding_dimension; d++) {
                    float weighted_sum = 0.0f;
                    
                    // Apply attention weights to values
                    for (int j = 0; j < sequence_length; j++) {
                        int v_idx = ((b * num_heads * sequence_length) + (h * sequence_length) + j) * embedding_dimension + d;
                        weighted_sum += (scores[j] / (row_sum + eps)) * value[v_idx];
                    }
                    
                    // Store in output
                    int out_idx = ((b * num_heads * sequence_length) + (h * sequence_length) + i) * embedding_dimension + d;
                    output[out_idx] = weighted_sum;
                }
                
                delete[] scores;
            }
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

template <typename T>
T* allocateAndInitializeDeviceMemory(size_t size, bool initializeToZero = false, bool initializeToNegativeInfinity = false) {
    T* device_ptr;
    cudaMalloc(&device_ptr, size);
    checkCudaError("Memory allocation failed");

    if (initializeToZero) {
        cudaMemset(device_ptr, 0, size);
        checkCudaError("Memory initialization to zero failed");
    } else if (initializeToNegativeInfinity) {
        // First allocate and initialize host memory
        T* host_ptr = new T[size / sizeof(T)];
        for (size_t i = 0; i < size / sizeof(T); i++) {
            host_ptr[i] = -INFINITY;
        }
        // Then copy to device
        cudaMemcpy(device_ptr, host_ptr, size, cudaMemcpyHostToDevice);
        checkCudaError("Memory copy to device failed");
        delete[] host_ptr;
    } else {
        curandGenerator_t generator;
        curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
        checkCudaError("CURAND generator creation failed");
        
        curandSetGeneratorOffset(generator, time(0));
        checkCudaError("CURAND generator offset setting failed");
        
        curandGenerateUniform(generator, reinterpret_cast<float*>(device_ptr), size / sizeof(T));
        checkCudaError("CURAND uniform generation failed");
        
        curandDestroyGenerator(generator);
        checkCudaError("CURAND generator destruction failed");
    }

    return device_ptr;
}

// Verify results between CPU and GPU implementations
bool verifyResults(const float* A, const float* B, int size, float tolerance = 1e-4) {
    int mismatches = 0;
    for (int i = 0; i < size; i++) {
        if (std::abs(A[i] - B[i]) > tolerance) {
            if (mismatches < 10) {  // Only print first 10 mismatches to avoid flooding console
                printf("Mismatch at position %d: CPU=%f, GPU=%f, diff=%f\n", 
                       i, A[i], B[i], std::abs(A[i] - B[i]));
            }
            mismatches++;
        }
    }
    
    if (mismatches > 0) {
        printf("Total %d mismatches out of %d elements\n", mismatches, size);
        return false;
    }
    return true;
}

// Run benchmark for specific parameters
void runBenchmark(int batch_size, int num_heads, int sequence_length, int embedding_dimension) {
    printf("\n===== Parameters: Batch=%d, Heads=%d, Seq_Length=%d, Embedding_Dim=%d =====\n", 
           batch_size, num_heads, sequence_length, embedding_dimension);
    
    // Block size parameters
    const int block_size_columns = min(32, sequence_length);  // Adjust for smaller sequences
    const int block_size_rows = min(32, sequence_length);     // Adjust for smaller sequences

    // Derived dimensions
    const int total_columns_in_blocks = ceil((float)sequence_length / block_size_columns);
    const int total_rows_in_blocks = ceil((float)sequence_length / block_size_rows);
    const float softmax_scale = 1.0f / sqrtf(embedding_dimension);

    // Calculate sizes for memory allocation
    size_t matrix_size = batch_size * num_heads * sequence_length * embedding_dimension * sizeof(float);
    size_t vector_size = batch_size * num_heads * sequence_length * sizeof(float);
    
    // Allocate host memory for CPU computation
    float* query_host = new float[batch_size * num_heads * sequence_length * embedding_dimension];
    float* key_host = new float[batch_size * num_heads * sequence_length * embedding_dimension];
    float* value_host = new float[batch_size * num_heads * sequence_length * embedding_dimension];
    float* output_cpu = new float[batch_size * num_heads * sequence_length * embedding_dimension];
    float* sum_matrix_cpu = new float[batch_size * num_heads * sequence_length];
    float* max_matrix_cpu = new float[batch_size * num_heads * sequence_length];
    float* output_gpu_host = new float[batch_size * num_heads * sequence_length * embedding_dimension];

    // Initialize host arrays with random values for consistency
    std::mt19937 rng(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (int i = 0; i < batch_size * num_heads * sequence_length * embedding_dimension; i++) {
        query_host[i] = dist(rng);
        key_host[i] = dist(rng);
        value_host[i] = dist(rng);
    }
    
    // Initialize CPU arrays
    for (int i = 0; i < batch_size * num_heads * sequence_length; i++) {
        sum_matrix_cpu[i] = 0.0f;
        max_matrix_cpu[i] = -INFINITY;
    }
    
    for (int i = 0; i < batch_size * num_heads * sequence_length * embedding_dimension; i++) {
        output_cpu[i] = 0.0f;
    }
    
    // Device memory allocation
    float* query_device = allocateAndInitializeDeviceMemory<float>(matrix_size);
    float* key_device = allocateAndInitializeDeviceMemory<float>(matrix_size);
    float* value_device = allocateAndInitializeDeviceMemory<float>(matrix_size);
    float* output_device = allocateAndInitializeDeviceMemory<float>(matrix_size, true);
    float* sum_matrix_device = allocateAndInitializeDeviceMemory<float>(vector_size, true);
    float* max_matrix_device = allocateAndInitializeDeviceMemory<float>(vector_size, false, true);
    
    // Copy input data to device
    cudaMemcpy(query_device, query_host, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(key_device, key_host, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(value_device, value_host, matrix_size, cudaMemcpyHostToDevice);
    
    // Shared memory size calculation and check
    const int shared_memory_size = (3 * block_size_columns * embedding_dimension + 
                                    block_size_rows * block_size_columns) * sizeof(float);
    
    int max_shared_memory_size;
    cudaDeviceGetAttribute(&max_shared_memory_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    
    if (shared_memory_size > max_shared_memory_size) {
        printf("WARNING: Required shared memory (%d bytes) exceeds device limit (%d bytes).\n",
               shared_memory_size, max_shared_memory_size);
        printf("Consider reducing embedding_dimension or block sizes.\n");
        return;
    }
    
    // CPU implementation timing
    auto cpu_start = high_resolution_clock::now();
    attention_cpu(query_host, key_host, value_host, output_cpu, sum_matrix_cpu, max_matrix_cpu,
                batch_size, num_heads, sequence_length, embedding_dimension);
    auto cpu_end = high_resolution_clock::now();
    auto cpu_duration = duration_cast<microseconds>(cpu_end - cpu_start);
    
    // GPU kernel launch configuration
    dim3 grid_dim(batch_size, num_heads);
    dim3 block_dim(block_size_columns);
    
    // GPU kernel timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    forward_kernel<<<grid_dim, block_dim, shared_memory_size>>>(
        query_device, key_device, value_device, sequence_length,
        embedding_dimension, total_columns_in_blocks, total_rows_in_blocks, block_size_columns,
        block_size_rows, softmax_scale, sum_matrix_device, max_matrix_device, output_device);
    cudaEventRecord(stop);
    
    // Check for kernel launch errors
    checkCudaError("Kernel launch failed");
    
    cudaEventSynchronize(stop);
    float kernel_ms = 0;
    cudaEventElapsedTime(&kernel_ms, start, stop);
    
    // Copy results back to host
    cudaMemcpy(output_gpu_host, output_device, matrix_size, cudaMemcpyDeviceToHost);
    
    // Verify results
    bool results_match = verifyResults(output_cpu, output_gpu_host, 
                                      batch_size * num_heads * sequence_length * embedding_dimension);
    
    // Report performance
    printf("Performance Results:\n");
    printf("CPU time:                  %ld microseconds\n", cpu_duration.count());
    printf("GPU Kernel Time:           %.3f microseconds\n", kernel_ms * 1000);
    
    // Compute approximate FLOPS
    // Each element requires: 
    // - sequence_length * embedding_dimension multiplications for dot product
    // - sequence_length exponentiations
    // - sequence_length multiplications for value weighting
    double operations = (double)batch_size * num_heads * sequence_length * 
                       (sequence_length * embedding_dimension * 2 +  // Dot products
                        sequence_length * 2 +                       // Exp and sum
                        sequence_length * embedding_dimension);     // Weighted sum
    
    double cpu_gflops = operations / (cpu_duration.count() * 1000); // Giga FLOPS
    double gpu_gflops = operations / (kernel_ms * 1000000); // Giga FLOPS
    
    printf("Approximate GFLOPS - CPU:  %.3f\n", cpu_gflops);
    printf("Approximate GFLOPS - GPU:  %.3f\n", gpu_gflops);
    
    // Compute speedup
    float speedup = cpu_duration.count() / (kernel_ms * 1000);
    
    printf("Speedup (GPU vs CPU):      %.2fx\n", speedup);
    printf("Results match:             %s\n", results_match ? "Yes" : "No");
    
    // Clean up
    delete[] query_host;
    delete[] key_host;
    delete[] value_host;
    delete[] output_cpu;
    delete[] sum_matrix_cpu;
    delete[] max_matrix_cpu;
    delete[] output_gpu_host;
    
    cudaFree(query_device);
    cudaFree(key_device);
    cudaFree(value_device);
    cudaFree(output_device);
    cudaFree(sum_matrix_device);
    cudaFree(max_matrix_device);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    // Test with various configurations
    struct BenchmarkConfig {
        int batch_size;
        int num_heads;
        int sequence_length;
        int embedding_dimension;
    };
    
    BenchmarkConfig configs[] = {
        {1, 1, 32, 32},     // Small model (BERT-mini)
        {1, 1, 64, 64},     // Original test case
        {1, 4, 128, 64},    // Medium-sized model
        {1, 8, 256, 64},    // Medium-large model
        {1, 12, 512, 64},   // Large model
        {2, 16, 512, 128}   // Very large model
    };
    
    int num_configs = sizeof(configs) / sizeof(configs[0]);
    
    for (int i = 0; i < num_configs; i++) {
        runBenchmark(
            configs[i].batch_size,
            configs[i].num_heads,
            configs[i].sequence_length,
            configs[i].embedding_dimension
        );
    }
    
    return 0;
}