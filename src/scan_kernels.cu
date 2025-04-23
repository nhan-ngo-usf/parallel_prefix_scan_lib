#include "scan_kernels.cuh"
#include "utils.cuh"
// Histogram kernel
__global__ void generate_histogram(int* input_array, int input_size, int* histogram, int numBits, int buckets)
{
    extern __shared__ int shared_histogram[]; // Shared memory for histogram

    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    // Initialize shared histogram to 0
    for (int i = tid; i < buckets; i += blockDim.x) {
        shared_histogram[i] = 0;
    }
    __syncthreads(); // Synchronize to make sure shared memory is initialized

    // Each thread computes the histogram index and updates the shared histogram
    for (int i = idx; i < input_size; i += stride) {
        uint h = bfe(input_array[i], 0, numBits);
        atomicAdd(&shared_histogram[h], 1);
    }

    __syncthreads(); // Synchronize to make sure all updates are complete

    // Each block contributes its histogram to the global histogram
    for (int i = tid; i < buckets; i += blockDim.x) {
        atomicAdd(&histogram[i], shared_histogram[i]);
    }
}

//define the prefix scan kernel here
__global__ void prefixScan(int* histogram, int* prefix_sum, int size)
{
    extern __shared__ int temp[];  // Shared memory for the scan
    int tid = threadIdx.x;
    int idx = tid * 2;
    // Load histogram into shared memory
    if (idx < size){
        temp[idx] = histogram[idx];
    }
    if (idx + 1 < size) {
        temp[idx + 1] = histogram[idx + 1];
    }
    __syncthreads();

    for (int offset = 1; offset < size; offset *= 2) {
        int i = (tid + 1) * offset * 2 - 1; 
        if (i < size) {
            temp[i] += temp[i - offset];
        }
        __syncthreads();  // Ensure all threads complete this step
    }

    // Clear the root node for exclusive scan
    if (tid == 0) {
        temp[size - 1] = 0;
    }
    __syncthreads();

    for (int offset = size / 2; offset > 0; offset /= 2) {
        int i = (tid + 1) * offset * 2 - 1;  
        if (i < size) {
            int t = temp[i - offset];
            temp[i - offset] = temp[i];
            temp[i] += t;
        }
        __syncthreads();  // Ensure all threads complete this step
    }

    
    // Write results back to global memory
    if (idx < size){
        prefix_sum[idx] = temp[idx];
    }
    if (idx + 1 < size) {
        prefix_sum[idx + 1] = temp[idx + 1];
    }
}


//define the reorder kernel here
__global__ void Reorder(int* input_array, int* prefix_sum, int* output_array, int input_size, int numBits, int buckets)
{
    extern __shared__ int s_offsets[]; // Shared memory for local offsets

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialize local offsets
    if (threadIdx.x < buckets) {
        s_offsets[threadIdx.x] = 0;
    }
    __syncthreads();

    // Coalesced memory access
    if (idx < input_size) {
        int local_val = input_array[idx];
        int local_bin = bfe(local_val, 0, numBits);

        // Compute global and local offsets
        int global_base_offset = prefix_sum[local_bin];
        int local_offset = atomicAdd(&s_offsets[local_bin], 1);
        int global_offset = global_base_offset + local_offset;

        // Bounds-checked reordering
        if (global_offset < input_size) {
            output_array[global_offset] = local_val;
        }
    }
}