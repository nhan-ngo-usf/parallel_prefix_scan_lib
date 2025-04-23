#pragma once

__global__ void generate_histogram(int* input_array, int input_size, int* histogram, int numBits, int buckets);
__global__ void prefixScan(int* histogram, int* prefix_sum, int size);
__global__ void Reorder(int* input_array, int* prefix_sum, int* output_array, int input_size, int numBits, int buckets);