#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include "scan_kernels.cuh"
#include "utils.cuh"
#include <string>
#include <vector>
#include <fstream>
#include <sys/stat.h>  // for mkdir()

// Utility to check for CLI flags
bool hasFlag(int argc, char const* argv[], const std::string& flag) {
    for (int i = 3; i < argc; ++i) {
        if (flag == argv[i]) return true;
    }
    return false;
}

int main(int argc, char const* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <num_keys> <buckets> [--dump]\n";
        return -1;
    }

    bool dump_output = hasFlag(argc, argv, "--dump");

    int num_keys = atoi(argv[1]);
    int buckets = atoi(argv[2]);
    int numBits = std::ceil(std::log2(buckets));

    int* h_keys;
    cudaMallocHost(&h_keys, sizeof(int) * num_keys);  // Pinned memory
    dataGenerator(h_keys, num_keys, 0, 1);

    int* h_histogram = (int*)malloc(buckets * sizeof(int));
    memset(h_histogram, 0, sizeof(int) * buckets);

    int *d_keys, *d_histogram;
    cudaMalloc(&d_keys, sizeof(int) * num_keys);
    cudaMalloc(&d_histogram, sizeof(int) * buckets);
    cudaMemcpy(d_keys, h_keys, sizeof(int) * num_keys, cudaMemcpyHostToDevice);
    cudaMemset(d_histogram, 0, sizeof(int) * buckets);

    // Timers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Histogram kernel
    int num_threads = 512;
    int num_blocks = (num_keys + num_threads - 1) / num_threads;
    size_t shared_mem = sizeof(int) * buckets;
    generate_histogram<<<num_blocks, num_threads, shared_mem>>>(d_keys, num_keys, d_histogram, numBits, buckets);
    cudaDeviceSynchronize();

    // Prefix scan
    int* d_prefix_sum;
    cudaMalloc(&d_prefix_sum, sizeof(int) * buckets);
    prefixScan<<<1, buckets / 2, sizeof(int) * buckets>>>(d_histogram, d_prefix_sum, buckets);
    cudaDeviceSynchronize();

    // Reorder
    int* d_output;
    cudaMalloc(&d_output, sizeof(int) * num_keys);
    num_blocks = (num_keys + num_threads * 4 - 1) / (num_threads * 4);
    Reorder<<<num_blocks, num_threads, shared_mem>>>(d_keys, d_prefix_sum, d_output, num_keys, numBits, buckets);
    cudaDeviceSynchronize();

    // Copy results back
    int* h_prefix_sum = (int*)malloc(sizeof(int) * buckets);
    int* h_output_array = (int*)malloc(sizeof(int) * num_keys);  // 🔧 FIXED
    cudaMemcpy(h_histogram, d_histogram, sizeof(int) * buckets, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_prefix_sum, d_prefix_sum, sizeof(int) * buckets, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_array, d_output, sizeof(int) * num_keys, cudaMemcpyDeviceToHost);  // 🔧 FIXED

    for (int i = 0; i < buckets; i++) {
        printf("partition %d: offset %d, number of keys %d\n", i, h_prefix_sum[i], h_histogram[i]);
    }

    // Stop timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("******** Total Running Time of Kernel = %0.5f sec *******\n", elapsed / 1000.0f);

    if (dump_output) {
        mkdir("../output", 0777);  // Ensures output dir exists


        FILE* fprefix = fopen("../output/offset.txt", "w");
        if (fprefix) {
            for (int i = 0; i < buckets; ++i)
                fprintf(fprefix, "%d\n", h_prefix_sum[i]);
            fclose(fprefix);
            std::cout << "[INFO] Offset dumped to output/offset.txt\n";
        }

        FILE* fhist = fopen("../output/histogram.txt", "w");
        if (fhist) {
            for (int i = 0; i < buckets; ++i)
                fprintf(fhist, "%d\n", h_histogram[i]);
            fclose(fhist);
            std::cout << "[INFO] Histogram dumped to output/histogram.txt\n";
        }
    }

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFreeHost(h_keys);
    cudaFree(d_keys);
    cudaFree(d_histogram);
    cudaFree(d_prefix_sum);
    cudaFree(d_output);
    free(h_histogram);
    free(h_prefix_sum);
    free(h_output_array);  // 🔧 FIXED

    return 0;
}
