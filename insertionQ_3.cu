#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <math.h>

#define RANGE 99.0

void push(float * queue, float item, int max_size);
__global__ void d_push(float * queue, float item, int max_size);
float pop(float * d_queue, int max_size);
__global__ void d_pop(float * d_queue, float * d_ret, int max_size, int ofs);

void swap(float* &a, float* &b) {
    float *temp = a;
    a = b;
    b = temp;
}

void check_sorted(float * d_queue, int n) {
    float * temp, prev;
    prev = -INFINITY;
    temp = (float *) malloc((n + 1) * sizeof(float));
    cudaMemcpy(temp, d_queue, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        if (temp[i] < prev) {
            printf("error: queue not sorted properly\n");
            exit(1);
        }
        prev = temp[i];
    }
    printf("success: queue is sorted\n");
    free(temp);
}

void print_d_queue(float * d_queue, int n) {
    float * temp, prev;
    prev = -INFINITY;
    temp = (float *) malloc((n + 1) * sizeof(float));
    cudaMemcpy(temp, d_queue, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        printf("%f ", temp[i]);

        // check if d_queue is sorted
        if (temp[i] < prev) {
            printf("error: queue not sorted properly\n");
            exit(1);
        }
        prev = temp[i];
    }
    printf("\n");
    free(temp);
}

// Corrected run_tests function
void run_tests(float *d_queue, int n, int thread_count, int block_count) {
    float item;

    // Start timing
    cudaDeviceSynchronize(); // Ensure previous GPU tasks are complete
    clock_t start = clock();

    // Push operations
    for (int i = 0; i < n; i++) {
        item = ((float)rand() / (float)(RAND_MAX)) * RANGE;
        d_push<<<block_count, thread_count>>>(d_queue, item, n);
    }

    // Pop operations
    for (int i = 0; i < n; i++) {
        pop(d_queue, n);
    }

    // Ensure all GPU operations are complete
    cudaDeviceSynchronize();

    // Stop timing
    clock_t stop = clock();

    // Check if the queue is sorted
    check_sorted(d_queue, n);

    // Output the total time taken
    printf("Total time taken = %lf seconds\n", (double)(stop - start) / CLOCKS_PER_SEC);
}

int main(int argc, char *argv[]) {

    cudaError_t err;
    int n = 0, thread_count = 0, block_count = 0;

    if (argc != 6) { // Updated argument count
        printf("Usage: ./insertionQ_3 n push_ratio device_index thread_count block_count\n");
        printf("  n: number of operations\n");
        printf("  push_ratio: ratio of push operations (50 ~ 100)\n");
        printf("  device_index: GPU device index (0 or 1)\n");
        printf("  thread_count: number of threads per block\n");
        printf("  block_count: number of blocks\n");
        exit(1);
    }

    n = atoi(argv[1]);
    int push_ratio = atoi(argv[2]);
    int device_index = atoi(argv[3]);
    thread_count = atoi(argv[4]);
    block_count = atoi(argv[5]);

    cudaSetDevice(device_index);

    // Validate thread and block counts
    if (thread_count < 1 || thread_count > 1024) {
        printf("Error: Thread count must be between 1 and 1024.\n");
        exit(1);
    }
    if (block_count < 1 || block_count > 65535) {
        printf("Error: Block count must be between 1 and 65535.\n");
        exit(1);
    }

    // Allocate memory for the queue
    float *d_queue;
    err = cudaMalloc((void **) &d_queue, (n + 1) * sizeof(float));
    if (err != cudaSuccess) {
        printf("Queue could not be allocated, exiting...\n");
        exit(1);
    }

    // Initialize the queue
    float *temp = (float *) malloc((n + 1) * sizeof(float));
    for (int i = 0; i < n + 1; i++) {
        temp[i] = INFINITY;
    }
    err = cudaMemcpy(d_queue, temp, (n + 1) * sizeof(float), cudaMemcpyHostToDevice);
    free(temp);

    // Run the tests
    run_tests(d_queue, n, thread_count, block_count);

    // Free GPU memory
    cudaFree(d_queue);

    return 0;
}

void push(float * queue, float item, int max_size) {
    d_push<<<1, 1>>>(queue, item, max_size);
}

__global__ void d_push(float * queue, float item, int max_size) {
    int i;
    float temp;
    for (i = 0; i < max_size; i++) {
        if (item < queue[i]) {
            temp = queue[i];
            queue[i] = item;
            item = temp;
        }
    }
}

float pop(float * d_queue, int max_size) {
    float *d_ret;
    cudaMalloc((void **) &d_ret, sizeof(float));
    for (int i = 0; i < ceil(max_size / 512.0); i++) {
        d_pop<<<1, 512>>>(d_queue, d_ret, max_size, i * 512);
    }
    float temp;
    cudaMemcpy(&temp, d_ret, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_ret);
    return temp;
}

__global__ void d_pop(float * d_queue, float * d_ret, int max_size, int ofs) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid == 0 && ofs == 0) {
        *d_ret = d_queue[0];
    }
    if (tid + ofs >= max_size) return;
    d_queue[tid + ofs] = d_queue[tid + ofs + 1];
}
