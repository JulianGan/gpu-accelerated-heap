#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <math.h>

#define RANGE 99.0

// --- GLOBAL VARIABLE ---
int size = 0, max_size = 0;
float * d_queue, * buffer;

// --- FUNCTION DECLARATION ---
void init_queue(int n);
void free_queue();
void push(float item);
__global__ void find_index(float * d_queue, float item, int * d_ret, int max_size, int ofs);
__global__ void d_push(float * d_queue, float * buffer, float item, int index, int max_size, int ofs);
float pop();
__global__ void d_pop(float * d_queue, float * buffer, float * d_ret, int max_size, int size, int ofs);

// --- UTILITY FUNCTION ---
void swap(float * &a, float * &b){
    float * temp = a;
    a = b;
    b = temp;
}

float random_float(){
    return ((float)rand()/(float)(RAND_MAX)) * RANGE;
}

void print_queue(){
    float * temp;
    temp = (float *) malloc(max_size * sizeof(float));
    cudaMemcpy(temp, d_queue, max_size * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < max_size; i++){
        printf("%f ", temp[i]);
    }
    printf("\n");
    free(temp);
}

void run_tests(int ops, int ratio){
void run_tests(float *d_queue, int n, int thread_count, int block_count) {
    float item;

    // Start timing
    cudaDeviceSynchronize(); // Ensure any previous GPU tasks are complete
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

    // Synchronize to ensure all GPU operations are complete before stopping the clock
    cudaDeviceSynchronize();

    // Stop timing
    clock_t stop = clock();

    // Check if the queue is sorted (optional)
    check_sorted(d_queue, n);

    // Output the total time taken
    printf("Total time taken = %lf seconds\n", (double)(stop - start) / CLOCKS_PER_SEC);
}


}

int main(int argc, char *argv[]) {
    cudaError_t err;
    int n = 0; // Problem size
    int thread_count = 0; // Threads per block
    int block_count = 0; // Number of blocks

    if (argc != 5) { // Updated argument count
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

    // Check for thread and block limits
    if (thread_count < 1 || thread_count > 1024) {
        printf("Error: Thread count must be between 1 and 1024.\n");
        exit(1);
    }
    if (block_count < 1 || block_count > 65535) {
        printf("Error: Block count must be between 1 and 65535.\n");
        exit(1);
    }

    // Initialize queue and run the tests
    float *d_queue;
    err = cudaMalloc((void **)&d_queue, (n + 1) * sizeof(float));
    if (err != cudaSuccess) {
        printf("Queue could not be allocated, exiting...\n");
        exit(1);
    }

    // Pass thread and block configuration to the kernels
    run_tests(d_queue, n, thread_count, block_count);

    cudaFree(d_queue);

    return 0;
}


// --- FUNCTION DEFINITION ---
void init_queue(int n){

    cudaError_t err;

    max_size = n;

    err = cudaMalloc((void **) &d_queue, n * sizeof(float));
    if (err != cudaSuccess){
        printf("error: queue could not be allocated, exiting...\n");
        exit(1);
    }
    err = cudaMalloc((void **) &buffer, n * sizeof(float));
    if (err != cudaSuccess){
        printf("error: queue could not be allocated, exiting...\n");
        exit(1);
    }

    do {
        float * temp;
        temp = (float *) malloc(n * sizeof(float));
        for (int i = 0; i < n; i++){
            temp[i] = INFINITY;
        }
        err = cudaMemcpy(d_queue, temp, n * sizeof(float), cudaMemcpyHostToDevice);
        free(temp);
    } while (0);
}

void free_queue(){
    cudaFree(d_queue); cudaFree(buffer);
}

void push(float item){

    dim3 gridDim(32);
    dim3 blockDim(250);

    int * d_ret, index, ofs;
    cudaMalloc((void **) &d_ret, sizeof(int));
    for (ofs = 0; ofs < size; ofs += gridDim.x * blockDim.x){
        find_index<<<gridDim, blockDim>>>(d_queue, item, d_ret, size, ofs);
    }
    cudaMemcpy(&index, d_ret, sizeof(int), cudaMemcpyDeviceToHost);

    for (ofs = 0; ofs < size; ofs += gridDim.x * blockDim.x){
        d_push<<<gridDim, blockDim>>>(d_queue, buffer, item, index, size, ofs);
    }
    ++size;
    cudaFree(d_ret);
    swap(d_queue, buffer);
}

__global__ void find_index(float * d_queue, float item, int * d_ret, int max_size, int ofs){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid + ofs >= max_size) return;

    if (tid + ofs == 0){
        if (item < d_queue[0]){
            *d_ret = 0;
        }
        return;
    }
    if (d_queue[tid + ofs - 1] <= item && item < d_queue[tid + ofs]){
        *d_ret = tid + ofs;
    }
}

__global__ void d_push(float * d_queue, float * buffer, float item, int index, int max_size, int ofs){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid + ofs >= max_size) return;

    if (tid + ofs > index){
        buffer[tid + ofs] = d_queue[tid + ofs - 1]; return;
    }
    if (tid + ofs < index){
        buffer[tid + ofs] = d_queue[tid + ofs]; return;
    }
    if (tid + ofs == index){
        buffer[tid + ofs] = item;
    }

}

float pop(){

    dim3 gridDim(32);
    dim3 blockDim(250);

    int ofs;
    float * d_ret;
    cudaMalloc((void **) &d_ret, sizeof(float));
    for (ofs = 0; ofs < size; ofs += gridDim.x * blockDim.x){
        d_pop<<<gridDim, blockDim>>>(d_queue, buffer, d_ret, size, size, ofs);
    }
    --size;

    float temp;
    cudaMemcpy(&temp, d_ret, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_ret);
    swap(d_queue, buffer);
    return temp;
}

__global__ void d_pop(float * d_queue, float * buffer, float * d_ret, int max_size, int size, int ofs){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid + ofs >= max_size) return;

    if (tid + ofs == 0){
        *d_ret = d_queue[0];
        return;
    }

    buffer[tid + ofs - 1] = d_queue[tid + ofs];

    if (tid + ofs == size - 1){
        buffer[tid + ofs] = INFINITY;
        d_queue[tid + ofs] = INFINITY;
    }
}