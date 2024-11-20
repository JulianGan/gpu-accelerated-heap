#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <math.h>

#define RANGE 99.0

// --- GLOBAL VARIABLE ---
int size = 0, max_size = 0;
float * d_queue, * buffer;
int blocks, threads;

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

    int i; // loop index
    clock_t start, stop;
    cudaDeviceSynchronize();

    start = clock();

    for (i = 0; i < int(ops * (ratio / 100.0)); i++){
        push( random_float() );
    }

    for (i = 0; i < int(ops * (1 - ratio / 100.0)); i++){
        pop();
    }
    cudaDeviceSynchronize();
    stop = clock();
    
    #ifdef DEBUG
    print_queue();
    #endif

    printf("Total time taken = %lf\n", (double)(stop - start) / CLOCKS_PER_SEC);

}

int main(int argc, char *argv[]){

    cudaError_t err;
    int ops = 0, ratio = 0;

    if(argc != 6){
		printf("usage: ./insertionQ <total_operations> <ratio_of_push> <device_number> <blocks> <threads>\n");
		printf("total_operations = number of push/pop operations to simulate\n");
        printf("ratio_of_push = percent of operations that are the push operation * 100\n");
        printf("device_number = index of device (0 or 1 on cims machine)\n");
        printf("blocks = total number of blocks\n");
        printf("threads = number of threads per block\n");
		exit(1);
	}

    err = cudaSetDevice(atoi(argv[3]));
    if (err != cudaSuccess){
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        printf("available devices: %d\n", deviceCount);
        printf("error: could not set device %d\n", atoi(argv[2]));
        exit(1);
    }

    ops = atoi(argv[1]);
    ratio = atoi(argv[2]);
    blocks = atoi(argv[4]);
    threads = atoi(argv[5]);

    init_queue(ops);

    run_tests(ops, ratio);

    free_queue();

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

float peek(){
    float temp;
    cudaMemcpy(&temp, &d_queue[0], sizeof(float), cudaMemcpyDeviceToHost);
    return temp;
}

void push(float item){

    dim3 gridDim(blocks);
    dim3 blockDim(threads);

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

    dim3 gridDim(blocks);
    dim3 blockDim(threads);

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
