#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <math.h>

#define RANGE 99.0

void push(float * queue, float item, int max_size);
__global__ void d_push(float * queue, float item, int max_size);
float pop(float * queue, int max_size);
__global__ void d_pop(float * queue, int max_size, int ofs);

float min(float * old, float val){
    if (val < *old){
        float temp = *old;
        *old = val;
        return temp;
    } else {
        return *old;
    }
}

void print_queue(float * queue, int n){
    for (int i = 0; i < n; i++){
        printf("%f ", queue[i]);
        if ((i != 0) && (i % 10 == 0)) printf("\n");
    }
    if (n % 10 != 0) printf("\n");
}

void run_tests(float * queue, int max_size){
    float item;
    for (int i = 0; i < max_size; i++){
        item = ((float)rand()/(float)(RAND_MAX)) * RANGE;
        push(queue, item, max_size);
    }
    print_queue(queue, max_size);
}

int main(int argc, char *argv[]){

    cudaError_t err;
    int n = 0;

    if(argc != 2){
		printf("usage:  ./insertionQ n\n");
		printf("n = number of elements in array\n");
		exit(1);
	}

    n = atoi(argv[1]);

    float * queue;
    err = cudaMallocManaged((void **) &queue, n * sizeof(float));
    if (err != cudaSuccess){
        printf("queue could not be allocated, exiting...");
        exit(1);
    }

    for (int i = 0; i < n; i++){
        queue[i] = INFINITY;
    }

    run_tests(queue, n);

    cudaFree(queue);

    return 0;
}

void push(float * queue, float item, int max_size){
    d_push<<<1, 1>>>(queue, item, max_size);
}

__global__ void d_push(float * queue, float item, int max_size){
    int i;
    float old;
    for (i = 0; i < max_size; i++){
        old = min(&queue[i], item);
        if (isinf(old)) break;
        if (old > item) item = old;
    }
}

float pop(float * queue, int max_size){
    float val = queue[0];
    dim3 gridDim(128);
    dim3 blockDim(250);
    for (int i = 0; i < ceil(max_size / (gridDim.x * blockDim.x)); i++){
        d_pop<<<gridDim, blockDim>>>(queue, max_size, i * gridDim.x * blockDim.x);
    }
}

__global__ void d_pop(float * queue, int max_size, int ofs){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float temp = queue[tid + ofs];
    cudaThreadSynchronize();
    if (tid + ofs == 0) return;
    queue[tid + ofs - 1] = temp;
}