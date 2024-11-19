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

void swap(float* &a, float* &b){
  float *temp = a;
  a = b;
  b = temp;
}

void check_sorted(float * d_queue, int n){
    float * temp, prev;
    prev = -INFINITY;
    temp = (float *) malloc((n + 1) * sizeof(float));
    cudaMemcpy(temp, d_queue, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++){
        if (temp[i] < prev){
            printf("error: queue not sorted properly\n");
            exit(1);
        }
        prev = temp[i];
    }
    printf("success: queue is sorted\n");
    free(temp);
}

void print_d_queue(float * d_queue, int n){
    float * temp, prev;
    prev = -INFINITY;
    temp = (float *) malloc((n + 1) * sizeof(float));
    cudaMemcpy(temp, d_queue, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++){
        printf("%f ", temp[i]);

        // check if d_queue is sorted
        if (temp[i] < prev){
            printf("error: queue not sorted properly\n");
            exit(1);
        }
        prev = temp[i];
    }
    printf("\n");
    free(temp);
}

void run_tests(float * d_queue, int max_size){
    float item;
    for (int i = 0; i < max_size; i++){
        item = ((float)rand()/(float)(RAND_MAX)) * RANGE;
        push(d_queue, item, max_size);
    }
    //print_d_queue(d_queue, max_size);
    for (int i = 0; i < 100; i++){
        pop(d_queue, max_size);
        item = ((float)rand()/(float)(RAND_MAX)) * RANGE;
        push(d_queue, item, max_size);
    }
    //print_d_queue(d_queue, max_size);
    check_sorted(d_queue, max_size);
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

    float * d_queue;
    err = cudaMalloc((void **) &d_queue, (n + 1) * sizeof(float));
    if (err != cudaSuccess){
        printf("queue could not be allocated, exiting...");
        exit(1);
    }

    do {
        float * temp;
        temp = (float *) malloc((n + 1) * sizeof(float));
        for (int i = 0; i < n + 1; i++){
            temp[i] = INFINITY;
        }
        err = cudaMemcpy(d_queue, temp, (n + 1) * sizeof(float), cudaMemcpyHostToDevice);
        free(temp);
    } while (0);

    run_tests(d_queue, n);

    cudaFree(d_queue);

    return 0;
}

void push(float * queue, float item, int max_size){
    d_push<<<1, 1>>>(queue, item, max_size);
}

__global__ void d_push(float * queue, float item, int max_size){
    int i;
    float temp;
    for (i = 0; i < max_size; i++){
        if (item < queue[i]) {
            temp = queue[i];
            queue[i] = item;
            item = temp;
        }
    }
}

float pop(float * d_queue, int max_size){
    float * d_ret;
    cudaMalloc((void **) &d_ret, sizeof(float));
    for (int i = 0; i < ceil(max_size / 512.0); i++){
        d_pop<<<1, 512>>>(d_queue, d_ret, max_size, i * 512);
    }
    float temp;
    cudaMemcpy(&temp, d_ret, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_ret);
    // printf("popped item: %f\n", temp);
    return temp;
}

__global__ void d_pop(float * d_queue, float * d_ret, int max_size, int ofs){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid == 0 && ofs == 0){
        *d_ret = d_queue[0];
    }
    if (tid + ofs >= max_size) return;
    d_queue[tid + ofs] = d_queue[tid + ofs + 1];
}