#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <math.h>

#define RANGE 99.0

// --- GLOBAL VARIABLE ---
int size = 0;

// --- FUNCTION DECLARATION ---
void push(float * d_queue, float * buffer, float item, int max_size);
__global__ void find_index(float * d_queue, float item, int * d_ret, int max_size, int ofs);
__global__ void d_push(float * d_queue, float * buffer, float item, int index, int max_size, int ofs);
float pop(float * d_queue, float * buffer, int max_size);
__global__ void d_pop(float * d_queue, float * buffer, float * d_ret, int max_size, int size, int ofs);

// --- UTILITY FUNCTION ---
void test_find_index(){
    int n = 8;
    float arr[] = {1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 9.0, 10.0};
    float * d_arr;
    cudaMalloc((void **) &d_arr, n * sizeof(float));
    cudaMemcpy(d_arr, arr, n * sizeof(float), cudaMemcpyHostToDevice);

    int * d_ret, index;
    cudaMalloc((void **) &d_ret, sizeof(int));
    find_index<<<1, n>>>(d_arr, 7.0, d_ret, n, 0);
    cudaMemcpy(&index, d_ret, sizeof(int), cudaMemcpyDeviceToHost);

    printf("d_ret should be 5, d_ret: %d\n", index);

    cudaFree(d_arr); cudaFree(d_ret);
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
    int sorted = 1;
    prev = -INFINITY;
    temp = (float *) malloc((n + 1) * sizeof(float));
    cudaMemcpy(temp, d_queue, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++){
        printf("%f ", temp[i]);

        // check if d_queue is sorted
        if (temp[i] < prev){
            sorted = 0;
        }
        prev = temp[i];
    }
    printf("\n");
    if (!sorted){
        printf("error: queue not sorted properly\n");
        exit(1);
    }
    free(temp);
}

void swap(float* &a, float* &b){
  float *temp = a;
  a = b;
  b = temp;
}

void run_tests(float * d_queue, float * buffer, int max_size){
    float item;
    for (int i = 0; i < max_size; i++){
        item = ((float)rand()/(float)(RAND_MAX)) * RANGE;
        push(d_queue, buffer, item, max_size);
        swap(buffer, d_queue);
    }
    print_d_queue(d_queue, max_size);
    for (int i = 0; i < 6; i++){
        item = pop(d_queue, buffer, max_size);
        printf("item: %f\n", item);
        swap(buffer, d_queue);
        print_d_queue(d_queue, max_size);
    }

    for (int i = 0; i < 100; i++){
        pop(d_queue, buffer, max_size);
        swap(buffer, d_queue);
        item = ((float)rand()/(float)(RAND_MAX)) * RANGE;
        push(d_queue, buffer, item, max_size);
        swap(buffer, d_queue);
    }

    print_d_queue(d_queue, max_size);
    check_sorted(d_queue, max_size);
}

int main(int argc, char *argv[]){

    cudaError_t err;
    int n = 0;

    if(argc != 3){
		printf("usage:  ./insertionQ n m\n");
		printf("n = number of elements in array\n");
        printf("m = device number\n");
		exit(1);
	}

    err = cudaSetDevice(atoi(argv[2]));
    if (err != cudaSuccess){
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        printf("available devices: %d\n", deviceCount);
        printf("error: could not set device %d\n", atoi(argv[2]));
        exit(1);
    }

    n = atoi(argv[1]);

    float * d_queue, * buffer;
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

    run_tests(d_queue, buffer, n);

    cudaFree(d_queue); cudaFree(buffer);

    return 0;
}

// --- FUNCTION DEFINITION ---

void push(float * d_queue, float * buffer, float item, int max_size){

    dim3 gridDim(32);
    dim3 blockDim(250);

    int * d_ret, index, ofs;
    cudaMalloc((void **) &d_ret, sizeof(int));
    for (ofs = 0; ofs < max_size; ofs += gridDim.x * blockDim.x){
        find_index<<<gridDim, blockDim>>>(d_queue, item, d_ret, max_size, ofs);
    }
    cudaMemcpy(&index, d_ret, sizeof(int), cudaMemcpyDeviceToHost);

    for (ofs = 0; ofs < max_size; ofs += gridDim.x * blockDim.x){
        d_push<<<gridDim, blockDim>>>(d_queue, buffer, item, index, max_size, ofs);
    }
    ++size;
    cudaFree(d_ret);
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

float pop(float * d_queue, float * buffer, int max_size){

    dim3 gridDim(32);
    dim3 blockDim(250);

    int ofs;
    float * d_ret;
    cudaMalloc((void **) &d_ret, sizeof(float));
    for (ofs = 0; ofs < max_size; ofs += gridDim.x * blockDim.x){
        d_pop<<<gridDim, blockDim>>>(d_queue, buffer, d_ret, max_size, size, ofs);
    }
    --size;

    float temp;
    cudaMemcpy(&temp, d_ret, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_ret);
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
