#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <math.h>
#include <assert.h>

int x = 0;

typedef struct _{
    size_t size = 0;
    size_t n;
    size_t m;
    size_t max_size;
    float * data;
} PrioQ;

/* FUNCTION DECLARATION */
PrioQ * create_PrioQ(size_t n, size_t m);
void delete_PrioQ(PrioQ * queue);
void push(PrioQ * queue, float item);
float pop(PrioQ * queue);

__global__ void d_push(float * data, float item, size_t n, size_t m);


/* UTILITY FUNCTION */
void print_PrioQ_unstructured(PrioQ * queue){
    if (queue){
        if (queue->data){
            printf("queue max size at %d\n", queue->n * queue->m);
            printf("printing %d items from PrioQ\n", queue->size);
            for (size_t i = 0; i < queue->size; i++){
                printf("%f ", queue->data[i]);
                if ((i != 0) && (i % 10 == 0)) printf("\n");
            }
            if ( !(queue->size % 10 == 0)) printf("\n");
            return;
        }
    }
    printf("PrioQ could not be printed\n");
}

void run_tests(PrioQ * queue){
    if (!queue){
        printf("error: running tests impossible on queue where queue is null pointer\n");
        return;
    }
    if (!queue->data){
        printf("error: running tests impossible on queue where data is null pointer\n");
        return;
    }
    push(queue, 2);
    push(queue, 20);
    push(queue, 7);
    push(queue, 31);
    print_PrioQ_unstructured(queue);
}

int main(int argc, char *argv[]){

    size_t n = 0;

    if(argc != 2){
		printf("usage:  ./digestQ n\n");
		printf("n = number of elements in a segment\n");
		exit(1);
	}

    n = atoi(argv[1]);

    size_t m = 10;

    PrioQ * queue = create_PrioQ(n, m);
    if (!queue){
        printf("queue could not be initialized\n");
        exit(1);
    }

    printf("number of items per segment n = %d\n", n);
    printf("number of segments m = %d\n", m);

    run_tests(queue);

    delete_PrioQ(queue);

    return 0;
}

PrioQ * create_PrioQ(size_t n, size_t m){
    PrioQ * queue = (PrioQ *) malloc(sizeof(PrioQ));
    if (!queue) return NULL;
    queue->n = n;
    queue->m = m;
    queue->max_size = n * m;

    cudaError_t err;
    err = cudaMallocManaged((void **) &queue->data, n * m * sizeof(float));
    if (err != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return NULL;
    }

    return queue;
}

void delete_PrioQ(PrioQ * queue){
    if (queue){
        if (queue->data) cudaFree(queue->data);
        free(queue);
    }
}

void push(PrioQ * queue, float item){
    if (!queue){
        printf("error: push to queue where queue is null pointer\n");
        return;
    }

    // check if queue is full
    if (queue->size == queue->max_size){
        printf("push failed: queue full\n");
        return;
    }

    if (!queue->data){
        printf("error: push to queue where data is null poiner\n");
        return;
    }

    cudaError_t err;

    d_push<<<1, 1>>>(queue->data, item, queue->n, x++);
    err = cudaDeviceSynchronize();

    if (err != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return;
    }
    ++queue->size;
}

float pop(PrioQ * queue){
    return 0;
}

__global__ void d_push(float * data, float item, size_t n, size_t idx){
    data[idx] = item;
}