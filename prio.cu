#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <math.h>
#include <assert.h>

struct PriorityQueue{
	float * data;
	size_t size;
	size_t max_size;
};

struct PriorityQueue * new_PriorityQueue(size_t max_size);
void delete_PriorityQueue(struct PriorityQueue * queue);
float peek(struct PriorityQueue * queue);
bool push(struct PriorityQueue * queue, float item);
float pop(struct PriorityQueue * queue);

int main(int argc, char *argv[]){

	struct PriorityQueue * queue = new_PriorityQueue(10);
	assert(queue != NULL);

	int i;
	float j;
	for (i = 0; i < 10; i++){
		push(queue, (float) i);
	}
	for (i = 0; i < 10; i++){
		j = pop(queue);
		if (!isnan(j))
		printf("%f\n", j);
	}

	delete_PriorityQueue(queue);

	return 0;
}

struct PriorityQueue * new_PriorityQueue(size_t max_size){

	struct PriorityQueue * queue;
	queue = (struct PriorityQueue *) malloc(sizeof(struct PriorityQueue));
	if (queue == NULL) return NULL;

	queue->data = (float *) malloc(max_size * sizeof(float));
	if (queue->data == NULL) return NULL;

	queue->size = 0;
	queue->max_size = max_size;

	return queue;
}

void delete_PriorityQueue(struct PriorityQueue * queue){
	if (queue != NULL){
		free(queue->data);
		free(queue);
	}
}

float peek(struct PriorityQueue * queue){
	/*
	returns the first item in the queue
	if there are no items return NaN
	*/
	if (queue->size > 0) return queue->data[0];
	return NAN; 
}

bool push(struct PriorityQueue * queue, float item){
	/*
	inserts an item into the queue
	returns 1 on success
	returns 0 on error
	*/

	//check capacity
	if (queue->size == queue->max_size) return 0;

	// insert item
	queue->data[queue->size] = item;
	++queue->size;

	return 1;
}

float pop(struct PriorityQueue * queue){
	/*
	pop first item in queue and return its value
	if there are no items return NaN
	*/
	if (queue->size > 0) return queue->data[--queue->size];
	return NAN;
}