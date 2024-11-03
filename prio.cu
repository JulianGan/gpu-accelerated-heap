#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <math.h>
#include <assert.h>

// array implementation of a max heap
struct PriorityQueue{
	float * data;
	size_t size;
	size_t max_size;
};

/* FUNCTION DECLARATION */
struct PriorityQueue * new_PriorityQueue(size_t max_size);
void delete_PriorityQueue(struct PriorityQueue * queue);
float peek(struct PriorityQueue * queue);
bool push(struct PriorityQueue * queue, float item);
float pop(struct PriorityQueue * queue);

/* UTILITY FUNCTION */
void swap(float * a, float * b){
	float temp;
	temp = * a;
	* a = * b;
	* b = temp;
}

void print_PriorityQueue(struct PriorityQueue * queue){
	for (int i = 0; i < queue->size; i++){
		printf("%f ", queue->data[i]);
	}
	printf("\n");
}

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
		if (!isnan(j)) printf("%f\n", j);
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

	// check capacity
	if (queue->size == queue->max_size) return false;

	// insert item
	queue->data[queue->size] = item;
	++queue->size;

	// swim up
	size_t idx = queue->size - 1;
	while (idx > 0 && queue->data[idx] > queue->data[(idx - 1) / 2]){
		swap(&queue->data[idx], &queue->data[(idx - 1) / 2]);
		idx -= 1; idx /= 2;
	}

	return true;
}

float pop(struct PriorityQueue * queue){
	/*
	pop first item in queue and return its value
	if there are no items return NaN
	*/

	// check if there are any items to pop
	if (queue->size == 0) return NAN;
	
	// get item
	float ans = queue->data[0];

	// pop item
	swap(&queue->data[0], &queue->data[--queue->size]);

	// swim down
	size_t n = queue->size;
	size_t idx = 0;
	size_t l, r;
	while (true){
		l = idx * 2 + 1;
		r = idx * 2 + 2;
		if (l < n && queue->data[idx] < queue->data[l] && r < n && queue->data[idx] < queue->data[r]){
			if (queue->data[l] > queue->data[r]){
				swap(&queue->data[idx], &queue->data[l]);
				idx = l;
			} else {
				swap(&queue->data[idx], &queue->data[r]);
				idx = r;
			}
			continue;
		}
		if (l < n && queue->data[idx] < queue->data[l]){
			swap(&queue->data[idx], &queue->data[l]);
			idx = l;
			continue;
		}
		if (r < n && queue->data[idx] < queue->data[r]){
			swap(&queue->data[idx], &queue->data[r]);
			idx = r;
			continue;
		}
		break;
	}

	return ans;
}