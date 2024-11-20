### GPU Accelerated Priority Queue

**Overview**
<br>
Here, we present a gpu implementation of the priority queue. The algorithm is just an insertion sort using gpu.
- For inserting (push) a new item, we search for the correct index of the new item in the sorted queue and put the item in a buffer array using this index. Then, we move the rest of the elements over from the queue to the buffer. Finally, we perform a pointer swap, and the queue will be sorted with the new item.
- For removeMin (pop), we return the first element of the queue. Then, we move all the items from queue to a buffer and perform a pointer swap. When we move the items from queue to buffer, the index of the items decrement by 1 to make up for the removal of the first item.

**How to Run**
```
Compile with:
nvcc insertionQ.cu -o insertionQ

Call:
./insertionQ a b c

|    where 'a' is number of operations
|          'b' is ratio of push operations
|          'c' is device index (0 or 1 on cims machine)


Example:
./insertionQ 100 80 1
```

**Files**
- insertionQ.cu (our work on gpu accelerated PQ)
- bufferQ_3.cu (also our work on gpu accelerated PQ, different call format)
