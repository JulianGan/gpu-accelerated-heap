### GPU Accelerated Priority Queue

**Overview**
<br>
This project provides a GPU-based implementation of a priority queue, designed using CUDA. The algorithm uses an insertion sort approach optimized for GPU parallelism.
- **Push (Insert):** To insert a new item, the algorithm identifies the correct position in the sorted queue and inserts the item into a buffer array. The remaining elements are moved accordingly, and a pointer swap finalizes the operation.
- **Pop (RemoveMin):** The smallest element (front of the queue) is returned, and the remaining elements are shifted to maintain the sorted order. A pointer swap ensures the consistency of the queue.

---

### How to Run

#### **Using `insertionQ_2.cu`**
```
Compile with:
nvcc insertionQ_2.cu -o insertionQ_2
```

Call:
```
./insertionQ_2 a b c
```

| Parameter      | Description                                       |
|----------------|---------------------------------------------------|
| `a`            | Number of operations                             |
| `b`            | Ratio of push operations (50 ~ 100)              |
| `c`            | GPU device index (e.g., 0 or 1)                  |

**Example**:
```
./insertionQ_2 100000 80 0
```
This example runs the program with:
- 100,000 total operations
- 80% push operations
- GPU device 0

---

#### **Using `insertionQ_3.cu`**
```
Compile with:
nvcc insertionQ_3.cu -o insertionQ_3
```

Call:
```
./insertionQ_3 a b c d e
```

| Parameter      | Description                                       |
|----------------|---------------------------------------------------|
| `a`            | Number of operations                             |
| `b`            | Ratio of push operations (50 ~ 100)              |
| `c`            | GPU device index (e.g., 0 or 1)                  |
| `d`            | Number of threads per block (1 ~ 1024)           |
| `e`            | Number of blocks (1 ~ 65535)                     |

**Example**:
```
./insertionQ_3 100000 80 0 256 16
```
This example runs the program with:
- 100,000 total operations
- 80% push operations
- GPU device 0
- 256 threads per block
- 16 blocks

---

### Files

- `insertionQ_2.cu`: Original GPU-accelerated priority queue with a simpler interface.
- `insertionQ_3.cu`: Updated version with configurable thread and block counts.
- `README.md`: Documentation for the usage and functionality of both versions.

---
