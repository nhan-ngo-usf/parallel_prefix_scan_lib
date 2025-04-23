# 🚀 CUDA Parallel Prefix Scan & Reordering Toolkit

This project implements an efficient, fully parallelized system for:
- Histogram generation
- Exclusive prefix scan (using Blelloch's algorithm)
- Data reordering based on scan results

It is designed for CUDA-enabled GPUs and uses **shared memory**, **atomic operations**, and **coalesced memory access** patterns for high performance on large-scale data.

---

## 📚 Project Summary

This project is built to demonstrate performance benefits from **GPU parallelization** in large-scale data processing tasks. It uses CUDA to accelerate:

### ⚙️ Components:
- **Histogram Kernel**: Each thread computes the histogram bin of a subset of elements using `atomicAdd`. Grid-stride loops improve scalability.
- **Prefix Scan Kernel**: Implements an exclusive scan via Blelloch's algorithm using shared memory to reduce global memory overhead.
- **Reorder Kernel**: Reorders keys based on histogram prefix sums. Atomic operations and shared memory are used to assign positions within each bin.

---

## 📁 Project Structure
parallel_scan/ 
├── include/ # Header files 
├── src/ # CUDA source files 
├── output/ # Optional output files (--dump) 
├── CMakeLists.txt # Build configuration 
└── README.md # This file

---

## 🧠 Optimization Techniques

- **Shared Memory**: Used in all kernels to reduce latency vs. global memory.
- **Grid-stride Loops**: Ensure scalability across arbitrary input sizes.
- **AtomicAdd**: Maintains thread safety during histogram and reordering steps.
- **Coalesced Memory Access**: Minimizes transaction overhead during global loads and stores.

---

## 🚀 Build Instructions

Make sure you have CUDA and CMake installed:
```bash
mkdir build
cd build
cmake ..
make

---

## 🧪 Run Instructions
./parallel_scan <num_keys> <buckets> [--dump]

---

## Sample output
Refer to sample/