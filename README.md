# Parallel-2D-Convolution-Engine-for-Deep-Learning-
Developed a distributed 2D convolution engine in C++ using MPI on an HPC cluster. Implemented filter-level parallelism with MPI_Bcast, MPI_Scatterv, and MPI_Gatherv. Benchmarked up to 32 cores, reducing runtime from 1.83 s to 0.52 s (3.52× speedup). Optimized memory layout and validated correctness against a serial baseline.
