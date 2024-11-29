#!/bin/bash
module load gcc
gcc -O3 -o serial ../src/cpu/serial.c
gcc -O3 -fopenmp -o openmp ../src/cpu/openmp.c
module load openmpi/4.1.4-gcc11
mpicc -O3 -o mpi ../src/cpu/mpi.c
./serial 5000 5000 > /scratch/$USER/serial.performance.5000.txt
./serial 5000 5000 > /scratch/$USER/serial.performance.5000.txt
./serial 5000 5000 > /scratch/$USER/serial.performance.5000.txt
./serial 10000 5000 > /scratch/$USER/serial.performance.10000.txt
./serial 10000 5000 > /scratch/$USER/serial.performance.10000.txt
./serial 10000 5000 > /scratch/$USER/serial.performance.10000.txt
./openmp 5000 5000 20 /scratch/$USER/openmp.performance.5000.txt
./openmp 5000 5000 20 /scratch/$USER/openmp.performance.5000.txt
./openmp 5000 5000 20 /scratch/$USER/openmp.performance.5000.txt
./openmp 10000 5000 20 /scratch/$USER/openmp.performance.10000.txt
./openmp 10000 5000 20 /scratch/$USER/openmp.performance.10000.txt
./openmp 10000 5000 20 /scratch/$USER/openmp.performance.10000.txt
mpirun -n 20 mpi 5000 5000 20 /scratch/$USER/mpi.performance.5000.txt
mpirun -n 20 mpi 5000 5000 20 /scratch/$USER/mpi.performance.5000.txt
mpirun -n 20 mpi 5000 5000 20 /scratch/$USER/mpi.performance.5000.txt
mpirun -n 20 mpi 10000 5000 20 /scratch/$USER/mpi.performance.10000.txt
mpirun -n 20 mpi 10000 5000 20 /scratch/$USER/mpi.performance.10000.txt
mpirun -n 20 mpi 10000 5000 20 /scratch/$USER/mpi.performance.10000.txt
rm serial openmp mpi
