/*  
Name:   Nolan McKivergan
Email:  npmckivergan@crimson.ua.edu
Course: CS 581
Homework #: 5
Instructions to compile the program: nvcc gpu_paper.cu -O3 -o gpu_paper
Instructions to run the program: ./gpu_paper <grid size> <max gens> <output file path>
*/ 

#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <string.h>

#define TILE_DIM 16

int size;
int max_gens;
char **curr_matrix;
char **next_matrix;

void print_board(char **matrix, int size, FILE *output_file);
double gettime();

// Kernel function
__global__ void updateKernel(char *d_curr_matrix, char *d_next_matrix, int width, int height, int *d_change_flag) {
    __shared__ char tile[TILE_DIM + 2][TILE_DIM + 2];
    __shared__ int local_change_flag;

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;

    // Initialize local change flag
    if (threadIdx.x == 0 && threadIdx.y == 0) local_change_flag = 0;
    __syncthreads();

    // Load cell and neighbors into shared memory
    if (x < width && y < height) {
        tile[ty][tx] = d_curr_matrix[y * width + x];
        if (threadIdx.x == 0) tile[ty][0] = (x > 0) ? d_curr_matrix[y * width + (x - 1)] : 0;
        if (threadIdx.x == TILE_DIM - 1) tile[ty][TILE_DIM + 1] = (x < width - 1) ? d_curr_matrix[y * width + (x + 1)] : 0;
        if (threadIdx.y == 0) tile[0][tx] = (y > 0) ? d_curr_matrix[(y - 1) * width + x] : 0;
        if (threadIdx.y == TILE_DIM - 1) tile[TILE_DIM + 1][tx] = (y < height - 1) ? d_curr_matrix[(y + 1) * width + x] : 0;
        if (threadIdx.x == 0 && threadIdx.y == 0) tile[0][0] = (x > 0 && y > 0) ? d_curr_matrix[(y - 1) * width + (x - 1)] : 0;
        if (threadIdx.x == TILE_DIM - 1 && threadIdx.y == 0) tile[0][TILE_DIM + 1] = (x < width - 1 && y > 0) ? d_curr_matrix[(y - 1) * width + (x + 1)] : 0;
        if (threadIdx.x == 0 && threadIdx.y == TILE_DIM - 1) tile[TILE_DIM + 1][0] = (x > 0 && y < height - 1) ? d_curr_matrix[(y + 1) * width + (x - 1)] : 0;
        if (threadIdx.x == TILE_DIM - 1 && threadIdx.y == TILE_DIM - 1) tile[TILE_DIM + 1][TILE_DIM + 1] = (x < width - 1 && y < height - 1) ? d_curr_matrix[(y + 1) * width + (x + 1)] : 0;
    }

    __syncthreads();

    // Process cell
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int numNeighbors = tile[ty - 1][tx - 1] + tile[ty - 1][tx] + tile[ty - 1][tx + 1] +
                           tile[ty][tx - 1] + tile[ty][tx + 1] +
                           tile[ty + 1][tx - 1] + tile[ty + 1][tx] + tile[ty + 1][tx + 1];

        int idx = y * width + x;
        if (tile[ty][tx] == 1) {
            if (numNeighbors < 2 || numNeighbors > 3) {
                d_next_matrix[idx] = 0;
                local_change_flag = 1;
            } else {
                d_next_matrix[idx] = 1;
            }
        } else {
            if (numNeighbors == 3) {
                d_next_matrix[idx] = 1;
                local_change_flag = 1;
            } else {
                d_next_matrix[idx] = 0;
            }
        }
    }

    __syncthreads();

    // Update global change flag
    if (threadIdx.x == 0 && threadIdx.y == 0 && local_change_flag) {
        atomicExch(d_change_flag, 1);
    }
}

// Game of Life function
void gameOfLife(char *grid, int width, int height, int steps) {
    char *d_curr_matrix, *d_next_matrix;
    int *d_change_flag;
    size_t size = width * height * sizeof(char);

    cudaMalloc(&d_curr_matrix, size);
    cudaMalloc(&d_next_matrix, size);
    cudaMalloc(&d_change_flag, sizeof(int));
    cudaMemcpy(d_curr_matrix, grid, size, cudaMemcpyHostToDevice);

    dim3 blockSize(TILE_DIM, TILE_DIM);
    dim3 gridSize((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);

    for (int step = 0; step < steps; ++step) {
        int change_flag = 0;
        cudaMemcpy(d_change_flag, &change_flag, sizeof(int), cudaMemcpyHostToDevice);

        updateKernel<<<gridSize, blockSize>>>(d_curr_matrix, d_next_matrix, width, height, d_change_flag);

        cudaMemcpy(&change_flag, d_change_flag, sizeof(int), cudaMemcpyDeviceToHost);
        if (change_flag == 0) break;

        std::swap(d_curr_matrix, d_next_matrix);
    }

    cudaMemcpy(grid, d_curr_matrix, size, cudaMemcpyDeviceToHost);
    cudaFree(d_curr_matrix);
    cudaFree(d_next_matrix);
    cudaFree(d_change_flag);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: ./a.exe <matrix size> <max generations> <output file path>\n");
        return 1;
    }
    size = atoi(argv[1]) + 2;
    max_gens = atoi(argv[2]);
    const char *output_file = argv[3];
    FILE *file = fopen(output_file, "w");

    curr_matrix = (char**)malloc(size * sizeof(char*));
    next_matrix = (char**)malloc(size * sizeof(char*));
    for (int x = 0; x < size; x++) {
        curr_matrix[x] = (char*)malloc(size * sizeof(char));
        next_matrix[x] = (char*)malloc(size * sizeof(char));
    }

    for (int x = 0; x < size; x++) {
        for (int y = 0; y < size; y++) {
            curr_matrix[x][y] = 0;
        }
    }

    for (int x = 1; x < size - 1; x++) {
        for (int y = 1; y < size - 1; y++) {
            curr_matrix[x][y] = rand() % 2;
        }
    }

    char *grid = new char[size * size];
    for (int x = 0; x < size; x++) {
        for (int y = 0; y < size; y++) {
            grid[x * size + y] = curr_matrix[x][y];
        }
    }

    double start_time = gettime();
    gameOfLife(grid, size, size, max_gens);
    double end_time = gettime();
    printf("Time taken: %lf seconds\n", end_time - start_time);

    for (int x = 0; x < size; x++) {
        for (int y = 0; y < size; y++) {
            curr_matrix[x][y] = grid[x * size + y];
        }
    }

    print_board(curr_matrix, size, file);

    for (int x = 0; x < size; x++) {
        free(curr_matrix[x]);
        free(next_matrix[x]);
    }
    free(curr_matrix);
    free(next_matrix);
    delete[] grid;
    return 0;
}

void print_board(char **matrix, int size, FILE *output_file) {
        for (int x = 1; x < size - 1; x++) {
            for (int y = 1; y < size - 1; y++) {
                fprintf(output_file, "%c ", matrix[x][y] ? 'O' : '.');
            }
            fprintf(output_file, "\n");
        }
        fprintf(output_file, "\n");
}

double gettime(void) {
    struct timeval tval;
    gettimeofday(&tval, NULL);
    return (double)tval.tv_sec + (double)tval.tv_usec / 1000000.0;
}
