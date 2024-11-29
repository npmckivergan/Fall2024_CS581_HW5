/*  
Name:   Nolan McKivergan
Email:  npmckivergan@crimson.ua.edu
Course: CS 581
Homework #: 5
Instructions to compile the program: nvcc gpu_global.cu -O3 -o gpu_global
Instructions to run the program: ./gpu_global <grid size> <max gens> <output file path>
*/ 

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>

// Function to print the current board state
void print_board(char *matrix, int size, int generation, int print_all, FILE *output_file) {
    if (print_all || generation == -1) {
        // printf("Generation %d:\n", generation);
        for (int x = 1; x < size - 1; x++) {
            for (int y = 1; y < size - 1; y++) {
                fprintf(output_file, "%c ", matrix[x * size + y] ? 'O' : '.');
            }
            fprintf(output_file, "\n");
        }
        fprintf(output_file, "\n");
    }
}

int size;
int max_gens;
int change_flag_host;
char *d_curr_matrix;
char *d_next_matrix;

__device__ int count_alive_neighbors(char *matrix, int x, int y, int size) {
    int sum = 0;
    sum += matrix[(x * size) + (y + 1)];
    sum += matrix[((x + 1) * size) + (y + 1)];
    sum += matrix[((x + 1) * size) + y];
    sum += matrix[((x + 1) * size) + (y - 1)];
    sum += matrix[(x * size) + (y - 1)];
    sum += matrix[((x - 1) * size) + (y - 1)];
    sum += matrix[((x - 1) * size) + y];
    sum += matrix[((x - 1) * size) + (y + 1)];
    return sum;
}

__global__ void update_matrix_kernel(char *curr_matrix, char *next_matrix, int size, int *change_flag) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int y = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (x < size - 1 && y < size - 1) {
        int alive_neighbors = count_alive_neighbors(curr_matrix, x, y, size);
        int index = x * size + y;

        if (curr_matrix[index] == 1) {
            if (alive_neighbors < 2 || alive_neighbors > 3) {
                next_matrix[index] = 0;
                *change_flag = 1;
            } else {
                next_matrix[index] = 1;
            }
        } else {
            if (alive_neighbors == 3) {
                next_matrix[index] = 1;
                *change_flag = 1;
            } else {
                next_matrix[index] = 0;
            }
        }
    }
}

double gettime(void) {
    struct timeval tval;
    gettimeofday(&tval, NULL);
    return ((double)tval.tv_sec + (double)tval.tv_usec / 1000000.0);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: ./game_of_life <matrix size> <max generations> <output file path>\n");
        return 1;
    }
    size = atoi(argv[1]) + 2;
    max_gens = atoi(argv[2]);
    const char *output_file = argv[3];
    FILE *file = fopen(output_file, "w");

    // Allocate host matrices
    char *h_curr_matrix = (char*)malloc(size * size * sizeof(char));
    char *h_next_matrix = (char*)malloc(size * size * sizeof(char));

    // Zero matrices and initialize random values
    for (int i = 0; i < size * size; i++) {
        h_curr_matrix[i] = 0;
    }
    for (int x = 1; x < size - 1; x++) {
        for (int y = 1; y < size - 1; y++) {
            h_curr_matrix[x * size + y] = rand() % 2;
        }
    }

    // Allocate device matrices
    cudaMalloc(&d_curr_matrix, size * size * sizeof(char));
    cudaMalloc(&d_next_matrix, size * size * sizeof(char));
    int *d_change_flag;
    cudaMalloc(&d_change_flag, sizeof(int));

    cudaMemcpy(d_curr_matrix, h_curr_matrix, size * size * sizeof(char), cudaMemcpyHostToDevice);

    dim3 block_size(16, 16);
    dim3 grid_size((size - 2 + block_size.x - 1) / block_size.x, (size - 2 + block_size.y - 1) / block_size.y);

    double start_time = gettime();
    for (int gen = 0; gen <= max_gens; gen++) {
        change_flag_host = 0;
        cudaMemcpy(d_change_flag, &change_flag_host, sizeof(int), cudaMemcpyHostToDevice);

        update_matrix_kernel<<<grid_size, block_size>>>(d_curr_matrix, d_next_matrix, size, d_change_flag);
        cudaDeviceSynchronize();

        // Copy current matrix back to the host for printing
        cudaMemcpy(h_curr_matrix, d_curr_matrix, size * size * sizeof(char), cudaMemcpyDeviceToHost);

        cudaMemcpy(&change_flag_host, d_change_flag, sizeof(int), cudaMemcpyDeviceToHost);
        if (change_flag_host == 0) {
            printf("No changes detected. Ending simulation early.\n");
            break;
        }

        // Swap the matrices
        char *temp = d_curr_matrix;
        d_curr_matrix = d_next_matrix;
        d_next_matrix = temp;
    }
    double end_time = gettime();

    // Print the final board
    print_board(h_curr_matrix, size, -1, 0, file);

    printf("Time taken: %lf seconds\n", end_time - start_time);

    // Free memory
    free(h_curr_matrix);
    free(h_next_matrix);
    cudaFree(d_curr_matrix);
    cudaFree(d_next_matrix);
    cudaFree(d_change_flag);

    return 0;
}
