/*  
Name:   Nolan McKivergan
Email:  npmckivergan@crimson.ua.edu
Course: CS 581
Homework #: 5
Instructions to compile the program: gcc openmp.c -O3 -fopenmp -o openmp
Instructions to run the program: ./openmp <grid size> <max gens> <output file path>
*/ 

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <omp.h>

int size;
int max_gens;
int thread_count;
int change_flag;
char **curr_matrix;
char **next_matrix;

void print();
void print_to_file(FILE *file);
void print_matrix(char **matrix);
void update_matrix(int start_column, int end_column);
int update_cell(int x, int y);
int count_alive_neighbors(int x, int y);
double gettime();

int main(int argc, char **argv) {
    //Parse args
    if (argc < 4) {
        printf("Usage: ./hw3 <matrix size> <max generations> <number of threads> <output file path (optional)>\n");
        return 1;
    }
    size = atoi(argv[1]) + 2;
    max_gens = atoi(argv[2]);
    thread_count = atoi(argv[3]);
    const char *output_file;
    if (argc > 4) {
        output_file = argv[4];
    }

    //Allocate matrices
    curr_matrix = (char**)malloc(size * sizeof(char*));
    next_matrix = (char**)malloc(size * sizeof(char*));
    for (int x = 0; x < size; x++) {
        curr_matrix[x] = (char*)malloc(size * sizeof(char));
        next_matrix[x] = (char*)malloc(size * sizeof(char));
    }

    //Zero matrix
    for (int x = 0; x < size; x++) {
        for (int y = 0; y < size; y++) {
            curr_matrix[x][y] = 0;
            next_matrix[x][y] = 0;
        }
    }

    // Initialize random values
    for (int x = 1; x < size - 1; x++) {
        for (int y = 1; y < size - 1; y++) {
            curr_matrix[x][y] = rand() % 2;
        }
    }

    // Code to run game max gen times
    change_flag = 0;
    char **temp;
    double start_time = gettime();
    int tid;

    //Start of parallel region
    #pragma omp parallel num_threads(thread_count) default(none) \
        private(tid)  shared(size, max_gens, curr_matrix, next_matrix, change_flag, thread_count, temp)
        {
            //Partition matrix by columns and assign to threads
            tid = omp_get_thread_num();
            int partition = (size - 2) / thread_count;
            int start_column = tid * partition;
            int end_column = start_column + partition;
            int remainder = (size - 2) % thread_count;
            if (tid == thread_count - 1) {
                end_column += remainder;
            }

            //Iterate
            for (int x = 0; x < max_gens; x++) {
                //Debug code to print board at the start of each generation
                // #pragma omp single
                // {
                //     print();
                // }

                //Update columns assigned to thread
                update_matrix(start_column, end_column);

                //Now wait for all threads before proceeding
                #pragma omp barrier

                //Only needs to be executed by one thread
                #pragma omp single
                {
                    //Break if no change in board or max gens have occured
                    if (change_flag == 0 || x == max_gens - 1) {
                        // printf("Exited after %d iterations\n", x+1);
                        x = max_gens;
                    }
                    change_flag = 0;

                    //Change matrix pointers
                    temp = curr_matrix;
                    curr_matrix = next_matrix;
                    next_matrix = temp;
                }
            }
        }

    //Calculate total time and print
    double end_time = gettime();
    printf("Time taken: %lf seconds\n", end_time - start_time);

    //Write final board to a file
    if (argc > 4) {
        FILE *file = fopen(output_file, "w");
        print_to_file(file);
    }

    //Free memory
    for (int x = 0; x < size; x++) {
        free(curr_matrix[x]);
        free(next_matrix[x]);
    }
    free(curr_matrix);
    free(next_matrix);
    return 0;
}

void update_matrix(int start_column, int end_column) {
    for (int x = start_column + 1; x < end_column + 1; x++) {
        for (int y = 1; y < size-1; y++) {
            if (update_cell(x,y) == 1) {
                change_flag = 1;
            }
        }
    }
}

int update_cell(int x, int y) {
    //Game rules
    int alive_neighbors = count_alive_neighbors(x,y);
    if (curr_matrix[x][y] == 1) {
        if (alive_neighbors < 2) {
            next_matrix[x][y] = 0;
            return 1;
        }
        else if (alive_neighbors > 3) {
            next_matrix[x][y] = 0;
            return 1;
        }
        else {
            next_matrix[x][y] = 1;
        }
    }
    else if (curr_matrix[x][y] == 0) {
        if (alive_neighbors == 3) {
            next_matrix[x][y] = 1;
            return 1;
        }
        else {
            next_matrix[x][y] = 0;
        }
    }
    return 0;
}

int count_alive_neighbors(int x, int y) {
    int sum = 0;
    //Clockwise from top

    //Top
    sum += curr_matrix[x][y+1];
    //Top right
    sum += curr_matrix[x+1][y+1];
    //Right
    sum += curr_matrix[x+1][y];
    //Bottom right
    sum += curr_matrix[x+1][y-1];
    //Bottom
    sum += curr_matrix[x][y-1];
    //Bottom left
    sum += curr_matrix[x-1][y-1];
    //Left
    sum += curr_matrix[x-1][y];
    //Top left
    sum += curr_matrix[x-1][y+1];
    return sum;
}

void print() {
    for (int x = 0; x < size; x++) {
        for (int y = 0; y < size; y++) {
            printf("%d  ", curr_matrix[x][y]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_to_file(FILE *file) {
    for (int x = 1; x < size - 1; x++) {
        for (int y = 1; y < size - 1; y++) {
            if (curr_matrix[x][y] == 0) {
                fprintf(file, ". ");
            }
            else {
                fprintf(file, "O ");
            }
        }
        fprintf(file, "\n");
    }
}

double gettime(void) {
  struct timeval tval;

  gettimeofday(&tval, NULL);

  return( (double)tval.tv_sec + (double)tval.tv_usec/1000000.0 );
}
