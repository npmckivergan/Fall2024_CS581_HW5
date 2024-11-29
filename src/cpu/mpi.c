/* 
Fall 2024: CS 581 High Performance Computing 
Homework-4
Name: Nolan McKivergan
Email: npmckivergan@crimson.ua.edu
Course Section: CS 581-001
Homework #: 4
Instructions to compile the program: mpicc -O3 -o hw4_1 hw4_1.c
Instructions to run the program: mpiexec -n <number of processes> hw4_1 <grid size> <max generations> <number of processes> <output file path>
*/ 

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <sys/time.h>

#define DIES   0
#define ALIVE  1

void Check_for_error(int local_ok, char fname[], char message[], MPI_Comm comm);
void Allocate_vectors_for_life(int** local_x_pp, int local_n, int n, MPI_Comm comm);
void Read_vector(int local_a[], int *counts, int *displs, int n, char vec_name[], int my_rank, MPI_Comm comm);
void compute_local(int local_x[], int n, int counts[], int my_rank, int comm_sz, MPI_Comm comm);
double gettime(void);
void printarray_1d(int *a, int N, int k, const char *output_file_path);
int **allocarray(int P, int Q);
void freearray(int **a);
void print_life(int **life, int nRowsGhost, int nColsGhost, int my_rank, int local_n);
int compute(int **life, int **temp, int nRows, int nCols);
int max_gens;
const char *output_file;

int main(int argc, char **argv) {
    int n, local_n, i, remain;
    int comm_sz, my_rank, *counts;
    int *local_x;
    MPI_Comm comm;
    int *displs;

    MPI_Init(NULL, NULL);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &comm_sz);
    MPI_Comm_rank(comm, &my_rank);

    n = atoi(argv[1]);
    max_gens = atoi(argv[2]);
    output_file = argv[4];

    /* compute counts and displacements */
    counts = (int *)malloc(comm_sz * sizeof(int));
    displs = (int *)malloc(comm_sz * sizeof(int));
    remain = n % comm_sz;
    for (i = 0; i < comm_sz; i++) {
        counts[i] = n / comm_sz + ((i < remain) ? 1 : 0);
        counts[i] = counts[i] * n;
    }
    displs[0] = 0;
    for (i = 1; i < comm_sz; i++) {
        displs[i] = displs[i - 1] + counts[i - 1];
    }
    local_n = counts[my_rank];

    Allocate_vectors_for_life(&local_x, local_n, n, comm);
    Read_vector(local_x, counts, displs, n, "x", my_rank, comm);
    compute_local(local_x, n, counts, my_rank, comm_sz, comm);

    // Gather the final board state
    int *final_board = NULL;
    if (my_rank == 0) {
        final_board = (int *)malloc(n * n * sizeof(int));
    }
    MPI_Gatherv(local_x, local_n, MPI_INT, final_board, counts, displs, MPI_INT, 0, comm);

    if (my_rank == 0) {
        printarray_1d(final_board, n, max_gens, output_file);
        free(final_board);
    }

    free(local_x);
    free(counts);
    free(displs);

    MPI_Finalize();
    return 0;
}

void Check_for_error(int local_ok, char fname[], char message[], MPI_Comm comm) {
    int ok;
    MPI_Allreduce(&local_ok, &ok, 1, MPI_INT, MPI_MIN, comm);
    if (ok == 0) {
        int my_rank;
        MPI_Comm_rank(comm, &my_rank);
        if (my_rank == 0) {
            // fprintf(stderr, "Proc %d > In %s, %s\n", my_rank, fname, message);
            fflush(stderr);
        }
        MPI_Finalize();
        exit(-1);
    }
}

void Allocate_vectors_for_life(int** local_x_pp, int local_n, int n, MPI_Comm comm) {
    int local_ok = 1;
    char* fname = "Allocate_vectors";
    *local_x_pp = malloc(local_n * sizeof(int));
    if (*local_x_pp == NULL) local_ok = 0;
    Check_for_error(local_ok, fname, "Can't allocate local vector(s)", comm);
}

void Read_vector(int local_a[], int *counts, int *displs, int n, char vec_name[], int my_rank, MPI_Comm comm) {
    int* a = NULL;
    int i, local_n;
    int local_ok = 1;
    char* fname = "Read_vector";

    local_n = counts[my_rank];
    if (my_rank == 0) {
        a = malloc(n * n * sizeof(int));
        if (a == NULL) local_ok = 0;
        Check_for_error(local_ok, fname, "Can't allocate temporary vector", comm);
        for (i = 0; i < n * n; i++) {
            a[i] = rand() % 2;
        }
        MPI_Scatterv(a, counts, displs, MPI_INT, local_a, local_n, MPI_INT, 0, comm);
        free(a);
    } else {
        Check_for_error(local_ok, fname, "Can't allocate temporary vector", comm);
        MPI_Scatterv(a, counts, displs, MPI_INT, local_a, local_n, MPI_INT, 0, comm);
    }
}

void compute_local(int local_x[], int n, int counts[], int my_rank, int comm_sz, MPI_Comm comm) {
    int i, j, local_n;
    int **life = NULL, **temp = NULL, **ptr;
    local_n = counts[my_rank];
    int nCols = n;
    int nRows = local_n / n;
    int upper_rank, down_rank;
    double t1, t2;
    int flag = 1, k;

    MPI_Status status;

    int nRowsGhost = nRows + 2;
    int nColsGhost = nCols + 2;
    life = allocarray(nRowsGhost, nColsGhost);
    temp = allocarray(nRowsGhost, nColsGhost);

    int row = 0;
    int col = 0;

    /* Initialize the boundaries of the life matrix */
    for (i = 0; i < nRowsGhost; i++) {
        for (j = 0; j < nColsGhost; j++) {
            if (i == 0 || j == 0 || i == nRowsGhost - 1 || j == nColsGhost - 1) {
                life[i][j] = DIES;
                temp[i][j] = DIES;
            }
        }
    }

    for (i = 0; i < local_n; i++) {
        row = i / n;
        col = i % n;
        row = row + 1;
        col = col + 1;
        life[row][col] = local_x[i];
    }

    upper_rank = my_rank + 1;
    if (upper_rank >= comm_sz) upper_rank = MPI_PROC_NULL;
    down_rank = my_rank - 1;
    if (down_rank < 0) down_rank = MPI_PROC_NULL;
    int NTIMES = max_gens;
    if (my_rank == 0) {
        t1 = gettime();
    }

    /* Play the game of life for given number of iterations */
    for (k = 0; k < NTIMES; k++) {
        flag = 0;

        if ((my_rank % 2) == 0) {
            MPI_Sendrecv(&(life[nRows][0]), nColsGhost, MPI_INT, upper_rank, 0,
                         &(life[nRows + 1][0]), nColsGhost, MPI_INT, upper_rank, 0,
                         comm, &status);
        } else {
            MPI_Sendrecv(&(life[1][0]), nColsGhost, MPI_INT, down_rank, 0,
                         &(life[0][0]), nColsGhost, MPI_INT, down_rank, 0,
                         comm, &status);
        }

        if ((my_rank % 2) == 1) {
            MPI_Sendrecv(&(life[nRows][0]), nColsGhost, MPI_INT, upper_rank, 1,
                         &(life[nRows + 1][0]), nColsGhost, MPI_INT, upper_rank, 1,
                         comm, &status);
        } else {
            MPI_Sendrecv(&(life[1][0]), nColsGhost, MPI_INT, down_rank, 1,
                         &(life[0][0]), nColsGhost, MPI_INT, down_rank, 1,
                         comm, &status);
        }

        flag = compute(life, temp, nRows, nCols);

        int reduction_flag = 0;
        MPI_Allreduce(&flag, &reduction_flag, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        if (my_rank == 0) {
            if (!reduction_flag) {
                // printf("The sum of all flag is %d after k=%d.\n", reduction_flag, k);
            }
        }
        if (!reduction_flag) {
            break;
        }

        MPI_Barrier(comm);

        ptr = life;
        life = temp;
        temp = ptr;
    }
    if (my_rank == 0) {
        t2 = gettime();
        printf("Completed after %d generations\n", k);
        printf("Elapsed time: %f seconds\n", t2 - t1);
    }

    // Copy the final state back to local_x
    for (i = 1; i <= nRows; i++) {
        for (j = 1; j <= nCols; j++) {
            local_x[(i - 1) * n + (j - 1)] = life[i][j];
        }
    }

    freearray(life);
    freearray(temp);
}

double gettime(void) {
    struct timeval tval;
    gettimeofday(&tval, NULL);
    return ((double)tval.tv_sec + (double)tval.tv_usec / 1000000.0);
}

void printarray_1d(int *a, int N, int k, const char *output_file_path) {
    FILE *file = fopen(output_file_path, "w");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(file, "%d ", a[i * N + j]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

int **allocarray(int P, int Q) {
    int i, *p, **a;
    p = (int *)malloc(P * Q * sizeof(int));
    a = (int **)malloc(P * sizeof(int *));
    for (i = 0; i < P; i++) {
        a[i] = &p[i * Q];
    }
    return a;
}

void freearray(int **a) {
    free(&a[0][0]);
    free(a);
}

int compute(int **life, int **temp, int nRows, int nCols) {
    int i, j, value, flag = 0;
    for (i = 1; i < nRows + 1; i++) {
        for (j = 1; j < nCols + 1; j++) {
            value = life[i - 1][j - 1] + life[i - 1][j] + life[i - 1][j + 1] +
                    life[i][j - 1] + life[i][j + 1] +
                    life[i + 1][j - 1] + life[i + 1][j] + life[i + 1][j + 1];
            if (life[i][j]) {
                if (value < 2 || value > 3) {
                    temp[i][j] = DIES;
                    flag++;
                } else {
                    temp[i][j] = ALIVE;
                }
            } else {
                if (value == 3) {
                    temp[i][j] = ALIVE;
                    flag++;
                } else {
                    temp[i][j] = DIES;
                }
            }
        }
    }
    return flag;
}
