#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <string.h>

int size;
int max_gens;
int change_flag;
char **curr_matrix;
char **next_matrix;

void print();
void print_matrix(char **matrix);
void update_matrix();
int update_cell(int x, int y);
int count_alive_neighbors(int x, int y);
double gettime();

int main(int argc, char **argv) {
    //Parse args
    if (argc < 3) {
        printf("Usage: ./a.exe <matrix size> <max generations>\n");
        return 1;
    }
    size = atoi(argv[1]) + 2;
    max_gens = atoi(argv[2]);

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
    for (int x = 0; x < max_gens; x++) {
        // print();
        update_matrix();
        if (change_flag == 0) {
            break;
        }
        change_flag = 0;

        //Change matrix pointers
        temp = curr_matrix;
        curr_matrix = next_matrix;
        next_matrix = temp;
    }
    print();
    double end_time = gettime();
    //  printf("Time taken: %lf seconds\n", end_time - start_time);

    //Free memory
    for (int x = 0; x < size; x++) {
        free(curr_matrix[x]);
        free(next_matrix[x]);
    }
    free(curr_matrix);
    free(next_matrix);
    return 0;
}

void update_matrix() {
    for (int x = 1; x < size-1; x++) {
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
    for (int x = 1; x < size - 1; x++) {
        for (int y = 1; y < size - 1; y++) {
            if (curr_matrix[x][y] == 0) {
                printf(". ");
            }
            else {
                printf("O ");
            }
        }
        printf("\n");
    }
    printf("\n");
}

double gettime(void) {
  struct timeval tval;

  gettimeofday(&tval, NULL);

  return( (double)tval.tv_sec + (double)tval.tv_usec/1000000.0 );
}