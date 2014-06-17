#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

double **random_data(int dim) {
  double *data = (double*)malloc(sizeof(double)*dim*dim);
  double **ret = (double**)malloc(sizeof(double*)*dim);
  for (int i = 0; i < dim; i++) {
    ret[i] = data + (i * dim);
  }
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      ret[i][j] = rand();
    }
  }
  return ret;
}


int main(int argc, char *argv[]) {
  srand(time(NULL));
  const int dim = atoi(argv[1]);
  
  double **a = random_data(dim);
  double **b = random_data(dim);
  double **c = random_data(dim);
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      c[i][j] = 0;
    }
  }
  
  double start = MPI_Wtime();
  
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      for (int k = 0; k < dim; k++) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  
  double end = MPI_Wtime();
  
  printf("%d,%f\n", dim*dim, end-start);
}

