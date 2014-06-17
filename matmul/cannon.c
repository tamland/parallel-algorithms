#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

typedef struct block {
  short i;
  short j;
  double *data;
} block;

void swap(void **a, void **b) {
  void *tmp;
  tmp = *a;
  *a = *b;
  *b = tmp;
}

void multiply_local(double *a, double *b, double *c, int dim) {
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      for (int k = 0; k < dim; k++) {
        c[i*dim+j] += a[i*dim+k] * b[k*dim+j];
      }
    }
  }
}

int main(int argc, char *argv[]) {
  srand(time(NULL));
  int pid;
  int np; 
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Status status;
  
  const int p_dim = (int)sqrt(np);
  const int dim = atoi(argv[1]) / p_dim;

  if (dim <= 0) {
    return 1;
  }
  
  int p_i = pid / p_dim;
  int p_j = pid % p_dim;
  
  double *a = (double*)malloc(sizeof(double)*dim*dim);
  double *b = (double*)malloc(sizeof(double)*dim*dim);
  double *c = (double*)malloc(sizeof(double)*dim*dim);
  double *recv_buf = (double*)malloc(sizeof(double)*dim*dim);

  for (int i = 0; i < dim*dim; i++) {
    a[i] = rand(); b[i] = rand(); c[i] = 0; recv_buf[i] = 0;
  }

  double start = MPI_Wtime();
  double start_comm = 0;
  double time_comm = 0;
  
  if (p_i > 0) {
    int dst_i = p_i;
    int dst_j = (p_j - p_i + p_dim) % p_dim;
    
    int src_i = p_i;
    int src_j = (p_j + p_i) % p_dim;
  
    start_comm = MPI_Wtime();
      MPI_Sendrecv(
        a,        dim*dim, MPI_DOUBLE, dst_i*p_dim+dst_j, 1,
        recv_buf, dim*dim, MPI_DOUBLE, src_i*p_dim+src_j, 1,
        MPI_COMM_WORLD, &status
      );
    time_comm += MPI_Wtime() - start_comm;

    swap(&a, &recv_buf);
  }

  if (p_j > 0) {
    int dst_i = (p_i - p_j + p_dim) % p_dim;
    int dst_j = p_j;
    
    int src_i = (p_i + p_j) % p_dim;
    int src_j = p_j;

    start_comm = MPI_Wtime();
      MPI_Sendrecv(
        b,        dim*dim, MPI_DOUBLE, dst_i*p_dim+dst_j, 2,
        recv_buf, dim*dim, MPI_DOUBLE, src_i*p_dim+src_j, 2,
        MPI_COMM_WORLD, &status
      );
    time_comm += MPI_Wtime() - start_comm;
    
    swap(&b, &recv_buf);
  }
 
  multiply_local(a, b, c, dim);
  
  for (int _i = 0; _i < p_dim-1; _i++) {
    int dst_i = p_i;
    int dst_j = (p_j - 1 + p_dim) % p_dim;
    int dst = dst_i*p_dim + dst_j;
    
    int src_i = p_i;
    int src_j = (p_j + 1) % p_dim;
    int src = src_i*p_dim + src_j;
    
    start_comm = MPI_Wtime();
      MPI_Sendrecv(
        a,        dim*dim, MPI_DOUBLE, dst, 3,
        recv_buf, dim*dim, MPI_DOUBLE, src, 3,
        MPI_COMM_WORLD, &status
      );
    time_comm += MPI_Wtime() - start_comm;
    
    swap(&a, &recv_buf);
    
    
    dst_i = (p_i - 1 + p_dim) % p_dim;
    dst_j = p_j;
    dst = dst_i*p_dim + dst_j;
    
    src_i = (p_i + 1) % p_dim;
    src_j = p_j;
    src = src_i*p_dim + src_j;

    start_comm = MPI_Wtime();
      MPI_Sendrecv(
        b,        dim*dim, MPI_DOUBLE, dst, 4,
        recv_buf, dim*dim, MPI_DOUBLE, src, 4,
        MPI_COMM_WORLD, &status
      );
    time_comm += MPI_Wtime() - start_comm;
    
    swap(&b, &recv_buf);

    multiply_local(a, b, c, dim);
  }
  
  double time = MPI_Wtime() - start;
  double time_comp = time - time_comm;
  if (pid == 0) {
    /*R*/ printf("%d,%d,%f,%f,%f\n", np, p_dim, time_comp, time_comm, time);
  }
  
  MPI_Finalize();
  return 0;
}
