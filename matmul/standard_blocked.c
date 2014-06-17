#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>


typedef struct block {
  short i;
  short j;
  double **data;
} block;


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

void add(block *a, block *b, block *c, int dim) {
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      c->data[c->i+i][c->j+j] = a->data[a->i+i][a->j+j] + b->data[b->i+i][b->j+j];
    }
  }
}

double ***alloc_temp_matrices(int _dim, int levels) {
  double ***ret = (double***)malloc(sizeof(double*)*levels);
  int dim = _dim/2;
  for (int i = 0; i < levels; i++) {
    ret[i] = random_data(dim);
    for (int j = 0; j < dim; j++) {
      for (int k = 0; k < dim; k++) {
        ret[i][j][k] = 0;
      }
    }
    dim = dim/2;
  }
  return ret;
}

static double ***g_tmp1;
static double ***g_tmp2;

void multiply(block *a, block *b, block *c, int limit, int dim, int level) {
  if (dim <= limit) {
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        c->data[c->i+i][c->j+j] = 0;
        for (int k = 0; k < dim; k++) {
          c->data[c->i+i][c->j+j] += a->data[a->i+i][a->j+k] * b->data[b->i+k][b->j+j];
        }
      }
    }
    return;
  }
  
  int bdim = dim / 2;
  
  block a11 = { a->i,      a->j,      a->data };
  block a12 = { a->i,      a->j+bdim, a->data };
  block a21 = { a->i+bdim, a->j     , a->data };
  block a22 = { a->i+bdim, a->j+bdim, a->data };
  
  block b11 = { b->i,      b->j,      b->data };
  block b12 = { b->i,      b->j+bdim, b->data };
  block b21 = { b->i+bdim, b->j,      b->data };
  block b22 = { b->i+bdim, b->j+bdim, b->data };
  
  block c11 = { c->i,      c->j,      c->data };
  block c12 = { c->i,      c->j+bdim, c->data };
  block c21 = { c->i+bdim, c->j,      c->data };
  block c22 = { c->i+bdim, c->j+bdim, c->data };
  
  block r = { 0, 0, g_tmp1[level] };
  block q = { 0, 0, g_tmp2[level] };
  
  //c11
  multiply(&a11, &b11, &q, limit, bdim, level+1);
  multiply(&a12, &b21, &r, limit, bdim, level+1);
  add(&q, &r, &c11, bdim);
  
  //c12
  multiply(&a11, &b12, &q, limit, bdim, level+1);
  multiply(&a12, &b22, &r, limit, bdim, level+1);
  add(&q, &r, &c12, bdim);
  
  //c21
  multiply(&a21, &b11, &q, limit, bdim, level+1);
  multiply(&a22, &b21, &r, limit, bdim, level+1);
  add(&q, &r, &c21, bdim);
  
  //c11
  multiply(&a21, &b12, &q, limit, bdim, level+1);
  multiply(&a22, &b22, &r, limit, bdim, level+1);
  add(&q, &r, &c22, bdim);
  
}


int main(int argc, char *argv[]) {
  srand(time(NULL));
  const int dim = atoi(argv[1]);
  const int limit = atoi(argv[2]);
  const int levels = log2(dim)-log2(limit);
  
  g_tmp1 = alloc_temp_matrices(dim, levels);
  g_tmp2 = alloc_temp_matrices(dim, levels);
  
  block a = { 0, 0, random_data(dim) };
  block b = { 0, 0, random_data(dim) };
  block c = { 0, 0, random_data(dim) };
  
  double start = MPI_Wtime();
  multiply(&a, &b, &c, limit, dim, 0);
  double end = MPI_Wtime();
  
  /*R*/ printf("%d,%f\n", levels, end-start);
}



