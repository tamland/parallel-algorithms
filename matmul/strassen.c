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

void sub(block *a, block *b, block *c, int dim) {
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      c->data[c->i+i][c->j+j] = a->data[a->i+i][a->j+j] - b->data[b->i+i][b->j+j];
    }
  }
}

static double ***g_tmp;

void alloc_temp_matrices(int dim, int levels) {
  g_tmp = (double***)malloc(sizeof(double*)*levels);
  int tmp_dim = dim/2;
  for (int i = 0; i < levels; i++) {
    g_tmp[i] = random_data(tmp_dim*3);
    
    for (int j = 0; j < tmp_dim*3; j++) {
      for (int k = 0; k < tmp_dim*3; k++) {
        g_tmp[i][j][k] = 0;
      }
    }
    tmp_dim = tmp_dim/2;
  }
}

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
  
  
  block m1 = { 0, 0,         g_tmp[level] };
  block m2 = { 0, bdim,      g_tmp[level] };
  block m3 = { 0, bdim+bdim, g_tmp[level] };
  block m4 = { bdim, 0,         g_tmp[level] };
  block m5 = { bdim, bdim,      g_tmp[level] };
  block m6 = { bdim, bdim+bdim, g_tmp[level] };
  block m7 = { bdim+bdim, 0,         g_tmp[level] };
  block r  = { bdim+bdim, bdim,      g_tmp[level] };
  block q  = { bdim+bdim, bdim+bdim, g_tmp[level] };
  
  
  //m1
  add(&a11, &a22, &r, bdim);
  add(&b11, &b22, &q, bdim);
  multiply(&r, &q, &m1, limit, bdim, level+1);
  
  //m2
  add(&a21, &a22, &r, bdim);
  multiply(&r, &b11, &m2, limit, bdim, level+1);
  
  
  //m3
  sub(&b12, &b22, &r, bdim);
  multiply(&a11, &r, &m3, limit, bdim, level+1);
  
  //m4
  sub(&b21, &b11, &r, bdim);
  multiply(&a22, &r, &m4, limit, bdim, level+1);
  
  //m5
  add(&a11, &a12, &r, bdim);
  multiply(&r, &b22, &m5, limit, bdim, level+1);
  
  //m6
  sub(&a21, &a11, &r, bdim);
  add(&b11, &b12, &q, bdim);
  multiply(&r, &q, &m6, limit, bdim, level+1);
  
  //m7
  sub(&a12, &a22, &r, bdim);
  add(&b21, &b22, &q, bdim);
  multiply(&r, &q, &m7, limit, bdim, level+1);
  
  //c11 = m1 + m4 - m5 + m7
  add(&m1, &m4, &c11, bdim);
  sub(&c11, &m5, &c11, bdim);
  add(&c11, &m7, &c11, bdim);
  
  //c12 = m3 + m5
  add(&m3, &m5, &c12, bdim);

  //c21 = m2 + m4
  add(&m2, &m4, &c21, bdim);
  
  //c22 = m1 - m2 + m3 + m6
  sub(&m1, &m2, &c22, bdim);
  add(&c22, &m3, &c22, bdim);
  add(&c22, &m6, &c22, bdim);
  
}


int main(int argc, char *argv[]) {
  srand(time(NULL));
  const int dim = atoi(argv[1]);
  const int limit = atoi(argv[2]);
  const int levels = log2(dim)-log2(limit);
  
  alloc_temp_matrices(dim, levels);
  
  block a = { 0, 0, random_data(dim) };
  block b = { 0, 0, random_data(dim) };
  block c = { 0, 0, random_data(dim) };
  
  double start = MPI_Wtime();
  multiply(&a, &b, &c, limit, dim, 0);
  double end = MPI_Wtime();
  
  /*R*/ printf("%d,%f\n", levels, end-start);
}



