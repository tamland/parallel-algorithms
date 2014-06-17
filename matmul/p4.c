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

void swap(void **a, void **b) {
  void *tmp;
  tmp = *a;
  *a = *b;
  *b = tmp;
}

void print_block(block *a, int dim) {
  for (int i = 0; i < dim; i++) {
    printf("\n\t");
    for (int j = 0; j < dim; j++) {
      printf("%.0f  ", a->data[a->i+i][a->j+j]);
    }
  }
  printf("\n\n");
}

void add(block *a, block *b, block *c, int dim, int pdim, int p_i, int p_j) {
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      c->data[c->i+i][c->j+j] = a->data[a->i+i][a->j+j] + b->data[b->i+i][b->j+j];      
    }
  }
}

void sub(block *a, block *b, block *c, int dim, int pdim, int p_i, int p_j) {
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      c->data[c->i+i][c->j+j] = a->data[a->i+i][a->j+j] - b->data[b->i+i][b->j+j];      
    }
  }
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

double *g_recv_buf;

void cannon(int dim , int p_dim, int p_i, int p_j, double *a, double *b, double *c, double *recv_buf) {
  MPI_Status status;
  
  if (dim <= 0) {
    exit(1);
  }

  if (p_i > 0) {
    int dst_i = p_i;
    int dst_j = (p_j - p_i + p_dim) % p_dim;
    int src_i = p_i;
    int src_j = (p_j + p_i) % p_dim;

    MPI_Sendrecv(
      a,        dim*dim, MPI_DOUBLE, dst_i*p_dim+dst_j, 1,
      recv_buf, dim*dim, MPI_DOUBLE, src_i*p_dim+src_j, 1,
      MPI_COMM_WORLD, &status
    );

    swap(&a, &recv_buf);
  }
  if (p_j > 0) {
    int dst_i = (p_i - p_j + p_dim) % p_dim;
    int dst_j = p_j;
    int src_i = (p_i + p_j) % p_dim;
    int src_j = p_j;

    MPI_Sendrecv(
      b,        dim*dim, MPI_DOUBLE, dst_i*p_dim+dst_j, 2,
      recv_buf, dim*dim, MPI_DOUBLE, src_i*p_dim+src_j, 2,
      MPI_COMM_WORLD, &status
    );

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

    MPI_Sendrecv(
      a,        dim*dim, MPI_DOUBLE, dst, 3,
      recv_buf, dim*dim, MPI_DOUBLE, src, 3,
      MPI_COMM_WORLD, &status
    );

    swap(&a, &recv_buf);
    
    
    dst_i = (p_i - 1 + p_dim) % p_dim;
    dst_j = p_j;
    dst = dst_i*p_dim + dst_j;
    src_i = (p_i + 1) % p_dim;
    src_j = p_j;
    src = src_i*p_dim + src_j;

    MPI_Sendrecv(
      b,        dim*dim, MPI_DOUBLE, dst, 4,
      recv_buf, dim*dim, MPI_DOUBLE, src, 4,
      MPI_COMM_WORLD, &status
    );

    
    swap(&b, &recv_buf);
    multiply_local(a, b, c, dim);
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


double *a_seq, *b_seq, *c_seq, *buf_seq;

void multiply(block *a, block *b, block *c, int limit, int dim, int level, int pdim, int pi, int pj) {
  if (dim <= limit) {
    //divide into p blocks
    int ldim = dim / pdim;
    
    //copy block to sequential memory
    for (int i = 0; i < ldim; i++) {
      for (int j = 0; j < ldim; j++) {
        a_seq[i*ldim+j] = a->data[a->i+i][a->j+j];
        b_seq[i*ldim+j] = b->data[b->i+i][b->j+j];
        c_seq[i*ldim+j] = 0;
        buf_seq[i*ldim+j] = 0;
      }
    }
    
    cannon(ldim , pdim, pi, pj, a_seq, b_seq, c_seq, buf_seq);
    
    //copy result back to c block
    for (int i = 0; i < ldim; i++) {
      for (int j = 0; j < ldim; j++) {
        c->data[c->i+(pi*ldim)+i][c->j+(pj*ldim)+j] = c_seq[i*ldim+j];
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
  add(&a11, &a22, &r, bdim, pdim, pi, pj);
  add(&b11, &b22, &q, bdim, pdim, pi, pj);
  multiply(&r, &q, &m1, limit, bdim, level+1, pdim, pi, pj);
  
  //m2
  add(&a21, &a22, &r, bdim, pdim, pi, pj);
  multiply(&r, &b11, &m2, limit, bdim, level+1, pdim, pi, pj);
  
  //m3
  sub(&b12, &b22, &r, bdim, pdim, pi, pj);
  multiply(&a11, &r, &m3, limit, bdim, level+1, pdim, pi, pj);
  
  //m4
  sub(&b21, &b11, &r, bdim, pdim, pi, pj);
  multiply(&a22, &r, &m4, limit, bdim, level+1, pdim, pi, pj);
 
  //m5
  add(&a11, &a12, &r, bdim, pdim, pi, pj);
  multiply(&r, &b22, &m5, limit, bdim, level+1, pdim, pi, pj);
  
  //m6
  sub(&a21, &a11, &r, bdim, pdim, pi, pj);
  add(&b11, &b12, &q, bdim, pdim, pi, pj);
  multiply(&r, &q, &m6, limit, bdim, level+1, pdim, pi, pj);
  
  //m7
  sub(&a12, &a22, &r, bdim, pdim, pi, pj);
  add(&b21, &b22, &q, bdim, pdim, pi, pj);
  multiply(&r, &q, &m7, limit, bdim, level+1, pdim, pi, pj);
  
  //c11 = m1 + m4 - m5 + m7
  add(&m1, &m4, &c11, bdim, pdim, pi, pj);
  sub(&c11, &m5, &c11, bdim, pdim, pi, pj);
  add(&c11, &m7, &c11, bdim, pdim, pi, pj);

  //c12 = m3 + m5
  add(&m3, &m5, &c12, bdim, pdim, pi, pj);

  //c21 = m2 + m4
  add(&m2, &m4, &c21, bdim, pdim, pi, pj);
  
  //c22 = m1 - m2 + m3 + m6
  sub(&m1, &m2, &c22, bdim, pdim, pi, pj);
  add(&c22, &m3, &c22, bdim, pdim, pi, pj);
  add(&c22, &m6, &c22, bdim, pdim, pi, pj);
}


int main(int argc, char *argv[]) {
  srand(time(NULL));
  const int dim = atoi(argv[1]);
  const int limit = atoi(argv[2]);
  const int levels = log2(dim)-log2(limit);
  
  int pid;
  int np; 
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  
  int pdim = (int)sqrt(np);
  int pi = pid / pdim;
  int pj = pid % pdim;
  
  alloc_temp_matrices(dim, levels);
  int cannon_block = limit / pdim;
  a_seq = (double*)malloc(sizeof(double)*cannon_block*cannon_block);
  b_seq = (double*)malloc(sizeof(double)*cannon_block*cannon_block);
  c_seq = (double*)malloc(sizeof(double)*cannon_block*cannon_block);
  buf_seq = (double*)malloc(sizeof(double)*cannon_block*cannon_block);
  
  
  block a = { 0, 0, random_data(dim) };
  block b = { 0, 0, random_data(dim) };
  block c = { 0, 0, random_data(dim) };
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      c.data[i][j] = 0;
    }
  }
 
  double start = MPI_Wtime();
  multiply(&a, &b, &c, limit, dim, 0, pdim, pi, pj);
  double end = MPI_Wtime();
  
  if (dim <= 16) {
    MPI_Barrier(MPI_COMM_WORLD);
    int count = 0;
    while (count < np) {
      if (pid == count) {
        printf("p %d:\n", pid);
        print_block(&c, dim);
      }
      fflush(stdout);
      count++;
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }
  
  if (pid == 0) {
    /*R*/ printf("%d,%d,%d,%f\n", np, limit, levels, end-start);
  }
  
  MPI_Finalize();
  return 0;
}



