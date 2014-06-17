#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <limits.h>
#include <stdint.h>
#include <math.h>

typedef unsigned long long int uint64;

uint64* random_data(int size) {
  srand(time(NULL));
  uint64 *ret = (uint64*)malloc(sizeof(uint64)*size);
  for (int i = 0; i < size; i++) {
    uint64 a = ((uint64)rand()) << 33; //first 31 bits
    uint64 b = ((uint64)rand()) << 2; //next 31
    uint64 c = rand() % 4; //last 2
    ret[i] = a | b | c;
  }
  return ret;
}

int get_bucket_num(uint64 x, int offset, int mask) {
  return (x >> offset) & mask;
}

int min(int a, int b) {
  if (a > b) return b;
  return a;
}

int main(int argc, char *argv[]) {
  const int n = atoi(argv[1]);
  const int bits_per_bucket = atoi(argv[2]);
  const int total_bits = sizeof(uint64)*8;
  
  const int n_buckets = 1 << bits_per_bucket;
  const int mask = ~((~0) << bits_per_bucket);

  uint64 *a = random_data(n);
  uint64 *a_sorted =  (uint64*)malloc(sizeof(uint64)*n);


  int num_th;
  float mt1;
  
  #pragma omp parallel
  #pragma omp single
  num_th = omp_get_num_threads();
  
  
  int **count = (int**)malloc(sizeof(int*)*num_th);
  int **prefix_sum = (int**)malloc(sizeof(int*)*num_th);
  for (int i = 0; i < num_th; i++) {
    count[i] = (int*)malloc(sizeof(int)*n_buckets);
    prefix_sum[i] = (int*)malloc(sizeof(int)*n_buckets);
  }
  
  mt1 = omp_get_wtime();
  
  int chunk_size = n > num_th ? ceil(n/(double)num_th) : n;
  
  for (int offset = 0; offset < total_bits; offset += bits_per_bucket) {
    prefix_sum[0][0] = 0;
    
    #pragma omp parallel
    {
      int pid = omp_get_thread_num();
      int start = pid * chunk_size;
      int end = min(start + chunk_size, n);
      
      for (int bucket = 0; bucket < n_buckets; bucket++) {
        count[pid][bucket] = 0;
      }
      
      for (int i = start; i < end; i++) {
        count[pid][ get_bucket_num(a[i], offset, mask) ]++;
      }
      
      #pragma omp barrier
      #pragma omp single
      for (int bucket = 0; bucket < n_buckets; bucket++) {
        if (bucket > 0)
          prefix_sum[0][bucket] = prefix_sum[num_th-1][bucket-1] + count[num_th-1][bucket-1];
        
        for (int th = 1; th < num_th; th++) {
          prefix_sum[th][bucket] = prefix_sum[th-1][bucket] + count[th-1][bucket];
        }
      }
      #pragma omp barrier
      
      for (int i = start; i < end; i++) {
        int *pos = &prefix_sum[pid][get_bucket_num(a[i], offset, mask)];
        a_sorted[*pos] = a[i];
        (*pos)++;
      }
    }
    
    uint64 *tmp = a;
    a = a_sorted;
    a_sorted = tmp;
  }
  
  uint64 *tmp = a;
  a = a_sorted;
  a_sorted = tmp;
  
  
  float mt2 = omp_get_wtime();
  float time = mt2-mt1;

  for (int i = 1; i < n; i++) {
    if (a_sorted[i-1] > a_sorted[i]) {
      printf("not sorted");
      return 1;
    }
  }
  
  /*R*/ printf("%d,%d,%d,%f\n", num_th, n, bits_per_bucket, time);
}

