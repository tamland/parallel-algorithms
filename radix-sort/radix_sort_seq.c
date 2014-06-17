#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <limits.h>
#include <stdint.h>

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

int main(int argc, char *argv[]) {
  const int n = atoi(argv[1]);
  const int bits_per_bucket = atoi(argv[2]);
  const int total_bits = sizeof(uint64)*8;
  
  const int n_buckets = 1 << bits_per_bucket;
  const int mask = ~((~0) << bits_per_bucket);

  uint64 *a =  random_data(n);
  uint64 *a_sorted =  (uint64*)malloc(sizeof(uint64)*n);

  int *count = (int*)malloc(sizeof(int)*n_buckets);
  int *prefix_sum = (int*)malloc(sizeof(int)*n_buckets);

  float mt1 = omp_get_wtime();
  
  for (int offset = 0; offset < total_bits; offset += bits_per_bucket) {
    for (int i = 0; i < n_buckets; i++) {
      count[i] = 0;
    }
    
    //count number of element that goes in each bucket
    for (int i = 0; i < n; i++) {
      count[ get_bucket_num(a[i], offset, mask) ]++;
    }
    
    //calculate the prefix sum i.e. where each bucket starts
    prefix_sum[0] = 0;
    for (int i = 1; i < n_buckets; i++) {
      prefix_sum[i] = prefix_sum[i-1] + count[i-1];
    }
    
    //put the elements in the right position using the prefix_sum array to store the next index
    for (int i = 0; i < n; i++) {
      int *pos = &prefix_sum[ get_bucket_num(a[i], offset, mask) ];
      a_sorted[*pos] = a[i];
      (*pos)++; 
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

  /*R*/ printf("%d,%d,%d,%f\n", 1, n, bits_per_bucket, time);
}

