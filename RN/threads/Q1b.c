#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define N 10000000
#define NUM_THREADS 10

float *A,*B,*C;

typedef struct{
int start;
int end;
} ThreadData;

void* add_arrays(void* arg){
ThreadData* data = (ThreadData*)arg;
int start = data->start;
int end = data->end;

for(int i=start;i<end;i++){
C[i] = A[i]+B[i];
}
pthread_exit(NULL);
}


int main(){
A = (float *)malloc(N * sizeof(float));
B = (float *)malloc(N * sizeof(float));
C = (float *)malloc(N * sizeof(float));

if (A == NULL || B == NULL || C == NULL){
printf("Memory allocation failed!\n");
return 1;
}

for (size_t i = 0; i < N; i++){
A[i] = i*1.0f;
B[i] = i*2.0f;
}

pthread_t threads[NUM_THREADS];
ThreadData thread_data[NUM_THREADS];

int chunk_size = N / NUM_THREADS;

clock_t start = clock();

for(int i = 0; i < NUM_THREADS; i++){
thread_data[i].start = i * chunk_size;
thread_data[i].end = (i == NUM_THREADS - 1) ? N : (i + 1) * chunk_size;
pthread_create(&threads[i], NULL, add_arrays, &thread_data[i]);
}

for (int i = 0; i < NUM_THREADS; i++){
pthread_join(threads[i], NULL);
}

clock_t end = clock();
double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

printf("C[0] = %f\n",C[0]);
printf("C[1] = %f\n",C[1]);
printf("C[SIZE-1] = %f\n",C[N-1]);
printf("Parallel Execution Time: %f seconds\n", time_taken);

free(A);
free(B);
free(C);
return 0;
}
