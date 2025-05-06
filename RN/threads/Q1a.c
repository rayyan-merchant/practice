#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define N 10000000  

int main(){
float *A, *B, *C;
A = (float *)malloc(N * sizeof(float));
B = (float *)malloc(N * sizeof(float));
C = (float *)malloc(N * sizeof(float));


if (A == NULL || B == NULL || C == NULL){
    printf("Memory allocation failed!\n");
    return 1;
}

clock_t start = clock();

for (int i = 0; i < N; i++) {
    A[i] = i*1.0f;
    B[i] = i*2.0f;
}

clock_t end = clock();
double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

printf("C[0] = %f\n",C[0]);
printf("C[1] = %f\n",C[1]);
printf("C[SIZE-1] = %f\n",C[N-1]);
printf("Time taken: %.6f seconds\n", time_spent);

free(A);
free(B);
free(C);

return 0;
}
