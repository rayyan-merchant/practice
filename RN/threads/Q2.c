#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

typedef struct{
int* arr;
int size;
} Data;

void* findavg(void* arg){
Data* data = (Data*)arg;
double* avg = malloc(sizeof(double));
int sum = 0;
for(int i = 0; i < data->size; i++){
sum += data->arr[i];
}
*avg = (double)sum / data->size;
return (void*)avg;
}

void* findmax(void* arg){
Data* data = (Data*)arg;
int* max = malloc(sizeof(int));
*max = data->arr[0];
for (int i = 1; i < data->size; i++){
if (*max < data->arr[i]){
*max = data->arr[i];
}
}
return (void*)max;
}

void* findmin(void* arg){
Data* data = (Data*)arg;
int* min = malloc(sizeof(int));
*min = data->arr[0];
for (int i = 1; i < data->size; i++){
if (*min > data->arr[i]){
*min = data->arr[i];
}
}
return (void*)min;
}

int main(int argc, char* argv[]){
if(argc < 2){
printf("Usage: %s <list of integers>\n", argv[0]);
return 1;
}

int size = argc - 1;
int* numbers = malloc(size * sizeof(int));
for(int i = 0; i < size; i++){
numbers[i] = atoi(argv[i + 1]);
}

Data data = { numbers, size };

pthread_t avg_thread, min_thread, max_thread;
void* avg_res;
void* min_res;
void* max_res;

pthread_create(&avg_thread, NULL, findavg, &data);
pthread_create(&min_thread, NULL, findmin, &data);
pthread_create(&max_thread, NULL, findmax, &data);

pthread_join(avg_thread, &avg_res);
pthread_join(min_thread, &min_res);
pthread_join(max_thread, &max_res);

printf("The average value is %.2f\n", *(double*)avg_res);
printf("The minimum value is %d\n", *(int*)min_res);
printf("The maximum value is %d\n", *(int*)max_res);

free(numbers);
free(avg_res);
free(min_res);
free(max_res);

return 0;
}
