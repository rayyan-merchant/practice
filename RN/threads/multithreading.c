#include<pthread.h>
#include<stdio.h>
#include<stdlib.h>

#define NUM_THREADS 4
#define ARRAY_SIZE 10000

int global_array[ARRAY_SIZE];

void initialize_array(){
	for(int i=0; i<ARRAY_SIZE; i++){
		global_array[i] = rand() % 1000;
	}
}

void *sum_array(void *param){
	int thread_id = *((int *)param);
	int start = thread_id * (ARRAY_SIZE/NUM_THREADS);
	int end = start + (ARRAY_SIZE/NUM_THREADS);
	long sum = 0;
	
	for(int i=start; i<end; i++){
		sum += global_array[i];
	}
	
	return (void *)sum;

}



int main(){
	pthread_t threads[NUM_THREADS];
	int threads_args[NUM_THREADS];
	void *threads_results[NUM_THREADS];
	long total = 0;
	
	initialize_array();
	
	for(int i=0; i<NUM_THREADS; i++){
		threads_args[i] = i;
		pthread_create(&threads[i], NULL, sum_array, (void *)&threads_args[i]);
	}
	
	for(int i=0; i<NUM_THREADS; i++){
		pthread_join(threads[i], &threads_results[i]);
		total += (long)threads_results[i];
	}
	
	printf("Total Sum: %ld\n", total);
	
	return 0;
}