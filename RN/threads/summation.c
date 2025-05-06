#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

int sum = 0;

void *runner(void *param){
	int i;
	int n = *((int *)param);
	
	printf("Thread received: %d\n", n);
	if(n>0){
		for(i=1; i<=n; i++){
			sum+=i;
		}
	}
	pthread_exit(0);
}

int main(int argc, char *argv[]){
	pthread_t threadID;
	pthread_attr_t attributes;
	int num = 5;
	
	pthread_attr_init(&attributes);
	
	printf("Main: before thread\n");
	pthread_create(&threadID, &attributes, runner, (void *)&num);
	printf("Main: after thread\n");
	pthread_join(threadID, NULL);
	printf("Main: thread joined\n");
	printf("Sum: %d\n", sum);


}


// returning value through pthread_exit snippet
void *thread_function(){
	printf("Thread Executing\n");
	pthread_exit((void *)42);
}

int main(){
	pthread_t thread;
	pthread_create(&thread, NULL, thread_function, NULL);
	void* exit_status;
	pthread_join(thread, &exit_status);
	printf("Thread exited with status: %ld\n", (long)exit_status);
}





