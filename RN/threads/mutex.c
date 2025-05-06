#include<pthread.h>
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>

int counter = 0;
pthread_mutex_t mutex;

void *count(void *arg){
    int id = *((int *)arg);
    
    pthread_mutex_lock(&mutex);
    for(int i = 0; i < 5; i++){
        //pthread_mutex_lock(&mutex);
        counter++;
        printf("thread %d counter %d\n", id, counter);
        //pthread_mutex_unlock(&mutex);  // no differnce
    }
    pthread_mutex_unlock(&mutex);


    return NULL;
}

int main() {
    int a=1, b=2, c=3;
    pthread_t p1, p2, p3;
    
    pthread_mutex_init(&mutex, NULL);

    pthread_create(&p1, NULL, count, &a);
    pthread_create(&p2, NULL, count, &b);
    pthread_create(&p3, NULL, count, &c);
    
    pthread_join(p1, NULL);
    pthread_join(p2, NULL);
    pthread_join(p3, NULL);
    
    pthread_mutex_destroy(&mutex);
    return 0;
}
