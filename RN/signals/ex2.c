#define _XOPEN_SOURCE 700
#include<signal.h>
#include<stdio.h>
#include<sys/resource.h>
#include<sys/wait.h>

void proc_exit(){
	int wstat;
	pid_t pid;
	
	while(1){
		pid = wait3(&wstat, WNOHANG, NULL);
		
		if(pid==0 || pid==-1){
			fprintf(stdout, "return value of wait3() is %d\n ", pid)
			return;
		}
		fprintf(stdout, "return code: %d\n", wstat);
	}
}

int main(){
	signal(SIGCILD, proc_exit);
	
	switch(fork()){
		case -1:
			perror("main: fork");
			exit(0);
		
		case 0:
			printf("I'm alive temporarily\n");
			int ret_code = rand();
			printf("Return code is %d", ret_code);
			return ret_code;
			
		default:
			pause();
	}
	exit(0);
}