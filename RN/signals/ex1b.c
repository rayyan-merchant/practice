#define _XOPEN_SOURCE 700
#include<signal.h>
#include<stdio.h>
#include<stdlib.h>

void sigint_handler(int signum){
	fprintf(stdout, "Caught SIGINT signal (%d)\n", signum);
	exit(signum);
}

int main(){
	struct sigaction sa;
	sa.sa_handler = sigint_handler;
	sa.sa_flags = 0;    // No special flags
	sigemptyset(&sa.sa_mask);  // clearing the signal mask
	
	if( sigaction(SIGINT, &sa, NULL)==-1){
		perror("sigaction");
		return EXIT_FAILURE;
	}
	
	while(1){
		
	}
	
	return EXIT_SUCCESS;
	
	
}
