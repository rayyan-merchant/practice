#define _XOPEN_SOURCE 700  // Required for wait3() and related features

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/resource.h>

// Signal handler for SIGCHLD
void proc_exit(int signum) {
    int wstat;
    pid_t pid;

    // Reap all terminated children
    while ((pid = wait3(&wstat, WNOHANG, NULL)) > 0) {
        fprintf(stdout, "Child with PID %d exited with code: %d\n",
                pid, WEXITSTATUS(wstat));
    }

    // If pid == 0: no children have exited yet
    // If pid == -1: no child processes remain, or error
}

int main() {
    struct sigaction sa;

    // Clear the struct and set the handler
    sa.sa_handler = proc_exit;
    sigemptyset(&sa.sa_mask);   // No extra signals blocked during handler
    sa.sa_flags = SA_RESTART;   // Restart interrupted syscalls like pause()

    // Install the SIGCHLD handler using sigaction
    if (sigaction(SIGCHLD, &sa, NULL) == -1) {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }

    int num_children = 3;
    pid_t pid;

    // Fork multiple child processes
    for (int i = 0; i < num_children; i++) {
        pid = fork();
        if (pid == -1) {
            perror("fork");
            exit(EXIT_FAILURE);
        } else if (pid == 0) {
            // Child process
            printf("Child #%d (PID=%d) is alive\n", i + 1, getpid());
            int ret_code = rand() % 100;
            sleep(1 + rand() % 3);  // Sleep a bit to simulate work
            printf("Child #%d (PID=%d) exiting with code %d\n", i + 1, getpid(), ret_code);
            exit(ret_code);
        }
    }

    // Parent process waits for children to terminate (via signal handler)
    // This will be interrupted each time a child exits
    while (1) {
        pause();  // Suspend until any signal is received
        // When SIGCHLD is received, proc_exit() will be invoked
    }

    return 0;
}
