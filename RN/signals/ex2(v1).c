#define _XOPEN_SOURCE 700  // Enables POSIX extensions for wait3() and other features

#include <signal.h>        // For signal handling
#include <stdio.h>         // For printf, fprintf
#include <stdlib.h>        // For exit(), rand()
#include <sys/resource.h>  // For wait3()
#include <sys/wait.h>      // For wait3(), WNOHANG, WEXITSTATUS
#include <unistd.h>        // For fork(), pause()

// Signal handler to reap child processes
void proc_exit(int signum) {
    int wstat;         // Variable to hold child's exit status
    pid_t pid;         // Will store PID of each reaped child

    // Loop to handle multiple children that may have exited
    while (1) {
        pid = wait3(&wstat, WNOHANG, NULL);  // Reap a terminated child without blocking

        // If no child has exited yet, or no more children left
        if (pid == 0 || pid == -1) {
            fprintf(stdout, "return value of wait3() is %d\n", pid);  // 0 = no exited child, -1 = error/no children
            return;
        }

        // Print the PID of the reaped child and its exit code
        fprintf(stdout, "Child with PID %d exited with code: %d\n", pid, WEXITSTATUS(wstat));
    }
}

int main() {
    // Register the SIGCHLD handler (called when child process exits)
    signal(SIGCHLD, proc_exit);

    // Create a child process
    switch (fork()) {
        case -1:
            // Error creating child
            perror("main: fork");
            exit(1);  // Exit with error

        case 0:
            // Child process code
            printf("I'm alive temporarily\n");
            int ret_code = rand() % 100;  // Generate a random return code (0-99)
            printf("Return code is %d\n", ret_code);
            exit(ret_code);  // Exit with the generated return code

        default:
            // Parent process code
            pause();  // Wait here until any signal is received (in this case, SIGCHLD)
    }

    return 0;  // Unreachable due to pause(), but good practice
}
