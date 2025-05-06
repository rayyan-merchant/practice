#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <signal.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <string.h>

#define NUM_THREADS 4

pthread_t threads[NUM_THREADS];  // Array to hold thread identifiers

/**
 * Signal handler for SIGUSR1.
 * Prints process ID, parent process ID, logical thread ID (pthread), and kernel thread ID (tid).
 */
void sigusr1_handler(int signum, siginfo_t *info, void *context) {
    pid_t pid = getpid();                        // Current process ID
    pid_t ppid = getppid();                      // Parent process ID
    pthread_t tid = pthread_self();              // POSIX thread ID (logical)
    pid_t k_tid = syscall(SYS_gettid);           // Kernel thread ID (Linux-specific)

    fprintf(stdout,
        "Thread (pthread ID: %lu) received SIGUSR1 signal "
        "[parent PID=%d, process PID=%d, kernel TID=%ld]\n",
        (unsigned long)tid, ppid, pid, (long)k_tid);
}

/**
 * Thread function that keeps the thread alive indefinitely.
 * Uses sleep to avoid busy looping.
 */
void* thread_function(void* arg) {
    while (1) {
        sleep(1);  // Keeps thread alive and CPU usage low
    }
    return NULL;
}

int main() {
    // Set up sigaction structure
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));                   // Zero out the structure
    sa.sa_sigaction = sigusr1_handler;            // Set handler function
    sa.sa_flags = SA_SIGINFO;                     // Use extended signal info
    sigemptyset(&sa.sa_mask);                     // Don't block any signals during handler

    // Register SIGUSR1 handler using sigaction
    if (sigaction(SIGUSR1, &sa, NULL) != 0) {
        perror("sigaction failed");
        exit(EXIT_FAILURE);
    }

    // Create multiple threads
    for (int i = 0; i < NUM_THREADS; ++i) {
        if (pthread_create(&threads[i], NULL, thread_function, NULL) != 0) {
            perror("pthread_create failed");
            exit(EXIT_FAILURE);
        }
    }

    // Print process/thread information
    pid_t pid = getpid();
    pid_t ppid = getppid();
    pthread_t main_tid = pthread_self();

    fprintf(stdout, "Parent PID: %d, Process PID: %d, main() Thread ID: %lu\n",
            ppid, pid, (unsigned long)main_tid);

    fprintf(stdout, "Worker Thread IDs: 0=%lu, 1=%lu, 2=%lu, 3=%lu\n",
            (unsigned long)threads[0], (unsigned long)threads[1],
            (unsigned long)threads[2], (unsigned long)threads[3]);

    // Send SIGUSR1 to the entire process (OS will choose one thread to handle it)
    kill(pid, SIGUSR1);

    // Send SIGUSR1 directly to the 3rd thread (index 2)
    pthread_kill(threads[2], SIGUSR1);

    // Join threads to prevent program exit (threads run infinitely)
    for (int i = 0; i < NUM_THREADS; ++i) {
        pthread_join(threads[i], NULL);
    }

    return 0;
}
