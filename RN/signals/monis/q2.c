#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <sys/stat.h>
#include <string.h>
#include <time.h>

#define SPECIAL_CHARS "@#$%"  // Special characters for folder name

/**
 * Signal handler for SIGUSR1.
 * Creates a folder with a name that includes special characters.
 */
void sigusr1_handler(int signum) {
    char folder_name[100];
    time_t now = time(NULL);
    struct tm *t = localtime(&now);

    // Format: folder_@#$%_created_YYYYMMDD_HHMMSS
    snprintf(folder_name, sizeof(folder_name),
             "folder_%s_created_%04d%02d%02d_%02d%02d%02d",
             SPECIAL_CHARS,
             t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
             t->tm_hour, t->tm_min, t->tm_sec);

    if (mkdir(folder_name, 0755) == 0) {
        printf("[✓] Folder '%s' created successfully.\n", folder_name);
    } else {
        perror("[X] Failed to create folder");
    }
}

int main() {
    pid_t pid;

    // Set up signal handler in parent
    struct sigaction sa;
    sa.sa_handler = sigusr1_handler;
    sa.sa_flags = 0;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGUSR1, &sa, NULL);

    pid = fork();

    if (pid < 0) {
        perror("fork failed");
        exit(EXIT_FAILURE);
    }

    if (pid == 0) {
        // Child process: sleep briefly, then send SIGUSR1 to parent
        sleep(2);  // Give parent time to set up handler
        kill(getppid(), SIGUSR1);
        printf("[Child] Sent SIGUSR1 to parent.\n");
        exit(0);
    } else {
        // Parent process: wait for signal
        printf("[Parent] Waiting for signal from child...\n");
        pause();  // Suspend until signal is received
        printf("[Parent] Signal handled. Exiting.\n");
    }

    return 0;
}
