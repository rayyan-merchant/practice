#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <time.h>

#define SIZE 20          // Size of the array
#define DELAY 200000     // Microseconds (0.2 seconds)

// Global array and flag
int arr[SIZE];
volatile sig_atomic_t interrupted = 0;

/**
 * Signal handler for SIGINT (Ctrl+C).
 * Sets an interrupt flag and displays the current array.
 */
void sigint_handler(int signum) {
    interrupted = 1; // Signal received
    printf("\n[!] SIGINT received. Current array state:\n");
    for (int i = 0; i < SIZE; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n\n");
}

/**
 * Swaps two integers with a short delay.
 */
void swap(int *a, int *b) {
    usleep(DELAY); // Delay to simulate long operation
    int temp = *a;
    *a = *b;
    *b = temp;
}

/**
 * Bubble sort with a delay on each swap.
 * Can be interrupted using Ctrl+C.
 */
void bubble_sort(int *array, int size) {
    for (int i = 0; i < size - 1 && !interrupted; i++) {
        for (int j = 0; j < size - i - 1 && !interrupted; j++) {
            if (array[j] > array[j + 1]) {
                swap(&array[j], &array[j + 1]);
            }
        }
    }

    if (!interrupted) {
        printf("\n[✓] Sorting complete. Final array:\n");
        for (int i = 0; i < SIZE; i++) {
            printf("%d ", array[i]);
        }
        printf("\n");
    } else {
        printf("[!] Sorting was interrupted by SIGINT.\n");
    }
}

/**
 * Initializes array with random integers.
 */
void generate_random_array(int *array, int size) {
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        array[i] = rand() % 100;
    }
}

/**
 * Main function.
 */
int main() {
    // Register SIGINT handler
    struct sigaction sa;
    sa.sa_handler = sigint_handler;
    sa.sa_flags = 0;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);

    generate_random_array(arr, SIZE);

    printf("Initial array:\n");
    for (int i = 0; i < SIZE; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n\nStarting bubble sort... Press Ctrl+C to interrupt.\n");

    bubble_sort(arr, SIZE);

    return 0;
}
