import random
import numpy as np

N = 6
GENS = 50
CR = 0.8
MR = 0.2

task_times = np.array([5, 8, 4, 7, 6, 3, 9])
facility_cap = np.array([24, 30, 28])
costs = np.array([
    [10, 12, 9],
    [15, 14, 16],
    [8, 9, 7],
    [12, 10, 13],
    [14, 13, 12],
    [9, 8, 10],
    [11, 12, 13]
])

def calc_fitness(current):
    current_costs = sum(costs[i][current[i] - 1] for i in range(7))
    times = [sum([task_times[i] for i in range(7) if current[i] - 1 == j]) for j in range(3)]
    penalties = sum(max(0, time - cap) * 9999 for time, cap in zip(times, facility_cap))
    return -current_costs - penalties

def generate_population():
    return [np.random.choice([1, 2, 3], 7) for _ in range(N)]

def select(population):
    fitness = [calc_fitness(curr) for curr in population]
    total_fitness = sum(fitness)
    return population[np.random.choice(len(population), p=[f / total_fitness for f in fitness])]

def crossover(parent1, parent2):
    if np.random.random() > CR:
        return parent1, parent2
    idx = np.random.randint(1, 6)
    return np.concatenate([parent1[:idx], parent2[idx:]]), np.concatenate([parent2[:idx], parent1[idx:]])

def mutate(current):
    if np.random.random() <= MR:
        i, j = np.random.randint(0, 7, size=2)
        current[i], current[j] = current[i], current[j]
    return current

def genetic_algorithm():
    population = generate_population()

    for _ in range(GENS):
        next_generation = []
        for _ in range(N):
            parent1, parent2 = select(population), select(population)
            child1, child2 = crossover(parent1, parent2)
            next_generation += [mutate(child1), mutate(child2)]
        population = next_generation

    best = min(population, key=calc_fitness)
    return best, -calc_fitness(best)

best, best_cost = genetic_algorithm()
print(f"best={best}, best_cost={best_cost}")
