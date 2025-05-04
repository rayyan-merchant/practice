# 8 Queen:

```py
import random

N = 8
pCount = 100

# Generate Population
def Generate_Population():
    population = []
    for _ in range(pCount):
        chromosome = [random.randint(1, N) for _ in range(N)]  # Random row for each queen
        population.append(chromosome)
    return population

# Fitness Function: Count number of non-attacking pairs
def Fitness(chromosome):
    non_attacking_pairs = 0
    for i in range(N):
        for j in range(i + 1, N):
            if chromosome[i] != chromosome[j] and abs(chromosome[i] - chromosome[j]) != abs(i - j):
                non_attacking_pairs += 1
    return non_attacking_pairs

# Select the best two parents
def Select(pop, fit):
    sorted_pop = sorted(zip(pop, fit), key=lambda x: x[1], reverse=True)
    return sorted_pop[0][0], sorted_pop[1][0]

# Crossover: Single Point Crossover
def crossover(p1, p2):
    point = random.randint(0, N - 1)
    child = p1[:point] + p2[point:]
    return child

# Mutation: Randomly change the position of a queen
def mutate(chromosome):
    i = random.randint(0, N - 1)
    chromosome[i] = random.randint(1, N)
    return chromosome

# Genetic Algorithm Main Loop
popuation = Generate_Population()
maxGen = 1000
mut_rate = 0.3

for gen in range(maxGen):
    fitness_score = [Fitness(i) for i in popuation]
    best_fit = max(fitness_score)
    print(f"Generation: {gen} BestFit: {best_fit}")

    if best_fit == 28:  # Maximum fitness for 8 queens
        break

    newPop = []
    for _ in range(pCount):
        p1, p2 = Select(popuation, fitness_score)
        child = crossover(p1, p2)
        if random.random() < mut_rate:
            child = mutate(child)
        newPop.append(child)

    popuation = newPop

# Output the solution
solution = popuation[fitness_score.index(best_fit)]
print("Solution:", solution)
```

# Knapsack:

```py
import random

pCount = 4

def Generate_Population():
    population = []
    for i in range(pCount):
        population.append("".join([random.choice("01") for _ in range(4)]))
        
    return population

# Fitness Function

def Fitness(bitstr):
    weight = [5,3,7,2]
    cost = [12,5,10,7]
    tWeight = 0
    tValue = 0
    for i in range(pCount):
        k = int(bitstr[i])
        tWeight += k*weight[i]
        tValue += k*cost[i]
        
    if tWeight > 12:
        return 0
    return tValue

# print(Fitness('0011'))

def Select(pop,fit):
    sortedpop = sorted(zip(pop,fit),key=lambda x:x[1],reverse=True)
    return (sortedpop[0][0],sortedpop[1][0])

def crossover(p1,p2):
    i = random.randint(0,len(p1))
    child = list(p1)
    child[:i] = list(p2[:i])
    print(i,'\n')
    return "".join(child)

# print(crossover("1111","0000"))

def mutate(bitstr):
    bitstr = list(bitstr)
    i = random.randint(0,len(bitstr)-1)
    if bitstr[i] == '1':
        bitstr[i] = '0'
        return "".join(bitstr)

    bitstr[i] = '1'
    return "".join(bitstr)

# print(mutate('1101'))

# GENETIC ALGO

popuation = Generate_Population()
maxGen = 100
mut_rate = 0.4  # k +- (gen*maxGen) +- k
for gen in range(maxGen):
    fitness_score = [Fitness(i) for i in popuation]
    best_fit = max(fitness_score)
    print(f"Generation: {gen} BestFit {best_fit}")
    
    newPop = []
    
    for i in range(pCount):
        parents = Select(popuation,fitness_score)
        p1,p2 = parents
        child = crossover(p1,p2)
        if random.random() < mut_rate:
            child = mutate(child)
        newPop.append(child)
    popuation = newPop
```

# TSP:

```py
import numpy as np
import random

# Distance matrix for 10 cities (symmetric TSP)
distance_matrix = np.random.randint(10, 100, size=(10, 10))
np.fill_diagonal(distance_matrix, 0)

# Parameters
N = 10
pCount = 100
maxGen = 100
mutation_rate = 0.1

# Generate Population
def generate_population():
    population = []
    for _ in range(pCount):
        chromosome = random.sample(range(N), N)  # Randomly shuffle cities
        population.append(chromosome)
    return population

# Fitness Function
def fitness(chromosome):
    total_distance = 0
    for i in range(N - 1):
        total_distance += distance_matrix[chromosome[i], chromosome[i + 1]]
    total_distance += distance_matrix[chromosome[-1], chromosome[0]]  # Returning to the starting city
    return total_distance

# Select the best two parents
def select(population, fitness_scores):
    sorted_population = sorted(zip(population, fitness_scores), key=lambda x: x[1])
    parent1 = sorted_population[0][0]  # Best fitness
    parent2 = sorted_population[1][0]  # Second best fitness
    return parent1, parent2

# Crossover: Single Point Crossover
def crossover(p1, p2):
    point = random.randint(0, N - 1)
    child = p1[:point]
    for gene in p2:
        if gene not in child:
            child.append(gene)
    return child

# Mutation: Swap two cities
def mutate(chromosome):
    i, j = random.sample(range(N), 2)
    chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
    return chromosome

# Genetic Algorithm Main Loop
population = generate_population()

for gen in range(maxGen):
    fitness_scores = [fitness(individual) for individual in population]
    best_fit = min(fitness_scores)
    print(f"Generation: {gen} BestFit: {best_fit}")

    if best_fit == 0:  # Ideal case (not likely for TSP)
        break

    new_population = []
    for _ in range(pCount):
        p1, p2 = select(population, fitness_scores)
        child = crossover(p1, p2)
        if random.random() < mutation_rate:
            child = mutate(child)
        new_population.append(child)

    population = new_population

# Output the solution
best_index = fitness_scores.index(best_fit)
solution = population[best_index]
print(f"Solution: {solution}, Best fit: {best_fit}")
print("Distance Matrix:")
print(distance_matrix)
```
