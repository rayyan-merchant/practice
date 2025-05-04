import numpy as np
import random
import matplotlib.pyplot as plt
from math import sqrt

# Constants
NUM_CITIES = 10
POPULATION_SIZE = 100
GENERATIONS = 500
MUTATION_RATE = 0.02
ELITISM = True  # Keep best individual from each generation

def generate_cities(num_cities):
    cities = []
    for _ in range(num_cities):
        x, y = random.random() * 100, random.random() * 100
        cities.append((x, y))
    return cities

def distance(city1, city2):
    x1, y1 = city1
    x2, y2 = city2
    return sqrt((x2 - x1)**2 + (y2 - y1)**2)

def route_distance(route, cities):
    total = 0
    for i in range(len(route)):
        current_city = cities[route[i]]
        next_city = cities[route[(i + 1) % len(route)]]
        total += distance(current_city, next_city)
    return total

def initialize_population(pop_size, num_cities):
    population = []
    for _ in range(pop_size):
        individual = list(range(num_cities))
        random.shuffle(individual)
        population.append(individual)
    return population

def tournament_selection(population, fitness, tournament_size=3):
    selected = []
    for _ in range(len(population)):
        contestants = random.sample(range(len(population)), tournament_size)
        contestant_fitness = [fitness[i] for i in contestants]
        winner = contestants[np.argmin(contestant_fitness)]  # Lower distance is better
        selected.append(population[winner])
    return selected

def ordered_crossover(parent1, parent2):
    size = len(parent1)
    child = [-1] * size
    
    a, b = sorted(random.sample(range(size), 2))
    
    child[a:b] = parent1[a:b]
    
    parent2_ptr = 0
    for i in list(range(b, size)) + list(range(0, b)):
        if child[i] == -1:
            while parent2[parent2_ptr] in child:
                parent2_ptr += 1
            child[i] = parent2[parent2_ptr]
    
    return child

def swap_mutation(individual):
    if random.random() < MUTATION_RATE:
        a, b = random.sample(range(len(individual)), 2)
        individual[a], individual[b] = individual[b], individual[a]
    return individual

def genetic_algorithm(cities, pop_size, generations):
    population = initialize_population(pop_size, len(cities))
    best_distance = float('inf')
    best_route = None
    distance_history = []
    
    for gen in range(generations):
        fitness = [route_distance(ind, cities) for ind in population]
        
        current_best = min(fitness)
        if current_best < best_distance:
            best_distance = current_best
            best_route = population[np.argmin(fitness)]
            print(f"Generation {gen}: New best distance = {best_distance:.2f}")
        
        distance_history.append(best_distance)
        
        selected = tournament_selection(population, fitness)
        
        children = []
        for i in range(0, pop_size, 2):
            parent1, parent2 = selected[i], selected[i+1]
            child1 = ordered_crossover(parent1, parent2)
            child2 = ordered_crossover(parent2, parent1)
            children.extend([child1, child2])
        
        mutated_children = [swap_mutation(child) for child in children]
        
        if ELITISM:
            best_idx = np.argmin(fitness)
            mutated_children[0] = population[best_idx]
        
        # New generation
        population = mutated_children
    
    return best_route, best_distance, distance_history

cities = generate_cities(NUM_CITIES)
best_route, best_distance, history = genetic_algorithm(cities, POPULATION_SIZE, GENERATIONS)

plt.figure(figsize=(12, 5))

# Plot cities and route
plt.subplot(1, 2, 1)
x = [city[0] for city in cities]
y = [city[1] for city in cities]
plt.scatter(x, y, color='red')

# Draw the best route
best_route.append(best_route[0])  # Return to start
route_x = [cities[i][0] for i in best_route]
route_y = [cities[i][1] for i in best_route]
plt.plot(route_x, route_y, linestyle='-', color='blue')
plt.title(f"Best Route (Distance = {best_distance:.2f})")

# Plot convergence
plt.subplot(1, 2, 2)
plt.plot(history)
plt.title("Convergence")
plt.xlabel("Generation")
plt.ylabel("Best Distance")

plt.tight_layout()
plt.show()

print(f"Best route: {best_route[:-1]}")
print(f"Best distance: {best_distance:.2f}")
