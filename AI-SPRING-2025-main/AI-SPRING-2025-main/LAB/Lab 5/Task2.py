import math
import random

def hill_climbing_tsp(coordinates, max_iterations=10000):
    def distance(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    
    def route_distance(route):
        total = 0
        for i in range(len(route)):
            total += distance(route[i], route[(i+1)%len(route)])
        return total
    
    current_route = coordinates.copy()
    random.shuffle(current_route)
    current_distance = route_distance(current_route)
    
    for _ in range(max_iterations):
        # Generate neighbor by swapping two random cities
        neighbor_route = current_route.copy()
        i, j = random.sample(range(len(neighbor_route)), 2)
        neighbor_route[i], neighbor_route[j] = neighbor_route[j], neighbor_route[i]
        
        neighbor_distance = route_distance(neighbor_route)
        
        if neighbor_distance < current_distance:
            current_route = neighbor_route
            current_distance = neighbor_distance
    
    return current_route, current_distance

# Example delivery locations (coordinates)
delivery_points = [
    (0, 0),  # Depot
    (2, 4),
    (3, 1),
    (5, 2),
    (4, 5),
    (1, 3)
]

optimized_route, total_distance = hill_climbing_tsp(delivery_points)

print("Optimized Route:")
for point in optimized_route:
    print(point)
print(f"\nTotal Distance: {total_distance:.2f}")
