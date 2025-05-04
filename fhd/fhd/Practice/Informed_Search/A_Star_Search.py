# Graph with different edge costs
graph = {
'A': {'B': 2, 'C': 1},
'B': {'D': 4, 'E': 3},
'C': {'F': 1, 'G': 5},
'D': {'H': 2},
'E': {},
'F': {'I': 6},
'G': {},
'H': {},
'I': {}
}
# Heuristic function (estimated cost to reach goal 'I')
heuristic = {
'A': 7,
'B': 6,
'C': 5,
'D': 4,
'E': 7,
'F': 3,
'G': 6,
'H': 2,
'I': 0 # Goal node
}
# A* Search Function
def a_star(graph, start, goal):
    frontier = [(start, 0 + heuristic[start])] # List-based priority queue(sorted manually)
    visited = set() # Set to keep track of visited nodes
    g_costs = {start: 0} # Cost to reach each node from start
    came_from = {start: None} # Path reconstruction

    while frontier:
        # Sort frontier by f(n) = g(n) + h(n)
        frontier.sort(key=lambda x: x[1])
        current_node, current_f = frontier.pop(0) # Get node with lowest f(n)
        if current_node in visited:
            continue

        print(current_node, end=" ") # Print visited node
        visited.add(current_node)
        
        # If goal is reached, reconstruct path
        if current_node == goal:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = came_from[current_node]
            path.reverse()
            print(f"\nGoal found with A*. Path: {path}")
            return

        # Explore neighbors
        for neighbor, cost in graph[current_node].items():
            new_g_cost = g_costs[current_node] + cost # Path cost from start to neighbor
            f_cost = new_g_cost + heuristic[neighbor] # f(n) = g(n) + h(n)
            if neighbor not in g_costs or new_g_cost < g_costs[neighbor]:
                g_costs[neighbor] = new_g_cost
                came_from[neighbor] = current_node
                frontier.append((neighbor, f_cost))
    print("\nGoal not found")

# Run A* Search
print("\nFollowing is the A* Search:")
a_star(graph, 'A', 'I')