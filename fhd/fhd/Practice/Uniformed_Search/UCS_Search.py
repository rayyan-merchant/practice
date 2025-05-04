# UCS Function with Frontier and Visited
def ucs(graph, start, goal):
    # Initialize the frontier with the start node and cost 0
    frontier = [(start, 0)] # (node, cost)
    visited = set() # Set to keep track of visited nodes
    cost_so_far = {start: 0} # Cost to reach each node
    came_from = {start: None} # Path reconstruction
    while frontier:
        # Sort frontier by cost, simulate priority queue
        frontier.sort(key=lambda x: x[1])
        # Pop the node with the lowest cost
        current_node, current_cost = frontier.pop(0)
        # If we've already visited this node, skip it
        if current_node in visited:
            continue
        # Mark the current node as visited
        visited.add(current_node)
        # If we reach the goal, reconstruct the path and return
        if current_node == goal:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = came_from[current_node]
            path.reverse()
            print(f"Goal found with UCS. Path: {path}, Total Cost: {current_cost}")
            return
        # Explore neighbors
        for neighbor, cost in graph[current_node].items():
            new_cost = current_cost + cost
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current_node
                frontier.append((neighbor, new_cost)) # Add to frontier
    print("Goal not found")
    
    # Graph with different edge costs for UCS
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
# Run UCS with updated costs, using frontier and visited
ucs(graph, 'A', 'I')