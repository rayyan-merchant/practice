from queue import PriorityQueue
# Example graph represented as an adjacency list
graph = {
'A': [('B', 5), ('C', 8)],
'B': [('D', 10)],
'C': [('E', 3)],
'D': [('F', 7)],
'E': [('F', 2)],
'F': []
}
def best_first_search(graph, start, goal):
    visited = set()
    pq = PriorityQueue()
    pq.put((0, start)) # priority queue with priority as the heuristic value
    while not pq.empty():
        cost, node = pq.get()
        if node not in visited:
            print(node, end=' ')
            visited.add(node)
        if node == goal:
            print("\nGoal reached!")
            return True
        for neighbor, weight in graph[node]:
            if neighbor not in visited:
                pq.put((weight, neighbor))
    print("\nGoal not reachable!")
    return False

# Example usage:
print("Best-First Search Path:")
best_first_search(graph, 'A', 'F')