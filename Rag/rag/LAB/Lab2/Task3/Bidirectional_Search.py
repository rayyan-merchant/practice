from collections import deque

def bidirectional_search(graph, start, goal):
    if start == goal:
        return [start]

    forward_queue = deque([[start]])
    backward_queue = deque([[goal]])

    forward_visited = {start}
    backward_visited = {goal}

    while forward_queue and backward_queue:
        result = expand_queue(graph, forward_queue, forward_visited, backward_visited)
        if result:
            return result
        result = expand_queue(graph, backward_queue, backward_visited, forward_visited)
        if result:
            return result

    return "No path found"

def expand_queue(graph, queue, visited, other_visited):
    path = queue.popleft()
    node = path[-1]

    for neighbor in graph.get(node, []):
        if neighbor not in visited:
            new_path = path + [neighbor]
            visited.add(neighbor)
            queue.append(new_path)

            if neighbor in other_visited:
                return new_path
    return None

graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'G'],
    'F': ['C', 'G'],
    'G': ['E', 'F']
}

print(bidirectional_search(graph, 'A', 'G'))
