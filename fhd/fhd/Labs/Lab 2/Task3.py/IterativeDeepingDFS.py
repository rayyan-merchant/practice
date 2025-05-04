graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E','C'],
    'C': ['F', 'G'],
    'D': ['H'],
    'E': [],
    'F': ['I'],
    'G': [],
    'H': [],
    'I': []
}

def dls(start, goal, depth_limit, path):
    if depth_limit < 0:
        return False  

    path.append(start)  
    if start == goal:
        return True  

    for neighbor in graph.get(start, []):
        if neighbor not in path: 
            if dls(neighbor, goal, depth_limit - 1, path):
                return True 

    path.pop()  
    return False

def iterative_deepening(start, goal, max_depth):
    for depth in range(max_depth + 1):
        print(f"Depth: {depth}")
        path = []
        if dls(start, goal, depth, path):
            print("\nPath to goal:", " → ".join(path)) 
            return
    print("Goal not found within depth limit.")

# Test Iterative Deepening
start_node = 'A'
goal_node = 'I'
max_search_depth = 3
iterative_deepening(start_node, goal_node, max_search_depth)
