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

def dls(start, goal, depthLimit, path):
    if depthLimit < 0:
        return False  

    path.append(start)  
    if start == goal:
        return True  

    for neighbor in graph.get(start, []):
        if neighbor not in path: 
            if dls(neighbor, goal, depthLimit - 1, path):
                return True 

    path.pop()  
    return False

def iterativeDeepening(start, goal, maxDepth):
    for depth in range(maxDepth + 1):
        print(f"Depth: {depth}")
        path = []
        if dls(start, goal, depth, path):
            print("\nPath to goal:", " → ".join(path)) 
            return
    print("Goal not found within depth limit.")

start = 'A'
goal = 'I'
maxSearchDepth = 3
iterativeDeepening(start, goal, maxSearchDepth)
