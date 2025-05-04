tree = {
'A': ['B', 'C'],
'B': ['D', 'E'],
'C': ['F', 'G'],
'D': ['H'],
'E': [],
'F': ['I'],
'G': [],
'H': [],
'I': []
}
def dls(node, goal, depth, path):
    if depth == 0:
        return False
    if node == goal:
        path.append(node)
        return True
    if node not in tree:
        return False
    for child in tree[node]:
        if dls(child, goal, depth - 1, path):
            path.append(node) # Store nodes while backtracking
            return True
        return False
def iterative_deepening(start, goal, max_depth):
    for depth in range(max_depth + 1):
        print(f"Depth: {depth}")
        path = []
        if dls(start, goal, depth, path):
            print("\nPath to goal:", " → ".join(reversed(path))) # Print path correctly
            return
    print("Goal not found within depth limit.")
    
# Test Iterative Deepening
start_node = 'A'
goal_node = 'I'
max_search_depth = 5
iterative_deepening(start_node, goal_node, max_search_depth)