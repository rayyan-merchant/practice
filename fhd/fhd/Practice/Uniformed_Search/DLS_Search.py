# DLS Function
def dls(graph, start, goal, depth_limit):
    visited = []
    def dfs(node, depth):
        if depth > depth_limit:
            return None # Limit reached
        visited.append(node)
        if node == goal:
            print(f"Goal found with DLS. Path: {visited}")
            return 1
        # return visited
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                path = dfs(neighbor, depth + 1)
            if path:
                return path
        visited.pop() # Backtrack if goal not found
        return None

    return dfs(start, 0)

graph = {
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
dls(graph, 'A', 'I', 3)