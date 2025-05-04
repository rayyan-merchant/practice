class TSP_DFS:
    def __init__(self, graph):
        self.graph = graph
        self.n = len(graph)
        self.min_cost = float('inf')
        self.best_path = []

    def dfs(self, node, visited, path, cost, start):
        if len(visited) == self.n:  
            total_cost = cost + self.graph[node][start]  
            if total_cost < self.min_cost:
                self.min_cost = total_cost
                self.best_path = path[:] + [start]
            return

        for neighbor in self.graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                self.dfs(neighbor, visited, path, cost + self.graph[node][neighbor], start)
                path.pop()
                visited.remove(neighbor)  

    def solve(self, start):
        self.dfs(start, {start}, [start], 0, start)
        return self.best_path, self.min_cost

graph = {
    1: {2: 10, 3: 15, 4: 20},
    2: {1: 10, 3: 35, 4: 25},
    3: {1: 15, 2: 35, 4: 30},
    4: {1: 20, 2: 25, 3: 30}
}

tsp_solver = TSP_DFS(graph)
best_path, min_cost = tsp_solver.solve(1)
print(f"Best DFS TSP Path: {best_path}, Minimum Cost: {min_cost}")
