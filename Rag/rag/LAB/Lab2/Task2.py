class TravelingSalesmanDFS:
    def __init__(self, graph):
        self.graph = graph
        self.num_cities = len(graph)
        self.min_cost = float('inf')
        self.best_path = []

    def dfs(self, current_city, visited, path, cost):
        if len(path) == self.num_cities:
            total_cost = cost + self.graph[current_city][path[0]]
            if total_cost < self.min_cost:
                self.min_cost = total_cost
                self.best_path = path[:] + [path[0]]
            return

        for next_city in range(self.num_cities):
            if next_city not in visited:
                visited.add(next_city)
                path.append(next_city)
                self.dfs(next_city, visited, path, cost + self.graph[current_city][next_city])
                path.pop()
                visited.remove(next_city) 

    def find_shortest_path(self, start=0):
        visited = {start}
        self.dfs(start, visited, [start], 0)
        return self.best_path, self.min_cost

graph = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

tsp = TravelingSalesmanDFS(graph)
best_path, min_cost = tsp.find_shortest_path()
print(f"Shortest Path: {best_path} with cost {min_cost}")
