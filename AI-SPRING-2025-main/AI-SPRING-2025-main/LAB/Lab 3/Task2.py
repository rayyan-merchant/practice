import heapq
import random
import time

class DynamicMazeSolver:
    def __init__(self, maze, start, goal):
        self.maze = maze
        self.start = start
        self.goal = goal
        self.rows = len(maze)
        self.cols = len(maze[0])
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        self.edge_costs = {((x, y), (x + dx, y + dy)): 1 for x in range(self.rows) for y in range(self.cols) for dx, dy in self.directions if self.is_valid(x + dx, y + dy)}

    def is_valid(self, x, y):
        return 0 <= x < self.rows and 0 <= y < self.cols and self.maze[x][y] != '#'

    def heuristic(self, point):
        return abs(point[0] - self.goal[0]) + abs(point[1] - self.goal[1])

    def update_edge_costs(self):
        for edge in self.edge_costs:
            self.edge_costs[edge] = random.randint(1, 5)

    def a_star(self):
        pq = []
        heapq.heappush(pq, (0, self.start, [self.start]))
        g_costs = {self.start: 0}
        visited = set()

        while pq:
            _, current, path = heapq.heappop(pq)

            if current in visited:
                continue

            visited.add(current)

            if current == self.goal:
                return path

            for dx, dy in self.directions:
                new_x, new_y = current[0] + dx, current[1] + dy
                if self.is_valid(new_x, new_y):
                    neighbor = (new_x, new_y)
                    edge = (current, neighbor)
                    new_g_cost = g_costs[current] + self.edge_costs.get(edge, float('inf'))
                    
                    if neighbor not in g_costs or new_g_cost < g_costs[neighbor]:
                        g_costs[neighbor] = new_g_cost
                        f_cost = new_g_cost + self.heuristic(neighbor)
                        heapq.heappush(pq, (f_cost, neighbor, path + [neighbor]))
            
            self.update_edge_costs()
            time.sleep(0.1)
        
        return None

maze = [
    ['S', '.', '.', '#', '.', 'G'],
    ['#', '#', '.', '#', '.', '.'],
    ['.', '.', '.', '.', '#', '.'],
    ['.', '#', '#', '.', '#', '.'],
    ['.', '.', '.', '.', '.', '.']
]

start = (0, 0)
goal = (0, 5)
solver = DynamicMazeSolver(maze, start, goal)
optimal_path = solver.a_star()

if optimal_path:
    print("Optimal Path:", optimal_path)
else:
    print("No path found")
