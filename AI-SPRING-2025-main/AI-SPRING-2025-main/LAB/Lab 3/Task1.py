import heapq

class MazeSolver:
    def __init__(self, maze, start, goals):
        self.maze = maze
        self.start = start
        self.goals = set(goals)
        self.rows = len(maze)
        self.cols = len(maze[0])
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def is_valid(self, x, y):
        return 0 <= x < self.rows and 0 <= y < self.cols and self.maze[x][y] != '#'

    def heuristic(self, point, goals):
        return min(abs(point[0] - gx) + abs(point[1] - gy) for gx, gy in goals)

    def bfs_with_goals(self):
        pq = []
        heapq.heappush(pq, (0, self.start, [self.start], self.goals))
        visited = set()

        while pq:
            _, current, path, remaining_goals = heapq.heappop(pq)

            if current in visited:
                continue

            visited.add(current)

            if current in remaining_goals:
                remaining_goals = set(remaining_goals)
                remaining_goals.remove(current)
                if not remaining_goals:
                    return path

            for dx, dy in self.directions:
                new_x, new_y = current[0] + dx, current[1] + dy
                if self.is_valid(new_x, new_y) and (new_x, new_y) not in visited:
                    new_point = (new_x, new_y)
                    new_path = path + [new_point]
                    heapq.heappush(pq, (self.heuristic(new_point, remaining_goals), new_point, new_path, remaining_goals))
        return None

maze = [
    ['S', '.', '.', '#', '.', 'G'],
    ['#', '#', '.', '#', '.', '.'],
    ['.', '.', '.', '.', '#', '.'],
    ['.', '#', '#', '.', '#', '.'],
    ['.', '.', '.', '.', '.', 'G']
]

start = (0, 0)
goals = [(0, 5), (4, 5)]
solver = MazeSolver(maze, start, goals)
shortest_path = solver.bfs_with_goals()

if shortest_path:
    print("Shortest Path:", shortest_path)
else:
    print("No path found")
