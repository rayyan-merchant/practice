from collections import deque

# Grid
grid = [
    [1, 1, 0, 1, 1],
    [1, 0, 1, 0, 1],
    [1, 1, 1, 1, 1],
    [0, 1, 0, 1, 1],
    [1, 1, 1, 0, 1]
]

ROWS, COLS = len(grid), len(grid[0])
directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up

def bfs(start_x, start_y):
    queue = deque([(start_x, start_y)])
    visited = set()
    visited.add((start_x, start_y))

    while queue:
        x, y = queue.popleft()
        print(f"({x}, {y})", end=" ")

        # Explore neighbors
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < ROWS and 0 <= ny < COLS and (nx, ny) not in visited and grid[nx][ny] == 1:
                visited.add((nx, ny))
                queue.append((nx, ny))

# Run BFS from top-left corner
bfs(0, 0)
