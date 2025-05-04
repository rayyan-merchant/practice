# Grid
grid = [
    [1, 1, 0, 1, 1],
    [1, 0, 1, 0, 1],
    [1, 1, 1, 1, 1],
    [0, 1, 0, 1, 1],
    [1, 1, 1, 0, 1]
]

ROWS, COLS = len(grid), len(grid[0])
visited = set()
directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up

def dfs(x, y):
    if (x, y) in visited or grid[x][y] == 0:
        return
    
    visited.add((x, y))
    print(f"({x}, {y})", end=" ")

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < ROWS and 0 <= ny < COLS:
            dfs(nx, ny)

# Run DFS from top-left corner
dfs(0, 0)
