from collections import deque
from typing import List, Tuple, Dict, Optional
from ortools.sat.python import cp_model # type: ignore
import time

def read_strings(filename):
    with open(filename, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def read_grid(file_grid):
    return [[int(file_grid[i * 9 + j]) for j in range(9)] for i in range(9)]

def solve(grid):
    rows = [[False for _ in range(10)] for _ in range(9)]
    cols = [[False for _ in range(10)] for _ in range(9)]
    subgrids = [[False for _ in range(10)] for _ in range(9)]
    domains = {(r, c): set(range(1, 10)) for r in range(9) for c in range(9)}

    def ac3():
        q = deque((r, c) for r in range(9) for c in range(9) if grid[r][c] == 0)
        while q:
            r, c = q.popleft()
            si = (r // 3) * 3 + c // 3
            legals = {num for num in domains[(r, c)] if not rows[r][num] and not cols[c][num] and not subgrids[si][num]}
            if not legals:
                return False
            if len(legals) == 1 and legals != domains[(r, c)]:
                num = next(iter(legals))
                assign(r, c, num)
                q.extend((nr, nc) for nc in range(9) for nr in range(9) if (nr == r or nc == c or (nr // 3 == r // 3 and nc // 3 == c // 3)) and (nr, nc) != (r, c) and grid[nr][nc] == 0)
            domains[(r, c)] = legals
        return True

    def assign(r, c, num):
        grid[r][c] = num
        rows[r][num] = cols[c][num] = subgrids[(r // 3) * 3 + c // 3][num] = True
        domains[(r, c)] = {num}

    def undo_assign(r, c, num):
        grid[r][c] = 0
        rows[r][num] = cols[c][num] = subgrids[(r // 3) * 3 + c // 3][num] = False

    def backtrack():
        empty = [(r, c) for r in range(9) for c in range(9) if grid[r][c] == 0]
        if not empty:
            return True
        r, c = empty[0]
        si = (r//3)*3 + c//3

        for num in domains[(r, c)]:
            if rows[r][num] or cols[c][num] or subgrids[si][num]:
                continue
            assign(r, c, num)
            if backtrack():
                return True
            undo_assign(r, c, num)
        return False

    # init
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                continue
            num = grid[r][c]
            if rows[r][num] or cols[c][num] or subgrids[(r//3)*3 + c//3][num]:
                return False
            assign(r, c, num)

    if ac3() and backtrack():
        return grid

# OR Tools
def solve_or(grid):
    model = cp_model.CpModel()

    domains = [[model.NewIntVar(1, 9, f"{r}{c}") for c in range(9)] for r in range(9)]

    # init
    for i in range(9):
        for j in range(9):
            if not grid[i][j]:
                continue
            model.Add(domains[i][j] == grid[i][j])
            model.AddAllDifferent(domains[i])
            model.AddAllDifferent([domains[r][j] for r in range(9)])
            if i % 3 == 0 and j % 3 == 0:
                model.AddAllDifferent([
                    domains[i + r][j + c]
                    for r in range(3) for c in range(3)
                ])

    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
        return [[solver.Value(domains[r][c]) for c in range(9)] for r in range(9)]
    
# GPT
class SudokuSolver:
    def __init__(self, board: List[List[int]]):
        self.board = board
        self.size = 9
        self.subgrid_size = 3
        self.domains = self.initialize_domains()

    def initialize_domains(self) -> Dict[Tuple[int, int], List[int]]:
        """Initialize domains for each cell."""
        domains = {}
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] == 0:
                    domains[(r, c)] = list(range(1, 10))
                else:
                    domains[(r, c)] = [self.board[r][c]]
        return domains

    def get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get all neighbors of a cell."""
        neighbors = set()
        for i in range(self.size):
            neighbors.add((row, i))  # Row neighbors
            neighbors.add((i, col))  # Column neighbors
        subgrid_row, subgrid_col = row // self.subgrid_size, col // self.subgrid_size
        for r in range(subgrid_row * self.subgrid_size, (subgrid_row + 1) * self.subgrid_size):
            for c in range(subgrid_col * self.subgrid_size, (subgrid_col + 1) * self.subgrid_size):
                neighbors.add((r, c))
        neighbors.discard((row, col))  # Exclude the cell itself
        return list(neighbors)

    def ac3(self) -> bool:
        """Perform AC-3 algorithm to enforce arc consistency."""
        queue = deque((x, y) for x in self.domains for y in self.get_neighbors(*x))
        while queue:
            x, y = queue.popleft()
            if self.revise(x, y):
                if not self.domains[x]:
                    return False
                for neighbor in self.get_neighbors(*x):
                    if neighbor != y:
                        queue.append((neighbor, x))
        return True

    def revise(self, x: Tuple[int, int], y: Tuple[int, int]) -> bool:
        """Revise domain of x based on constraints with y."""
        revised = False
        for value in self.domains[x][:]:
            if all(value == other for other in self.domains[y]):
                self.domains[x].remove(value)
                revised = True
        return revised

    def select_unassigned_variable(self) -> Tuple[int, int]:
        """Select the unassigned variable with the fewest legal values (MRV heuristic)."""
        return min(
            (var for var in self.domains if self.board[var[0]][var[1]] == 0),
            key=lambda var: len(self.domains[var]),
        )

    def order_domain_values(self, var: Tuple[int, int]) -> List[int]:
        """Order values for a variable by Least Constraining Value (LCV) heuristic."""
        neighbors = self.get_neighbors(*var)
        return sorted(
            self.domains[var],
            key=lambda val: sum(val in self.domains[neighbor] for neighbor in neighbors),
        )

    def is_consistent(self, var: Tuple[int, int], value: int) -> bool:
        """Check if assigning a value is consistent with constraints."""
        row, col = var
        for neighbor in self.get_neighbors(row, col):
            if value in self.domains[neighbor] and len(self.domains[neighbor]) == 1:
                return False
        return True

    def backtrack(self) -> Optional[List[List[int]]]:
        """Use backtracking search to solve the Sudoku."""
        if all(len(self.domains[var]) == 1 for var in self.domains):
            return [[self.domains[(r, c)][0] for c in range(self.size)] for r in range(self.size)]

        var = self.select_unassigned_variable()
        for value in self.order_domain_values(var):
            if self.is_consistent(var, value):
                self.board[var[0]][var[1]] = value
                old_domains = self.domains.copy()
                self.domains[var] = [value]
                if self.ac3():
                    result = self.backtrack()
                    if result:
                        return result
                self.domains = old_domains
                self.board[var[0]][var[1]] = 0
        return None

    def solve(self) -> Optional[List[List[int]]]:
        """Solve the Sudoku puzzle."""
        if not self.ac3():
            return None
        return self.backtrack()

def print_grid(grid):
    if not grid:
        return ""
    for i in range(9):
        if i != 0 and i % 3 == 0:
            print("-" * (9 + 2))
        for j in range(9):
            if j != 0 and j % 3 == 0:
                print("|", end="")
            print(grid[i][j] and grid[i][j] or '.', end="")
        print()

def timer_function(callback, *params, n=10):
    total = 0
    for i in range(n):
        start_time = time.time()
        callback(*params)
        end_time = time.time()
        total += end_time - start_time
    return total / n

def time_solution(solver, grid_strings):
    print(solver.__name__ + " started")
    total = 0
    for string in grid_strings:
        total += timer_function(solver, read_grid(string), n=10)
    return total / len(grid_strings)

grids = read_strings("Q03.txt")
print("Mine:", time_solution(solve, grids))
print("OR Tools:", time_solution(solve_or, grids))
print("GPT:", time_solution(lambda grid: SudokuSolver(grid).solve(), grids))
print()
print_grid(solve_or(read_grid(grids[1])))
