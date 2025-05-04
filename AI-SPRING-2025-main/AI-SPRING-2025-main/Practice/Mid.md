# AGENT + BSF:

```py
class Environment:
    def __init__(self,graph):
        self.graph = graph
    
    def get_percept(self,node):
        return node
    
    def BFS(self,start,goal):
        visited = []
        queue = []
        
        visited.append(start)
        queue.append(start)
        
        while queue:
            node = queue.pop(0)
            print('visited: ',node)
            
            if node == goal:
                return f"Goal {node} reached!"
            
            for neighbors in self.graph[node]:
                if neighbors not in visited:
                    visited.append(neighbors)
                    queue.append(neighbors)
                    
        return "Goal not found!"
    
class Agent:
    def __init__(self,goal):
        self.goal = goal
        
    def act(self,percept,environment):
        if percept == self.goal:
            return f"Goal {self.goal} found!"
        else:
            return environment.BFS(percept,self.goal)
        
def run(agent,environment,node):
    percept = environment.get_percept(node)
    action = agent.act(percept,environment)
    print(action)
    
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

start = 'A'
goal = 'I'

agent = Agent(goal)
environment = Environment(graph)

run(agent,environment,start)
```

# AGENT + DFS:

```py
class Environment:
    def __init__(self,graph):
        self.graph = graph
        
    def get_percept(self,node):
        return node
    
    def DFS(self,start,goal):
        visited = []
        stack = []
        
        visited.append(start)
        stack.append(start)
        
        while stack:
            node = stack.pop()
            print(f"visited: ",node)
            
            if node == goal:
                return f"Goal {node} reached!"
            
            for neighbors in reversed(self.graph[node]):
                if neighbors not in visited:
                    visited.append(neighbors)
                    stack.append(neighbors)
                    
        return f"Goal not found!"
    
class Agent:
    def __init__(self,goal):
        self.goal = goal
        
    def act(self,percept,environment):
        if percept == self.goal:
            return f"Goal {self.goal} found!"
        else:
            return environment.DFS(percept,self.goal)
        
def run(agent,environment,start):
    percept = environment.get_percept(start)
    action = agent.act(percept,environment)
    print(action)
    
    
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
    
start = 'A'
goal = 'I'
    
env = Environment(graph)
agent = Agent(goal)
run(agent,env,start)
```

# Grid + Agent + BFS:

```py
class Environment:
    def __init__(self, maze):
        self.maze = maze
        self.rows = len(maze)
        self.cols = len(maze[0])

    def get_percept(self, position):
        return position

    def is_valid(self, x, y, visited):
        return 0 <= x < self.rows and 0 <= y < self.cols and self.maze[x][y] != 1 and (x, y) not in visited

    def BFS(self, start, goal):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        visited = set()
        queue = [(start, [start])]

        while queue:
            (x, y), path = queue.pop(0)
            visited.add((x, y))

            if (x, y) == goal:
                return f"Goal {goal} reached! Path: {path}"

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if self.is_valid(nx, ny, visited):
                    queue.append(((nx, ny), path + [(nx, ny)]))

        return "Goal not found!"


class Agent:
    def __init__(self, goal):
        self.goal = goal

    def act(self, percept, environment):
        if percept == self.goal:
            return f"Goal {self.goal} found!"
        else:
            return environment.BFS(percept, self.goal)


def run(agent, environment, start):
    percept = environment.get_percept(start)
    action = agent.act(percept, environment)
    print(action)


# Maze representation (0 = open path, 1 = wall)
maze = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
]

start = (0, 0)
goal = (4, 4)

agent = Agent(goal)
environment = Environment(maze)

run(agent, environment, start)
```

# Grid + Agent + DFS:

```py
class Environment:
    def __init__(self, maze):
        self.maze = maze
        self.rows = len(maze)
        self.cols = len(maze[0])

    def get_percept(self, position):
        return position

    def is_valid(self, x, y, visited):
        return 0 <= x < self.rows and 0 <= y < self.cols and self.maze[x][y] != 1 and (x, y) not in visited

    def DFS(self, start, goal):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        visited = set()
        stack = [(start, [start])]

        while stack:
            (x, y), path = stack.pop()
            visited.add((x, y))

            if (x, y) == goal:
                return f"Goal {goal} reached! Path: {path}"

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if self.is_valid(nx, ny, visited):
                    stack.append(((nx, ny), path + [(nx, ny)]))

        return "Goal not found!"


class Agent:
    def __init__(self, goal):
        self.goal = goal

    def act(self, percept, environment):
        if percept == self.goal:
            return f"Goal {self.goal} found!"
        else:
            return environment.DFS(percept, self.goal)


def run(agent, environment, start):
    percept = environment.get_percept(start)
    action = agent.act(percept, environment)
    print(action)


# Maze representation (0 = open path, 1 = wall)
maze = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
]

start = (0, 0)
goal = (4, 4)

agent = Agent(goal)
environment = Environment(maze)

run(agent, environment, start)
```

# A star:

```py
from queue import PriorityQueue

graph = {
'A': [('B', 5, 9), ('C', 8, 5)], # (neighbor, cost, heuristic)
'B': [('D', 10, 4)], # (neighbor, cost, heuristic)
'C': [('E', 3, 7)], # (neighbor, cost, heuristic)
'D': [('F', 7, 5)], # (neighbor, cost, heuristic)
'E': [('F', 2, 1)], # (neighbor, cost, heuristic)
'F': [] # (neighbor, cost, heuristic)
}

def astar_search(graph, start, goal):
    visited = set() 
    pq = PriorityQueue()
    pq.put((0, start))
    while not pq.empty():
        cost, node = pq.get()
        if node not in visited:
            print(node, end=' ')
            visited.add(node)
            if node == goal:
                print("\nGoal reached!")
                return True
            for neighbor, edge_cost, heuristic in graph[node]:
                if neighbor not in visited:
                    f_value = cost + edge_cost + heuristic
                    pq.put((f_value, neighbor))
    print("\nGoal not reachable!")
    return False

print("A* Search Path:")
astar_search(graph, 'A', 'F')
```

# A star + Grid:

```py
from queue import PriorityQueue

class Environment:
    def __init__(self, maze):
        self.maze = maze
        self.rows = len(maze)
        self.cols = len(maze[0])
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
    
    def is_valid(self, x, y):
        return 0 <= x < self.rows and 0 <= y < self.cols and self.maze[x][y] != 1

    def heuristic(self, node, goal):
        # Manhattan distance
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])
    
    def astar(self, start, goal):
        pq = PriorityQueue()
        pq.put((0, [start]))  # (f_value, path)
        visited = set()
        
        while not pq.empty():
            cost, path = pq.get()
            current = path[-1]
            
            if current == goal:
                return path  # Goal reached!
            
            if current in visited:
                continue
                
            visited.add(current)
            
            for dx, dy in self.directions:
                nx, ny = current[0] + dx, current[1] + dy
                
                if self.is_valid(nx, ny):
                    new_path = path + [(nx, ny)]
                    g_cost = len(new_path) - 1
                    h_cost = self.heuristic((nx, ny), goal)
                    f_cost = g_cost + h_cost
                    pq.put((f_cost, new_path))
                    
        return None
        
maze = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
]

start = (0, 0)
goal = (4, 4)

env = Environment(maze)
path = env.astar(start, goal)

if path:
    print("A* Search Path:", path)
else:
    print("Goal not reachable!")
```

# A star Paper Question:

```py
from queue import PriorityQueue

maze = [
    [1, 2, 3, '#', 4],
    [1, '#', 1, 2, 2],
    [2, 3, 1, '#', 1],
    ['#', '#', 2, 1, 1],
    [1, 1, 2, 2, 1]
]

class Environment:
    def __init__(self, maze):
        self.maze = maze
        self.rows = len(maze)
        self.cols = len(maze[0])

    def get_percept(self, position):
        return position

    def is_valid(self, x, y, visited):
        return 0 <= x < self.rows and 0 <= y < self.cols and self.maze[x][y] != '#' and (x, y) not in visited

    def heuristic(self, node, goal):
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

    def Astar(self, start, goal):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        visited = set()
        pq = PriorityQueue()
        pq.put((0, [start]))

        while not pq.empty():
            cost, path = pq.get()
            current = path[-1]

            if current == goal:
                return f'Goal reached: {path}, Total Cost: {cost}'

            if current in visited:
                continue

            visited.add(current)

            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy

                if self.is_valid(nx, ny, visited):
                    new_path = path + [(nx, ny)]
                    g_cost = 0
                    for x, y in new_path:
                        g_cost += self.maze[x][y]
                    h_cost = self.heuristic((nx, ny), goal)
                    f_cost = g_cost + h_cost
                    pq.put((f_cost, new_path))

        return None

env = Environment(maze)
start = (0, 0)
goal = (4, 4)

print(env.Astar(start, goal))
```

# Agent + BFS + DFS Paper Question:

```py
class Environment:
    def __init__(self, maze):
        self.maze = maze
        self.rows = len(maze)
        self.cols = len(maze[0])

    def get_percept(self, position):
        return position

    def is_valid(self, x, y, visited):
        return 0 <= x < self.rows and 0 <= y < self.cols and self.maze[x][y] != 'X' and (x, y) not in visited

    def BFS(self, start, goal):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        visited = set()
        queue = [(start, [start])]

        while queue:
            (x, y), path = queue.pop(0)
            visited.add((x, y))

            if (x, y) == goal:
                return path

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if self.is_valid(nx, ny, visited):
                    queue.append(((nx, ny), path + [(nx, ny)]))

        return None

    def DFS(self, start, goal):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        visited = set()
        stack = [(start, [start])]

        while stack:
            (x, y), path = stack.pop()
            visited.add((x, y))

            if (x, y) == goal:
                return path

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if self.is_valid(nx, ny, visited):
                    stack.append(((nx, ny), path + [(nx, ny)]))

        return None


class Agent:
    def __init__(self, goal):
        self.goal = goal

    def act(self, percept, env, strategy):
        if strategy == "BFS":
            return env.BFS(percept, self.goal)
        elif strategy == "DFS":
            return env.DFS(percept, self.goal)


def run(agent, env, start, strategy="BFS"):
    percept = env.get_percept(start)
    path = agent.act(percept, env, strategy)
    if path:
        print(f"{strategy} Path: {path}")
    else:
        print("Goal not found!")


maze = [
    ['O', 'O', 'X', 'O', 'T'],
    ['O', 'X', 'O', 'O', 'X'],
    ['P', 'O', 'O', 'X', 'O'],
    ['X', 'X', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'X', 'O']
]

start = (2, 0)
goal = (0, 4)

env = Environment(maze)
agent = Agent(goal)

print("Running BFS:")
run(agent, env, start, strategy="BFS")

print("\nRunning DFS:")
run(agent, env, start, strategy="DFS")
```
