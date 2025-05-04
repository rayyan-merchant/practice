class Environment:
    def __init__(self, graph):
        self.graph = graph
        
    def UCS(self, start, goal):
        frontier = [(start, 0)]
        visited = set()
        cost_so_far = {start: 0}
        came_from = {start: None}
        
        while frontier:
            frontier.sort(key=lambda x: x[1])  
            cur_node, cur_cost = frontier.pop(0)

            if cur_node in visited:
                continue
            visited.add(cur_node)
            
            if cur_node == goal:
                path = []
                while cur_node is not None:
                    path.append(cur_node)
                    cur_node = came_from[cur_node]
                path.reverse()
                print(f"Path: {path}, Cost: {cur_cost}")
                return path  
            
            for neighbor, cost in self.graph.get(cur_node, {}).items():
                new_cost = cur_cost + cost
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    came_from[neighbor] = cur_node
                    frontier.append((neighbor, new_cost))
        
        return None  
    
class Agent:
    def __init__(self, goal):
        self.goal = goal
    
    def PerceptGoal(self, percept):
        if percept == self.goal:
            return "Goal Reached!" 
        else:
            return "Searching for Goal...."
        
    def act(self, percept, env):
        if self.PerceptGoal(percept) == "Goal Reached!":
            print(f"Already at Goal")
        else:
            out = env.UCS(percept, self.goal)
            if out is None:
                print("No Path Found")

def Run_Agent(agent, env, start_node):
    agent.act(start_node, env)

# Graph definition
graph = {
    'A': {'B': 2, 'C': 1},
    'B': {'D': 4, 'E': 3},
    'C': {'F': 1, 'G': 5},
    'D': {'H': 2},
    'E': {},
    'F': {'I': 6},
    'G': {},
    'H': {},
    'I': {}
}

start_node = 'A'
goal_node = 'I'

agent = Agent(goal_node)
environment = Environment(graph)
Run_Agent(agent, environment, start_node)
