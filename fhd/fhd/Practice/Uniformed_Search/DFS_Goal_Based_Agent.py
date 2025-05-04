class GoalBasedAgent:
    def __init__(self, goal):
        self.goal = goal
    def formulate_goal(self, percept):
        if percept == self.goal:
            return "Goal reached"
        return "Searching"
    def act(self, percept, environment):
        goal_status = self.formulate_goal(percept)
        if goal_status == "Goal reached":
            return f"Goal {self.goal} found!"
        else:
            return environment.dfs_search(percept, self.goal)
        
#BFS goal based agent
class Environment:
    def __init__(self, graph):
        self.graph = graph
    def get_percept(self, node):
        return node
    def dfs_search(self, start, goal):
        visited = [] # List for visited nodes
        stack = [] # Initialize a stack
        visited.append(start)
        stack.append(start)
        while stack:
            node = stack.pop() 
            print(f"Visiting: {node}")
            if node == goal: # Stop if goal is found
                return f"Goal {goal} found!"
            for neighbour in reversed(self.graph.get(node, [])):
                if neighbour not in visited:
                    visited.append(neighbour)
                    stack.append(neighbour)
        return "Goal not found"
    
    
def run_agent(agent, environment, start_node):
    percept = environment.get_percept(start_node)
    action = agent.act(percept, environment)
    print(action)
    
# Tree Representation
tree = {
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
# Define Start and Goal Nodes
start_node = 'A'
goal_node = 'I'
# Create instances of agent and environment
agent = GoalBasedAgent(goal_node)
environment = Environment(tree)
# Run the agent
run_agent(agent, environment, start_node)