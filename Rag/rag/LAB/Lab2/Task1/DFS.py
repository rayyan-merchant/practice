class Environment:
    def __init__(self,graph):
        self.graph = graph
        
    def DFS(self,start,goal):
        visited = []
        stack = []
        visited.append(start)
        stack.append(start)
        
        while(stack):
            node = stack.pop()
            
            if node == goal:
                print('Goal Reached!')
                return
            
            for neighbors in reversed(self.graph.get(node,[])):
                if neighbors not in visited:
                    visited.append(neighbors)
                    stack.append(neighbors)
                    
        print('Goal could not be Found!')
        
class Agent:
    def __init__(self,goal):
        self.goal = goal
        
    def Percept(self,percept):
        if percept == self.goal:
            return "Goal Reached!"
        else:
            return "Searching for Goal..."
        
    def act (self,percept,env):
        perception = self.Percept(percept)
        if perception == "Goal Reached!":
            print(f"Already at Goal!")
        else:
            env.DFS(percept,self.goal)
            
def Run_Agent(agent,env,start):
    agent.act(start,env)
    
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
start = 'A'
goal = 'I'

agent = Agent(goal)
environment = Environment(tree)
Run_Agent(agent, environment, start)
