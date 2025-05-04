class Environment:
    def __init__(self,graph):
        self.graph = graph
    
    def DLS(self, start,goal,maxDepth):
        visited = []
        def DFS(node,goal,depth):
            visited.append(node)
            if depth> maxDepth:
                return 0
            if node == goal:
                print(f"Path: {visited}")
                return 1
            for neighbors in self.graph.get(node,[]):
                if neighbors not in visited:
                    if DFS(neighbors,goal,depth+1):
                        return 1
            visited.pop()          
            return 0    
        
        return DFS(start,goal,0)
        
class Agent:
    def __init__(self, goal):
        self.goal = goal
    def PerceptGoal(self,percept):
        if percept == self.goal:
            return "Goal Reached!"
        else:
            return "Searching for Goal...."
        
    def act (self,percept,env,d_Limit):
        perception = self.PerceptGoal(percept)
        if perception == "Goal Reached":
            print(f"Already at Goal")
        else:
            return env.DLS(percept,self.goal,d_Limit)
       
def Run_Agent(agent,env,start, limit):
    if agent.act(start,env,limit) == 0:
        print("Goal not found")
        
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
maxDepth = 3
agent = Agent(goal)
environment = Environment(graph)
Run_Agent(agent, environment, start,maxDepth)
