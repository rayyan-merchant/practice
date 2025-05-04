class GoalBasedAgent:
    def __init__(self, goal):
        self.goal = goal
    def PerceptGoal(self,percept):
        if percept == self.goal:
            return "Goal Reached"
        else:
            return "Searching For Goal"
        
    def act (self,percept,env,d_Limit):
        perception = self.PerceptGoal(percept)
        if perception == "Goal Reached":
            print(f"Already at Goal")
        else:
            return env.DLS_Search(percept,self.goal,d_Limit)
        
class Environment:
    def __init__(self,graph):
        self.graph = graph
    
    def DLS_Search(self, start,goal,d_limit):
        visited = []
        def DFS(node,goal,depth):
            visited.append(node)
            if depth> d_limit:
                return 0
            if node == goal:
                print(f"Path: {visited}")
                return 1
            for neighbour in self.graph.get(node,[]):
                if neighbour not in visited:
                    if DFS(neighbour,goal,depth+1):
                        return 1
            visited.pop()          
            return 0
                        
            
        
        return DFS(start,goal,0)
        
       
       
def Run_Agent(agent,env,Start_Node, limit):
    if agent.act(Start_Node,env,limit) == 0:
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
start_node = 'A'
goal_node = 'I'
depth_limit = 3
agent = GoalBasedAgent(goal_node)
environment = Environment(graph)
Run_Agent(agent, environment, start_node,depth_limit)