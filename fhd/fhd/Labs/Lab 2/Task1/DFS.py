class GoalBasedAgent:
    def __init__(self, goal):
        self.goal = goal
    def PerceptGoal(self,percept):
        if percept == self.goal:
            return "Goal Reached"
        else:
            return "Searching For Goal"
        
    def act (self,percept,env):
        perception = self.PerceptGoal(percept)
        if perception == "Goal Reached":
            print(f"Already at Goal")
        else:
            env.DFS_Search(percept,self.goal)
        
class Environment:
    def __init__(self,graph):
        self.graph = graph
        
    def DFS_Search(self, start,goal):
        visited = []
        Stack =  []
        
        visited.append(start)
        Stack.append(start)
        
        while Stack:
            node = Stack.pop()
            print(f"Visiting Node: {node}")
            if node == goal:
                print("Goal Reached!!!!\n")
                return
            for neighbour in reversed(self.graph.get(node,[])):
                if neighbour not in visited:
                    visited.append(neighbour)
                    Stack.append(neighbour)
        print("Goal Not Found")
        
       
       
def Run_Agent(agent,env,Start_Node):
    agent.act(Start_Node,env)
        
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
start_node = 'A'
goal_node = 'I'

agent = GoalBasedAgent(goal_node)
environment = Environment(tree)
Run_Agent(agent, environment, start_node)