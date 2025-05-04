class Environment:
    
    def __init__(self, backups):
        self.backups = backups
    
    def get_backups(self, i):
        return self.backups[i]
    
    def update(self, i):
        self.backups[i] = "Completed"
    def display(self):
        print(f"Backups: {self.backups}")
        
class BackupManagementAgent:
    
    def __init__(self):
        pass
    
    def Scan(self,percept):
        if percept == "Completed":
            return 0
        else:
            return 1
        
def Run_Agent(agent, env):
    size = len(env.backups)
    env.display()
    for i in range(0,size):
        percept = env.get_backups(i) 
        act = agent.Scan(percept)
        if(act == 1):
            env.update(i)
    env.display()
    
env = Environment(["Failed","Completed","Failed","Completed","Failed","Failed"])
agent = BackupManagementAgent()

Run_Agent(agent,env)
