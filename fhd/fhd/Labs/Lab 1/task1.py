import random

class SystemEnviroment:
    def __init__(self):
        self.CC = [random.choice([0, 1]) for _ in range(9)]
        print(f"Initial State of System: {self.CC} \n")
    
class SecurityAgent:
    def __init__(self):
        self.patching = []
    
    def Scan(self, CC):
        
        for i in range(9):
            if CC[i] == 0:
                print("Warning \n")
                self.patching.append(i)
            else:
                print("Logged \n")
    
    def PatchingVulnerabilities(self, CC):
        
        psize = len(self.patching)
        
        for i in range(psize):
            print(f"Patching vulnerability at index {self.patching[i]}")
            CC[self.patching[i]] = 1
                
    def FinalSystemCheck(self, CC):
        print(f"Final State: {CC}")


env = SystemEnviroment()
agent = SecurityAgent()


agent.Scan(env.CC)
agent.PatchingVulnerabilities(env.CC)
agent.FinalSystemCheck(env.CC)

