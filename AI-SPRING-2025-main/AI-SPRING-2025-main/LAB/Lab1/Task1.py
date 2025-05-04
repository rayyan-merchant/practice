
import random

class Environment:
    def __init__(self):
        self.CC = [random.choice([0, 1]) for _ in range(9)]
        print(f"Initial State of System: {self.CC} \n")
        
class Agent:
    def __init__(self):
        self.patch = []
        
    def SystemScan(self, CC):
        for i in range(9):
            if CC[i] == 0:
                print("Warning \n")
                self.patch.append(i)
            else:
                print("Logged \n")
                
    def PatchingVul(self,CC):
        for i in range(len(self.patch)):
            CC[self.patch[i]] = 1
            
    def FinalSystemCheck(self, CC):
        print(f"Final State: {CC}")
        
env = Environment()
ag = Agent()


ag.SystemScan(env.CC)
ag.PatchingVul(env.CC)
ag.FinalSystemCheck(env.CC)
