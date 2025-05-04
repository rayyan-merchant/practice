class Environment:
    def __init__(self):
        self.CC = ['Safe', 'Low', 'Vulnerable', 'High', 'High', 'Low', 'Vulnerable', 'High']
        
    def display(self):
        print(f"Vulnerabilities: {self.CC}")
        
class Agent:
    def __init__(self):
        self.patch = []
        
    def scan(self,CC):
        for i in range(len(CC)):
            if CC[i] == "Vulnerable" or CC[i] == "Low" or CC[i] == "High":
                print('System is Vulnerable!\n')
            else:
                print('Congrats! System is not Vulnerable.')
                
    def patchVul(self,CC):
        for i in range(len(CC)):
            if CC[i] == "Low":
                self.patch.append(i)
                print('Vulnerability has been patched!\n')
            if CC[i] == "High":
                print("premium service is rwequired!")
                
    def finalSystemCheck(self,CC):
        x = 0
        for i in range(len(CC)):
            if(CC[i] == "Low"):
                x += 1
                
        if(len(self.patch) == x):
            print("All Low Risk Vulnerabilities have been patched!")
        else:
            print('Some Low Risk Vulnerabilities are not patched!')
            
env = Environment()
ag = Agent()
ag.scan(env.CC)
ag.patchVul(env.CC)
ag.finalSystemCheck(env.CC)
