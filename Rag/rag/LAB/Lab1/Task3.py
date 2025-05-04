class Environment:
    def __init__(self):
        self.lists = ['Completed','Failed','Failed','Completed','Completed']
        
    def display(self):
        print(f"List: {self.lists}")
        
class Agent:
    def scan(self,lists):
        for i in range(len(lists)):
            if lists[i] == 'Failed':
                print("Retrying.....\n")
                lists[i] = 'Completed'
                print('Successfully Completed!\n')
                
            else:
                print('No Action Needed!\n')
                

env = Environment()
ag = Agent()
ag.scan(env.lists)
env.display()
