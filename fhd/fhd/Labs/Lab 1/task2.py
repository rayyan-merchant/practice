class System:
    def __init__(self):
        self.servers = ["Underload", "Overload", "Balanced", "Overload", "Overload"]
    
    def Display(self):
        print(self.servers)

class LoadBalancer:
    def __init__(self):
        pass

    def DistributeLoad(self, ite, servers):
        for i in range(0, 5):
            if servers[i] == "Underload" and i != ite:
                servers[i] = "Balanced"
                servers[ite] = "Balanced"
                return True
        return False
                
    def Scan(self, servers):
        for i in range(0, 5):
            if servers[i] == "Overload":
                if not self.DistributeLoad(i, servers):  
                    print(f"Load cannot be Distributed for server {i} because all servers are balanced")
                else:
                    print(f"Load Distributed for server {i}")
                
env = System()
env.Display()
agent = LoadBalancer()
agent.Scan(env.servers)
env.Display()

