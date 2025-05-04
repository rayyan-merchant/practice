class Environment:
    def __init__(self):
        self.servers = ["Underload", "Overload", "Balanced", "Overload", "Overload"]

    def display(self):
        print(f"Current Load Status: {self.servers}")


class Agent:
    def __init__(self):
        pass

    def system_scan(self, servers):
        overloaded_indices = []
        underloaded_indices = []

        for i in range(len(servers)):
            if servers[i] == "Overload":
                overloaded_indices.append(i)
            elif servers[i] == "Underload":
                underloaded_indices.append(i)

        # Balance tasks using while loop
        while len(overloaded_indices) > 0 and len(underloaded_indices) > 0:
            overloaded_index = overloaded_indices.pop(0)
            underloaded_index = underloaded_indices.pop(0)

            servers[overloaded_index] = "Balanced"
            servers[underloaded_index] = "Balanced"

    def display(self, servers):
        print(f"Updated Load Status: {servers}")

env = Environment()
ag = Agent()

print("Before Balancing:")
env.display()

ag.system_scan(env.servers)

print("After Balancing:")
ag.display(env.servers)
