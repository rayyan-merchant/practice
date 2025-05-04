class Environment:
    def __init__(self):
        self.rooms = {"a": "No Fire", "b": "No Fire", "c": "Fire", "d": "No Fire",
                      "e": "Fire", "f": "No Fire", "g": "No Fire", "h": "No Fire", "j": "Fire"}
        self.path = ["a", "b", "c", "d", "e", "f", "g", "h", "j"]

    def display(self, cur_room):
        j = 1
        for i in self.path:
            if i == cur_room:
                print("🤖", end="")
            else:
                print("  ", end="")  
            if self.rooms[i] == "Fire":
                print("🔥", end=" ")
            else:
                print("🧯", end=" ")
            if j % 3 == 0:
                print()  
            j += 1
        print()  


class Agent:
    def __init__(self):
        pass

    def Movement(self, env):
        print("Initial State")
        env.display(env.path[0])  
        for i in env.path:
            if env.rooms[i] == "Fire":
                print(f"Extinguishing Fire in room {i}\n")
                env.rooms[i] = "No Fire"
            else:
                print(f"Moving to Next Room {i}\n")
            env.display(i)


# Initialize environment and agent
env = Environment()
agent = Agent()
agent.Movement(env)
