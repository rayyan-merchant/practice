class UtilityBasedAgent:
    def __init__(self):
        self.utility = {'Dirty': -10, 'Clean': 10}
    def calculate_utility(self, percept):
        return self.utility[percept]
    def select_action(self, percept):
        if percept == 'Dirty':
            return 'Clean the room'
        else:
            return 'No action needed'
    def act(self, percept):
        action = self.select_action(percept)
        return action

class Environment:
    def __init__(self, state='Dirty'):
        self.state = state
    def get_percept(self):
        return self.state
    def clean_room(self):
        self.state = 'Clean'

def run_agent(agent, environment, steps):
    total_utility = 0
    for step in range(steps):
        percept = environment.get_percept()
        action = agent.act(percept)
        utility = agent.calculate_utility(percept)
        print(f"Step {step + 1}: Percept - {percept}, Action - {action}, Utility - {utility}")
        total_utility += utility
        if percept == 'Dirty':
            environment.clean_room()
        print("Total Utility:", total_utility)

# Create instances of agent and environment
agent = UtilityBasedAgent()
environment = Environment()
# Run the agent in the environment for
run_agent(agent, environment, 5)