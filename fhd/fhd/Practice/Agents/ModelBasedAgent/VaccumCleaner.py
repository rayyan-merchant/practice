class ModelBasedAgent:
    def __init__(self):
        self.model = {}
        # d= {'current': clean}
        
    def update_model(self, percept):
        self.model['current'] = percept
        print(self.model)
    def predict_action(self):
        if self.model['current'] == 'Dirty':
            return 'Clean the room'
        else:
            return 'Room is clean'
    def act(self, percept):
        self.update_model(percept)
        return self.predict_action()
    
class Environment:
    def __init__(self, state='Dirty'):
        self.state = state
    def get_percept(self):
        return self.state
    def clean_room(self):
        self.state = 'Clean'

def run_agent(agent, environment, steps):
    for step in range(steps):
        percept = environment.get_percept()
        action = agent.act(percept)
        print(f"Step {step + 1}: Percept - {percept}, Action - {action}")
        if percept == 'Dirty':
            environment.clean_room()

# Create instances of agent and environment
agent = ModelBasedAgent()
environment = Environment()
# Run the agent in the environment for 5 steps
run_agent(agent, environment, 5)