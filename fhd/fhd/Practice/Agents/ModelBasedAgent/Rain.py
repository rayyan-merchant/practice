class Environment:
    def __init__(self, rain='No', windows_open='Open'):
            self.rain = rain
            self.windows_open = windows_open
    
    def get_percept(self):
        """Returns the current percept (rain status and window status)."""
        return {'rain': self.rain, 'windows_open': self.windows_open}
    
    def close_windows(self):
        """Closes the windows if they are open."""
        if self.windows_open == 'Open':
            self.windows_open = 'Closed'
            
class ModelBasedAgent:
    def __init__(self):
        self.model = {'rain': 'No', 'windows_open': 'Open'}
    def act(self, percept):
        """Decides action based on the model and current percept."""
        # Update the model with the current percept
        self.model.update(percept)
        # Check the model to decide action
        if self.model['rain'] == 'Yes' and self.model['windows_open'] == 'Open':
            return 'Close the windows'
        else:
            return 'No action needed'
    def display(self):
        print(self.model)
        
def run_agent(agent, environment, steps):
    for step in range(steps):
        # Get the current percept from the environment
        percept = environment.get_percept()
        # Agent makes a decision based on the current percept
        action = agent.act(percept)
        # Print the current percept and the agent's action
        print(f"Step {step + 1}: Percept - {percept}, Action - {action}")
        # If the agent decided to close the windows, update the environment
        if action == 'Close the windows':
            environment.close_windows()
            
# Create instances of agent and environment
agent = ModelBasedAgent()
environment = Environment(rain='Yes', windows_open='Open') # It's raining and windows are open
# Run the agent in the environment for 5 steps
run_agent(agent, environment, 5)
