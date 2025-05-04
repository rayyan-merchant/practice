import numpy as np

states = ["Sunny", "Cloudy", "Rainy"]
transition_matrix = np.array([
    [0.6, 0.3, 0.1],  # From Sunny
    [0.2, 0.5, 0.3],  # From Cloudy
    [0.3, 0.3, 0.4]   # From Rainy
])

def simulate_markov_process(initial_state, num_steps):
    current_state = initial_state
    state_sequence = [current_state]
    for _ in range(num_steps):
        if current_state == "Sunny":
            next_state = np.random.choice(states, p=transition_matrix[0])
        elif current_state == "Cloudy":
            next_state = np.random.choice(states, p=transition_matrix[1])
        else:  # Rainy
            next_state = np.random.choice(states, p=transition_matrix[2])
        state_sequence.append(next_state)
        current_state = next_state
    return state_sequence

initial_state = "Sunny"
num_steps = 10
state_sequence = simulate_markov_process(initial_state, num_steps)

print(f"State sequence for {num_steps} steps starting from {initial_state}:")
print(" -> ".join(state_sequence))

def calculate_rain_probability():
    rainy_counts = 0
    for _ in range(10):
        sequence = simulate_markov_process("Sunny", 10)
        if sequence.count("Rainy") >= 3:
            rainy_counts += 1
    return rainy_counts / 10

probability = calculate_rain_probability()
print(f"\nProbability of at least 3 rainy days in 10 days: {probability}")
