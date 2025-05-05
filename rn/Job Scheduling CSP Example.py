
# Import the OR-Tools CP-SAT solver for constraint satisfaction problems
from ortools.sat.python import cp_model

# Create a CP model to define variables, constraints, and objective
# In the hospital example, this model represents the operating room schedule
model = cp_model.CpModel()

# Define the jobs (surgeries) with their durations and deadlines
# Each tuple: (duration in hours, deadline in hours from 7 AM)
# S1: Appendectomy (2 hours, by 12 PM = 5 hours from 7 AM)
# S2: Biopsy (1 hour, by 11 AM = 4 hours from 7 AM)
# S3: Knee Arthroscopy (3 hours, by 1 PM = 6 hours from 7 AM)
jobs = [
    (2, 5),  # Surgery 1: duration=2, deadline=5
    (1, 4),  # Surgery 2: duration=1, deadline=4
    (3, 6)   # Surgery 3: duration=3, deadline=6
]
num_jobs = len(jobs)  # Number of surgeries (3)

# Define the time horizon: maximum possible time needed
# Sum of durations (2 + 1 + 3 = 6 hours) as a conservative estimate
horizon = sum(job[0] for job in jobs)

# Create variables for start times of each surgery
# Each start_var[i] is an integer variable representing the start time (hours from 7 AM)
# Domain: 0 to horizon (0 to 6 hours)
# Example: start_vars[0] is the start time of S1 (Appendectomy)
start_vars = [model.NewIntVar(0, horizon, f'start_{i}') for i in range(num_jobs)]

# Add constraints to the model
# 1. Deadline constraints: Ensure each surgery completes before its deadline
# For each surgery i, start time + duration <= deadline
# Example: For S1, start_vars[0] + 2 <= 5 (ends by 12 PM)
for i, (duration, deadline) in enumerate(jobs):
    model.Add(start_vars[i] + duration <= deadline)

# 2. Non-overlap constraints: Ensure no two surgeries run simultaneously
# Since there's one operating room, surgeries must be sequential
# For each pair of surgeries (i, j), either i finishes before j starts or vice versa
for i in range(num_jobs):
    for j in range(i + 1, num_jobs):
        # Create a boolean variable to decide the order (i before j or j before i)
        # Example: i_before_j means S1 finishes before S2 starts
        i_before_j = model.NewBoolVar(f'{i}_before_{j}')
        # Constraint: If i_before_j is true, start_i + duration_i <= start_j
        # Example: If S1 before S2, start_vars[0] + 2 <= start_vars[1]
        model.Add(start_vars[i] + jobs[i][0] <= start_vars[j]).OnlyEnforceIf(i_before_j)
        # Constraint: If i_before_j is false, start_j + duration_j <= start_i
        # Example: If S2 before S1, start_vars[1] + 1 <= start_vars[0]
        model.Add(start_vars[j] + jobs[j][0] <= start_vars[i]).OnlyEnforceIf(i_before_j.Not())

# Define the objective: Minimize the makespan
# Makespan is the time when the last surgery ends (max of start + duration)
# Create a variable for makespan, bounded by 0 to horizon
makespan = model.NewIntVar(0, horizon, 'makespan')
# Ensure makespan is at least the end time of each surgery
# Example: makespan >= start_vars[0] + 2 (end of S1)
for i, (duration, _) in enumerate(jobs):
    model.Add(makespan >= start_vars[i] + duration)
# Set the objective to minimize makespan
model.Minimize(makespan)

# Create a solver to find a solution
solver = cp_model.CpSolver()

# Solve the model
# The solver uses backtracking and constraint propagation to find a feasible solution
# It also optimizes the makespan (smallest possible completion time)
status = solver.Solve(model)

# Print the results
# Check if a solution (optimal or feasible) was found
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print("=== Operating Room Schedule ===")
    # Print start and end times for each surgery
    # Convert hours from 7 AM to clock time for readability
    for i in range(num_jobs):
        start = solver.Value(start_vars[i])  # Start time in hours from 7 AM
        duration = jobs[i][0]
        # Convert to clock time (e.g., 7 AM + start hours)
        start_time = f"{7 + start}:00 AM"
        end_time = f"{7 + start + duration}:00 AM"
        print(f"Surgery {i+1}: Start={start_time}, End={end_time}, Duration={duration}h")
    # Print the makespan (time of last surgery's end)
    print(f"Makespan: {solver.Value(makespan)} hours (Ends at {7 + solver.Value(makespan)}:00 AM)")
else:
    print("No feasible schedule found.")
