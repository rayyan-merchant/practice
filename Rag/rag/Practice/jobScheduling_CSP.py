from ortools.sat.python import cp_model

# Initialize the model
model = cp_model.CpModel()

# Define job data
jobs = {
    'Job1': {'duration': 2},
    'Job2': {'duration': 1},
    'Job3': {'duration': 2},
}
horizon = 6  # Maximum time

# Create start variables and intervals for each job
start_vars = {}
end_vars = {}
intervals = {}

for job, data in jobs.items():
    duration = data['duration']
    start = model.NewIntVar(0, horizon - duration, f'{job}_start')
    end = model.NewIntVar(0, horizon, f'{job}_end')
    interval = model.NewIntervalVar(start, duration, end, f'{job}_interval')
    
    start_vars[job] = start
    end_vars[job] = end
    intervals[job] = interval

# Constraint 1: No overlapping jobs (using interval variables)
model.AddNoOverlap(intervals.values())

# Constraint 2: Job1 must finish before Job3 starts
model.Add(end_vars['Job1'] <= start_vars['Job3'])

# Constraint 3: Job2 must start at or after time 1
model.Add(start_vars['Job2'] >= 1)

# Solver and solution printer
solver = cp_model.CpSolver()

class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, starts, durations):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._starts = starts
        self._durations = durations
        self._solution_count = 0

    def on_solution_callback(self):
        self._solution_count += 1
        print(f"Solution {self._solution_count}:")
        for job in self._starts:
            start = self.Value(self._starts[job])
            print(f"  {job} starts at {start}, ends at {start + self._durations[job]}")
        print()

# Search for all feasible solutions
solution_printer = SolutionPrinter(start_vars, {job: jobs[job]['duration'] for job in jobs})
solver.SearchForAllSolutions(model, solution_printer)
