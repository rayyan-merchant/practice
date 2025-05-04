
### 1. **Basic Job Scheduling on Single Machine**

**Objective:** Minimize the total completion time of all jobs.

```python
from ortools.sat.python import cp_model

def job_scheduling_single_machine(jobs):
    # Create the model
    model = cp_model.CpModel()

    # Number of jobs
    n = len(jobs)

    # Create the variables (start times for each job)
    start = [model.new_int_var(0, sum(jobs), f"start_{i}") for i in range(n)]
    end = [model.new_int_var(0, sum(jobs), f"end_{i}") for i in range(n)]

    # Create the constraints (end = start + duration)
    for i in range(n):
        model.add_constraint(end[i] == start[i] + jobs[i])

    # Minimize the total completion time (sum of end times)
    model.minimize(sum(end))

    # Create the solver and solve the model
    solver = cp_model.CpSolver()
    status = solver.solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"Total Completion Time: {solver.ObjectiveValue()}")
        for i in range(n):
            print(f"Job {i} starts at {solver.Value(start[i])} and ends at {solver.Value(end[i])}")
    else:
        print("No solution found")

# Sample input: job durations
jobs = [5, 8, 3, 7, 6]  # Job durations
job_scheduling_single_machine(jobs)
```

### 2. **Job Scheduling with Deadlines and Penalties**

**Objective:** Minimize total penalties for jobs completed after their deadlines.

```python
from ortools.sat.python import cp_model

def job_scheduling_with_penalties(jobs, deadlines, penalties):
    # Create the model
    model = cp_model.CpModel()

    # Number of jobs
    n = len(jobs)

    # Create the variables (start times for each job)
    start = [model.new_int_var(0, sum(jobs), f"start_{i}") for i in range(n)]
    end = [model.new_int_var(0, sum(jobs), f"end_{i}") for i in range(n)]
    late = [model.new_int_var(0, sum(jobs), f"late_{i}") for i in range(n)]

    # Create the constraints (end = start + duration)
    for i in range(n):
        model.add_constraint(end[i] == start[i] + jobs[i])
        model.add_constraint(late[i] == max(0, end[i] - deadlines[i]))

    # Minimize total penalties
    penalty = sum(late[i] * penalties[i] for i in range(n))
    model.minimize(penalty)

    # Create the solver and solve the model
    solver = cp_model.CpSolver()
    status = solver.solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"Total Penalty: {solver.ObjectiveValue()}")
        for i in range(n):
            print(f"Job {i} starts at {solver.Value(start[i])}, ends at {solver.Value(end[i])}, late: {solver.Value(late[i])}")
    else:
        print("No solution found")

# Sample input: job durations, deadlines, and penalties
jobs = [5, 8, 3, 7, 6]
deadlines = [7, 10, 5, 8, 6]  # Deadlines for each job
penalties = [2, 3, 1, 4, 2]  # Penalty for each job if late
job_scheduling_with_penalties(jobs, deadlines, penalties)
```

### 3. **Job Scheduling on Parallel Machines (Unrelated Machines)**

**Objective:** Minimize the makespan (the time at which the last job finishes).

```python
from ortools.sat.python import cp_model

def job_scheduling_parallel_machines(jobs, machines):
    # Create the model
    model = cp_model.CpModel()

    # Number of jobs and machines
    n = len(jobs)

    # Create the variables: job assignment to machines and finish time on each machine
    x = [model.new_int_var(0, machines - 1, f"x_{i}") for i in range(n)]  # job-to-machine assignment
    finish = [model.new_int_var(0, sum(jobs), f"finish_{i}") for i in range(machines)]  # finish times per machine

    # Create the constraints: each job should be assigned to exactly one machine
    for i in range(n):
        model.add_constraint(x[i] < machines)

    # Create constraints for each machine
    for j in range(machines):
        model.add_constraint(sum(x[i] == j for i in range(n)) == 1)  # Each machine gets at least one job

    # Minimize the makespan (maximize the time it takes to finish all jobs)
    model.minimize(max(finish))

    # Create the solver and solve the model
    solver = cp_model.CpSolver()
    status = solver.solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"Total Makespan: {solver.ObjectiveValue()}")
        for i in range(n):
            print(f"Job {i} is assigned to machine {solver.Value(x[i])}, finishes at {solver.Value(finish[i])}")
    else:
        print("No solution found")

# Sample input: job durations and number of machines
jobs = [5, 8, 3, 7, 6]
machines = 3
job_scheduling_parallel_machines(jobs, machines)
```

### 4. **Job Precedence Constraints (Task Dependencies)**

**Objective:** Schedule jobs considering precedence constraints (job A must finish before job B can start).

```python
from ortools.sat.python import cp_model

def job_scheduling_with_precedence(jobs, dependencies):
    # Create the model
    model = cp_model.CpModel()

    # Number of jobs
    n = len(jobs)

    # Create the variables (start times for each job)
    start = [model.new_int_var(0, sum(jobs), f"start_{i}") for i in range(n)]
    end = [model.new_int_var(0, sum(jobs), f"end_{i}") for i in range(n)]

    # Create the constraints (end = start + duration)
    for i in range(n):
        model.add_constraint(end[i] == start[i] + jobs[i])

    # Precedence constraints
    for dep in dependencies:
        # Dep is a tuple (before_job, after_job)
        before, after = dep
        model.add_constraint(start[after] >= end[before])

    # Minimize the makespan (maximum completion time)
    model.minimize(max(end))

    # Create the solver and solve the model
    solver = cp_model.CpSolver()
    status = solver.solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"Total Makespan: {solver.ObjectiveValue()}")
        for i in range(n):
            print(f"Job {i} starts at {solver.Value(start[i])} and ends at {solver.Value(end[i])}")
    else:
        print("No solution found")

# Sample input: job durations and precedence relations
jobs = [5, 8, 3, 7, 6]
# Example: job 1 must finish before job 2 starts, job 3 before job 4, etc.
dependencies = [(0, 1), (1, 2), (3, 4)]  
job_scheduling_with_precedence(jobs, dependencies)
```

---

Here are code samples for **Job Scheduling variants (5-8)** using **constraint programming (CSP)** approach with OR-Tools:

### 5. **Job Scheduling with Job Release Times**

**Objective:** Minimize the total completion time of jobs considering release times (when jobs are ready to be scheduled).

```python
from ortools.sat.python import cp_model

def job_scheduling_with_release_times(jobs, release_times):
    # Create the model
    model = cp_model.CpModel()

    # Number of jobs
    n = len(jobs)

    # Create the variables (start times for each job)
    start = [model.new_int_var(0, sum(jobs), f"start_{i}") for i in range(n)]
    end = [model.new_int_var(0, sum(jobs), f"end_{i}") for i in range(n)]

    # Create the constraints (end = start + duration)
    for i in range(n):
        model.add_constraint(end[i] == start[i] + jobs[i])
        model.add_constraint(start[i] >= release_times[i])

    # Minimize the makespan (maximum completion time)
    model.minimize(max(end))

    # Create the solver and solve the model
    solver = cp_model.CpSolver()
    status = solver.solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"Total Makespan: {solver.ObjectiveValue()}")
        for i in range(n):
            print(f"Job {i} starts at {solver.Value(start[i])} and ends at {solver.Value(end[i])}")
    else:
        print("No solution found")

# Sample input: job durations and release times
jobs = [5, 8, 3, 7, 6]  # Job durations
release_times = [2, 1, 0, 4, 3]  # Release times for each job
job_scheduling_with_release_times(jobs, release_times)
```

### 6. **Job Scheduling with Job Priorities**

**Objective:** Schedule jobs such that higher priority jobs are completed first.

```python
from ortools.sat.python import cp_model

def job_scheduling_with_priorities(jobs, priorities):
    # Create the model
    model = cp_model.CpModel()

    # Number of jobs
    n = len(jobs)

    # Create the variables (start times for each job)
    start = [model.new_int_var(0, sum(jobs), f"start_{i}") for i in range(n)]
    end = [model.new_int_var(0, sum(jobs), f"end_{i}") for i in range(n)]

    # Create the constraints (end = start + duration)
    for i in range(n):
        model.add_constraint(end[i] == start[i] + jobs[i])

    # Create the constraints to prioritize jobs
    for i in range(n):
        for j in range(i + 1, n):
            if priorities[i] < priorities[j]:
                model.add_constraint(start[i] >= end[j])

    # Minimize the makespan (maximum completion time)
    model.minimize(max(end))

    # Create the solver and solve the model
    solver = cp_model.CpSolver()
    status = solver.solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"Total Makespan: {solver.ObjectiveValue()}")
        for i in range(n):
            print(f"Job {i} starts at {solver.Value(start[i])} and ends at {solver.Value(end[i])}")
    else:
        print("No solution found")

# Sample input: job durations and job priorities
jobs = [5, 8, 3, 7, 6]  # Job durations
priorities = [2, 1, 4, 3, 5]  # Priorities (higher priority number indicates higher priority)
job_scheduling_with_priorities(jobs, priorities)
```

### 7. **Job Scheduling with Machine Setup Time**

**Objective:** Minimize the total makespan considering setup times between jobs.

```python
from ortools.sat.python import cp_model

def job_scheduling_with_setup_time(jobs, setup_times):
    # Create the model
    model = cp_model.CpModel()

    # Number of jobs
    n = len(jobs)

    # Create the variables (start times for each job)
    start = [model.new_int_var(0, sum(jobs), f"start_{i}") for i in range(n)]
    end = [model.new_int_var(0, sum(jobs), f"end_{i}") for i in range(n)]

    # Create the constraints (end = start + duration)
    for i in range(n):
        model.add_constraint(end[i] == start[i] + jobs[i])

    # Create the setup time constraints
    for i in range(n - 1):
        model.add_constraint(start[i + 1] >= end[i] + setup_times[i])

    # Minimize the makespan (maximum completion time)
    model.minimize(max(end))

    # Create the solver and solve the model
    solver = cp_model.CpSolver()
    status = solver.solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"Total Makespan: {solver.ObjectiveValue()}")
        for i in range(n):
            print(f"Job {i} starts at {solver.Value(start[i])} and ends at {solver.Value(end[i])}")
    else:
        print("No solution found")

# Sample input: job durations and setup times
jobs = [5, 8, 3, 7, 6]  # Job durations
setup_times = [2, 1, 4, 3]  # Setup times between jobs
job_scheduling_with_setup_time(jobs, setup_times)
```

### 8. **Job Scheduling with Resource Constraints**

**Objective:** Schedule jobs such that they do not exceed the available resources at any time.

```python
from ortools.sat.python import cp_model

def job_scheduling_with_resources(jobs, resource_requirements, available_resources):
    # Create the model
    model = cp_model.CpModel()

    # Number of jobs
    n = len(jobs)

    # Create the variables (start times for each job)
    start = [model.new_int_var(0, sum(jobs), f"start_{i}") for i in range(n)]
    end = [model.new_int_var(0, sum(jobs), f"end_{i}") for i in range(n)]

    # Create the constraints (end = start + duration)
    for i in range(n):
        model.add_constraint(end[i] == start[i] + jobs[i])

    # Create the resource constraints
    for t in range(sum(jobs)):
        total_resources_used = sum(
            (start[i] <= t < end[i]) * resource_requirements[i] for i in range(n)
        )
        model.add_constraint(total_resources_used <= available_resources)

    # Minimize the makespan (maximum completion time)
    model.minimize(max(end))

    # Create the solver and solve the model
    solver = cp_model.CpSolver()
    status = solver.solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"Total Makespan: {solver.ObjectiveValue()}")
        for i in range(n):
            print(f"Job {i} starts at {solver.Value(start[i])} and ends at {solver.Value(end[i])}")
    else:
        print("No solution found")

# Sample input: job durations, resource requirements, and available resources
jobs = [5, 8, 3, 7, 6]  # Job durations
resource_requirements = [2, 3, 1, 4, 2]  # Resource requirements for each job
available_resources = 5  # Maximum available resources at any given time
job_scheduling_with_resources(jobs, resource_requirements, available_resources)
```

---

### Explanation for each of these variants:

1. **Job Scheduling with Release Times**: Jobs can only be started once their release time arrives. The objective is to minimize the makespan.

2. **Job Scheduling with Job Priorities**: Higher-priority jobs should be scheduled first, with the objective being to minimize the makespan.

3. **Job Scheduling with Machine Setup Time**: Each job requires a setup time before it can start, and the total makespan is minimized.

4. **Job Scheduling with Resource Constraints**: Each job requires a certain amount of resources, and the goal is to minimize the makespan while ensuring that the total resources used at any time do not exceed the available resources.

