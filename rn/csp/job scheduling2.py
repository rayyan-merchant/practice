from ortools.sat.python import cp_model

def job_scheduling_template():
    model = cp_model.CpModel()

    num_jobs = 3
    max_time = 20

    # Variables: start time for each job
    starts = [model.new_int_var(0, max_time, f"start_{i}") for i in range(num_jobs)]
    durations = [5, 3, 2]

    # End time
    ends = [model.new_int_var(0, max_time, f"end_{i}") for i in range(num_jobs)]

    for i in range(num_jobs):
        model.add(ends[i] == starts[i] + durations[i])

    # Jobs must not overlap (1-machine constraint)
    for i in range(num_jobs):
        for j in range(i + 1, num_jobs):
            model.add_no_overlap([cp_model.IntervalVar(starts[i], durations[i], ends[i]),
                                  cp_model.IntervalVar(starts[j], durations[j], ends[j])])

    # Objective: minimize makespan
    makespan = model.new_int_var(0, max_time, "makespan")
    model.add_max_equality(makespan, ends)
    model.minimize(makespan)

    solver = cp_model.CpSolver()
    status = solver.solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"Makespan = {solver.value(makespan)}")
        for i in range(num_jobs):
            print(f"Job {i} starts at {solver.value(starts[i])} and ends at {solver.value(ends[i])}")
    else:
        print("No solution found.")
