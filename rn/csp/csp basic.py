from ortools.sat.python import cp_model

def csp_template():
    model = cp_model.CpModel()

    # Variables
    var1 = model.new_int_var(0, 9, "var1")
    var2 = model.new_int_var(0, 9, "var2")

    # Constraints
    model.add(var1 != var2)
    model.add(var1 + var2 <= 10)

    # Objective (optional)
    # model.maximize(var1 + var2)

    # Solve
    solver = cp_model.CpSolver()
    status = solver.solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"var1 = {solver.value(var1)}, var2 = {solver.value(var2)}")
    else:
        print("No solution found.")
