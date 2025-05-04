from ortools.sat.python import cp_model

def solve_warehouse_assignment():
    # Sample input
    frequencies = [15, 8, 20]     # Product retrieval frequency
    volumes = [2, 1, 3]           # Product volumes
    distances = [1, 2, 3]         # Slot distances from dispatch

    num_products = len(frequencies)
    num_slots = len(distances)

    model = cp_model.CpModel()

    # Variables: assign[p][s] = 1 if product p is assigned to slot s
    assign = []
    for p in range(num_products):
        assign.append([
            model.new_bool_var(f'assign_p{p}_s{s}') for s in range(num_slots)
        ])

    # Constraint 1: Each product must be assigned to exactly one slot
    for p in range(num_products):
        model.add(sum(assign[p]) == 1)

    # Constraint 2: Each slot can hold only one product (no overlaps)
    for s in range(num_slots):
        model.add(sum(assign[p][s] for p in range(num_products)) <= 1)

    # Objective: minimize total weighted distance (frequency × distance)
    objective_terms = []
    for p in range(num_products):
        for s in range(num_slots):
            cost = frequencies[p] * distances[s]
            objective_terms.append(assign[p][s] * cost)

    model.minimize(sum(objective_terms))

    # Solve
    solver = cp_model.CpSolver()
    status = solver.solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("✅ Solution:")
        for p in range(num_products):
            for s in range(num_slots):
                if solver.value(assign[p][s]):
                    print(f"Product {p+1} assigned to Slot {s+1} "
                          f"(Freq: {frequencies[p]}, Distance: {distances[s]})")
    else:
        print("❌ No solution found.")

if __name__ == "__main__":
    solve_warehouse_assignment()
