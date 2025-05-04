from ortools.sat.python import cp_model

def main():
    products = [
        {"id": 1, "freq": 15, "size": 2},
        {"id": 2, "freq": 8, "size": 1},
        {"id": 3, "freq": 20, "size": 3}
    ]
    
    slots = [
        {"id": 1, "dist": 1},
        {"id": 2, "dist": 2}, 
        {"id": 3, "dist": 3}
    ]

    model = cp_model.CpModel()

    assignment = {}

    for product in products:
        product_id = product["id"]
        assignment[product_id] = model.NewIntVar(1, 3, f"slot_for_product_{product_id}")

    all_slot_assignments = []
    for product in products:
        all_slot_assignments.append(assignment[product["id"]])
    
    model.AddAllDifferent(all_slot_assignments)

    total_cost = 0
    for p in products:
        for s in slots:
            in_slot = model.NewBoolVar(f'in_slot_{p["id"]}_{s["id"]}')
            model.Add(assignment[p["id"]] == s["id"]).OnlyEnforceIf(in_slot)
            model.Add(assignment[p["id"]] != s["id"]).OnlyEnforceIf(in_slot.Not())
            total_cost += p["freq"] * s["dist"] * in_slot

    model.Minimize(total_cost)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL:
        print("Best arrangement:")
        for p in products:
            slot = solver.Value(assignment[p["id"]])
            print(f"Product {p['id']} (freq {p['freq']}) → Slot {slot} (distance {slots[slot-1]['dist']})")
        print(f"Total cost: {solver.ObjectiveValue()}")
    else:
        print("No solution found")

if __name__ == "__main__":
    main()
