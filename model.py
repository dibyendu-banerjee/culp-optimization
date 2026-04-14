import pulp
import matplotlib.pyplot as plt
from tabulate import tabulate

def run_model():
    nurses = ["A", "B", "C"]
    days = list(range(7))
    required_nurses_per_day = 2

    model_lp = pulp.LpProblem("CULP", pulp.LpMinimize)

    x = pulp.LpVariable.dicts("shift", (nurses, days), 0, 1, pulp.LpBinary)

    total_shifts = {
        i: pulp.lpSum(x[i][d] for d in days) for i in nurses
    }

    avg_shifts = pulp.lpSum(total_shifts[i] for i in nurses) / len(nurses)

    penalty_terms = []

    # Hard constraints
    for i in nurses:
        model_lp += total_shifts[i] <= 5

    for d in days:
        model_lp += pulp.lpSum(x[i][d] for i in nurses) >= required_nurses_per_day

    # Soft constraints
    for i in nurses:
        slack = pulp.LpVariable(f"slack_off_{i}", 0)
        off_days = pulp.lpSum(1 - x[i][d] for d in days)
        model_lp += off_days + slack >= 2
        penalty_terms.append(0.2 * slack)

    for i in nurses:
        slack = pulp.LpVariable(f"slack_fair_{i}", 0)
        model_lp += total_shifts[i] - avg_shifts <= 1 + slack
        model_lp += avg_shifts - total_shifts[i] <= 1 + slack
        penalty_terms.append(0.25 * slack)

    model_lp += pulp.lpSum(penalty_terms) + 0.1 * pulp.lpSum(total_shifts[i] for i in nurses)

    model_lp.solve(pulp.PULP_CBC_CMD(msg=0))

    # Output
    table = []
    headers = ["Nurse"] + [f"D{d+1}" for d in days]

    for i in nurses:
        row = [i] + [int(pulp.value(x[i][d])) for d in days]
        table.append(row)

    print("\n=== Schedule Table ===")
    print(tabulate(table, headers=headers, tablefmt="grid"))

    values = [pulp.value(total_shifts[i]) for i in nurses]

    plt.figure()
    plt.bar(nurses, values)
    plt.title("Workload Distribution")
    plt.xlabel("Nurse")
    plt.ylabel("Shifts")
    plt.show()
