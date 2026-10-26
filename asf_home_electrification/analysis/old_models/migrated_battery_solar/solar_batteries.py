import pandas as pd
import pulp


def run_linear_cost_optimisation_solar_battery(
    time_series: list,
    demand_profile: list,  # kWh
    solar_profile: list,  # kWh - generation per half hour
    import_prices: list,  # p/kWh
    export_prices: list,  # p/kWh
    capacity: float,
    inverter_power: float,
    lifespan: int,
    max_life_cycles: int,
    soc_init: float,
    soc_min: float,
    soc_max: float,
    delta_t: float = 0.5,
    charge_efficiency: float = 0.95,
    discharge_efficiency: float = 0.95,
) -> pd.DataFrame:

    n = len(time_series)
    max_energy_change = inverter_power * delta_t
    # max_import = inverter_power * delta_t + max(demand_profile)

    prob = pulp.LpProblem("solar_battery_optimisation", pulp.LpMinimize)

    # Decision variables
    charge = [pulp.LpVariable(f"charge_{t}", lowBound=0, upBound=max_energy_change) for t in range(n)]
    discharge = [pulp.LpVariable(f"discharge_{t}", lowBound=0, upBound=max_energy_change) for t in range(n)]
    grid_import = [pulp.LpVariable(f"grid_import_{t}", lowBound=0) for t in range(n)]
    grid_export = [pulp.LpVariable(f"grid_export_{t}", lowBound=0) for t in range(n)]
    export_solar = [pulp.LpVariable(f"export_solar_{t}", lowBound=0) for t in range(n)]
    export_battery = [pulp.LpVariable(f"export_battery_{t}", lowBound=0) for t in range(n)]
    soc = [pulp.LpVariable(f"soc_{t}", lowBound=soc_min * capacity, upBound=soc_max * capacity) for t in range(n)]
    is_charging = [pulp.LpVariable(f"is_charging_{t}", cat="Binary") for t in range(n)]

    # Objective: minimise import cost minus export revenue
    prob += pulp.lpSum(import_prices[t] * grid_import[t] - export_prices[t] * grid_export[t] for t in range(n))

    for t in range(n):
        # Total export is sum of solar and battery export
        prob += grid_export[t] == export_solar[t] + export_battery[t]

        # Solar export capped at generation
        prob += export_solar[t] <= solar_profile[t]

        # Battery export cannot exceed discharge, prevents pass-through arbitrage
        prob += export_battery[t] <= discharge[t]

        # Energy balance: demand + charging = solar + discharge + grid_import - grid_export
        prob += demand_profile[t] + charge[t] + grid_export[t] == solar_profile[t] + discharge[t] + grid_import[t]

        # Prevent simultaneous charge and discharge using binary variable
        prob += charge[t] <= max_energy_change * is_charging[t]
        prob += discharge[t] <= max_energy_change * (1 - is_charging[t])

        # Battery charging dynamics
        if t == 0:
            prob += soc[t] == soc_init * capacity + charge_efficiency * charge[t] - discharge[t] / discharge_efficiency
        else:
            prob += soc[t] == soc[t - 1] + charge_efficiency * charge[t] - discharge[t] / discharge_efficiency

        # Cap grid import at battery max and demand to prevent unbounded imports
        prob += grid_import[t] <= max_energy_change + demand_profile[t]

        # Negative export prices: prevent export entirely
        if export_prices[t] < 0:
            prob += grid_export[t] == 0

    # Cycle limit
    max_daily_cycles = max_life_cycles / (lifespan * 365)
    usable_capacity = capacity * (soc_max - soc_min)
    prob += pulp.lpSum(discharge[t] / discharge_efficiency for t in range(n)) <= usable_capacity * max_daily_cycles

    # Daily battery export cap — limit to one usable cycle per day
    prob += pulp.lpSum(export_battery[t] for t in range(n)) <= usable_capacity

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"LP solver failed with status: {pulp.LpStatus[prob.status]}")

    # Extract results
    charge_vals = [pulp.value(charge[t]) for t in range(n)]
    discharge_vals = [pulp.value(discharge[t]) for t in range(n)]
    grid_import_vals = [pulp.value(grid_import[t]) for t in range(n)]
    grid_export_vals = [pulp.value(grid_export[t]) for t in range(n)]
    export_solar_vals = [pulp.value(export_solar[t]) for t in range(n)]
    export_battery_vals = [pulp.value(export_battery[t]) for t in range(n)]
    soc_vals = [pulp.value(soc[t]) for t in range(n)]
    import_cost_vals = [import_prices[t] * grid_import_vals[t] for t in range(n)]
    export_cost_vals = [export_prices[t] * grid_export_vals[t] for t in range(n)]
    export_solar_cost_vals = [export_prices[t] * export_solar_vals[t] for t in range(n)]
    export_battery_cost_vals = [export_prices[t] * export_battery_vals[t] for t in range(n)]

    solar_self_consumption_vals = [solar_profile[t] - export_solar_vals[t] for t in range(n)]

    equivalent_full_cycles = (sum(discharge_vals) / discharge_efficiency) / usable_capacity
    print(f"Equivalent full cycles: {equivalent_full_cycles:.2f}, max daily: {max_daily_cycles:.2f}")

    return pd.DataFrame(
        {
            "timestamp": time_series,
            "demand_kwh": demand_profile,
            "solar_kwh": solar_profile,
            "soc_kwh": soc_vals,
            "charge_kwh": charge_vals,
            "discharge_kwh": discharge_vals,
            "grid_import_kwh": grid_import_vals,
            "grid_export_kwh": grid_export_vals,
            "export_solar_kwh": export_solar_vals,
            "export_battery_kwh": export_battery_vals,
            "import_price_p_per_kwh": import_prices,
            "export_price_p_per_kwh": export_prices,
            "import_cost_p": import_cost_vals,
            "export_cost_p": export_cost_vals,
            "export_solar_p": export_solar_cost_vals,
            "export_battery_p": export_battery_cost_vals,
            "solar_self_consumption_kwh": solar_self_consumption_vals,
        }
    )


def run_linear_cost_optimisation_solar_battery_plugin(
    time_series: list,
    demand_profile: list,  # kWh
    solar_profile: list,  # kWh - generation per half hour
    import_prices: list,  # p/kWh
    capacity: float,
    inverter_power: float,
    lifespan: int,
    max_life_cycles: int,
    soc_init: float,
    soc_min: float,
    soc_max: float,
    delta_t: float = 0.5,
    charge_efficiency: float = 0.95,
    discharge_efficiency: float = 0.95,
) -> pd.DataFrame:
    """No export capability."""

    n = len(time_series)
    max_energy_change = inverter_power * delta_t
    # max_import = inverter_power * delta_t + max(demand_profile)

    prob = pulp.LpProblem("solar_battery_plugin_optimisation", pulp.LpMinimize)

    # Decision variables
    charge = [pulp.LpVariable(f"charge_{t}", lowBound=0, upBound=max_energy_change) for t in range(n)]
    discharge = [pulp.LpVariable(f"discharge_{t}", lowBound=0, upBound=max_energy_change) for t in range(n)]
    grid_import = [pulp.LpVariable(f"grid_import_{t}", lowBound=0) for t in range(n)]
    surplus = [pulp.LpVariable(f"curtail_{t}", lowBound=0) for t in range(n)]
    soc = [pulp.LpVariable(f"soc_{t}", lowBound=soc_min * capacity, upBound=soc_max * capacity) for t in range(n)]
    is_charging = [pulp.LpVariable(f"is_charging_{t}", cat="Binary") for t in range(n)]

    # Objective: minimise import cost only — no export revenue
    prob += pulp.lpSum(import_prices[t] * grid_import[t] for t in range(n))

    for t in range(n):
        # Energy balance: demand + charging + unused surplus = solar + discharge + grid_import
        prob += demand_profile[t] + charge[t] + surplus[t] == solar_profile[t] + discharge[t] + grid_import[t]

        # Prevent simultaneous charge and discharge using binary variable
        prob += charge[t] <= max_energy_change * is_charging[t]
        prob += discharge[t] <= max_energy_change * (1 - is_charging[t])

        # Battery charging dynamics
        if t == 0:
            prob += soc[t] == soc_init * capacity + charge_efficiency * charge[t] - discharge[t] / discharge_efficiency
        else:
            prob += soc[t] == soc[t - 1] + charge_efficiency * charge[t] - discharge[t] / discharge_efficiency

        # Cap grid import at battery max and demand to prevent unbounded imports
        prob += grid_import[t] <= max_energy_change + demand_profile[t]

    # Cycle limit
    max_daily_cycles = max_life_cycles / (lifespan * 365)
    usable_capacity = capacity * (soc_max - soc_min)
    prob += pulp.lpSum(discharge[t] / discharge_efficiency for t in range(n)) <= usable_capacity * max_daily_cycles

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"LP solver failed with status: {pulp.LpStatus[prob.status]}")

    # Extract results
    charge_vals = [pulp.value(charge[t]) for t in range(n)]
    discharge_vals = [pulp.value(discharge[t]) for t in range(n)]
    grid_import_vals = [pulp.value(grid_import[t]) for t in range(n)]
    surplus_vals = [pulp.value(surplus[t]) for t in range(n)]
    soc_vals = [pulp.value(soc[t]) for t in range(n)]
    import_cost_vals = [import_prices[t] * grid_import_vals[t] for t in range(n)]

    # Solar self consumption = generation minus curtailed surplus
    solar_self_consumption_vals = [solar_profile[t] - surplus_vals[t] for t in range(n)]

    return pd.DataFrame(
        {
            "timestamp": time_series,
            "demand_kwh": demand_profile,
            "solar_kwh": solar_profile,
            "soc_kwh": soc_vals,
            "charge_kwh": charge_vals,
            "discharge_kwh": discharge_vals,
            "grid_import_kwh": grid_import_vals,
            "surplus_kwh": surplus_vals,
            "import_price_p_per_kwh": import_prices,
            "import_cost_p": import_cost_vals,
            "solar_self_consumption_kwh": solar_self_consumption_vals,
        }
    )
