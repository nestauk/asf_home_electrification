import numpy as np
import pandas as pd
import pulp


class Battery:
    """
    Represents a domestic battery storage system.
    """

    def __init__(
        self,
        capacity_kwh: float,
        max_power_kw: float,
        lifespan: float,
        lifecycles_max: float,
        charge_efficiency: float = 0.95,
        discharge_efficiency: float = 0.95,
        state_of_charge_min: float = 0.1,
        state_of_charge_max: float = 0.9,
        state_of_charge_init: float = 0.0,
    ):
        """
        Initializes the Battery instance.

        Attributes:
            capacity_kwh: Total energy capacity of the battery in kilowatt-hours.
            max_power_kw: Maximum rate of charge/discharge in kilowatts, based on inverter.
            lifespan: Life of battery in years.
            lifecycles_max: Lifecycles available before battery degradation.
            charge_efficiency: Efficiency factor of charging (0.0 to 1.0). Default 0.95.
            discharge_efficiency: Efficiency factor of discharging (0.0 to 1.0). Default 0.95.
            state_of_charge_min: Minimum state of charge (0.0 to 1.0). Default 0.1.
            state_of_charge_max: Maximum state of charge (0.0 to 1.0). Default 0.9.
            state_of_charge_init: Initial stae of charge (0.0 to 1.0). Default 0.0 (empty).

        """
        self.total_capacity = capacity_kwh
        self.max_power_kw = max_power_kw
        self.lifespan = lifespan
        self.lifecycles_max = lifecycles_max
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.state_of_charge_min = state_of_charge_min
        self.state_of_charge_max = state_of_charge_max

        # Dynamic state variables
        self.current_charge = state_of_charge_init  # Starts empty (in kWh)

    def run_linear_cost_optimisation(
        self,
        demand_profile: pd.DataFrame,
        dp_settlement_date_col: str,
        dp_coefficients_col: str,
        cost: pd.DataFrame,
        cost_settlement_date_col: str,
        cost_col: str,
        total_annual_consumption: float,
        delta_t: float = 0.5,
    ) -> pd.DataFrame:
        """
        Optimises battery charge and discharge.

        Attributes:
            demand_profile: Pandas DataFrame of intra-daily demand.
            dp_settlement_date_col: demand_profile column indicating the date.
            dp_coefficients_col: demand profile column indicating the consumption coefficients.
            cost: Pandas DataFrame of costs (should be the same length as demand_profile)
            cost_settlement_date_col: cost column indicating the date.
            cost_settlement_period_col: cost column indicating the settlement period.
            cost_col: cost column name.
            total_annual_consumption: hosuehold consumption value to apply to demand profiles.
            delta_t: Time step in hours. Defaults to 0.5 for half-hourly data.
        """
        ### Battery power limit - the maximum power of the battery inverter in a settlement period.
        max_energy_change = self.max_power_kw * delta_t  # kWh per timestep

        ### Generate consumption values for given total consumption.
        scale_factor = total_annual_consumption / demand_profile[dp_coefficients_col].sum()
        profile = demand_profile.copy()
        profile[dp_coefficients_col] *= scale_factor

        all_days = []
        ### Set up a daily optimisation problem
        for i, date in enumerate(profile[dp_settlement_date_col].unique()):
            # the load profile is the consumption over the 24 hour period of [date]
            load_profile = profile.loc[lambda df: df[dp_settlement_date_col] == date, dp_coefficients_col].tolist()
            # costs are the cost associated with consumption over the 24 hour period of [date]
            costs = cost.loc[lambda df: df[cost_settlement_date_col] == date, cost_col].tolist()

            ### Time index - the number of unique settlement periods in a day.
            n = len(load_profile)

            # LP problem definition
            prob = pulp.LpProblem(f"battery_cost_optimisation_{i}", pulp.LpMinimize)

            ### Add decision variables to LP problem
            charge = [pulp.LpVariable(f"charge_{i}_{t}", lowBound=0, upBound=max_energy_change) for t in range(n)]
            discharge = [pulp.LpVariable(f"discharge_{i}_{t}", lowBound=0, upBound=max_energy_change) for t in range(n)]
            grid = [pulp.LpVariable(f"grid_{i}_{t}", lowBound=0) for t in range(n)]  # no export assumed
            soc = [
                pulp.LpVariable(
                    f"soc_{i}_{t}",
                    lowBound=self.state_of_charge_min * self.total_capacity,
                    upBound=self.state_of_charge_max * self.total_capacity,
                )
                for t in range(n)
            ]  # bounded by charge range limits

            ### Objective function: Minimise consumer bill for the whole day
            prob += pulp.lpSum(costs[t] * grid[t] for t in range(n))

            ### Subject to constraints
            for t in range(n):
                # Grid import = demand + charging - discharging
                prob += grid[t] == load_profile[t] + charge[t] - discharge[t]

                # Battery charging dynamics
                if t == 0:
                    prob += (
                        soc[t]
                        == self.current_charge * self.total_capacity
                        + self.charge_efficiency * charge[t]
                        - discharge[t] / self.discharge_efficiency
                    )
                else:
                    prob += (
                        soc[t]
                        == soc[t - 1] + self.charge_efficiency * charge[t] - discharge[t] / self.discharge_efficiency
                    )

            # Limit daily cycles
            max_daily_cycles = self.lifecycles_max / (self.lifespan * 365)
            usable_capacity = self.total_capacity * (self.state_of_charge_max - self.state_of_charge_min)
            prob += (
                pulp.lpSum(discharge[t] / self.discharge_efficiency for t in range(n))
                <= usable_capacity * max_daily_cycles
            )

            ### Solve using Coin-or Branch and Cut (CBC solver)
            prob.solve(pulp.PULP_CBC_CMD(msg=0))

            if pulp.LpStatus[prob.status] != "Optimal":
                raise RuntimeError(f"LP solver failed with status: {pulp.LpStatus[prob.status]}")

            ### Extract results
            charge_vals = [pulp.value(charge[t]) for t in range(n)]
            discharge_vals = [pulp.value(discharge[t]) for t in range(n)]
            grid_vals = [pulp.value(grid[t]) for t in range(n)]
            soc_vals = [pulp.value(soc[t]) for t in range(n)]
            cost_vals = [grid_vals[t] * costs[t] for t in range(n)]

            ### Report cycling
            equivalent_full_cycles = (sum(discharge_vals) / self.discharge_efficiency) / usable_capacity
            print(
                f"{str(date)} - Equivalent full cycles : {equivalent_full_cycles:.2f}, maximum daily cycles ({max_daily_cycles:.2f})"
            )

            temp = pd.DataFrame(
                {
                    "timestamp": profile.loc[
                        lambda df: df[dp_settlement_date_col] == date, dp_settlement_date_col
                    ].tolist(),
                    "demand_kwh": load_profile,
                    "soc_kwh": soc_vals,
                    "charge_kwh": charge_vals,
                    "discharge_kwh": discharge_vals,
                    "grid_kwh": grid_vals,
                    "price_p_per_kwh": costs,
                    "cost_p": cost_vals,
                }
            )
            temp["settlement_date"] = date
            all_days.append(temp)

            prob.assignVarsVals({})
            del prob
            ### Update current charge at end of day
            self.current_charge = soc_vals[-1] / self.total_capacity

        return pd.concat(all_days, ignore_index=True).reset_index(drop=True)

    def __str__(self) -> str:
        return f"Battery: {self.current_charge:.2f}/{self.total_capacity} kWh ({self.soc:.1f}% SoC)"


def run_rule_based_tariff_driven_battery(
    time_series: list,
    demand_profile: list,  # kWh
    prices: list,
    low_price_threshold: float,
    high_price_threshold: float,
    capacity: float,
    inverter_power: float,
    lifespan: int,
    max_life_cycles: int,
    soc_init: float,
    soc_min: float,
    soc_max: float,
    delta_t: float = 0.5,  # hours
    charge_efficiency: float = 0.95,
    discharge_efficiency: float = 0.95,
) -> pd.DataFrame:

    n = len(time_series)

    # battery dynamics
    max_energy_change = inverter_power * delta_t  # max kWh change per timestep

    max_daily_cycles = max_life_cycles / (lifespan * 365)
    usable_capacity = capacity * (soc_max - soc_min)
    max_daily_throughput = usable_capacity * max_daily_cycles

    # initialise outputs
    soc = np.zeros(n)
    charge = np.zeros(n)
    discharge = np.zeros(n)
    grid = np.zeros(n)
    cost = np.zeros(n)
    energy_throughput = np.zeros(n)

    # initialise battery state of charge
    soc[0] = soc_init * capacity

    # initialise daily energy throughput
    energy_throughput = 0

    for t in range(n):
        demand_t = demand_profile[t]
        price_t = prices[t]
        soc_t = soc[t]
        charge_t = 0
        discharge_t = 0

        # charge when cheap
        if price_t <= low_price_threshold and soc_t < soc_max * capacity:
            # charge as much as allowed by inverter and remaining space
            charge_t = min(
                max_energy_change,  # physically can't charge faster than this bc of inverter limit
                (soc_max * capacity - soc_t) / charge_efficiency,  # remaining battery space
            )

        # discharge when expensive
        elif price_t >= high_price_threshold and soc_t > soc_min * capacity:
            # discharge to meet demand
            discharge_t = min(
                max_energy_change,  # physically can't discharge faster than this bc of inverter limit
                (soc_t - soc_min * capacity) * discharge_efficiency,  # available usable energy in battery
                demand_t,  # only enough to cover demand
            )

        # budget from cycle limit
        remaining_budget = max(0, max_daily_throughput - energy_throughput)
        if remaining_budget <= 0:
            discharge_t = 0
        else:
            discharge_t = min(discharge_t, remaining_budget * discharge_efficiency)
        energy_throughput += discharge_t / discharge_efficiency

        # battery state of charge
        soc_next = soc_t + (charge_efficiency * charge_t) - (discharge_t / discharge_efficiency)
        soc_next = max(0, min(capacity, soc_next))

        # Energy balance
        grid_t = demand_t + charge_t - discharge_t

        # Store output
        charge[t] = charge_t
        discharge[t] = discharge_t
        grid[t] = grid_t
        cost[t] = grid_t * price_t

        if t < n - 1:
            soc[t + 1] = soc_next

    df = pd.DataFrame(
        {
            "timestamp": time_series,
            "demand_kwh": demand_profile,  # kWh
            "soc_kwh": soc,  # kWh
            "charge_kwh": charge,  # kWh
            "discharge_kwh": discharge,  # kWh
            "grid_kwh": grid,  # kWh
            "price_p_per_kwh": prices,  # p/kWh
            "cost_p": cost,  # p
        }
    )

    # Note: rule-based approach cannot enforce terminal SOC — report drift instead
    terminal_soc = soc[-1]
    target_soc = soc_init * capacity
    print(
        f"Terminal SOC : {terminal_soc:.2f} kWh  (target: {target_soc:.2f} kWh, drift: {terminal_soc - target_soc:+.2f} kWh)"
    )
    print(
        f"Equivalent full cycles : {energy_throughput / usable_capacity:.2f}, maximum daily cycles ({max_daily_cycles:.2f})"
    )

    return df


def run_linear_cost_optimisation_battery(
    time_series: list,
    demand_profile: list,  # kWh
    prices: list,
    capacity: float,
    inverter_power: float,
    lifespan: int,
    max_life_cycles: int,
    soc_init: float,  # fraction
    soc_min: float,  # fraction
    soc_max: float,  # fraction
    delta_t: float = 0.5,  # hours
    charge_efficiency: float = 0.95,
    discharge_efficiency: float = 0.95,
) -> pd.DataFrame:

    ### Time index
    n = len(time_series)

    ### Battery power limit
    max_energy_change = inverter_power * delta_t  # kWh per timestep

    # LP problem definition
    prob = pulp.LpProblem("battery_cost_optimisation", pulp.LpMinimize)

    ### Add decision variables to LP problem
    charge = [pulp.LpVariable(f"charge_{t}", lowBound=0, upBound=max_energy_change) for t in range(n)]
    discharge = [pulp.LpVariable(f"discharge_{t}", lowBound=0, upBound=max_energy_change) for t in range(n)]
    grid = [pulp.LpVariable(f"grid_{t}", lowBound=0) for t in range(n)]  # no export assumed
    soc = [
        pulp.LpVariable(f"soc_{t}", lowBound=soc_min * capacity, upBound=soc_max * capacity) for t in range(n)
    ]  # bounded by charge range limits

    ### Objective function: Minimise consumer bill for the whole day
    prob += pulp.lpSum(prices[t] * grid[t] for t in range(n))

    ### Subject to constraints
    for t in range(n):
        # Grid import = demand + charging - discharging
        prob += grid[t] == demand_profile[t] + charge[t] - discharge[t]

        # Battery charging dynamics
        if t == 0:
            prob += soc[t] == soc_init * capacity + charge_efficiency * charge[t] - discharge[t] / discharge_efficiency
        else:
            prob += soc[t] == soc[t - 1] + charge_efficiency * charge[t] - discharge[t] / discharge_efficiency

    # Terminal SOC — battery ends where it started because we are only simulating 1 day
    # prob += soc[n - 1] == soc_init * capacity

    # Limit daily cycles
    max_daily_cycles = max_life_cycles / (lifespan * 365)
    usable_capacity = capacity * (soc_max - soc_min)
    prob += pulp.lpSum(discharge[t] / discharge_efficiency for t in range(n)) <= usable_capacity * max_daily_cycles

    ### Solve using Coin-or Branch and Cut (CBC solver)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # Print solver status statements
    # print(f"Status : {pulp.LpStatus[prob.status]}")
    # print(f"Total daily cost : {pulp.value(prob.objective):.2f} p")

    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"LP solver failed with status: {pulp.LpStatus[prob.status]}")

    ### Extract results
    charge_vals = [pulp.value(charge[t]) for t in range(n)]
    discharge_vals = [pulp.value(discharge[t]) for t in range(n)]
    grid_vals = [pulp.value(grid[t]) for t in range(n)]
    soc_vals = [pulp.value(soc[t]) for t in range(n)]
    cost_vals = [grid_vals[t] * prices[t] for t in range(n)]

    ### Report cycling
    equivalent_full_cycles = (sum(discharge_vals) / discharge_efficiency) / usable_capacity
    print(f"Equivalent full cycles : {equivalent_full_cycles:.2f}, maximum daily cycles ({max_daily_cycles:.2f})")

    return pd.DataFrame(
        {
            "timestamp": time_series,
            "demand_kwh": demand_profile,
            "soc_kwh": soc_vals,
            "charge_kwh": charge_vals,
            "discharge_kwh": discharge_vals,
            "grid_kwh": grid_vals,
            "price_p_per_kwh": prices,
            "cost_p": cost_vals,
        }
    )


def run_linear_cost_optimisation_battery_with_export(
    time_series: list,
    demand_profile: list,  # kWh
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

    prob = pulp.LpProblem("battery_export_optimisation", pulp.LpMinimize)

    # Decision variables
    charge = [pulp.LpVariable(f"charge_{t}", lowBound=0, upBound=max_energy_change) for t in range(n)]
    discharge = [pulp.LpVariable(f"discharge_{t}", lowBound=0, upBound=max_energy_change) for t in range(n)]
    grid_import = [pulp.LpVariable(f"grid_import_{t}", lowBound=0) for t in range(n)]
    grid_export = [pulp.LpVariable(f"grid_export_{t}", lowBound=0) for t in range(n)]
    soc = [pulp.LpVariable(f"soc_{t}", lowBound=soc_min * capacity, upBound=soc_max * capacity) for t in range(n)]
    is_charging = [pulp.LpVariable(f"is_charging_{t}", cat="Binary") for t in range(n)]

    # Objective: minimise import cost minus export revenue
    prob += pulp.lpSum(import_prices[t] * grid_import[t] - export_prices[t] * grid_export[t] for t in range(n))

    for t in range(n):
        # Energy balance: import - export = demand + charging - discharging
        prob += grid_import[t] - grid_export[t] == demand_profile[t] + charge[t] - discharge[t]

        # Export cannot exceed discharge — prevents grid pass-through arbitrage
        prob += grid_export[t] <= discharge[t]

        # Discharge cannot exceed demand + export — battery can't discharge more than needed
        prob += discharge[t] <= demand_profile[t] + grid_export[t]

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

    # Daily export cap — limit to one usable cycle per day
    # Prevents excessive grid arbitrage while allowing legitimate surplus export
    prob += pulp.lpSum(grid_export[t] for t in range(n)) <= usable_capacity

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"LP solver failed with status: {pulp.LpStatus[prob.status]}")

    # Extract results
    charge_vals = [pulp.value(charge[t]) for t in range(n)]
    discharge_vals = [pulp.value(discharge[t]) for t in range(n)]
    grid_import_vals = [pulp.value(grid_import[t]) for t in range(n)]
    grid_export_vals = [pulp.value(grid_export[t]) for t in range(n)]
    soc_vals = [pulp.value(soc[t]) for t in range(n)]
    import_cost_vals = [import_prices[t] * grid_import_vals[t] for t in range(n)]
    export_cost_vals = [export_prices[t] * grid_export_vals[t] for t in range(n)]

    return pd.DataFrame(
        {
            "timestamp": time_series,
            "demand_kwh": demand_profile,
            "soc_kwh": soc_vals,
            "charge_kwh": charge_vals,
            "discharge_kwh": discharge_vals,
            "grid_import_kwh": grid_import_vals,
            "grid_export_kwh": grid_export_vals,
            "import_price_p_per_kwh": import_prices,
            "export_price_p_per_kwh": export_prices,
            "import_cost_p": import_cost_vals,
            "export_cost_p": export_cost_vals,
        }
    )


# def run_linear_cost_optimisation_battery_with_export(
#     time_series: list,
#     demand_profile: list,  # kWh
#     import_prices: list,  # p/kWh
#     export_prices: list,  # p/kWh
#     capacity: float,
#     inverter_power: float,
#     lifespan: int,
#     max_life_cycles: int,
#     soc_init: float,
#     soc_min: float,
#     soc_max: float,
#     delta_t: float = 0.5,
#     charge_efficiency: float = 0.95,
#     discharge_efficiency: float = 0.95,
# ) -> pd.DataFrame:

#     n = len(time_series)
#     max_energy_change = inverter_power * delta_t
#     max_import = inverter_power * delta_t + max(demand_profile)

#     prob = pulp.LpProblem("battery_export_optimisation", pulp.LpMinimize)

#     # Decision variables
#     charge = [pulp.LpVariable(f"charge_{t}", lowBound=0, upBound=max_energy_change) for t in range(n)]
#     discharge = [pulp.LpVariable(f"discharge_{t}", lowBound=0, upBound=max_energy_change) for t in range(n)]
#     grid_import = [pulp.LpVariable(f"grid_import_{t}", lowBound=0, upBound=max_import) for t in range(n)]
#     grid_export = [pulp.LpVariable(f"grid_export_{t}", lowBound=0) for t in range(n)]
#     soc = [pulp.LpVariable(f"soc_{t}", lowBound=soc_min * capacity, upBound=soc_max * capacity) for t in range(n)]

#     # Objective: minimise import cost minus export revenue
#     prob += pulp.lpSum(import_prices[t] * grid_import[t] - export_prices[t] * grid_export[t] for t in range(n))

#     for t in range(n):
#         # Energy balance: import - export = demand + charging - discharging
#         prob += grid_import[t] - grid_export[t] == demand_profile[t] + charge[t] - discharge[t]

#         # Export cannot exceed discharge — prevents grid pass-through arbitrage
#         prob += grid_export[t] <= discharge[t]

#         # Discharge cannot exceed demand + export — battery can't discharge more than needed
#         prob += discharge[t] <= demand_profile[t] + grid_export[t]

#         # Battery charging dynamics
#         if t == 0:
#             prob += soc[t] == soc_init * capacity + charge_efficiency * charge[t] - discharge[t] / discharge_efficiency
#         else:
#             prob += soc[t] == soc[t - 1] + charge_efficiency * charge[t] - discharge[t] / discharge_efficiency

#         # Negative import prices: cap grid import at battery max and demand to prevent unbounded imports
#         if import_prices[t] < 0:
#             prob += grid_import[t] <= max_energy_change + demand_profile[t]

#         # Negative export prices: prevent export entirely
#         if export_prices[t] < 0:
#             prob += grid_export[t] == 0

#     # Cycle limit
#     max_daily_cycles = max_life_cycles / (lifespan * 365)
#     usable_capacity = capacity * (soc_max - soc_min)
#     prob += pulp.lpSum(discharge[t] / discharge_efficiency for t in range(n)) <= usable_capacity * max_daily_cycles

#     prob.solve(pulp.PULP_CBC_CMD(msg=0))

#     if pulp.LpStatus[prob.status] != "Optimal":
#         raise RuntimeError(f"LP solver failed with status: {pulp.LpStatus[prob.status]}")

#     # Extract results
#     charge_vals = [pulp.value(charge[t]) for t in range(n)]
#     discharge_vals = [pulp.value(discharge[t]) for t in range(n)]
#     grid_import_vals = [pulp.value(grid_import[t]) for t in range(n)]
#     grid_export_vals = [pulp.value(grid_export[t]) for t in range(n)]
#     soc_vals = [pulp.value(soc[t]) for t in range(n)]
#     import_cost_vals = [import_prices[t] * grid_import_vals[t] for t in range(n)]
#     export_cost_vals = [export_prices[t] * grid_export_vals[t] for t in range(n)]

#     return pd.DataFrame(
#         {
#             "timestamp": time_series,
#             "demand_kwh": demand_profile,
#             "soc_kwh": soc_vals,
#             "charge_kwh": charge_vals,
#             "discharge_kwh": discharge_vals,
#             "grid_import_kwh": grid_import_vals,
#             "grid_export_kwh": grid_export_vals,
#             "import_price_p_per_kwh": import_prices,
#             "export_price_p_per_kwh": export_prices,
#             "import_cost_p": import_cost_vals,
#             "export_cost_p": export_cost_vals,
#         }
#     )
