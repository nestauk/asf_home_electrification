"""Home battery model and simulations of its operation with electricity tariffs.

Provides a `Battery` class that holds the technical limits of a home battery
and runs half-hourly simulations on a demand profile and price series: a fixed
charge schedule, a cost-minimising optimisation, and a cost-minimising
optimisation that allows export to the grid.
"""

import logging

import numpy as np
import pandas as pd
import pulp

logger = logging.getLogger(__name__)


class Battery:
    """Home battery storage unit with technical limits and a starting charge.

    Models the technical limits of a battery (capacity, power, efficiency,
    state-of-charge window, and degradation limits). Simulations start from
    `starting_charge_kwh` and do not modify the battery.
    """

    def __init__(
        self,
        capacity_kwh: float,
        max_power_kw: float,
        lifespan_years: float,
        cycle_life: float,
        charge_efficiency: float = 0.95,
        discharge_efficiency: float = 0.95,
        state_of_charge_min_frac: float = 0.1,
        state_of_charge_max_frac: float = 0.9,
        state_of_charge_init_frac: float = 0.1,
    ) -> None:
        """Initialise a battery with its technical parameters and starting charge.

        Args:
            capacity_kwh: Total energy storage capacity in kWh.
            max_power_kw: Maximum charge/discharge power in kW.
            lifespan_years: Expected battery lifespan in years.
            cycle_life: Rated number of full charge-discharge cycles
                before the battery reaches end of life.
            charge_efficiency: Fraction of input energy stored when charging,
                between 0 and 1.
            discharge_efficiency: Fraction of stored energy delivered when
                discharging, between 0 and 1.
            state_of_charge_min_frac: Minimum allowed state of charge as a
                fraction of total capacity, between 0 and 1.
            state_of_charge_max_frac: Maximum allowed state of charge as a
                fraction of total capacity, between 0 and 1.
            state_of_charge_init_frac: Initial state of charge as a fraction
                of total capacity, between the minimum and maximum.

        Raises:
            ValueError: If the state-of-charge fractions are inconsistent.
        """
        if not 0 <= state_of_charge_min_frac < state_of_charge_max_frac <= 1:
            raise ValueError("Require 0 <= min < max <= 1 for state-of-charge fractions.")
        if not state_of_charge_min_frac <= state_of_charge_init_frac <= state_of_charge_max_frac:
            raise ValueError("Initial state of charge must lie between the minimum and maximum.")

        self.capacity_kwh = capacity_kwh
        self.max_power_kw = max_power_kw
        self.lifespan_years = lifespan_years
        self.cycle_life = cycle_life
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.state_of_charge_min_frac = state_of_charge_min_frac
        self.state_of_charge_max_frac = state_of_charge_max_frac
        self.state_of_charge_init_frac = state_of_charge_init_frac

    def __repr__(self) -> str:
        """Return a string representation of the battery's technical parameters."""
        return (
            f"{type(self).__name__}("
            f"capacity_kwh={self.capacity_kwh}, "
            f"max_power_kw={self.max_power_kw}, "
            f"lifespan_years={self.lifespan_years}, "
            f"cycle_life={self.cycle_life}, "
            f"charge_efficiency={self.charge_efficiency}, "
            f"discharge_efficiency={self.discharge_efficiency}, "
            f"state_of_charge_min_frac={self.state_of_charge_min_frac}, "
            f"state_of_charge_max_frac={self.state_of_charge_max_frac}, "
            f"state_of_charge_init_frac={self.state_of_charge_init_frac})"
        )

    @property
    def starting_charge_kwh(self) -> float:
        """Stored energy at the start of a simulation, in kWh."""
        return self.capacity_kwh * self.state_of_charge_init_frac

    @property
    def soc_min_kwh(self) -> float:
        """Minimum allowed stored energy in kWh."""
        return self.capacity_kwh * self.state_of_charge_min_frac

    @property
    def soc_max_kwh(self) -> float:
        """Maximum allowed stored energy in kWh."""
        return self.capacity_kwh * self.state_of_charge_max_frac

    @property
    def usable_capacity_kwh(self) -> float:
        """Energy available between the minimum and maximum state of charge, in kWh."""
        return self.soc_max_kwh - self.soc_min_kwh

    @property
    def max_daily_cycles(self) -> float:
        """Daily cycle budget, spreading the rated cycle life evenly over the lifespan."""
        return self.cycle_life / (self.lifespan_years * 365)

    @property
    def max_daily_throughput_kwh(self) -> float:
        """Maximum energy removed from the battery per day, in kWh.

        Equal to the usable capacity multiplied by the daily cycle budget.
        Measured on the battery side, i.e. before discharge efficiency losses,
        so energy delivered to the home is this value times
        `discharge_efficiency`.
        """
        return self.usable_capacity_kwh * self.max_daily_cycles

    def run_fixed_schedule(
        self,
        demand_profile: pd.Series,
        import_price_profile: pd.Series,
        charge_schedule: pd.Series,
        delta_t: float = 0.5,
    ) -> pd.DataFrame:
        """Run the battery on a fixed charge/discharge schedule. No export.

        The battery charges from the grid in intervals where the schedule is 1,
        discharges to cover household demand where it is -1, and does nothing
        where it is 0. It never exports to the grid. Discharge is limited by
        the daily throughput budget, which resets each calendar day.

        The profiles can cover any period length, from a single day to
        multiple years, as long as they share the same DatetimeIndex and
        its interval matches `delta_t`. The battery starts the period at
        `starting_charge_kwh` and its state carries across days within the run.

        Args:
            demand_profile: Household demand per interval in kWh.
            import_price_profile: Electricity import price per interval in p/kWh.
            charge_schedule: Per interval: 1 to charge, -1 to discharge,
                0 to stay idle.
            delta_t: Interval length in hours. Defaults to 0.5 (half-hourly).

        Returns:
            One row per interval, on the same index as the inputs, with
            columns: demand_kwh, soc_kwh (end of interval), charge_kwh
            (drawn from the grid), discharge_kwh (delivered to the home),
            grid_import_kwh, import_price_p_per_kwh and cost_p (pence).

        Raises:
            ValueError: If the inputs do not share the same DatetimeIndex, or
                the schedule contains values other than -1, 0 and 1.
        """
        if not (
            demand_profile.index.equals(import_price_profile.index)
            and demand_profile.index.equals(charge_schedule.index)
        ):
            raise ValueError("Inputs must share the same index.")
        if not isinstance(demand_profile.index, pd.DatetimeIndex):
            raise ValueError("Inputs must be indexed by a DatetimeIndex.")
        if not charge_schedule.isin([-1, 0, 1]).all():
            raise ValueError("charge_schedule must contain only -1, 0 and 1.")

        demand = demand_profile.to_numpy(dtype=float)
        prices = import_price_profile.to_numpy(dtype=float)
        modes = charge_schedule.to_numpy(dtype=int)

        interval_dates = demand_profile.index.normalize()
        n = len(demand)

        max_energy_per_step = self.max_power_kw * delta_t  # kWh per interval

        charge = np.zeros(n)
        discharge = np.zeros(n)
        soc = np.zeros(n)

        soc_kwh = self.starting_charge_kwh
        day_throughput_kwh = 0.0

        for t in range(n):
            if t > 0 and interval_dates[t] != interval_dates[t - 1]:
                day_throughput_kwh = 0.0  # new day, reset the cycle budget

            if modes[t] == 1:
                charge[t] = max(
                    0.0,
                    min(
                        max_energy_per_step,  # inverter limit
                        (self.soc_max_kwh - soc_kwh) / self.charge_efficiency,  # space left
                    ),
                )
            elif modes[t] == -1:
                remaining_budget_kwh = max(0.0, self.max_daily_throughput_kwh - day_throughput_kwh)
                discharge[t] = max(
                    0.0,
                    min(
                        max_energy_per_step,  # inverter limit
                        (soc_kwh - self.soc_min_kwh) * self.discharge_efficiency,  # usable energy
                        demand[t],  # cover demand only, no export
                        remaining_budget_kwh * self.discharge_efficiency,  # cycle budget
                    ),
                )

            energy_removed_kwh = discharge[t] / self.discharge_efficiency
            day_throughput_kwh += energy_removed_kwh
            soc_kwh += self.charge_efficiency * charge[t] - energy_removed_kwh
            soc[t] = soc_kwh

        grid_import = demand + charge - discharge

        return pd.DataFrame(
            {
                "demand_kwh": demand,
                "soc_kwh": soc,
                "charge_kwh": charge,
                "discharge_kwh": discharge,
                "grid_import_kwh": grid_import,
                "import_price_p_per_kwh": prices,
                "cost_p": grid_import * prices,
            },
            index=demand_profile.index,
        )

    def run_cost_optimisation(
        self,
        demand_profile: pd.Series,
        import_price_profile: pd.Series,
        delta_t: float = 0.5,
    ) -> pd.DataFrame:
        """Minimise the electricity bill as a perfect-foresight linear programming problem.

        Chooses charge and discharge in every interval to minimise total import
        cost over the whole period, subject to inverter limits, state-of-charge
        limits, round-trip efficiency and a daily throughput budget. The
        battery never exports to the grid, and must end the period with at
        least as much stored energy as it started with. Export is not modelled.

        The profiles can cover any period length, from a single day to
        multiple years, as long as they share the same DatetimeIndex and
        its interval matches `delta_t`. The whole period is solved as one
        problem, so charge can carry across days.

        Charging and discharging can both occur in the same interval (the
        battery can switch within a half hour), but together they cannot
        exceed the inverter limit, and discharge cannot exceed household
        demand since export is not modelled.

        Args:
            demand_profile: Household demand per interval in kWh.
            import_price_profile: Electricity import price per interval in p/kWh.
            delta_t: Interval length in hours. Defaults to 0.5 (half-hourly).

        Returns:
            One row per interval, on the same index as the inputs, with
            columns: demand_kwh, soc_kwh (end of interval), charge_kwh
            (drawn from the grid), discharge_kwh (discharged to meet demand),
            grid_import_kwh, import_price_p_per_kwh and cost_p (pence).

        Raises:
            ValueError: If the inputs do not share the same DatetimeIndex.
            RuntimeError: If the solver does not find an optimal solution.
        """
        if not demand_profile.index.equals(import_price_profile.index):
            raise ValueError("Inputs must share the same index.")
        if not isinstance(demand_profile.index, pd.DatetimeIndex):
            raise ValueError("Inputs must be indexed by a DatetimeIndex.")

        demand = demand_profile.to_numpy(dtype=float)
        prices = import_price_profile.to_numpy(dtype=float)
        n = len(demand)

        max_energy_per_step = self.max_power_kw * delta_t  # kWh per interval

        prob = pulp.LpProblem("battery_cost_optimisation", pulp.LpMinimize)

        # Decision variables (all energy in kWh)
        charge = [prob.add_variable(f"charge_{t}", lowBound=0, upBound=max_energy_per_step) for t in range(n)]
        discharge = [prob.add_variable(f"discharge_{t}", lowBound=0, upBound=max_energy_per_step) for t in range(n)]
        grid = [prob.add_variable(f"grid_{t}", lowBound=0) for t in range(n)]  # no export
        soc = [prob.add_variable(f"soc_{t}", lowBound=self.soc_min_kwh, upBound=self.soc_max_kwh) for t in range(n)]

        # Objective: total electricity bill (pence)
        prob += pulp.lpSum(prices[t] * grid[t] for t in range(n))

        for t in range(n):
            # Grid import = demand + charging - discharging
            prob += grid[t] == demand[t] + charge[t] - discharge[t]

            # Discharge only serves the home
            prob += discharge[t] <= demand[t]

            # Combined charge and discharge cannot exceed the inverter's energy limit for the interval
            prob += charge[t] + discharge[t] <= max_energy_per_step

            # Battery dynamics
            previous_soc = self.starting_charge_kwh if t == 0 else soc[t - 1]
            prob += (
                soc[t] == previous_soc + self.charge_efficiency * charge[t] - discharge[t] / self.discharge_efficiency
            )

        # Daily throughput (energy removed from the battery) within budget
        interval_dates = demand_profile.index.normalize()
        for intervals_in_day in pd.Series(np.arange(n)).groupby(interval_dates).agg(list):
            prob += (
                pulp.lpSum(discharge[t] / self.discharge_efficiency for t in intervals_in_day)
                <= self.max_daily_throughput_kwh
            )

        # Don't finish the period with less stored energy than it started with
        prob += soc[n - 1] >= self.starting_charge_kwh

        # Solve with COIN-OR Branch and Cut solver
        stats = prob.solve(pulp.COIN_CMD(msg=False))
        if stats.status != pulp.LpSolveStatus.Optimal:
            raise RuntimeError(f"LP solver failed: {stats.status_str}")

        # Extract results
        charge_vals = np.array([charge[t].value() for t in range(n)])
        discharge_vals = np.array([discharge[t].value() for t in range(n)])
        grid_vals = np.array([grid[t].value() for t in range(n)])
        soc_vals = np.array([soc[t].value() for t in range(n)])

        # Flag any intervals with simultaneous charge and discharge
        simultaneous = (charge_vals > 1e-6) & (discharge_vals > 1e-6)
        if simultaneous.any():
            logger.warning(
                "Battery charges and discharges in %d of %d intervals.",
                simultaneous.sum(),
                n,
            )

        return pd.DataFrame(
            {
                "demand_kwh": demand,
                "soc_kwh": soc_vals,
                "charge_kwh": charge_vals,
                "discharge_kwh": discharge_vals,
                "grid_import_kwh": grid_vals,
                "import_price_p_per_kwh": prices,
                "cost_p": grid_vals * prices,
            },
            index=demand_profile.index,
        )

    def run_cost_optimisation_with_export(
        self,
        demand_profile: pd.Series,
        import_price_profile: pd.Series,
        export_price_profile: pd.Series,
        delta_t: float = 0.5,
    ) -> pd.DataFrame:
        """Minimise the net electricity bill, allowing export, as a perfect-foresight LP.

        Chooses charge and discharge in every interval to minimise import cost
        minus export revenue over the whole period, subject to inverter
        limits, state-of-charge limits, round-trip efficiency and a daily
        throughput budget. The battery must end the period with at least as
        much stored energy as it started with.

        There is no solar, so the battery is the only source of exported
        energy and the grid is the only source of charging energy. Export is
        not allowed in intervals where the export price is negative.

        The profiles can cover any period length, from a single day to
        multiple years, as long as they share the same DatetimeIndex and
        its interval matches `delta_t`. The whole period is solved as one
        problem, so charge can carry across days.

        Charging and discharging can both occur in the same interval (the
        battery can switch within a half hour), but together they cannot
        exceed the inverter limit. Exported energy can only come from the
        battery, and charging energy can only come from the grid.

        Args:
            demand_profile: Household demand per interval in kWh.
            import_price_profile: Electricity import price per interval in p/kWh.
            export_price_profile: Electricity export price per interval in p/kWh.
            delta_t: Interval length in hours. Defaults to 0.5 (half-hourly).

        Returns:
            One row per interval, on the same index as the inputs, with
            columns: demand_kwh, soc_kwh (end of interval), charge_kwh
            (drawn from the grid), discharge_kwh (delivered to the home or
            exported), grid_import_kwh, grid_export_kwh,
            import_price_p_per_kwh, export_price_p_per_kwh, import_cost_p,
            export_revenue_p and cost_p (net cost: import cost minus export
            revenue). Cost columns are in pence.

        Raises:
            ValueError: If the inputs do not share the same DatetimeIndex.
            RuntimeError: If the solver does not find an optimal solution.
        """
        if not (
            demand_profile.index.equals(import_price_profile.index)
            and demand_profile.index.equals(export_price_profile.index)
        ):
            raise ValueError("Inputs must share the same index.")
        if not isinstance(demand_profile.index, pd.DatetimeIndex):
            raise ValueError("Inputs must be indexed by a DatetimeIndex.")

        demand = demand_profile.to_numpy(dtype=float)
        import_prices = import_price_profile.to_numpy(dtype=float)
        export_prices = export_price_profile.to_numpy(dtype=float)
        n = len(demand)

        max_energy_per_step = self.max_power_kw * delta_t  # kWh per interval

        prob = pulp.LpProblem("battery_cost_optimisation_with_export", pulp.LpMinimize)

        # Decision variables (all energy in kWh)
        charge = [prob.add_variable(f"charge_{t}", lowBound=0, upBound=max_energy_per_step) for t in range(n)]
        discharge = [prob.add_variable(f"discharge_{t}", lowBound=0, upBound=max_energy_per_step) for t in range(n)]
        grid_import = [prob.add_variable(f"grid_import_{t}", lowBound=0) for t in range(n)]
        grid_export = [
            prob.add_variable(
                f"grid_export_{t}",
                lowBound=0,
                upBound=0 if export_prices[t] < 0 else None,  # no export at negative prices
            )
            for t in range(n)
        ]
        soc = [prob.add_variable(f"soc_{t}", lowBound=self.soc_min_kwh, upBound=self.soc_max_kwh) for t in range(n)]

        # Objective: net electricity bill (import cost minus export revenue, pence)
        prob += pulp.lpSum(import_prices[t] * grid_import[t] - export_prices[t] * grid_export[t] for t in range(n))

        for t in range(n):
            # Energy balance: import - export = demand + charging - discharging
            prob += grid_import[t] - grid_export[t] == demand[t] + charge[t] - discharge[t]

            # Exported energy can only come from the battery
            prob += grid_export[t] <= discharge[t]

            # Charging energy can only come from the grid
            prob += charge[t] <= grid_import[t]

            # The inverter can only do one thing at a time, so the time spent charging plus
            # the time spent discharging within the interval cannot exceed the interval length
            prob += charge[t] + discharge[t] <= max_energy_per_step

            # Battery dynamics
            previous_soc = self.starting_charge_kwh if t == 0 else soc[t - 1]
            prob += (
                soc[t] == previous_soc + self.charge_efficiency * charge[t] - discharge[t] / self.discharge_efficiency
            )

        # Daily throughput (energy removed from the battery) within budget
        interval_dates = demand_profile.index.normalize()
        for intervals_in_day in pd.Series(np.arange(n)).groupby(interval_dates).agg(list):
            prob += (
                pulp.lpSum(discharge[t] / self.discharge_efficiency for t in intervals_in_day)
                <= self.max_daily_throughput_kwh
            )

        # Don't finish the period with less stored energy than it started with
        prob += soc[n - 1] >= self.starting_charge_kwh

        # Solve with COIN-OR Branch and Cut solver
        stats = prob.solve(pulp.COIN_CMD(msg=False))
        if stats.status != pulp.LpSolveStatus.Optimal:
            raise RuntimeError(f"LP solver failed: {stats.status_str}")

        # Extract results
        charge_vals = np.array([charge[t].value() for t in range(n)])
        discharge_vals = np.array([discharge[t].value() for t in range(n)])
        import_vals = np.array([grid_import[t].value() for t in range(n)])
        export_vals = np.array([grid_export[t].value() for t in range(n)])
        soc_vals = np.array([soc[t].value() for t in range(n)])

        # Flag any intervals with simultaneous charge and discharge
        simultaneous = (charge_vals > 1e-6) & (discharge_vals > 1e-6)
        if simultaneous.any():
            logger.warning(
                "Battery charges and discharges in %d of %d intervals.",
                simultaneous.sum(),
                n,
            )

        # Flag grid pass-through: the battery charges from the grid and exports to the grid in the
        # same interval while the export price exceeds the import price. It earns the price spread
        # without storing anything useful, so it can inflate savings
        pass_through = (charge_vals > 1e-6) & (export_vals > 1e-6) & (export_prices > import_prices)
        if pass_through.any():
            logger.warning(
                "Battery imports and exports in the same interval at an export price above the "
                "import price in %d of %d intervals; this is grid pass-through arbitrage and may "
                "overstate savings.",
                pass_through.sum(),
                n,
            )

        import_cost = import_vals * import_prices
        export_revenue = export_vals * export_prices

        return pd.DataFrame(
            {
                "demand_kwh": demand,
                "soc_kwh": soc_vals,
                "charge_kwh": charge_vals,
                "discharge_kwh": discharge_vals,
                "grid_import_kwh": import_vals,
                "grid_export_kwh": export_vals,
                "import_price_p_per_kwh": import_prices,
                "export_price_p_per_kwh": export_prices,
                "import_cost_p": import_cost,
                "export_revenue_p": export_revenue,
                "cost_p": import_cost - export_revenue,
            },
            index=demand_profile.index,
        )
