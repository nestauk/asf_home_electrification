"""Home solar PV and battery system, with simulations of its operation against electricity tariffs.

Provides a `SolarBatterySystem` class that combines a `SolarPV` array with a
`Battery` and runs half-hourly simulations on a demand profile and price
series: rule-based operating modes (self-consumption, and self-consumption with
overnight grid charging) and cost-minimising optimisations with and without
export to the grid.
"""

import logging

import numpy as np
import pandas as pd
import pulp

from asf_home_electrification.pipeline.battery import Battery
from asf_home_electrification.pipeline.solar import SolarPV

logger = logging.getLogger(__name__)

INTERVAL_HOURS = 0.5  # PVGIS data is converted to half-hourly intervals


class SolarBatterySystem:
    """Home solar PV and battery storage system.

    Combines a `SolarPV` array with a `Battery`, so simulations can balance
    household demand against solar generation and stored energy. The battery
    and PV system are held by reference and are not modified by simulations.

    Simulations (each takes half-hourly profiles on a shared DatetimeIndex
    covering a full year, as solar generation is matched to the
    index using the latest year of PVGIS weather):
        run_self_consumption_only: Battery charges from surplus solar only.
        run_self_consumption_with_grid_charge: Battery also charges from the
            grid inside a set window, towards a target state of charge.
        run_solar_cost_optimisation_with_export: Minimise the bill with
            perfect foresight, allowing export to the grid.
        run_solar_cost_optimisation_no_export: Minimise the bill with perfect
            foresight, with no export to the grid.

    Attributes:
        battery (Battery): Battery storage unit with its technical limits.
        solar (SolarPV): Solar PV array with its size and orientation.
    """

    def __init__(self, battery: Battery, solar: SolarPV) -> None:
        """Initialise a solar and battery system from its two components."""
        self.battery = battery
        self.solar = solar

    def __repr__(self) -> str:
        """Return a string representation of the battery's and solar PV's technical parameters."""
        return f"{type(self).__name__}(battery={self.battery!r}, solar={self.solar!r})"

    def run_self_consumption_only(
        self,
        demand_profile: pd.Series,
        import_price_profile: pd.Series,
        export_price_profile: pd.Series,
        delta_t: float = 0.5,
    ) -> pd.DataFrame:
        """Run the system in self-consumption mode: the battery charges only from solar.

        Solar generation is used in the home first. Surplus generation charges
        the battery, and anything the battery cannot take is exported, or
        curtailed in intervals where the export price is negative. When solar
        falls short of demand the battery discharges to cover the shortfall, and
        any remainder is imported. The battery is never charged from the grid and
        never exports. Discharge is limited by the daily throughput budget, which
        resets each calendar day.

        Generation uses the latest year of PVGIS weather as a representative
        year, so the index must be a full half-hourly year (see
        `SolarPV.generation_profile`). The battery starts the period at
        `starting_charge_kwh`.

        Args:
            demand_profile (pd.Series): Household demand per half-hour in kWh.
            import_price_profile (pd.Series): Electricity import price per
                half-hour in p/kWh.
            export_price_profile (pd.Series): Electricity export price per
                half-hour in p/kWh.
            delta_t (float, optional): _description_. Defaults to 0.5.

        Raises:
            ValueError: If the inputs do not share the same DatetimeIndex, or the
                index cannot be matched to the PVGIS year.

        Returns:
            pd.DataFrame: One row per half-hour, on the same index as the
            inputs, with columns: demand_kwh, solar_generation_kwh,
            self_consumption_kwh (solar used directly in the home), soc_kwh
            (end of interval), charge_kwh (from surplus solar), discharge_kwh
            (delivered to the home), grid_import_kwh, grid_export_kwh,
            curtailed_kwh, import_price_p_per_kwh, export_price_p_per_kwh,
            import_cost_p, export_revenue_p and cost_p (net cost: import cost
            minus export revenue). Cost columns are in pence.
        """

        if not (
            demand_profile.index.equals(import_price_profile.index)
            and demand_profile.index.equals(export_price_profile.index)
        ):
            raise ValueError("Inputs must share the same index.")
        if not isinstance(demand_profile.index, pd.DatetimeIndex):
            raise ValueError("Inputs must be indexed by a DatetimeIndex.")

        battery = self.battery
        demand = demand_profile.to_numpy(dtype=float)
        generation = self.solar.generation_profile(demand_profile.index).to_numpy(dtype=float)
        import_prices = import_price_profile.to_numpy(dtype=float)
        export_prices = export_price_profile.to_numpy(dtype=float)

        interval_dates = demand_profile.index.normalize()
        n = len(demand)

        max_energy_per_step = battery.max_power_kw * delta_t  # kWh per interval

        # Solar is used in the home first, leaving either a surplus or a shortfall
        self_consumption = np.minimum(demand, generation)
        surplus = generation - self_consumption
        shortfall = demand - self_consumption

        charge = np.zeros(n)
        discharge = np.zeros(n)
        soc = np.zeros(n)

        soc_kwh = battery.starting_charge_kwh
        day_throughput_kwh = 0.0

        for t in range(n):
            if t > 0 and interval_dates[t] != interval_dates[t - 1]:
                day_throughput_kwh = 0.0  # new day, reset the cycle budget

            # Surplus solar charges the battery, up to the inverter limit and the space left
            charge[t] = max(
                0.0,
                min(
                    surplus[t],
                    max_energy_per_step,  # inverter limit
                    (battery.soc_max_kwh - soc_kwh) / battery.charge_efficiency,  # space left
                ),
            )

            # The battery covers demand that solar does not meet
            remaining_budget_kwh = max(0.0, battery.max_daily_throughput_kwh - day_throughput_kwh)
            discharge[t] = max(
                0.0,
                min(
                    shortfall[t],  # cover the shortfall only, no export
                    max_energy_per_step,  # inverter limit
                    (soc_kwh - battery.soc_min_kwh) * battery.discharge_efficiency,  # usable energy
                    remaining_budget_kwh * battery.discharge_efficiency,  # cycle budget
                ),
            )

            energy_removed_kwh = discharge[t] / battery.discharge_efficiency
            day_throughput_kwh += energy_removed_kwh
            soc_kwh += battery.charge_efficiency * charge[t] - energy_removed_kwh
            soc[t] = soc_kwh

        # Surplus that the battery could not take is exported, unless the export price is negative
        unstored_surplus = surplus - charge
        grid_export = np.where(export_prices < 0, 0.0, unstored_surplus)
        curtailed = unstored_surplus - grid_export
        grid_import = shortfall - discharge

        import_cost = grid_import * import_prices
        export_revenue = grid_export * export_prices

        return pd.DataFrame(
            {
                "demand_kwh": demand,
                "solar_generation_kwh": generation,
                "self_consumption_kwh": self_consumption,
                "soc_kwh": soc,
                "charge_kwh": charge,
                "discharge_kwh": discharge,
                "grid_import_kwh": grid_import,
                "grid_export_kwh": grid_export,
                "curtailed_kwh": curtailed,
                "import_price_p_per_kwh": import_prices,
                "export_price_p_per_kwh": export_prices,
                "import_cost_p": import_cost,
                "export_revenue_p": export_revenue,
                "cost_p": import_cost - export_revenue,
            },
            index=demand_profile.index,
        )

    def run_self_consumption_with_grid_charge(
        self,
        demand_profile: pd.Series,
        import_price_profile: pd.Series,
        export_price_profile: pd.Series,
        grid_charge_window: pd.Series,
        grid_charge_target_frac: float | None = None,
        delta_t: float = 0.5,
    ) -> pd.DataFrame:
        """Run the system in self-consumption mode, with the battery also charging from the grid.

        Solar generation is used in the home first. Surplus generation charges
        the battery, and anything the battery cannot take is exported, or
        curtailed in intervals where the export price is negative. In intervals
        inside the grid charge window the battery is also topped up from the
        grid, towards a target state of charge, and does not discharge. Outside
        the window the battery discharges to cover demand that solar does not
        meet, and any remainder is imported. The battery never exports.
        Discharge is limited by the daily throughput budget, which resets each
        calendar day.

        Generation uses the latest year of PVGIS weather as a representative
        year, so the index must be a full half-hourly year (see
        `SolarPV.generation_profile`). The battery starts the period at
        `starting_charge_kwh`.

        Args:
            demand_profile (pd.Series): Household demand per half-hour in kWh.
            import_price_profile (pd.Series): Electricity import price per
                half-hour in p/kWh.
            export_price_profile (pd.Series): Electricity export price per
                half-hour in p/kWh.
            grid_charge_window (pd.Series): Per half-hour: True where the
                battery may charge from the grid, e.g. the cheap overnight
                window of a time-of-use tariff.
            grid_charge_target_frac (float | None): State of charge, as a
                fraction of capacity, that grid charging stops at. Solar can
                still charge the battery above it. Defaults to None, meaning
                the battery's maximum state of charge.
            delta_t (float): Interval length in hours. Defaults to 0.5
                (half-hourly), which is the only value that works with the
                half-hourly PVGIS generation profile.

        Raises:
            ValueError: If the inputs do not share the same DatetimeIndex, the
                grid charge target is outside the battery's state-of-charge
                limits, or the index cannot be matched to the PVGIS year.

        Returns:
            pd.DataFrame: One row per half-hour, on the same index as the
            inputs, with columns: demand_kwh, solar_generation_kwh,
            self_consumption_kwh (solar used directly in the home), soc_kwh
            (end of interval), charge_kwh (total), charge_from_solar_kwh,
            charge_from_grid_kwh, discharge_kwh (delivered to the home),
            grid_import_kwh, grid_export_kwh, curtailed_kwh,
            import_price_p_per_kwh, export_price_p_per_kwh, import_cost_p,
            export_revenue_p and cost_p (net cost: import cost minus export
            revenue). Cost columns are in pence.
        """

        if not (
            demand_profile.index.equals(import_price_profile.index)
            and demand_profile.index.equals(export_price_profile.index)
            and demand_profile.index.equals(grid_charge_window.index)
        ):
            raise ValueError("Inputs must share the same index.")
        if not isinstance(demand_profile.index, pd.DatetimeIndex):
            raise ValueError("Inputs must be indexed by a DatetimeIndex.")

        battery = self.battery
        if grid_charge_target_frac is None:
            grid_charge_target_frac = battery.state_of_charge_max_frac
        if not battery.state_of_charge_min_frac <= grid_charge_target_frac <= battery.state_of_charge_max_frac:
            raise ValueError(
                "grid_charge_target_frac must lie between the battery's minimum and maximum state of charge fraction."
            )
        target_kwh = grid_charge_target_frac * battery.capacity_kwh

        demand = demand_profile.to_numpy(dtype=float)
        generation = self.solar.generation_profile(demand_profile.index).to_numpy(dtype=float)
        import_prices = import_price_profile.to_numpy(dtype=float)
        export_prices = export_price_profile.to_numpy(dtype=float)
        in_window = grid_charge_window.to_numpy(dtype=bool)

        interval_dates = demand_profile.index.normalize()
        n = len(demand)

        max_energy_per_step = battery.max_power_kw * delta_t  # kWh per interval

        # Solar is used in the home first, leaving either a surplus or a shortfall
        self_consumption = np.minimum(demand, generation)
        surplus = generation - self_consumption
        shortfall = demand - self_consumption

        charge_from_solar = np.zeros(n)
        charge_from_grid = np.zeros(n)
        discharge = np.zeros(n)
        soc = np.zeros(n)

        soc_kwh = battery.starting_charge_kwh
        day_throughput_kwh = 0.0

        for t in range(n):
            if t > 0 and interval_dates[t] != interval_dates[t - 1]:
                day_throughput_kwh = 0.0  # new day, reset the cycle budget

            # Surplus solar charges the battery, up to the inverter limit and the space left
            charge_from_solar[t] = max(
                0.0,
                min(
                    surplus[t],
                    max_energy_per_step,  # inverter limit
                    (battery.soc_max_kwh - soc_kwh) / battery.charge_efficiency,  # space left
                ),
            )
            soc_after_solar_kwh = soc_kwh + battery.charge_efficiency * charge_from_solar[t]

            if in_window[t]:
                # Top up from the grid towards the target, within the inverter limit left after solar
                charge_from_grid[t] = max(
                    0.0,
                    min(
                        max_energy_per_step - charge_from_solar[t],  # inverter limit left
                        (target_kwh - soc_after_solar_kwh) / battery.charge_efficiency,  # room to target
                    ),
                )
            else:
                # Outside the window the battery covers demand that solar does not meet
                remaining_budget_kwh = max(0.0, battery.max_daily_throughput_kwh - day_throughput_kwh)
                discharge[t] = max(
                    0.0,
                    min(
                        shortfall[t],  # cover the shortfall only, no export
                        max_energy_per_step,  # inverter limit
                        (soc_kwh - battery.soc_min_kwh) * battery.discharge_efficiency,  # usable energy
                        remaining_budget_kwh * battery.discharge_efficiency,  # cycle budget
                    ),
                )

            energy_removed_kwh = discharge[t] / battery.discharge_efficiency
            day_throughput_kwh += energy_removed_kwh
            soc_kwh = soc_after_solar_kwh + battery.charge_efficiency * charge_from_grid[t] - energy_removed_kwh
            soc[t] = soc_kwh

        # Surplus that the battery could not take is exported, unless the export price is negative
        unstored_surplus = surplus - charge_from_solar
        grid_export = np.where(export_prices < 0, 0.0, unstored_surplus)
        curtailed = unstored_surplus - grid_export
        grid_import = shortfall - discharge + charge_from_grid

        import_cost = grid_import * import_prices
        export_revenue = grid_export * export_prices

        return pd.DataFrame(
            {
                "demand_kwh": demand,
                "solar_generation_kwh": generation,
                "self_consumption_kwh": self_consumption,
                "soc_kwh": soc,
                "charge_kwh": charge_from_solar + charge_from_grid,
                "charge_from_solar_kwh": charge_from_solar,
                "charge_from_grid_kwh": charge_from_grid,
                "discharge_kwh": discharge,
                "grid_import_kwh": grid_import,
                "grid_export_kwh": grid_export,
                "curtailed_kwh": curtailed,
                "import_price_p_per_kwh": import_prices,
                "export_price_p_per_kwh": export_prices,
                "import_cost_p": import_cost,
                "export_revenue_p": export_revenue,
                "cost_p": import_cost - export_revenue,
            },
            index=demand_profile.index,
        )

    def run_solar_cost_optimisation_with_export(
        self,
        demand_profile: pd.Series,
        import_price_profile: pd.Series,
        export_price_profile: pd.Series,
    ) -> pd.DataFrame:
        """Minimise the net electricity bill of a solar and battery system, allowing export, as a perfect-foresight LP.

        Chooses charge, discharge, export and curtailment in every half-hour to
        minimise import cost minus export revenue over the whole period, subject
        to inverter limits, state-of-charge limits, round-trip efficiency and a
        daily throughput budget. The battery must end the period with at least
        as much stored energy as it started with.

        Each half-hour, demand + charge + export + curtailment = solar
        generation + discharge + import. Solar can charge the battery, serve the
        home, be exported or be curtailed, as the optimiser chooses. The grid can
        only supply the home and the battery, never be exported or curtailed.
        Discharged energy can only go to the home or the grid, not back into the
        charger. Export is not allowed when the export price is negative.

        Generation uses the latest year of PVGIS weather as a representative
        year, so the index must be a full half-hourly year (see
        `SolarPV.generation_profile`). The whole period is solved as one
        problem, so charge can carry across days.

        Charging and discharging can both occur in the same interval, but
        together they cannot exceed the inverter limit. A warning is logged if
        this happens, and if the system imports and exports in the same interval
        at an export price above the import price.

        Args:
            demand_profile (pd.Series): Household demand per half-hour in kWh.
            import_price_profile (pd.Series): Electricity import price per
                half-hour in p/kWh.
            export_price_profile (pd.Series): Electricity export price per
                half-hour in p/kWh.

        Raises:
            ValueError: If the inputs do not share the same DatetimeIndex, or the
                index cannot be matched to the PVGIS year.
            RuntimeError: If the solver does not find an optimal solution.

        Returns:
            pd.DataFrame: One row per half-hour, on the same index as the
            inputs, with columns: demand_kwh, solar_generation_kwh, soc_kwh (end
            of interval), charge_kwh (from solar and grid combined),
            discharge_kwh, grid_import_kwh, grid_export_kwh, curtailed_kwh,
            import_price_p_per_kwh, export_price_p_per_kwh, import_cost_p,
            export_revenue_p and cost_p (net cost: import cost minus export
            revenue). Cost columns are in pence.
        """

        if not (
            demand_profile.index.equals(import_price_profile.index)
            and demand_profile.index.equals(export_price_profile.index)
        ):
            raise ValueError("Inputs must share the same index.")
        if not isinstance(demand_profile.index, pd.DatetimeIndex):
            raise ValueError("Inputs must be indexed by a DatetimeIndex.")

        battery = self.battery
        demand = demand_profile.to_numpy(dtype=float)
        generation = self.solar.generation_profile(demand_profile.index).to_numpy(dtype=float)
        import_prices = import_price_profile.to_numpy(dtype=float)
        export_prices = export_price_profile.to_numpy(dtype=float)
        n = len(demand)

        max_energy_per_step = battery.max_power_kw * INTERVAL_HOURS  # kWh per interval

        prob = pulp.LpProblem("solar_battery_cost_optimisation_with_export", pulp.LpMinimize)

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
        # Only solar generation can be curtailed
        curtailed = [prob.add_variable(f"curtailed_{t}", lowBound=0, upBound=generation[t]) for t in range(n)]
        soc = [
            prob.add_variable(f"soc_{t}", lowBound=battery.soc_min_kwh, upBound=battery.soc_max_kwh) for t in range(n)
        ]

        # Objective: net electricity bill (import cost minus export revenue, pence)
        prob += pulp.lpSum(import_prices[t] * grid_import[t] - export_prices[t] * grid_export[t] for t in range(n))

        for t in range(n):
            # Energy balance: demand + charging + exported + curtailed = solar + discharging + imported
            prob += (
                demand[t] + charge[t] + grid_export[t] + curtailed[t] == generation[t] + discharge[t] + grid_import[t]
            )

            # Exported or curtailed energy can only come from solar or the battery, not the grid
            prob += grid_export[t] + curtailed[t] <= generation[t] + discharge[t]

            # Discharged energy can only go to the home or the grid, not back into the charger
            prob += discharge[t] <= demand[t] + grid_export[t]

            # The inverter can only do one thing at a time, so the time spent charging plus
            # the time spent discharging within the interval cannot exceed the interval length
            prob += charge[t] + discharge[t] <= max_energy_per_step

            # Battery dynamics
            previous_soc = battery.starting_charge_kwh if t == 0 else soc[t - 1]
            prob += (
                soc[t]
                == previous_soc + battery.charge_efficiency * charge[t] - discharge[t] / battery.discharge_efficiency
            )

        # Daily throughput (energy removed from the battery) within budget
        interval_dates = demand_profile.index.normalize()
        for intervals_in_day in pd.Series(np.arange(n)).groupby(interval_dates).agg(list):
            prob += (
                pulp.lpSum(discharge[t] / battery.discharge_efficiency for t in intervals_in_day)
                <= battery.max_daily_throughput_kwh
            )

        # Don't finish the period with less stored energy than it started with
        prob += soc[n - 1] >= battery.starting_charge_kwh

        # Solve with COIN-OR Branch and Cut solver
        stats = prob.solve(pulp.COIN_CMD(msg=False))
        if stats.status != pulp.LpSolveStatus.Optimal:
            raise RuntimeError(f"LP solver failed: {stats.status_str}")

        # Extract results
        charge_vals = np.array([charge[t].value() for t in range(n)])
        discharge_vals = np.array([discharge[t].value() for t in range(n)])
        import_vals = np.array([grid_import[t].value() for t in range(n)])
        export_vals = np.array([grid_export[t].value() for t in range(n)])
        curtailed_vals = np.array([curtailed[t].value() for t in range(n)])
        soc_vals = np.array([soc[t].value() for t in range(n)])

        # Flag any intervals with simultaneous charge and discharge
        simultaneous = (charge_vals > 1e-6) & (discharge_vals > 1e-6)
        if simultaneous.any():
            logger.warning(
                "Battery charges and discharges in %d of %d intervals.",
                simultaneous.sum(),
                n,
            )

        # Flag grid arbitrage: the system imports and exports in the same interval while the export
        # price exceeds the import price
        arbitrage = (import_vals > 1e-6) & (export_vals > 1e-6) & (export_prices > import_prices)
        if arbitrage.any():
            logger.warning(
                "System imports and exports in the same interval at an export price above the "
                "import price in %d of %d intervals; this is grid pass-through arbitrage and may overstate savings.",
                arbitrage.sum(),
                n,
            )

        import_cost = import_vals * import_prices
        export_revenue = export_vals * export_prices

        return pd.DataFrame(
            {
                "demand_kwh": demand,
                "solar_generation_kwh": generation,
                "soc_kwh": soc_vals,
                "charge_kwh": charge_vals,
                "discharge_kwh": discharge_vals,
                "grid_import_kwh": import_vals,
                "grid_export_kwh": export_vals,
                "curtailed_kwh": curtailed_vals,
                "import_price_p_per_kwh": import_prices,
                "export_price_p_per_kwh": export_prices,
                "import_cost_p": import_cost,
                "export_revenue_p": export_revenue,
                "cost_p": import_cost - export_revenue,
            },
            index=demand_profile.index,
        )

    def run_solar_cost_optimisation_no_export(
        self,
        demand_profile: pd.Series,
        import_price_profile: pd.Series,
    ) -> pd.DataFrame:
        """Minimise the electricity bill of a solar and battery system, with no export, as a perfect-foresight LP.

        Chooses charge, discharge and curtailment in every half-hour to minimise
        import cost over the whole period, subject to inverter limits,
        state-of-charge limits, round-trip efficiency and a daily throughput
        budget. The battery must end the period with at least as much stored
        energy as it started with.

        Each half-hour, demand + charge + curtailment = solar generation +
        discharge + import. Solar can serve the home, charge the battery or be
        curtailed, as the optimiser chooses, and the grid can charge the battery.
        Export is not allowed, so surplus solar is curtailed. Discharged energy
        can only go to the home, not back into the charger.

        Generation uses the latest year of PVGIS weather as a representative
        year, so the index must be a full half-hourly year (see
        `SolarPV.generation_profile`). The whole period is solved as one
        problem, so charge can carry across days.

        Charging and discharging can both occur in the same interval, but
        together they cannot exceed the inverter limit. A warning is logged if
        this happens.

        Args:
            demand_profile (pd.Series): Household demand per half-hour in kWh.
            import_price_profile (pd.Series): Electricity import price per
                half-hour in p/kWh.

        Raises:
            ValueError: If the inputs do not share the same DatetimeIndex, or the
                index cannot be matched to the PVGIS year.
            RuntimeError: If the solver does not find an optimal solution.

        Returns:
            pd.DataFrame: One row per half-hour, on the same index as the
            inputs, with columns: demand_kwh, solar_generation_kwh, soc_kwh (end
            of interval), charge_kwh (from solar and grid combined),
            discharge_kwh, grid_import_kwh, curtailed_kwh,
            import_price_p_per_kwh, import_cost_p and cost_p (pence).
        """

        if not (demand_profile.index.equals(import_price_profile.index)):
            raise ValueError("Inputs must share the same index.")
        if not isinstance(demand_profile.index, pd.DatetimeIndex):
            raise ValueError("Inputs must be indexed by a DatetimeIndex.")

        battery = self.battery
        demand = demand_profile.to_numpy(dtype=float)
        generation = self.solar.generation_profile(demand_profile.index).to_numpy(dtype=float)
        import_prices = import_price_profile.to_numpy(dtype=float)
        n = len(demand)

        max_energy_per_step = battery.max_power_kw * INTERVAL_HOURS  # kWh per interval

        prob = pulp.LpProblem("solar_battery_cost_optimisation_no_export", pulp.LpMinimize)

        # Decision variables (all energy in kWh)
        charge = [prob.add_variable(f"charge_{t}", lowBound=0, upBound=max_energy_per_step) for t in range(n)]
        discharge = [prob.add_variable(f"discharge_{t}", lowBound=0, upBound=max_energy_per_step) for t in range(n)]
        grid_import = [prob.add_variable(f"grid_import_{t}", lowBound=0) for t in range(n)]

        # Only solar generation can be curtailed
        curtailed = [prob.add_variable(f"curtailed_{t}", lowBound=0, upBound=generation[t]) for t in range(n)]
        soc = [
            prob.add_variable(f"soc_{t}", lowBound=battery.soc_min_kwh, upBound=battery.soc_max_kwh) for t in range(n)
        ]

        # Objective: total electricity bill (pence)
        prob += pulp.lpSum(import_prices[t] * grid_import[t] for t in range(n))

        for t in range(n):
            # Energy balance: demand + charging + curtailed = solar + discharging + imported
            prob += demand[t] + charge[t] + curtailed[t] == generation[t] + discharge[t] + grid_import[t]

            # Discharged energy can only go to the home, not back into the charger
            prob += discharge[t] <= demand[t]

            # The inverter can only do one thing at a time, so the time spent charging plus
            # the time spent discharging within the interval cannot exceed the interval length
            prob += charge[t] + discharge[t] <= max_energy_per_step

            # Battery dynamics
            previous_soc = battery.starting_charge_kwh if t == 0 else soc[t - 1]
            prob += (
                soc[t]
                == previous_soc + battery.charge_efficiency * charge[t] - discharge[t] / battery.discharge_efficiency
            )

        # Daily throughput (energy removed from the battery) within budget
        interval_dates = demand_profile.index.normalize()
        for intervals_in_day in pd.Series(np.arange(n)).groupby(interval_dates).agg(list):
            prob += (
                pulp.lpSum(discharge[t] / battery.discharge_efficiency for t in intervals_in_day)
                <= battery.max_daily_throughput_kwh
            )

        # Don't finish the period with less stored energy than it started with
        prob += soc[n - 1] >= battery.starting_charge_kwh

        # Solve with COIN-OR Branch and Cut solver
        stats = prob.solve(pulp.COIN_CMD(msg=False))
        if stats.status != pulp.LpSolveStatus.Optimal:
            raise RuntimeError(f"LP solver failed: {stats.status_str}")

        # Extract results
        charge_vals = np.array([charge[t].value() for t in range(n)])
        discharge_vals = np.array([discharge[t].value() for t in range(n)])
        import_vals = np.array([grid_import[t].value() for t in range(n)])
        curtailed_vals = np.array([curtailed[t].value() for t in range(n)])
        soc_vals = np.array([soc[t].value() for t in range(n)])

        # Flag any intervals with simultaneous charge and discharge
        simultaneous = (charge_vals > 1e-6) & (discharge_vals > 1e-6)
        if simultaneous.any():
            logger.warning(
                "Battery charges and discharges in %d of %d intervals.",
                simultaneous.sum(),
                n,
            )

        import_cost = import_vals * import_prices

        return pd.DataFrame(
            {
                "demand_kwh": demand,
                "solar_generation_kwh": generation,
                "soc_kwh": soc_vals,
                "charge_kwh": charge_vals,
                "discharge_kwh": discharge_vals,
                "grid_import_kwh": import_vals,
                "curtailed_kwh": curtailed_vals,
                "import_price_p_per_kwh": import_prices,
                "import_cost_p": import_cost,
                "cost_p": import_cost,
            },
            index=demand_profile.index,
        )
