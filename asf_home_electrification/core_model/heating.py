"""Half-hourly heat demand model for a TABULA GB archetype, with domestic hot water.

Steps indoor temperature forward in 30-minute timesteps using the TABULA heat balance
(EN ISO 13790, as implemented in the TABULA calculation method), with a thermal-mass
term. A heating system with a hot water tank shares its capacity between the tank
(served first) and space heating.

TABULA seasonal equations used (all quantities in W, K, W/K here rather than kWh/a):
    H_tr  = sum(b_tr,i * A_env,i * U_eff,i) + dU_tb * sum(A_env,i)        transmission coefficient
    H_ve  = c_p,air * (n_air,use + n_air,infiltr) * A_C,ref * h_room      ventilation coefficient
    Q_int = phi_int * A_C,ref                                              internal gains
    Q_sol = sum_j F_sh,j * (1 - F_F) * F_W * g_gl * A_window,j * I_sol,j   solar gains (per orientation)
    C     = c_m * A_C,ref                                                  internal heat capacity
    F_nu  multiplies H_tr and H_ve (non-uniform heating, "F_red" in the report; archetype parameter)

Conventions:
    - Timestamps are UK local time (Europe/London). Weather is looked up by UTC instant
      and mapped to the 2023 PVGIS year.
    - Energies are kWh over the half hour, temperatures are degC.
    - Indoor and tank temperatures are carried between steps by the caller, not stored
      on any object.

Typical use:
    dwelling = Dwelling(lat, lon, archetype_name)
    tank = HotWaterTank(210, 10.0, 55.0, 20.0, ref_standing_loss_kwh_per_day=1.01, ref_delta_t=45.0)
    heat_pump = HeatingSystem("heat pump", 8.0, fuel="electricity", efficiency=3.0, tank=tank)

    t_setpoint = thermostat.setpoints(start, end)
    t_target = thermostat.targets(t_setpoint)
    df = heat_pump.run_heat_balance_time_period(
        dwelling=dwelling, start=start, end=end, t_initial=16.0,
        t_target=t_target, t_setpoint=t_setpoint, dhw_draw_litres=draws,
    )
"""

import pandas as pd

from asf_home_electrification.core_model.dwelling import Dwelling
from asf_home_electrification.core_model.hot_water import HotWaterTank


class HeatingSystem:
    """A heating system that supplies space heating and, optionally, hot water.

    Args:
        type (str): Description of the system, e.g. "heat pump" or "gas boiler".
        system_size_kw (float): Maximum thermal output, in kW. Shared between hot water
            (served first) and space heating.
        fuel (str): Energy carrier consumed, e.g. "electricity" or "gas". Used to name
            the fuel demand column in results.
        efficiency (float): Heat delivered per unit of fuel energy for space heating
            (seasonal COP for a heat pump, boiler efficiency otherwise).
        efficiency_dhw (float | None, optional): The same ratio for hot water. Defaults to
            efficiency. Heat pumps usually have a lower value for hot water.
        tank (HotWaterTank | None, optional): Hot water tank. None means no hot water
            is modelled. Defaults to None.
    """

    def __init__(
        self,
        type: str,
        system_size_kw: float,
        fuel: str,
        efficiency: float,  # TODO allow for efficiency curve with t_out
        efficiency_dhw: float | None = None,  # TODO allow for efficiency curve with t_out
        tank: HotWaterTank | None = None,  # None means no hot water modelled
    ) -> None:
        self.type = type
        self.system_size_kw = system_size_kw
        self.fuel = fuel
        self.efficiency = efficiency
        self.efficiency_dhw = efficiency if efficiency_dhw is None else efficiency_dhw
        self.tank = tank

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(type={self.type!r}, system_size_kw={self.system_size_kw}, "
            f"fuel={self.fuel!r}, efficiency={self.efficiency}, "
            f"efficiency_dhw={self.efficiency_dhw}, tank={self.tank!r})"
        )

    @property
    def tank(self) -> HotWaterTank | None:
        """The hot water tank, or None if no hot water is modelled.

        Raises:
            TypeError: On assignment, if the value is not HotWaterTank or None.
        """
        return self._tank

    @tank.setter
    def tank(self, value: HotWaterTank | None) -> None:
        if value is not None and not isinstance(value, HotWaterTank):
            raise TypeError(f"tank must be a HotWaterTank or None, got {type(value).__name__}")
        self._tank = value

    def run_heat_balance_half_hour(
        self,
        *,
        dwelling: Dwelling,
        t_start: float,
        t_target: float,
        step_start: str,
        # --- TABULA standard values ---
        n_air_use: float = 0.4,
        h_room: float = 2.5,
        c_p_air: float = 0.34,
        f_f: float = 0.3,
        f_w: float = 0.9,
        f_sh_vertical: float = 0.6,
        f_sh_horizontal: float = 0.8,
        c_m: float = 45.0,
        phi_int: float = 3.0,
        # --- Hot water ---
        t_dhw_start: float | None = None,  # tank temperature at start of step (default: tank.t_hot_water)
        draw_litres: float = 0.0,
        t_dhw_target: float | None = None,  # tank target (default: tank.t_hot_water),
    ) -> dict:
        """Run the heat balance for one half-hour timestep, including hot water.

        If the system has a tank, the tank is stepped first and its heat takes priority
        for the system's capacity. Space heating then gets the capacity that is left.
        The tank heat does not enter the indoor balance, because it goes into the cylinder.

        Weather (outside temperature and solar irradiance for each window orientation)
        is looked up for step_start. All energies are in kWh over the half hour, and
        temperatures are in degC.

        Args:
            dwelling (Dwelling): The building being heated.
            t_start (float): Indoor temperature at the start of the step, in degC.
            t_target (float): Indoor temperature to heat towards, in degC.
            step_start (str | pd.Timestamp): Start of the half hour, in UK local time.
                A naive value is treated as Europe/London.
            n_air_use (float, optional): Air change rate from use, in 1/h. Defaults to 0.4.
            h_room (float, optional): Ventilation reference room height, in m. Defaults to 2.5.
            c_p_air (float, optional): Volumetric heat capacity of air (heat needed to warm
                1 m3 of air by 1 K), in Wh/(m3 K). Defaults to 0.34.
            f_f (float, optional): Fraction of the window that is frame and lets no sun
                through. Defaults to 0.3.
            f_w (float, optional): Correction for sunlight hitting windows at an angle.
                Defaults to 0.9.
            f_sh_vertical (float, optional): External shading correction for vertical windows
                (fraction of sunlight not blocked by neighbours, vegetation, overhangs).
                Defaults to 0.6.
            f_sh_horizontal (float, optional): External shading correction for horizontal
                windows. Defaults to 0.8.
            c_m (float, optional): Internal heat capacity per m2 of reference area, in
                Wh/(m2 K). Defaults to 45.0.
            phi_int (float, optional): Internal heat source density (people, appliances,
                lighting, cooking) per m2 of reference area, in W/m2. Defaults to 3.0.
            t_dhw_start (float | None, optional): Tank temperature at the start of the step,
                in degC. Defaults to the tank's t_hot_water. Ignored if there is no tank.
            draw_litres (float, optional): Hot water drawn from the tank during the step, in
                litres. Defaults to 0.0. Ignored if there is no tank.
            t_dhw_target (float | None, optional): Temperature to heat the tank towards,
                in degC. Defaults to the tank's t_hot_water. Ignored if there is no tank.

        Raises:
            ValueError: If any archetype parameter of the dwelling is None or an empty
                dict, or if draw_litres is outside the tank volume.
            KeyError: If step_start is not in the weather data.

        Returns:
            dict: Results for the half-hour step, with these keys:
                - q_transmission_kwh: heat lost through the building fabric (kWh).
                - q_ventilation_kwh: heat lost through ventilation and infiltration (kWh).
                - q_solar_kwh: solar gains through windows (kWh).
                - q_internal_kwh: internal gains from occupants, appliances and lighting (kWh).
                - q_mass_kwh: energy needed to bring the thermal mass from t_start to
                  t_target (kWh; negative if the house is warmer than the target).
                - q_required_kwh: space heating required to reach t_target (kWh), never
                  negative, before the capacity limit.
                - q_delivered_kwh: space heating actually delivered (kWh): q_required_kwh
                  limited to the capacity left after hot water.
                - q_dhw_kwh: heat delivered to the hot water tank (kWh). 0.0 with no tank.
                - t_out_c: outside temperature for the step (degC).
                - t_end_c: indoor temperature at the end of the step (degC).
                If the system has a tank, also:
                - q_dhw_draw_kwh: heat removed from the tank by the draw (kWh).
                - q_dhw_standing_kwh: heat lost from the tank to the room (kWh).
                - t_tank_end_c: tank temperature at the end of the step (degC).
        """

        # Unpack building archetype parameters
        empty = [key for key, value in dwelling.archetype_params.items() if value is None or value == {}]
        if empty:
            raise ValueError(f"Empty archetype params for {dwelling.archetype_name!r}: {empty}")
        params = dwelling.archetype_params
        a_ref = params["a_ref"]  # m2, floor area
        h_tr = params["h_tr"]  # W/K, transmission heat transfer coefficient including thermal bridging
        f_nu = params["f_nu"]  # -, temperature reduction factor for non-uniform heating (F_red in TABULA report)
        n_air_infiltr = params["n_air_infiltr"]  # 1/h, infiltration air change rate
        window_area = params["window_area"]  # dict, m2 by orientation, e.g. {"east": 22.7, "west": 23.8}
        g_gl = params["g_gl"]  # -, glazing solar transmittance

        # PVGIS weather data getter always uses latest year in dataset (2023)
        timestamp = pd.Timestamp(step_start)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("Europe/London")
        timestamp_utc = timestamp.tz_convert("UTC")
        if timestamp_utc.month == 2 and timestamp_utc.day == 29:  # 2023 has no leap day
            timestamp_utc = timestamp_utc.replace(day=28)
        day_and_time = timestamp_utc.replace(year=2023)
        irradiance_row = dwelling.solar_irradiance.loc[day_and_time]  # W/m2, one value per orientation
        t_out = dwelling.outside_temperature.loc[day_and_time]  # degC

        # Ventilation heat transfer coefficient
        h_ve = c_p_air * (n_air_use + n_air_infiltr) * a_ref * h_room  # W/K

        # Total heat loss coefficient
        h = f_nu * (h_tr + h_ve)  # W/K

        # Solar gains
        q_solar = 0.0  # W
        for orientation, area in window_area.items():  # area: m2
            f_sh = f_sh_horizontal if orientation == "horizontal" else f_sh_vertical
            q_solar += (
                f_sh
                * (1 - f_f)
                * f_w
                * g_gl  # - (dimensionless reduction factors)
                * area  # m2
                * irradiance_row[orientation]  # W/m2
            )

        # Internal free gains
        q_internal = phi_int * a_ref  # W

        # Timestep
        dt = 0.5  # hours
        w_to_kwh = dt / 1_000  # W over dt hours -> kWh

        # Thermal mass
        c_dt = c_m * a_ref / dt  # W/K, how much heat to raise temp by 1K in timestep
        q_mass = c_dt * (t_target - t_start)  # W

        # Required heating
        t_mean = 0.5 * (t_start + t_target)
        q_transmission = f_nu * h_tr * (t_mean - t_out)
        q_ventilation = f_nu * h_ve * (t_mean - t_out)
        q_required = max(
            q_transmission + q_ventilation - q_solar - q_internal + q_mass,
            0.0,
        )  # can't be negative

        # Domestic hot water: served first, so space heating gets the capacity that is left
        if self.tank is None:
            tank_result = {"q_dhw_kwh": 0.0}
        else:
            tank_result = self.tank.run_half_hour_step(
                t_start=self.tank.t_hot_water if t_dhw_start is None else t_dhw_start,
                draw_litres=draw_litres,
                heating_system_size_kw=self.system_size_kw,
                t_target=t_dhw_target,  # None means the tank uses its own t_hot_water
            )
        q_dhw = tank_result["q_dhw_kwh"] / w_to_kwh  # W averaged over the step

        # Heat delivered is constrained by system size
        capacity = self.system_size_kw * 1_000  # W
        q_delivered = min(q_required, max(capacity - q_dhw, 0.0))  # W, space heating only

        # Indoor temperature at end of step given heat delivered
        # Calculated by rearranging original balance
        # c_dt × (t_end − t_start)  =  q_delivered + q_solar + q_internal − q_loss
        # where q_loss = h × (t_indoor − t_out) where t_indoor is average of t_start and t_end
        t_end = (c_dt * t_start + q_delivered + q_solar + q_internal - h * (0.5 * t_start - t_out)) / (c_dt + 0.5 * h)

        return {
            **tank_result,  # q_dhw_kwh, plus q_dhw_draw_kwh, q_dhw_standing_kwh and t_tank_end_c if there is a tank
            "q_transmission_kwh": q_transmission * w_to_kwh,
            "q_ventilation_kwh": q_ventilation * w_to_kwh,
            "q_solar_kwh": q_solar * w_to_kwh,
            "q_internal_kwh": q_internal * w_to_kwh,
            "q_mass_kwh": q_mass * w_to_kwh,
            "q_required_kwh": q_required * w_to_kwh,
            "q_delivered_kwh": q_delivered * w_to_kwh,
            "t_out_c": t_out,
            "t_end_c": t_end,
        }

    def run_heat_balance_time_period(
        self,
        *,
        dwelling: Dwelling,
        start: str,  # UK local time, e.g. "2026-01-05" (inclusive)
        end: str,  # UK local time (exclusive)
        t_initial: float,
        t_target: pd.Series,  # heating targets based on pre-heating needed
        t_setpoint: pd.Series,  # scheduled setpoints
        dhw_draw_litres: pd.Series | None = None,  # litres drawn in each half hour (default: no draws)
        t_dhw_target: pd.Series | None = None,  # tank target per half hour (default: tank.t_hot_water)
        t_tank_initial: float | None = None,  # default: tank.t_hot_water
        **step_kwargs,
    ) -> pd.DataFrame:
        """Run the heat balance over a period, chaining half-hour steps together.

        The indoor temperature at the end of each step becomes the starting temperature of
        the next, and the same applies to the tank temperature. If the system has a tank,
        hot water is always modelled (with no draws unless dhw_draw_litres is given), so
        the tank still loses heat and is reheated.

        Args:
            dwelling (Dwelling): The building being heated.
            start (str): Start of the period, in UK local time (inclusive), e.g. "2026-01-05".
            end (str): End of the period, in UK local time (exclusive).
            t_initial (float): Indoor temperature at the start of the period, in degC.
            t_target (pd.Series): Indoor heating target for each half hour, in degC. Needs a
                Europe/London index covering the period. Usually the setpoints with preheating.
            t_setpoint (pd.Series): Scheduled comfort setpoint for each half hour, in degC.
                Same index as t_target. Only used for reporting, not for the heating.
            dhw_draw_litres (pd.Series | None, optional): Hot water drawn in each half hour,
                in litres. Defaults to no draws. Requires a tank.
            t_dhw_target (pd.Series | None, optional): Tank target temperature for each half
                hour, in degC. Defaults to the tank's t_hot_water. Requires a tank.
            t_tank_initial (float | None, optional): Tank temperature at the start of the
                period, in degC. Defaults to the tank's t_hot_water.
            **step_kwargs: Any TABULA standard value accepted by run_heat_balance_half_hour
                (e.g. c_m, phi_int), passed through unchanged.

        Raises:
            ValueError: If t_target, t_setpoint, dhw_draw_litres or t_dhw_target does not
                cover every half hour in the period, or if hot water inputs are given
                to a system with no tank.

        Returns:
            pd.DataFrame: One row per half hour, indexed by step_start (Europe/London), with:
                - t_start_c, t_target_c, t_setpoint_c: indoor temperature at the start of
                  the step, the heating target and the scheduled setpoint (degC).
                - All keys returned by run_heat_balance_half_hour, plus t_tank_start_c if
                  the system has a tank.
                - q_unmet_kwh: space heating required but not delivered (kWh). Overstates
                  demand during recovery, because the thermal-mass catch-up is re-counted
                  each step until the house reaches its target.
                - q_total_kwh: space heating plus hot water delivered (kWh).
                - "{fuel}_demand_kwh": fuel used (kWh), with space heating and hot water each
                  divided by their own efficiency.
        """

        timestamps = pd.date_range(start, end, freq="30min", inclusive="left", tz="Europe/London")
        targets = t_target.reindex(timestamps)
        if targets.isna().any():
            raise ValueError(f"t_target is missing for {int(targets.isna().sum())} step(s)")
        setpoints = t_setpoint.reindex(timestamps)
        if setpoints.isna().any():
            raise ValueError(f"t_setpoint is missing for {int(setpoints.isna().sum())} step(s)")

        # Hot water inputs: modelled whenever the system has a tank
        model_dhw = self.tank is not None
        if not model_dhw and (dhw_draw_litres is not None or t_dhw_target is not None):
            raise ValueError("hot water inputs were given but this HeatingSystem has no tank")

        t_tank = None
        if model_dhw:
            if dhw_draw_litres is None:
                draws = pd.Series(0.0, index=timestamps)
            else:
                draws = dhw_draw_litres.reindex(timestamps)
                if draws.isna().any():
                    raise ValueError(f"dhw_draw_litres is missing for {int(draws.isna().sum())} step(s)")
            if t_dhw_target is None:
                tank_targets = pd.Series(self.tank.t_hot_water, index=timestamps)
            else:
                tank_targets = t_dhw_target.reindex(timestamps)
                if tank_targets.isna().any():
                    raise ValueError(f"t_dhw_target is missing for {int(tank_targets.isna().sum())} step(s)")
            t_tank = self.tank.t_hot_water if t_tank_initial is None else t_tank_initial

        t_indoor = t_initial
        rows = []
        for timestamp, target in targets.items():
            # the house step also steps the tank (hot water first, house gets the capacity left)
            result = self.run_heat_balance_half_hour(
                t_start=t_indoor,
                t_target=target,
                dwelling=dwelling,
                step_start=timestamp,
                t_dhw_start=t_tank,
                draw_litres=draws[timestamp] if model_dhw else 0.0,
                t_dhw_target=tank_targets[timestamp] if model_dhw else None,
                **step_kwargs,
            )
            row = {"t_start_c": t_indoor, "t_target_c": target, **result}
            if model_dhw:
                row["t_tank_start_c"] = t_tank
                t_tank = result["t_tank_end_c"]
            rows.append(row)
            t_indoor = result["t_end_c"]

        df = pd.DataFrame(rows, index=timestamps)
        df.index.name = "step_start"
        df["q_unmet_kwh"] = df["q_required_kwh"] - df["q_delivered_kwh"]
        df["t_setpoint_c"] = setpoints
        df["q_total_kwh"] = df["q_delivered_kwh"] + df["q_dhw_kwh"]
        df[f"{self.fuel}_demand_kwh"] = df["q_delivered_kwh"] / self.efficiency + df["q_dhw_kwh"] / self.efficiency_dhw
        return df
