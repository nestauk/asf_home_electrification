"""
Functions for a simple half-hourly heat demand model for a TABULA GB archetype.

TABULA heat balance (EN ISO 13790, as implemented in the TABULA calculation method)
applied to one timestep, with a thermal-mass term so indoor temperature can be stepped forward.

TABULA seasonal equations used (all quantities in W, K, W/K here rather than kWh/a):
    H_tr  = sum(b_tr,i * A_env,i * U_eff,i) + dU_tb * sum(A_env,i)        transmission coefficient
    H_ve  = c_p,air * (n_air,use + n_air,infiltr) * A_C,ref * h_room      ventilation coefficient
    Q_int = phi_int * A_C,ref                                              internal gains
    Q_sol = sum_j F_sh,j * (1 - F_F) * F_W * g_gl * A_window,j * I_sol,j   solar gains (per orientation)
    C     = c_m * A_C,ref                                                  internal heat capacity
    F_nu  multiplies H_tr and H_ve (non-uniform heating, "F_red" in the report; archetype parameter)

Dropped from the seasonal method (they only make sense as season averages):
    eta_h,gn  gain utilisation factor - the stepping model finds usable gains itself via indoor temperature
    0.024 * d_hs  unit conversion / season length

Typical use, if modelling one day (48 half-hourly timesteps):
    archetype = load_archetype("archetypes/gb.eng.sfh.01.gen.reex.001.001.yaml")
    df = heat_demand_time_period(
        t_initial=20.0, t_out=7.4, t_target=20.0,
        solar_irradiance=TABULA_GB_MEAN_IRRADIANCE,
        max_heat_pump_capacity=10_000.0, steps=48, **archetype,
"""

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# GB average solar irradiance used in TABULA
# ---------------------------------------------------------------------------

HEATING_SEASON_HOURS = 243 * 24  # h, from the report (243 heating days)

TABULA_GB_IRRADIATION = {  # kWh/m2 over the heating season, from the report
    "horizontal": 392,
    "east": 257,
    "south": 403,
    "west": 257,
    "north": 149,
}

TABULA_GB_MEAN_IRRADIANCE = {  # W/m2, season average (includes nights)
    orientation: kwh * 1000 / HEATING_SEASON_HOURS for orientation, kwh in TABULA_GB_IRRADIATION.items()
}

# ---------------------------------------------------------------------------
# Archetype loading
# ---------------------------------------------------------------------------


def load_archetype(path):
    """Read a TABULA archetype YAML file into the dict that heat_demand_timestep expects."""
    raw = yaml.safe_load(Path(path).read_text())
    # heat loss through individual construction elements
    elements = raw["elements"]
    h_elements = sum(e["u"] * e["area"] * e["b_tr"] for e in elements)
    # thermal bridging is where heat bypasses insulation, modelled here as a flat surcharge
    h_bridging = raw["delta_u_tb"] * sum(e["area"] for e in elements)
    return dict(
        a_ref=raw["a_ref"],
        h_tr=h_elements + h_bridging,
        f_nu=raw["f_nu"],
        n_air_infiltr=raw["n_air_infiltr"],
        window_area=raw["window_area"],
        g_gl=raw["g_gl"],
    )


# ---------------------------------------------------------------------------
# TABULA GB climate: average solar irradiance
# ---------------------------------------------------------------------------

HEATING_SEASON_DAYS = 243  # from the report
HEATING_SEASON_HOURS = HEATING_SEASON_DAYS * 24  # h

TABULA_GB_IRRADIATION = {  # kWh/m2 over the heating season, from the report
    "horizontal": 392,
    "east": 257,
    "south": 403,
    "west": 257,
    "north": 149,
}

TABULA_GB_MEAN_IRRADIANCE = {  # W/m2, season average (includes nights)
    orientation: kwh * 1000 / HEATING_SEASON_HOURS for orientation, kwh in TABULA_GB_IRRADIATION.items()
}

# ---------------------------------------------------------------------------
# Input checks
# ---------------------------------------------------------------------------

VALID_ORIENTATIONS = {"horizontal", "north", "east", "south", "west"}


def check_solar_irradiance(window_area, solar_irradiance):
    """Raise if solar_irradiance isn't a valid W/m2 dict covering every orientation in window_area."""
    if not isinstance(solar_irradiance, dict):
        raise TypeError(
            f"solar_irradiance must be a dict of W/m2 by orientation, got {type(solar_irradiance).__name__}"
        )

    unknown = set(window_area) - VALID_ORIENTATIONS
    if unknown:
        raise ValueError(
            f"window_area has unrecognised orientations: {sorted(unknown)}; use {sorted(VALID_ORIENTATIONS)}"
        )

    missing = set(window_area) - set(solar_irradiance)
    if missing:
        raise KeyError(f"solar_irradiance is missing orientations present in window_area: {sorted(missing)}")

    bad = {
        k: solar_irradiance[k] for k in window_area if not math.isfinite(solar_irradiance[k]) or solar_irradiance[k] < 0
    }
    if bad:
        raise ValueError(f"solar_irradiance must be finite and non-negative: {bad}")


def _convert_scalar_to_profile(value, steps, name):
    """Scalar -> constant profile; sequence -> checked array of length steps."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = np.full(steps, arr)
    if arr.shape != (steps,):
        raise ValueError(f"{name} must be a scalar or have {steps} values, got shape {arr.shape}")
    return arr


# ---------------------------------------------------------------------------
# One timestep
# ---------------------------------------------------------------------------


def heat_demand_timestep(
    *,
    # --- state, control and weather for this timestep ---
    t_start,  # degC, indoor temperature at start of step
    t_out,  # degC
    t_target,  # degC, set-point temperature
    solar_irradiance,  # dict, solar power W/m2 on each orientation, e.g. {"east": 120, "west": 40}
    max_heat_pump_capacity,  # W thermal
    dt=1_800.0,  # s, length of timestep
    # --- archetype parameters (from the TABULA webtool report) ---
    a_ref,  # m2, floor area
    h_tr,  # W/K, transmission heat transfer coefficient (incl. thermal bridging)
    n_air_infiltr,  # 1/h, infiltration air change rate
    f_nu,  # -, temperature reduction factor for non-uniform heating (F_red in TABULA report)
    window_area,  # dict, m2 by orientation, e.g. {"east": 22.7, "west": 23.8}
    g_gl,  # -, glazing solar transmittance
    # --- TABULA standard values ---
    n_air_use=0.4,  # 1/h, use air change rate
    h_room=2.5,  # m, ventilation reference room height
    c_p_air=0.34,  # Wh/(m3 K)
    f_f=0.3,  # window frame fraction, only glass lets sun through
    f_w=0.9,  # non-perpendicular correction, sunlight at an angle
    f_sh_vertical=0.6,  # external shading correction, vertical windows
    f_sh_horizontal=0.8,  # external shading correction, horizontal windows
    c_m=45.0,  # Wh/(m2 K) internal heat capacity per m2 reference area
    phi_int=3.0,  # W/m2 internal heat source density
):

    # ventilation heat transfer coefficient
    h_ve = c_p_air * (n_air_use + n_air_infiltr) * a_ref * h_room  # W/K

    # total heat loss coefficient
    h = f_nu * (h_tr + h_ve)  # W/K

    # solar gains
    check_solar_irradiance(window_area, solar_irradiance)

    q_solar = 0.0  # W
    for orientation, area in window_area.items():  # area: m2
        f_sh = f_sh_horizontal if orientation == "horizontal" else f_sh_vertical
        q_solar += (
            f_sh
            * (1 - f_f)
            * f_w
            * g_gl  # - (dimensionless reduction factors)
            * area  # m2
            * solar_irradiance[orientation]  # W/m2
        )

    # internal free gains
    q_internal = phi_int * a_ref  # W

    # thermal mass
    c_dt = c_m * a_ref * 3600 / dt  # W/K, how much heat to raise temp by 1K in timestep
    q_mass = c_dt * (t_target - t_start)  # W

    # required heating
    t_mean = 0.5 * (t_start + t_target)
    q_transmission = f_nu * h_tr * (t_mean - t_out)
    q_ventilation = f_nu * h_ve * (t_mean - t_out)
    q_required = q_transmission + q_ventilation - q_solar - q_internal + q_mass

    # heat delivered constrained by heat pump capacity
    q_delivered = min(max(q_required, 0.0), max_heat_pump_capacity)

    # calculate indoor temperature at end of step given heat delivered
    # rearrange original balance
    # c_dt × (t_end − t_start)  =  q_delivered + q_solar + q_internal − q_loss
    # where q_loss = h × (t_indoor − t_out) where t_indoor is average of t_start and t_end
    t_end = (c_dt * t_start + q_delivered + q_solar + q_internal - h * (0.5 * t_start - t_out)) / (c_dt + 0.5 * h)

    return {
        "q_transmission": q_transmission,
        "q_ventilation": q_ventilation,
        "q_solar": q_solar,
        "q_internal": q_internal,
        "q_mass": q_mass,
        "q_required": q_required,
        "q_delivered": q_delivered,
        "t_end": t_end,
    }


# ---------------------------------------------------------------------------
# A time period (e.g. a day) of timesteps
# ---------------------------------------------------------------------------


def heat_demand_time_period(
    *, t_initial, t_out, t_target, solar_irradiance, max_heat_pump_capacity, steps, dt=1_800.0, **archetype
):
    """Step heat_demand_timestep through a period of `steps` timesteps, each `dt` seconds long.

    steps: number of timesteps in the period (48 with dt=1800 s is one day).
    t_out, t_target and each solar_irradiance value may be a scalar or `steps` values.
    Returns a DataFrame with one row per timestep; q_delivered (W) is the heat demand profile.
    """
    t_out = _convert_scalar_to_profile(t_out, steps, "t_out")
    t_target = _convert_scalar_to_profile(t_target, steps, "t_target")
    solar = {o: _convert_scalar_to_profile(v, steps, f"solar_irradiance['{o}']") for o, v in solar_irradiance.items()}

    t_indoor = t_initial
    rows = []
    for i in range(steps):
        result = heat_demand_timestep(
            t_start=t_indoor,
            t_out=t_out[i],
            t_target=t_target[i],
            solar_irradiance={o: series[i] for o, series in solar.items()},
            max_heat_pump_capacity=max_heat_pump_capacity,
            dt=dt,
            **archetype,
        )
        rows.append({"t_start": t_indoor, "t_out": t_out[i], "t_target": t_target[i], **result})
        t_indoor = result["t_end"]  # end of this step is the start of the next

    index = pd.timedelta_range("0h", periods=steps, freq=pd.Timedelta(seconds=dt), name="time")
    df = pd.DataFrame(rows, index=index)
    df["heat_kwh"] = df["q_delivered"] * dt / 3.6e6  # W x s -> kWh
    return df


def approximate_annual_heat_kwh(df, heating_days=HEATING_SEASON_DAYS, tol=0.05):
    """Scale one modelled day (output of heat_demand_time_period) to a heating season.

    heating_days: days of the year with heating (243 in the TABULA GB climate).
    The day is assumed to repeat, so it should end where it started; warns if indoor
    temperature drifts by more than tol (K), e.g. a cold start or an undersized heat pump.
    """
    drift = df["t_end"].iloc[-1] - df["t_start"].iloc[0]
    if abs(drift) > tol:
        warnings.warn(
            f"day is not periodic (indoor temperature moves {drift:+.2f} K from start to end); "
            "annual figure will include one-off recovery or storage effects",
            stacklevel=2,
        )
    return df["heat_kwh"].sum() * heating_days
