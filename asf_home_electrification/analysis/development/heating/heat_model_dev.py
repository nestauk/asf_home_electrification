# %%
from asf_home_electrification.analysis.development.heating import (
    simple_heat_model,
)

# %%
chosen_archetype = simple_heat_model.load_archetype("archetypes/gb.eng.sfh.04.gen.reex.001.003.yaml")

# %%
# for a theoretical half hour
simple_heat_model.heat_demand_timestep(
    t_start=15.0,
    t_out=7.4,
    t_target=20.0,
    dt=1_800.0,
    max_heat_pump_capacity=8000.0,
    solar_irradiance=simple_heat_model.TABULA_GB_MEAN_IRRADIANCE,
    **chosen_archetype,
)

# %%
# for a theoretical day
day_profile_df = simple_heat_model.heat_demand_time_period(
    t_initial=17.0,  # degC, indoor temperature at midnight of modelled day
    t_out=7.4,  # degC, scalar = same all day
    t_target=20.0,  # degC, scalar = same all day
    solar_irradiance=simple_heat_model.TABULA_GB_MEAN_IRRADIANCE,  # dict of scalars = same all day, W/m2 by orientation
    max_heat_pump_capacity=10_000.0,  # W thermal
    dt=1_800.0,
    steps=48,
    **chosen_archetype,
)

# %%
simple_heat_model.approximate_annual_heat_kwh(day_profile_df)

# %%
# for a theoretical year
year_profile_df = simple_heat_model.heat_demand_time_period(
    t_initial=17.0,  # degC, indoor temperature at midnight of modelled day
    t_out=7.4,  # degC, scalar = same all day, eventually we will move to outside temperature profile
    t_target=20.0,  # degC, scalar = same all day, eventually we will move to set-point temperature profile
    solar_irradiance=simple_heat_model.TABULA_GB_MEAN_IRRADIANCE,  # dict of scalars = same all day, W/m2 by orientation, eventually we will move to irradiance profile
    max_heat_pump_capacity=10_000.0,  # W thermal
    dt=1_800.0,
    steps=365 * 48,
    **chosen_archetype,
)

# %%
year_profile_df

# %%
year_profile_df["heat_kwh"].sum()

# %% [markdown]
# Next steps:
# - Replace constant outside temperature and solar irradiance with a year of outdoor temperaure and irradiance for a real location
# - Replace constant set-point target temperature with a schedule (e.g. warm overnight and during waking hours, lower daytime when noone's home, and separate schedules for weekday vs weekend) (f_nu should then be set as 1.0)
# - Add pre-heating: currently, controller asks for temperature ramp up in one half-hour which gives a morning spike (capped by heat pump capacity) then flattens. Real thermostat starts heating before the target time.
# - When converting to electricity demand, consider whether we want a variable COP that is a function of outdoor temperature (and flow temperature?)
# - Model domestic hot water requirements, on the same unit (usually takes priority over space heating)
# - Verify against measured use
# - Add in tariff price series input functionality to calculate heating bill
#
# Two response models to approximate heating bill based on a given tariff:
# 1. Rule-based: Set a t_target schedule that a human would plausibly set. (Develop this first)
# 2. Cost-minimising optimiser (smart controller): LP model with perfect foresight inside a comfort band, this would give the upper limit of what a household could save.
#
#
