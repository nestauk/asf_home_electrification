# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: asf-faster-further-electrification (3.13.1)
#     language: python
#     name: python3
# ---

# %%
import pandas

from asf_faster_further_electrification.getters import elexon_demand_profiles, carbon_intensity, octopus_tariffs
from asf_faster_further_electrification.utils.batteries_utils import calculate_bill_summary, calculate_import_summary, calculate_energy_flows_summary
from asf_faster_further_electrification.batteries import Battery

# %%
# April to June 2026 price cap
electricity_unit_rate = 24.67  # p/kWh
electricity_standing_charge = 57.21  # p/day

# %% [markdown]
# ### Get Elexon Profile

# %%
# Elexon profile every day in 2025
elexon_2025_df = elexon_demand_profiles.get_elexon_profile_use_daily(
    df=elexon_demand_profiles.get_raw_elexon_profile_coefficients(),
    annual_electricity_use=1,
    unit="kWh",
    start_date="01/01/2025",
    end_date="31/12/2025",
)

elexon_profile_class_1 = elexon_2025_df[elexon_2025_df["Profile Class ID"] == 1].reset_index(drop=True)

# %% [markdown]
# ### Get Tariff Prices
#
# #### Octopus Go

# %%
go_price_raw = octopus_tariffs.get_go_rates(region="E", start_date="2025-01-01", end_date="2026-01-01")
go_price = octopus_tariffs.expand_flux_to_half_hourly(go_price_raw)

# Convert UTC valid_from to local time to get correct settlement timestamps
go_price["valid_from_local"] = go_price["valid_from"].dt.tz_convert("Europe/London")
# Derive Settlement Date and timestamp from LOCAL time
go_price["Settlement Date"] = go_price["valid_from_local"].dt.date
go_price["Settlement Date"] = pandas.to_datetime(go_price["Settlement Date"])
go_price["Settlement timestamp"] = go_price["valid_from_local"].dt.strftime("%H:%M")

# Settlement Period ID from local time
go_price["Settlement Period ID"] = (
    pandas.to_timedelta(go_price["Settlement timestamp"] + ":00").dt.total_seconds() / 1800
).astype(int) + 1

# %%
df = octopus_tariffs.get_go_standing_charge(region="E", start_date="2025-01-01", end_date="2026-01-01")
df["valid_from"] = pandas.to_datetime(df["valid_from"])
df["valid_to"] = pandas.to_datetime(df["valid_to"])
df["days"] = (df["valid_to"] - df["valid_from"]).dt.total_seconds() / 86400
go_standing_charge = (df["value_inc_vat"] * df["days"]).sum() / df["days"].sum()

# %% [markdown]
# ### Octopus Agile

# %%
# # Run this first time only
# df = octopus_tariffs.get_agile_rates(region="E", start_date="2025-01-01", end_date="2026-01-01")
# df.to_pickle("agile_region_E_2025.pkl")

# %%
agile_import_price = pandas.read_pickle("agile_region_E_2025.pkl")

# Create clean date column
agile_import_price["valid_from"] = pandas.to_datetime(agile_import_price["valid_from"])
agile_import_price["Settlement Date"] = agile_import_price["valid_from"].dt.tz_convert(None).dt.floor("D")

# Create clean timestamp column
agile_import_price["Settlement timestamp"] = agile_import_price["valid_from"].dt.strftime("%H:%M")

# Sort values
agile_import_price = agile_import_price.sort_values(
    by=["Settlement Date", "valid_from"], ascending=[True, True]
).reset_index(drop=True)

# %%
agile_standing_charge_df = octopus_tariffs.get_agile_standing_charge(
    region="E", start_date="2025-01-01", end_date="2026-01-01"
)
agile_standing_charge = agile_standing_charge_df["value_inc_vat"][0]  # p/day

# %%
# Convert UTC to local time
agile_import_price["valid_from_local"] = agile_import_price["valid_from"].dt.tz_convert("Europe/London")

# Derive Settlement Date and timestamp from local time
agile_import_price["Settlement Date"] = pandas.to_datetime(agile_import_price["valid_from_local"].dt.date)
agile_import_price["Settlement timestamp"] = agile_import_price["valid_from_local"].dt.strftime("%H:%M")

# Settlement Period ID from local time
agile_import_price["Settlement Period ID"] = (
    pandas.to_timedelta(agile_import_price["Settlement timestamp"] + ":00").dt.total_seconds() / 1800
).astype(int) + 1

agile_price = agile_import_price

# %% [markdown]
# ### Grid Carbon Intensity

# %%
# # Run this first time only
# year = 2025
# start = datetime(year, 1, 1)
# end = datetime(year + 1, 1, 1)

# all_days = []
# current = start

# while current < end:
#     date_str = current.strftime("%Y-%m-%d")

#     try:
#         daily_df = carbon_intensity.get_carbon_intensity_for_date(date_str)
#         all_days.append(daily_df)

#     except Exception as e:
#         print(f"Failed {date_str}: {e}")

#     current += timedelta(days=1)

# grid_intensity = pd.concat(all_days, ignore_index=True)

# grid_intensity.to_pickle("grid_intensity_2025.pkl")

# %%
grid_intensity = pandas.read_pickle("grid_intensity_2025.pkl")

# Clean date columns
grid_intensity["from"] = pandas.to_datetime(grid_intensity["from"])
grid_intensity["Settlement Date"] = grid_intensity["from"].dt.tz_convert(None).dt.floor("D")

# Create clean timestamp column
grid_intensity["Settlementtimestamp"] = grid_intensity["from"].dt.strftime("%H:%M")

# %% [markdown]
# ## Conventional Installed Battery
#
# ### Elexon Profile 1 - Every day in 2025 - Two-rate tariff - Perfect foresight day-ahead optimisation battery model

# %%
# Set up a battery system with given characteristics
battery_5_368 = Battery(capacity_kwh=5,
                        max_power_kw=3.68,
                        lifespan=12,
                        lifecycles_max=6_000,
                        charge_efficiency=0.95,
                        discharge_efficiency=0.95,
                        state_of_charge_min=0.1,
                        state_of_charge_max=0.9,
                        state_of_charge_init=0.0)

# %%
# Get optimal battery usage for a year for given usage and costs
res5_go_2700 = battery_5_368.run_linear_cost_optimisation(demand_profile=elexon_profile_class_1,
                                                         dp_settlement_date_col="Settlement Date",
                                                         dp_coefficients_col="Electricity use (kWh)",
                                                         cost=go_price,
                                                         cost_settlement_date_col="Settlement Date",
                                                         cost_col="value_inc_vat",
                                                         total_annual_consumption=2_700,
                                                         delta_t=0.5)

# %%
calculate_bill_summary(electricity_unit_rate, electricity_standing_charge, 2700, elexon_profile_class_1, go_price, go_standing_charge, res5_go_2700)

# %%
calculate_import_summary(res5_go_2700)

# %%
calculate_energy_flows_summary(res5_go_2700, battery_5_368, 2700)

# %%
# Set up a battery system with given characteristics
battery_10_5 = Battery(capacity_kwh=10,
                       max_power_kw=5,
                       lifespan=12,
                       lifecycles_max=6_000,
                       charge_efficiency=0.95,
                       discharge_efficiency=0.95,
                       state_of_charge_min=0.1,
                       state_of_charge_max=0.9,
                       state_of_charge_init=0.0)

# %%
# Get optimal battery usage for a year for given usage and costs
res10_go_2700 = battery_10_5.run_linear_cost_optimisation(demand_profile=elexon_profile_class_1,
                                                         dp_settlement_date_col="Settlement Date",
                                                         dp_coefficients_col="Electricity use (kWh)",
                                                         cost=go_price,
                                                         cost_settlement_date_col="Settlement Date",
                                                         cost_col="value_inc_vat",
                                                         total_annual_consumption=2_700,
                                                         delta_t=0.5)

# %%
calculate_bill_summary(electricity_unit_rate, electricity_standing_charge, 2700, elexon_profile_class_1, go_price, go_standing_charge, res10_go_2700)

# %% [markdown]
# ### Elexon Profile 1 - Every day in 2025 - Dynamic tariff - Perfect foresight day-ahead optimisation battery model

# %%
# Get optimal battery usage for a year for given usage and costs
res5_agile_2700 = battery_5_368.run_linear_cost_optimisation(demand_profile=elexon_profile_class_1,
                                                      dp_settlement_date_col="Settlement Date",
                                                      dp_coefficients_col="Electricity use (kWh)",
                                                      cost=agile_price,
                                                      cost_settlement_date_col="Settlement Date",
                                                      cost_col="value_inc_vat",
                                                      total_annual_consumption=2_700,
                                                      delta_t=0.5)

# %%
calculate_bill_summary(electricity_unit_rate, electricity_standing_charge, 2700, elexon_profile_class_1, agile_price, agile_standing_charge, res5_agile_2700)

# %%
calculate_import_summary(res5_agile_2700)

# %%
calculate_energy_flows_summary(res5_agile_2700, battery_5_368, 2700)

# %%
# Get optimal battery usage for a year for given usage and costs
res10_agile_2700 = battery_10_5.run_linear_cost_optimisation(demand_profile=elexon_profile_class_1,
                                                      dp_settlement_date_col="Settlement Date",
                                                      dp_coefficients_col="Electricity use (kWh)",
                                                      cost=agile_price,
                                                      cost_settlement_date_col="Settlement Date",
                                                      cost_col="value_inc_vat",
                                                      total_annual_consumption=2_700,
                                                      delta_t=0.5)

# %%
calculate_bill_summary(electricity_unit_rate, electricity_standing_charge, 2700, elexon_profile_class_1, agile_price, agile_standing_charge, res10_agile_2700)

# %%
calculate_import_summary(res10_agile_2700)

# %%
calculate_energy_flows_summary(res10_agile_2700, battery_10_5, 2700)
