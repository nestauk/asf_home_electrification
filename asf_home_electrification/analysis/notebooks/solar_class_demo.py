# %%
import pandas as pd

from asf_home_electrification.getters import elexon_demand_profiles
from asf_home_electrification.pipeline.solar import SolarPV

# %% [markdown]
# ## Setting up demand and price profiles

# %% [markdown]
# Demand

# %%
# Elexon profile every day in 2025
annual_electricity_use_kwh = 2_500  # kWh

elexon_2025_df = elexon_demand_profiles.get_elexon_profile_use_daily(
    df=elexon_demand_profiles.get_raw_elexon_profile_coefficients(),
    annual_electricity_use=annual_electricity_use_kwh,
    unit="kWh",
    start_date="01/01/2025",
    end_date="31/12/2025",
)

elexon_profile_class_1 = elexon_2025_df[elexon_2025_df["Profile Class ID"] == 1].reset_index(drop=True)

# %%
# Create demand_profile arg as pd.Series
local_midnight = pd.to_datetime(elexon_profile_class_1["Settlement Date"]).dt.tz_localize("Europe/London")
timestamps = local_midnight + pd.to_timedelta((elexon_profile_class_1["Settlement Period ID"] - 1) * 30, unit="min")

demand_profile = pd.Series(
    elexon_profile_class_1["Electricity use (kWh)"].to_numpy(),
    index=pd.DatetimeIndex(timestamps, name="timestamp"),
).sort_index()

# %%
# Baseline bill with price cap rates
price_cap_rate = 26.32  # p / kWh
price_cap_standing_charge = 54.83  # p / day

n_days = elexon_profile_class_1["Settlement Date"].nunique()  # 365 for 2025

baseline_annual_bill = (demand_profile.values.sum() * price_cap_rate + n_days * price_cap_standing_charge) / 100  # £
baseline_annual_bill

# %% [markdown]
# Import prices - set as Standard Variable Tariff

# %%
price_cap_rate = 26.32  # p / kWh
price_cap_standing_charge = 54.83  # p / day

import_price_profile = pd.Series(price_cap_rate, index=demand_profile.index)

# %% [markdown]
# Export prices - set as Fixed Smart Export Guarantee

# %%
# https://octopus.energy/smart/smart-export-guarantee/
# 02/11/2026 fixed at 4.1 p/kWh
seg_rate = 4.1  # p /kWh
export_price_profile = pd.Series(seg_rate, index=demand_profile.index)

# %% [markdown]
# ---

# %% [markdown]
# ## Instantiate solar PV object

# %%
# Set representative household parameters
lat = 52.5  # West Midlands region (typical central England suburban/semi-rural conditions)
lon = -1.5
peakpower = 4.0  # kWp (typical, semi-detached)
loss = 14  # % (standard PVGIS default, realistic)
mountingplace = "building"
aspect = 0  # south-facing
angle = 35  # degrees (typical UK roof pitch)

# %%
my_solar = SolarPV(
    peak_power_kw=peakpower, lat=lat, lon=lon, loss=loss, mounting_place=mountingplace, aspect=aspect, angle=angle
)

# %% [markdown]
# ## 1. Run annual household energy balance with exports enabled

# %%
energy_balance = my_solar.run_household_energy_balance(
    demand_profile=demand_profile, import_price_profile=import_price_profile, export_price_profile=export_price_profile
)

# %%
# Results summary
run = energy_balance

# Annual energy totals (kWh)
demand_kwh = run["demand_kwh"].sum()
generation_kwh = run["solar_generation_kwh"].sum()
self_consumed_kwh = run["self_consumption_kwh"].sum()
exported_kwh = run["grid_export_kwh"].sum()
imported_kwh = run["grid_import_kwh"].sum()
curtailed_kwh = generation_kwh - self_consumed_kwh - exported_kwh

# Ratios
self_consumption_rate = self_consumed_kwh / generation_kwh  # share of generation used in the home
self_sufficiency = self_consumed_kwh / demand_kwh  # share of demand met by solar

# Bills (£, including standing charge)
standing_charge_gbp = price_cap_standing_charge * n_days / 100
import_cost_gbp = run["import_cost_p"].sum() / 100
export_revenue_gbp = run["export_revenue_p"].sum() / 100
bill_with_solar = import_cost_gbp - export_revenue_gbp + standing_charge_gbp

# No-solar bill on the same demand and prices, so the comparison is like for like
bill_no_solar = ((run["demand_kwh"] * run["import_price_p_per_kwh"]).sum() / 100) + standing_charge_gbp
saving_solar = bill_no_solar - bill_with_solar

print(f"Demand:                  {demand_kwh:,.0f} kWh")
print(f"Solar generation:        {generation_kwh:,.0f} kWh")
print(f"  Self-consumed:         {self_consumed_kwh:,.0f} kWh ({self_consumption_rate:.0%} of generation)")
print(f"  Exported:              {exported_kwh:,.0f} kWh")
print(f"  Curtailed:             {curtailed_kwh:,.0f} kWh")
print(f"Grid import:             {imported_kwh:,.0f} kWh (solar meets {self_sufficiency:.0%} of demand)")
print()
print(f"Import cost:             £{import_cost_gbp:,.2f}")
print(f"Export revenue:          £{export_revenue_gbp:,.2f}")
print(f"Standing charge:         £{standing_charge_gbp:,.2f}")
print(f"Bill with solar:         £{bill_with_solar:,.2f}")
print(f"Bill without solar:      £{bill_no_solar:,.2f}")
print(f"Saving from solar:       £{saving_solar:,.2f} ({saving_solar / bill_no_solar:.0%})")

# %% [markdown]
# ## 2. Run annual household energy balance with exports NOT enabled

# %%
energy_balance_no_export = my_solar.run_household_energy_balance_no_export(
    demand_profile=demand_profile, import_price_profile=import_price_profile
)

# %%
# Results summary (no export)
run = energy_balance_no_export

# Annual energy totals (kWh)
demand_kwh = run["demand_kwh"].sum()
generation_kwh = run["solar_generation_kwh"].sum()
self_consumed_kwh = run["self_consumption_kwh"].sum()
imported_kwh = run["grid_import_kwh"].sum()
curtailed_kwh = generation_kwh - self_consumed_kwh  # all surplus is lost, as export is not allowed

# Ratios
self_consumption_rate = self_consumed_kwh / generation_kwh  # share of generation used in the home
self_sufficiency = self_consumed_kwh / demand_kwh  # share of demand met by solar

# Bills (£, including standing charge)
standing_charge_gbp = price_cap_standing_charge * n_days / 100
import_cost_gbp = run["import_cost_p"].sum() / 100
bill_with_solar = import_cost_gbp + standing_charge_gbp

# No-solar bill on the same demand and prices, so the comparison is like for like
bill_no_solar = ((run["demand_kwh"] * run["import_price_p_per_kwh"]).sum() / 100) + standing_charge_gbp
saving_solar = bill_no_solar - bill_with_solar

print(f"Demand:                  {demand_kwh:,.0f} kWh")
print(f"Solar generation:        {generation_kwh:,.0f} kWh")
print(f"  Self-consumed:         {self_consumed_kwh:,.0f} kWh ({self_consumption_rate:.0%} of generation)")
print(f"  Curtailed:             {curtailed_kwh:,.0f} kWh ({curtailed_kwh / generation_kwh:.0%} of generation)")
print(f"Grid import:             {imported_kwh:,.0f} kWh (solar meets {self_sufficiency:.0%} of demand)")
print()
print(f"Import cost:             £{import_cost_gbp:,.2f}")
print(f"Standing charge:         £{standing_charge_gbp:,.2f}")
print(f"Bill with solar:         £{bill_with_solar:,.2f}")
print(f"Bill without solar:      £{bill_no_solar:,.2f}")
print(f"Saving from solar:       £{saving_solar:,.2f} ({saving_solar / bill_no_solar:.0%})")

# %%
