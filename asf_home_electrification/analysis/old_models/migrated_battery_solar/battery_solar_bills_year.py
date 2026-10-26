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
#     display_name: asf-faster-further-electrification (3.13.2)
#     language: python
#     name: python3
# ---

# %%
import altair as alt
import numpy as np
import pandas as pd
import requests

from asf_faster_further_electrification.getters import elexon_demand_profiles, pvgis, octopus_tariffs
from asf_faster_further_electrification.solar_batteries import (
    run_linear_cost_optimisation_solar_battery,
    run_linear_cost_optimisation_solar_battery_plugin,
)
from asf_faster_further_electrification.utils.helpers import apply_dst_adjustments

# %% [markdown]
# **Get Elexon demand profiles**

# %%
# Set annual electricity use
low_tdcv = 1_800
medium_tdcv = 2_700
high_tdcv = 4_100

annual_electricity_use = medium_tdcv

# %%
# Elexon profile every day in 2025
elexon_2025_df = elexon_demand_profiles.get_elexon_profile_use_daily(
    df=elexon_demand_profiles.get_raw_elexon_profile_coefficients(),
    annual_electricity_use=annual_electricity_use,
    unit="kWh",
    start_date="01/01/2025",
    end_date="31/12/2025",
)

elexon_profile_class_1 = elexon_2025_df[elexon_2025_df["Profile Class ID"] == 1].reset_index(drop=True)

# Elexon profile coefficients sum over profile year (Apr-Mar, not Jan-Dec)
scale_factor = annual_electricity_use / elexon_profile_class_1["Electricity use (kWh)"].sum()
elexon_profile_class_1["Electricity use (kWh)"] *= scale_factor

# %% [markdown]
# **Calculate baseline bill**

# %%
# April to June 2026 price cap
electricity_unit_rate = 24.6744  # p/kWh
electricity_standing_charge = 57.2061  # p/day

baseline_bill = ((annual_electricity_use * electricity_unit_rate) + (electricity_standing_charge * 365)) / 100  # £/year
baseline_bill

# %% [markdown]
# ## Tariffs

# %% [markdown]
# **Get Octopus Flux rates**

# %% [markdown]
# Import

# %%
# Octopus Flux
# Off-peak (02:00-05:00), Standard, Peak (16:00-19:00)

flux_import_price = octopus_tariffs.get_flux_rates(
    region="E", start_date="2025-01-01", end_date="2026-01-01", direction="import"
)
flux_import_price = octopus_tariffs.expand_flux_to_half_hourly(flux_import_price)

# Create clean date column
flux_import_price["valid_from"] = pd.to_datetime(flux_import_price["valid_from"])
flux_import_price["Settlement Date"] = flux_import_price["valid_from"].dt.tz_convert(None).dt.floor("D")

# Create clean timestamp column
flux_import_price["Settlement timestamp"] = flux_import_price["valid_from"].dt.strftime("%H:%M")

# Sort values
flux_import_price = flux_import_price.sort_values(
    by=["Settlement Date", "valid_from"], ascending=[True, True]
).reset_index(drop=True)

# Convert UTC to local time
flux_import_price["valid_from_local"] = flux_import_price["valid_from"].dt.tz_convert("Europe/London")

# Derive Settlement Date and timestamp from local time
flux_import_price["Settlement Date"] = pd.to_datetime(flux_import_price["valid_from_local"].dt.date)
flux_import_price["Settlement timestamp"] = flux_import_price["valid_from_local"].dt.strftime("%H:%M")

# Settlement Period ID from local time
flux_import_price["Settlement Period ID"] = (
    pd.to_timedelta(flux_import_price["Settlement timestamp"] + ":00").dt.total_seconds() / 1800
).astype(int) + 1

# %%
# Flux standing charge
flux_standing_charge_df = octopus_tariffs.get_flux_standing_charge(
    region="E", start_date="2025-01-01", end_date="2026-01-01"
)
df = flux_standing_charge_df.copy()
df["valid_from"] = pd.to_datetime(df["valid_from"])
df["valid_to"] = pd.to_datetime(df["valid_to"])
df["days"] = (df["valid_to"] - df["valid_from"]).dt.total_seconds() / 86400

flux_standing_charge = (df["value_inc_vat"] * df["days"]).sum() / df["days"].sum()  # p/day, weighted average

# %% [markdown]
# Export

# %%
flux_export_price = octopus_tariffs.get_flux_rates(
    region="E", start_date="2025-01-01", end_date="2026-01-01", direction="export"
)
flux_export_price = octopus_tariffs.expand_flux_to_half_hourly(flux_export_price)

# Create clean date column
flux_export_price["valid_from"] = pd.to_datetime(flux_export_price["valid_from"])
flux_export_price["Settlement Date"] = flux_export_price["valid_from"].dt.tz_convert(None).dt.floor("D")

# Create clean timestamp column
flux_export_price["Settlement timestamp"] = flux_export_price["valid_from"].dt.strftime("%H:%M")

# Sort values
flux_export_price = flux_export_price.sort_values(
    by=["Settlement Date", "valid_from"], ascending=[True, True]
).reset_index(drop=True)

# Convert UTC to local time
flux_export_price["valid_from_local"] = flux_export_price["valid_from"].dt.tz_convert("Europe/London")

# Derive Settlement Date and timestamp from local time
flux_export_price["Settlement Date"] = pd.to_datetime(flux_export_price["valid_from_local"].dt.date)
flux_export_price["Settlement timestamp"] = flux_export_price["valid_from_local"].dt.strftime("%H:%M")

# Settlement Period ID from local time
flux_export_price["Settlement Period ID"] = (
    pd.to_timedelta(flux_export_price["Settlement timestamp"] + ":00").dt.total_seconds() / 1800
).astype(int) + 1

# %%
# Flux import/export rates
avg_flux_import = flux_import_price["value_inc_vat"].mean()
avg_flux_export = flux_export_price["value_inc_vat"].mean()

print(f"Average Flux import rate: {avg_flux_import:.2f} p/kWh")
print(f"Average Flux export rate: {avg_flux_export:.2f} p/kWh")

# %% [markdown]
# **Get Octopus Agile rates**

# %% [markdown]
# Import rates

# %%
# # Run this first time only
# df = octopus_tariffs.get_agile_rates(region="E", start_date="2025-01-01", end_date="2026-01-01")
# df.to_pickle("agile_region_E_2025.pkl")

# %%
agile_import_price = pd.read_pickle("agile_region_E_2025.pkl")

# Create clean date column
agile_import_price["valid_from"] = pd.to_datetime(agile_import_price["valid_from"])
agile_import_price["Settlement Date"] = agile_import_price["valid_from"].dt.tz_convert(None).dt.floor("D")

# Create clean timestamp column
agile_import_price["Settlement timestamp"] = agile_import_price["valid_from"].dt.strftime("%H:%M")

# Sort values
agile_import_price = agile_import_price.sort_values(
    by=["Settlement Date", "valid_from"], ascending=[True, True]
).reset_index(drop=True)

# Convert UTC to local time
agile_import_price["valid_from_local"] = agile_import_price["valid_from"].dt.tz_convert("Europe/London")

# Derive Settlement Date and timestamp from local time
agile_import_price["Settlement Date"] = pd.to_datetime(agile_import_price["valid_from_local"].dt.date)
agile_import_price["Settlement timestamp"] = agile_import_price["valid_from_local"].dt.strftime("%H:%M")

# Settlement Period ID from local time
agile_import_price["Settlement Period ID"] = (
    pd.to_timedelta(agile_import_price["Settlement timestamp"] + ":00").dt.total_seconds() / 1800
).astype(int) + 1

# %%
agile_standing_charge_df = octopus_tariffs.get_agile_standing_charge(
    region="E", start_date="2025-01-01", end_date="2026-01-01"
)
agile_standing_charge = agile_standing_charge_df["value_inc_vat"][0]  # p/day

# %% [markdown]
# Export rates

# %%
# # Run this first time only
# df = octopus_tariffs.get_agile_export_rates(region="E", start_date="2025-01-01", end_date="2026-01-01")
# df.to_pickle("agile_export_region_E_2025.pkl")

# %%
agile_export_price = pd.read_pickle("agile_export_region_E_2025.pkl")

# Create clean date column
agile_export_price["valid_from"] = pd.to_datetime(agile_export_price["valid_from"])
agile_export_price["Settlement Date"] = agile_export_price["valid_from"].dt.tz_convert(None).dt.floor("D")

# Create clean timestamp column
agile_export_price["Settlement timestamp"] = agile_export_price["valid_from"].dt.strftime("%H:%M")

# Sort values
agile_export_price = agile_export_price.sort_values(
    by=["Settlement Date", "valid_from"], ascending=[True, True]
).reset_index(drop=True)

# Convert UTC to local time
agile_export_price["valid_from_local"] = agile_export_price["valid_from"].dt.tz_convert("Europe/London")

# Derive Settlement Date and timestamp from local time
agile_export_price["Settlement Date"] = pd.to_datetime(agile_export_price["valid_from_local"].dt.date)
agile_export_price["Settlement timestamp"] = agile_export_price["valid_from_local"].dt.strftime("%H:%M")

# Settlement Period ID from local time
agile_export_price["Settlement Period ID"] = (
    pd.to_timedelta(agile_export_price["Settlement timestamp"] + ":00").dt.total_seconds() / 1800
).astype(int) + 1

# %%
# Average Agile export rate 2025
avg_export_rate = agile_export_price["value_inc_vat"].mean()
avg_import_rate = agile_import_price["value_inc_vat"].mean()

print(f"Average Agile import rate 2025: {avg_import_rate:.2f} p/kWh")
print(f"Average Agile export rate 2025: {avg_export_rate:.2f} p/kWh")

# %% [markdown]
# **Get grid intensity**

# %%
grid_intensity = pd.read_pickle("grid_intensity_2025.pkl")

# Clean date columns
grid_intensity["from"] = pd.to_datetime(grid_intensity["from"])
grid_intensity["Settlement Date"] = grid_intensity["from"].dt.tz_convert(None).dt.floor("D")

# Create clean timestamp column
grid_intensity["Settlementtimestamp"] = grid_intensity["from"].dt.strftime("%H:%M")

# %%
# Convert UTC valid_from to local time to get correct settlement timestamps
grid_intensity["from_local"] = pd.to_datetime(grid_intensity["from"]).dt.tz_convert("Europe/London")
grid_intensity["Settlement Date"] = pd.to_datetime(grid_intensity["from_local"].dt.date)
grid_intensity["Settlement timestamp"] = grid_intensity["from_local"].dt.strftime("%H:%M")
grid_intensity["Settlement Period ID"] = (
    pd.to_timedelta(grid_intensity["Settlement timestamp"] + ":00").dt.total_seconds() / 1800
).astype(int) + 1

# Renumber sequentially per day to handle DST
grid_intensity["Settlement Period ID"] = grid_intensity.groupby("Settlement Date").cumcount() + 1

# %%
# Average grid intensity by settlement timestamp
avg_intensity_by_period = grid_intensity.groupby("Settlement timestamp")["intensity (gCO2/kWh)"].mean()

off_peak_times = [f"{h:02d}:{m:02d}" for h in range(0, 6) for m in [0, 30]]  # periods 00:00–06:00
standard_times = [f"{h:02d}:{m:02d}" for h in range(6, 16) for m in [0, 30]] + [
    f"{h:02d}:{m:02d}" for h in range(19, 24) for m in [0, 30]
]  # periods 06:00–16:00 and 19:00–00:00
peak_times = [f"{h:02d}:{m:02d}" for h in range(16, 19) for m in [0, 30]]  # periods 16:00–19:00

off_peak_intensity = avg_intensity_by_period[avg_intensity_by_period.index.isin(off_peak_times)].mean()
standard_intensity = avg_intensity_by_period[avg_intensity_by_period.index.isin(standard_times)].mean()
peak_intensity = avg_intensity_by_period[avg_intensity_by_period.index.isin(peak_times)].mean()
annual_avg = grid_intensity["intensity (gCO2/kWh)"].mean()

print(f"Off-peak (00:00-06:00) avg intensity : {off_peak_intensity:.1f} gCO2/kWh")
print(f"Standard (06:00-15:30, 19:00-23:30)  : {standard_intensity:.1f} gCO2/kWh")
print(f"Peak     (16:00-19:00) avg intensity : {peak_intensity:.1f} gCO2/kWh")
print(f"Annual average                       : {annual_avg:.1f} gCO2/kWh")
print(f"\nShifting 1 kWh from peak to off-peak saves : {peak_intensity - off_peak_intensity:.1f} gCO2")
print(
    f"  i.e. {(peak_intensity - off_peak_intensity) / peak_intensity * 100:.1f}% lower carbon intensity overnight vs peak"
)
print(f"  equivalent to {(peak_intensity - off_peak_intensity) / 1000:.4f} kgCO2 per kWh shifted")

# %% [markdown]
# ## Rooftop solar + Installed battery

# %% [markdown]
# **Generate solar yield profiles**

# %%
# Set representative household parameters
lat = 52.5  # West Midlands region (typical central England suburban/semi-rural conditions)
lon = -1.5
peakpower = 4.0  # kWp (typical, semi-detached)
loss = 14  # % (standard PVGIS default, realistic)
mountingplace = "building"
aspect = 0  # south-facing
angle = 35  # degrees (typical UK roof pitch)

df_hh = pvgis.get_solar_half_hourly_radiation(
    lat=lat, lon=lon, peakpower=peakpower, loss=loss, mountingplace=mountingplace, aspect=aspect, angle=angle
)

# Add settlement period and timestamps
df_hh["date"] = df_hh.index.strftime("%m-%d")
df_hh["settlement_period"] = df_hh.index.hour * 2 + df_hh.index.minute // 30 + 1  # 1-48
df_hh["settlement_timestamp"] = df_hh.index.strftime("%H:%M")

# Representative year for 2005-2023
typical_year = (
    df_hh.groupby(["date", "settlement_period", "settlement_timestamp"])["generation_kWh"].mean().reset_index()
)

radiation_profile = typical_year[typical_year["date"] != "02-29"]

# Add artificial year to radiation profile
radiation_profile["Settlement Date"] = pd.to_datetime("2025-" + radiation_profile["date"], format="%Y-%m-%d")

# Fix DST dates
radiation_profile = apply_dst_adjustments(
    df=radiation_profile,
    date_col="Settlement Date",
    timestamp_col="settlement_timestamp",
    period_id_col="settlement_period",
    clocks_forward_date=pd.Timestamp("2025-03-30"),
    clocks_back_date=pd.Timestamp("2025-10-26"),
)

# %%
print(f"Annual generation: {radiation_profile['generation_kWh'].sum():,.1f} kWh")

# %% [markdown]
# Harmonising demand, price and carbon intensity profiles

# %%
# Create daily lookups
dates = elexon_profile_class_1["Settlement Date"].unique().tolist()

daily_profile = {}  # 2025
daily_flux_import_rate = {}  # 2025 historical
daily_flux_export_rate = {}  # 2025 historical
daily_agile_import_rate = {}  # 2025 historical
daily_agile_export_rate = {}  # 2025 historical
daily_radiation_profile = {}
daily_grid_intensity = {}  # 2025 historical
for date in dates:
    daily_profile[date] = elexon_profile_class_1[elexon_profile_class_1["Settlement Date"] == date]
    daily_flux_import_rate[date] = flux_import_price[flux_import_price["Settlement Date"] == date]
    daily_flux_export_rate[date] = flux_export_price[flux_export_price["Settlement Date"] == date]
    daily_agile_import_rate[date] = agile_import_price[agile_import_price["Settlement Date"] == date]
    daily_agile_export_rate[date] = agile_export_price[agile_import_price["Settlement Date"] == date]
    daily_radiation_profile[date] = radiation_profile[radiation_profile["Settlement Date"] == date]
    daily_grid_intensity[date] = grid_intensity[grid_intensity["Settlement Date"] == date]

# %% [markdown]
# ### Octopus Flux

# %%
all_days = {}

soc_init = 0.5

capacity = 5.0

for date in dates:
    daily_result = run_linear_cost_optimisation_solar_battery(
        time_series=daily_profile.get(date)["Settlement timestamp"].tolist(),
        demand_profile=daily_profile.get(date)["Electricity use (kWh)"].tolist(),
        solar_profile=daily_radiation_profile.get(date)["generation_kWh"].tolist(),
        import_prices=daily_flux_import_rate.get(date)["value_inc_vat"].tolist(),
        export_prices=daily_flux_export_rate.get(date)["value_inc_vat"].tolist(),
        capacity=capacity,  # battery
        inverter_power=3.68,  # battery
        lifespan=12,  # battery
        max_life_cycles=6_000,  # battery
        soc_init=soc_init,  # battery
        soc_min=0.1,  # battery
        soc_max=0.9,  # battery
        delta_t=0.5,  # battery
        charge_efficiency=0.95,  # battery
        discharge_efficiency=0.95,  # battery
    )

    daily_result = daily_result.copy()
    daily_result["Settlement Date"] = date

    soc_init = daily_result["soc_kwh"].iloc[-1] / capacity

    all_days[date] = daily_result

# %%
# Solar and battery optimised bill
solar_battery_import_bill = sum(all_days[date]["import_cost_p"].sum() for date in dates) / 100
solar_battery_import_bill += (flux_standing_charge * 365) / 100

solar_battery_export = sum(all_days[date]["export_cost_p"].sum() for date in dates) / 100

# Tariff switch only bill
tariff_switch_import_bill = (
    sum(
        sum(
            d * p
            for d, p in zip(
                daily_profile[date]["Electricity use (kWh)"].tolist(),
                daily_flux_import_rate[date]["value_inc_vat"].tolist(),
            )
        )
        for date in dates
    )
    / 100
)
tariff_switch_import_bill += (flux_standing_charge * 365) / 100

solar_export_gain = sum(all_days[date]["export_solar_p"].sum() for date in dates) / 100
battery_export_gain = sum(all_days[date]["export_battery_p"].sum() for date in dates) / 100

print(f"Baseline SVT: £{baseline_bill:.2f}")
print(f"Flux - no battery or solar : £{tariff_switch_import_bill:.2f}")
print(f"Flux + battery + solar bill:  £{solar_battery_import_bill:.2f}")
print(f"Flux + battery + solar export gain:  £{solar_battery_export:.2f}")
print(f"  of which solar export: £{solar_export_gain:.2f}")
print(f"  of which battery export: £{battery_export_gain:.2f}")


# %%
solar_self_consumption = sum(all_days[date]["solar_self_consumption_kwh"].sum() for date in dates)
solar_generation = sum(all_days[date]["solar_kwh"].sum() for date in dates)
print(
    f"Solar self consumption: {solar_self_consumption:.0f} kWh ({solar_self_consumption / solar_generation * 100:.1f}%)"
)
print(f"Solar export: {sum(all_days[date]['export_solar_kwh'].sum() for date in dates):.1f} kWh")

# %%
summer_dates = [d for d in dates if d.month in [4, 5, 6, 7, 8, 9]]
winter_dates = [d for d in dates if d.month in [10, 11, 12, 1, 2, 3]]

for label, date_subset in [("Summer", summer_dates), ("Winter", winter_dates), ("Annual", dates)]:
    grid_import = sum(all_days[date]["grid_import_kwh"].sum() for date in date_subset)
    grid_export = sum(all_days[date]["grid_export_kwh"].sum() for date in date_subset)
    net_grid = grid_import - grid_export
    demand = sum(all_days[date]["demand_kwh"].sum() for date in date_subset)
    print(
        f"{label}: net grid {net_grid:.0f} kWh (import {grid_import:.0f}, export {grid_export:.0f}, demand {demand:.0f})"
    )

# %%
# Check whether any SPs has battery charging and discharging at same time (not physically possible)
results = pd.concat(all_days.values())
simultaneous = results[(results["charge_kwh"] > 1e-6) & (results["discharge_kwh"] > 1e-6)]
print(f"Simultaneous charge/discharge periods: {len(simultaneous)}")

# %%
total_charge = sum(all_days[date]["charge_kwh"].sum() for date in all_days)
total_export = sum(all_days[date]["grid_export_kwh"].sum() for date in all_days)
total_import = sum(all_days[date]["grid_import_kwh"].sum() for date in all_days)
total_discharge = sum(all_days[date]["discharge_kwh"].sum() for date in all_days)
annual_solar_export = sum(all_days[date]["export_solar_kwh"].sum() for date in all_days)
annual_battery_export = sum(all_days[date]["export_battery_kwh"].sum() for date in all_days)

print(f"Annual solar export   : {annual_solar_export:.1f} kWh")
print(f"Annual battery export : {annual_battery_export:.1f} kWh")
print(f"Annual total export   : {annual_solar_export + annual_battery_export:.1f} kWh")
print(f"Annual charge         : {total_charge:.1f} kWh")
print(f"Annual discharge         : {total_discharge:.1f} kWh")
print(f"Battery export / charge ratio: {annual_battery_export / total_charge:.2f}")


# %%
daily_battery_export_ratio = pd.Series(
    {
        date: all_days[date]["export_battery_kwh"].sum() / max(all_days[date]["discharge_kwh"].sum(), 0.001)
        for date in all_days
    }
)
print(f"Mean battery export/discharge : {daily_battery_export_ratio.mean():.2f}")
print(f"Max battery export/discharge  : {daily_battery_export_ratio.max():.2f}")
print(f"Days at export cap            : {(daily_battery_export_ratio > 0.99).sum()}")

# %%
# Carbon emissions comparison

# Calculate baseline annual emissions (no battery, full grid import)
baseline_emissions = (
    sum(
        (daily_profile[date]["Electricity use (kWh)"] * daily_grid_intensity[date]["intensity (gCO2/kWh)"].values).sum()
        for date in dates
    )
    / 1000
)  # convert gCO2 to kgCO2

# Calculate new annual emissions with battery (grid import only)
battery_emissions = (
    sum(
        (all_days[date]["grid_import_kwh"] * daily_grid_intensity[date]["intensity (gCO2/kWh)"].values).sum()
        for date in dates
    )
    / 1000
)  # kgCO2

emissions_saving = baseline_emissions - battery_emissions
saving_per_kwh = emissions_saving * 1000 / total_discharge

print(f"Baseline annual emissions            : {baseline_emissions:.1f} kgCO2")
print(f"Battery annual emissions             : {battery_emissions:.1f} kgCO2")
print(
    f"Emissions saving                     : {emissions_saving:.1f} kgCO2 ({emissions_saving / baseline_emissions * 100:.1f}%)"
)
print(f"Carbon saving per kWh shifted        : {saving_per_kwh:.1f} gCO2/kWh")

# %%
# Add solar tracking
solar_by_period = pd.Series(0.0, index=avg_intensity_by_period.index)
solar_self_cons_by_period = pd.Series(0.0, index=avg_intensity_by_period.index)
export_solar_by_period = pd.Series(0.0, index=avg_intensity_by_period.index)
export_battery_by_period = pd.Series(0.0, index=avg_intensity_by_period.index)

for date in dates:
    day = all_days[date]
    for _, row in day.iterrows():
        ts = row["timestamp"]
        if ts in solar_by_period.index:
            solar_by_period[ts] += row["solar_kwh"]
            solar_self_cons_by_period[ts] += row["solar_self_consumption_kwh"]
            export_solar_by_period[ts] += row["export_solar_kwh"]
            export_battery_by_period[ts] += row["export_battery_kwh"]

# Summarise
off_peak_solar = solar_by_period[solar_by_period.index.isin(off_peak_times)].sum()
standard_solar = solar_by_period[solar_by_period.index.isin(standard_times)].sum()
peak_solar = solar_by_period[solar_by_period.index.isin(peak_times)].sum()

off_peak_self_cons = solar_self_cons_by_period[solar_self_cons_by_period.index.isin(off_peak_times)].sum()
standard_self_cons = solar_self_cons_by_period[solar_self_cons_by_period.index.isin(standard_times)].sum()
peak_self_cons = solar_self_cons_by_period[solar_self_cons_by_period.index.isin(peak_times)].sum()

off_peak_exp_solar = export_solar_by_period[export_solar_by_period.index.isin(off_peak_times)].sum()
standard_exp_solar = export_solar_by_period[export_solar_by_period.index.isin(standard_times)].sum()
peak_exp_solar = export_solar_by_period[export_solar_by_period.index.isin(peak_times)].sum()

off_peak_exp_battery = export_battery_by_period[export_battery_by_period.index.isin(off_peak_times)].sum()
standard_exp_battery = export_battery_by_period[export_battery_by_period.index.isin(standard_times)].sum()
peak_exp_battery = export_battery_by_period[export_battery_by_period.index.isin(peak_times)].sum()

total_solar = sum(all_days[date]["solar_kwh"].sum() for date in all_days)
total_self_cons = sum(all_days[date]["solar_self_consumption_kwh"].sum() for date in all_days)
total_exp_solar = sum(all_days[date]["export_solar_kwh"].sum() for date in all_days)
total_exp_battery = sum(all_days[date]["export_battery_kwh"].sum() for date in all_days)

# Then add to your existing print statements:
print(f"\n=== SOLAR GENERATION PROFILE ===")
print(f"Off-peak (00:00-06:00)              : {off_peak_solar:.1f} kWh ({off_peak_solar / total_solar * 100:.1f}%)")
print(f"Standard (06:00-16:00, 19:00-00:00) : {standard_solar:.1f} kWh ({standard_solar / total_solar * 100:.1f}%)")
print(f"Peak     (16:00-19:00)              : {peak_solar:.1f} kWh ({peak_solar / total_solar * 100:.1f}%)")

print(f"\n=== SOLAR SELF-CONSUMPTION PROFILE ===")
print(
    f"Off-peak (00:00-06:00)              : {off_peak_self_cons:.1f} kWh ({off_peak_self_cons / total_self_cons * 100:.1f}%)"
)
print(
    f"Standard (06:00-16:00, 19:00-00:00) : {standard_self_cons:.1f} kWh ({standard_self_cons / total_self_cons * 100:.1f}%)"
)
print(f"Peak     (16:00-19:00)              : {peak_self_cons:.1f} kWh ({peak_self_cons / total_self_cons * 100:.1f}%)")

print(f"\n=== EXPORT PROFILE — SOLAR ===")
print(
    f"Off-peak (00:00-06:00)              : {off_peak_exp_solar:.1f} kWh ({off_peak_exp_solar / max(total_exp_solar, 0.001) * 100:.1f}%)"
)
print(
    f"Standard (06:00-16:00, 19:00-00:00) : {standard_exp_solar:.1f} kWh ({standard_exp_solar / max(total_exp_solar, 0.001) * 100:.1f}%)"
)
print(
    f"Peak     (16:00-19:00)              : {peak_exp_solar:.1f} kWh ({peak_exp_solar / max(total_exp_solar, 0.001) * 100:.1f}%)"
)

print(f"\n=== EXPORT PROFILE — BATTERY ===")
print(
    f"Off-peak (00:00-06:00)              : {off_peak_exp_battery:.1f} kWh ({off_peak_exp_battery / max(total_exp_battery, 0.001) * 100:.1f}%)"
)
print(
    f"Standard (06:00-16:00, 19:00-00:00) : {standard_exp_battery:.1f} kWh ({standard_exp_battery / max(total_exp_battery, 0.001) * 100:.1f}%)"
)
print(
    f"Peak     (16:00-19:00)              : {peak_exp_battery:.1f} kWh ({peak_exp_battery / max(total_exp_battery, 0.001) * 100:.1f}%)"
)

# %% [markdown]
# ### Octopus Agile

# %%
all_days = {}

soc_init = 0.5

capacity = 5.0

for date in dates:
    daily_result = run_linear_cost_optimisation_solar_battery(
        time_series=daily_profile.get(date)["Settlement timestamp"].tolist(),
        demand_profile=daily_profile.get(date)["Electricity use (kWh)"].tolist(),
        solar_profile=daily_radiation_profile.get(date)["generation_kWh"].tolist(),
        import_prices=daily_agile_import_rate.get(date)["value_inc_vat"].tolist(),
        export_prices=daily_agile_export_rate.get(date)["value_inc_vat"].tolist(),
        capacity=capacity,
        inverter_power=3.68,
        lifespan=12,
        max_life_cycles=6_000,
        soc_init=soc_init,
        soc_min=0.1,
        soc_max=0.9,
        delta_t=0.5,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
    )

    daily_result = daily_result.copy()
    daily_result["Settlement Date"] = date

    soc_init = daily_result["soc_kwh"].iloc[-1] / capacity

    all_days[date] = daily_result

# %%
# Solar and battery optimised bill
solar_battery_import_bill = sum(all_days[date]["import_cost_p"].sum() for date in dates) / 100
solar_battery_import_bill += (agile_standing_charge * 365) / 100

solar_battery_export = sum(all_days[date]["export_cost_p"].sum() for date in dates) / 100

# Tariff switch only bill
tariff_switch_import_bill = (
    sum(
        sum(
            d * p
            for d, p in zip(
                daily_profile[date]["Electricity use (kWh)"].tolist(),
                daily_agile_import_rate[date]["value_inc_vat"].tolist(),
            )
        )
        for date in dates
    )
    / 100
)
tariff_switch_import_bill += (agile_standing_charge * 365) / 100

solar_export_gain = sum(all_days[date]["export_solar_p"].sum() for date in dates) / 100
battery_export_gain = sum(all_days[date]["export_battery_p"].sum() for date in dates) / 100

print(f"Baseline SVT: £{baseline_bill:.2f}")
print(f"Agile - no battery or solar : £{tariff_switch_import_bill:.2f}")
print(f"Agile + battery + solar bill:  £{solar_battery_import_bill:.2f}")
print(f"Agile + battery + solar export gain:  £{solar_battery_export:.2f}")
print(f"  of which solar export: £{solar_export_gain:.2f}")
print(f"  of which battery export: £{battery_export_gain:.2f}")


# %%
solar_self_consumption = sum(all_days[date]["solar_self_consumption_kwh"].sum() for date in dates)
solar_generation = sum(all_days[date]["solar_kwh"].sum() for date in dates)
print(
    f"Solar self consumption: {solar_self_consumption:.0f} kWh ({solar_self_consumption / solar_generation * 100:.1f}%)"
)
print(f"Solar export: {sum(all_days[date]['export_solar_kwh'].sum() for date in dates):.1f} kWh")

# %%
summer_dates = [d for d in dates if d.month in [4, 5, 6, 7, 8, 9]]
winter_dates = [d for d in dates if d.month in [10, 11, 12, 1, 2, 3]]

for label, date_subset in [("Summer", summer_dates), ("Winter", winter_dates), ("Annual", dates)]:
    grid_import = sum(all_days[date]["grid_import_kwh"].sum() for date in date_subset)
    grid_export = sum(all_days[date]["grid_export_kwh"].sum() for date in date_subset)
    net_grid = grid_import - grid_export
    demand = sum(all_days[date]["demand_kwh"].sum() for date in date_subset)
    print(
        f"{label}: net grid {net_grid:.0f} kWh (import {grid_import:.0f}, export {grid_export:.0f}, demand {demand:.0f})"
    )

# %%
# Check whether any SPs has battery charging and discharging at same time (not physically possible)
results = pd.concat(all_days.values())
simultaneous = results[(results["charge_kwh"] > 1e-6) & (results["discharge_kwh"] > 1e-6)]
print(f"Simultaneous charge/discharge periods: {len(simultaneous)}")

# %%
total_charge = sum(all_days[date]["charge_kwh"].sum() for date in all_days)
total_export = sum(all_days[date]["grid_export_kwh"].sum() for date in all_days)
total_import = sum(all_days[date]["grid_import_kwh"].sum() for date in all_days)
total_discharge = sum(all_days[date]["discharge_kwh"].sum() for date in all_days)
annual_solar_export = sum(all_days[date]["export_solar_kwh"].sum() for date in all_days)
annual_battery_export = sum(all_days[date]["export_battery_kwh"].sum() for date in all_days)

print(f"Annual solar export   : {annual_solar_export:.1f} kWh")
print(f"Annual battery export : {annual_battery_export:.1f} kWh")
print(f"Annual total export   : {annual_solar_export + annual_battery_export:.1f} kWh")
print(f"Annual charge         : {total_charge:.1f} kWh")
print(f"Annual discharge         : {total_discharge:.1f} kWh")
print(f"Battery export / charge ratio: {annual_battery_export / total_charge:.2f}")


# %%
daily_battery_export_ratio = pd.Series(
    {
        date: all_days[date]["export_battery_kwh"].sum() / max(all_days[date]["discharge_kwh"].sum(), 0.001)
        for date in all_days
    }
)
print(f"Mean battery export/discharge : {daily_battery_export_ratio.mean():.2f}")
print(f"Max battery export/discharge  : {daily_battery_export_ratio.max():.2f}")
print(f"Days at export cap            : {(daily_battery_export_ratio > 0.99).sum()}")

# %%
# Carbon emissions comparison

# Calculate baseline annual emissions (no battery, full grid import)
baseline_emissions = (
    sum(
        (daily_profile[date]["Electricity use (kWh)"] * daily_grid_intensity[date]["intensity (gCO2/kWh)"].values).sum()
        for date in dates
    )
    / 1000
)  # convert gCO2 to kgCO2

# Calculate new annual emissions with battery (grid import only)
battery_emissions = (
    sum(
        (all_days[date]["grid_import_kwh"] * daily_grid_intensity[date]["intensity (gCO2/kWh)"].values).sum()
        for date in dates
    )
    / 1000
)  # kgCO2

emissions_saving = baseline_emissions - battery_emissions
saving_per_kwh = emissions_saving * 1000 / total_discharge

print(f"Baseline annual emissions            : {baseline_emissions:.1f} kgCO2")
print(f"Battery annual emissions             : {battery_emissions:.1f} kgCO2")
print(
    f"Emissions saving                     : {emissions_saving:.1f} kgCO2 ({emissions_saving / baseline_emissions * 100:.1f}%)"
)
print(f"Carbon saving per kWh shifted        : {saving_per_kwh:.1f} gCO2/kWh")

# %%
# Add solar tracking
solar_by_period = pd.Series(0.0, index=avg_intensity_by_period.index)
solar_self_cons_by_period = pd.Series(0.0, index=avg_intensity_by_period.index)
export_solar_by_period = pd.Series(0.0, index=avg_intensity_by_period.index)
export_battery_by_period = pd.Series(0.0, index=avg_intensity_by_period.index)

for date in dates:
    day = all_days[date]
    for _, row in day.iterrows():
        ts = row["timestamp"]
        if ts in solar_by_period.index:
            solar_by_period[ts] += row["solar_kwh"]
            solar_self_cons_by_period[ts] += row["solar_self_consumption_kwh"]
            export_solar_by_period[ts] += row["export_solar_kwh"]
            export_battery_by_period[ts] += row["export_battery_kwh"]

# Summarise
off_peak_solar = solar_by_period[solar_by_period.index.isin(off_peak_times)].sum()
standard_solar = solar_by_period[solar_by_period.index.isin(standard_times)].sum()
peak_solar = solar_by_period[solar_by_period.index.isin(peak_times)].sum()

off_peak_self_cons = solar_self_cons_by_period[solar_self_cons_by_period.index.isin(off_peak_times)].sum()
standard_self_cons = solar_self_cons_by_period[solar_self_cons_by_period.index.isin(standard_times)].sum()
peak_self_cons = solar_self_cons_by_period[solar_self_cons_by_period.index.isin(peak_times)].sum()

off_peak_exp_solar = export_solar_by_period[export_solar_by_period.index.isin(off_peak_times)].sum()
standard_exp_solar = export_solar_by_period[export_solar_by_period.index.isin(standard_times)].sum()
peak_exp_solar = export_solar_by_period[export_solar_by_period.index.isin(peak_times)].sum()

off_peak_exp_battery = export_battery_by_period[export_battery_by_period.index.isin(off_peak_times)].sum()
standard_exp_battery = export_battery_by_period[export_battery_by_period.index.isin(standard_times)].sum()
peak_exp_battery = export_battery_by_period[export_battery_by_period.index.isin(peak_times)].sum()

total_solar = sum(all_days[date]["solar_kwh"].sum() for date in all_days)
total_self_cons = sum(all_days[date]["solar_self_consumption_kwh"].sum() for date in all_days)
total_exp_solar = sum(all_days[date]["export_solar_kwh"].sum() for date in all_days)
total_exp_battery = sum(all_days[date]["export_battery_kwh"].sum() for date in all_days)

# Then add to your existing print statements:
print(f"\n=== SOLAR GENERATION PROFILE ===")
print(f"Off-peak (00:00-06:00)              : {off_peak_solar:.1f} kWh ({off_peak_solar / total_solar * 100:.1f}%)")
print(f"Standard (06:00-16:00, 19:00-00:00) : {standard_solar:.1f} kWh ({standard_solar / total_solar * 100:.1f}%)")
print(f"Peak     (16:00-19:00)              : {peak_solar:.1f} kWh ({peak_solar / total_solar * 100:.1f}%)")

print(f"\n=== SOLAR SELF-CONSUMPTION PROFILE ===")
print(
    f"Off-peak (00:00-06:00)              : {off_peak_self_cons:.1f} kWh ({off_peak_self_cons / total_self_cons * 100:.1f}%)"
)
print(
    f"Standard (06:00-16:00, 19:00-00:00) : {standard_self_cons:.1f} kWh ({standard_self_cons / total_self_cons * 100:.1f}%)"
)
print(f"Peak     (16:00-19:00)              : {peak_self_cons:.1f} kWh ({peak_self_cons / total_self_cons * 100:.1f}%)")

print(f"\n=== EXPORT PROFILE — SOLAR ===")
print(
    f"Off-peak (00:00-06:00)              : {off_peak_exp_solar:.1f} kWh ({off_peak_exp_solar / max(total_exp_solar, 0.001) * 100:.1f}%)"
)
print(
    f"Standard (06:00-16:00, 19:00-00:00) : {standard_exp_solar:.1f} kWh ({standard_exp_solar / max(total_exp_solar, 0.001) * 100:.1f}%)"
)
print(
    f"Peak     (16:00-19:00)              : {peak_exp_solar:.1f} kWh ({peak_exp_solar / max(total_exp_solar, 0.001) * 100:.1f}%)"
)

print(f"\n=== EXPORT PROFILE — BATTERY ===")
print(
    f"Off-peak (00:00-06:00)              : {off_peak_exp_battery:.1f} kWh ({off_peak_exp_battery / max(total_exp_battery, 0.001) * 100:.1f}%)"
)
print(
    f"Standard (06:00-16:00, 19:00-00:00) : {standard_exp_battery:.1f} kWh ({standard_exp_battery / max(total_exp_battery, 0.001) * 100:.1f}%)"
)
print(
    f"Peak     (16:00-19:00)              : {peak_exp_battery:.1f} kWh ({peak_exp_battery / max(total_exp_battery, 0.001) * 100:.1f}%)"
)

# %% [markdown]
# ## Plug-in solar with plug-in battery

# %% [markdown]
# - No export
# - Solar feeds home circuit via wall socket
# - Battery connects to home circuit via wall sock
# - Solar gen reduces net demand on circuit and battery charges from what is available on circuit

# %% [markdown]
# **Generate solar yield profiles**

# %%
# Set representative household parameters
lat = 52.5  # West Midlands region (typical central England suburban/semi-rural conditions)
lon = -1.5
peakpower = 0.8  # kWp (typical, semi-detached)
loss = 14  # % (standard PVGIS default, realistic)
mountingplace = "building"
aspect = 0  # south-facing
angle = 35  # degrees (typical UK roof pitch)

df_hh = pvgis.get_solar_half_hourly_radiation(
    lat=lat, lon=lon, peakpower=peakpower, loss=loss, mountingplace=mountingplace, aspect=aspect, angle=angle
)

# Add settlement period and timestamps
df_hh["date"] = df_hh.index.strftime("%m-%d")
df_hh["settlement_period"] = df_hh.index.hour * 2 + df_hh.index.minute // 30 + 1  # 1-48
df_hh["settlement_timestamp"] = df_hh.index.strftime("%H:%M")

# Representative year for 2005-2023
typical_year = (
    df_hh.groupby(["date", "settlement_period", "settlement_timestamp"])["generation_kWh"].mean().reset_index()
)

radiation_profile = typical_year[typical_year["date"] != "02-29"]

# Add artificial year to radiation profile
radiation_profile["Settlement Date"] = pd.to_datetime("2025-" + radiation_profile["date"], format="%Y-%m-%d")

# Fix DST dates
radiation_profile = apply_dst_adjustments(
    df=radiation_profile,
    date_col="Settlement Date",
    timestamp_col="settlement_timestamp",
    period_id_col="settlement_period",
    clocks_forward_date=pd.Timestamp("2025-03-30"),
    clocks_back_date=pd.Timestamp("2025-10-26"),
)

# %%
print(f"Annual generation: {radiation_profile['generation_kWh'].sum():,.1f} kWh")

# %%
daily_radiation_profile = {}
for date in dates:
    daily_radiation_profile[date] = radiation_profile[radiation_profile["Settlement Date"] == date]

# %% [markdown]
# ### Standard variable tariff

# %%
static_price = elexon_profile_class_1[["Settlement Date", "Settlement timestamp"]].copy()

# Import - standard variable tariff
static_price["value_inc_vat_import"] = electricity_unit_rate  # p/kWh

daily_svt_rate = {}
for date in dates:
    daily_svt_rate[date] = static_price[static_price["Settlement Date"] == date]

# %%
all_days = {}

soc_init = 0.5

capacity = 2.5

for date in dates:
    daily_result = run_linear_cost_optimisation_solar_battery_plugin(
        time_series=daily_profile.get(date)["Settlement timestamp"].tolist(),
        demand_profile=daily_profile.get(date)["Electricity use (kWh)"].tolist(),
        solar_profile=daily_radiation_profile.get(date)["generation_kWh"].tolist(),
        import_prices=daily_svt_rate.get(date)["value_inc_vat_import"].tolist(),
        capacity=capacity,
        inverter_power=0.8,
        lifespan=10,  # battery
        max_life_cycles=3_650,  # 1 cycle/day × 10 years × 365 days
        soc_init=soc_init,
        soc_min=0.1,
        soc_max=0.9,
        delta_t=0.5,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
    )

    daily_result = daily_result.copy()
    daily_result["Settlement Date"] = date

    soc_init = daily_result["soc_kwh"].iloc[-1] / capacity

    all_days[date] = daily_result

# %%
# Solar and battery optimised bill
solar_battery_import_bill = sum(all_days[date]["import_cost_p"].sum() for date in dates) / 100
solar_battery_import_bill += (electricity_standing_charge * 365) / 100

# Tariff switch only bill
tariff_switch_import_bill = (
    sum(
        sum(
            d * p
            for d, p in zip(
                daily_profile[date]["Electricity use (kWh)"].tolist(),
                daily_svt_rate[date]["value_inc_vat_import"].tolist(),
            )
        )
        for date in dates
    )
    / 100
)
tariff_switch_import_bill += (electricity_standing_charge * 365) / 100


print(f"Baseline SVT: £{baseline_bill:.2f}")
print(f"SVT - no battery or solar : £{tariff_switch_import_bill:.2f}")
print(f"SVT + battery + solar bill:  £{solar_battery_import_bill:.2f}")


# %%
total_solar = sum(all_days[date]["solar_kwh"].sum() for date in all_days)
total_surplus = sum(all_days[date]["surplus_kwh"].sum() for date in all_days)
total_self_cons = sum(all_days[date]["solar_self_consumption_kwh"].sum() for date in all_days)

print(f"Annual solar generation  : {total_solar:.1f} kWh")
print(f"Annual surplus : {total_surplus:.1f} kWh")
print(f"Annual self-consumption  : {total_self_cons:.1f} kWh")
print(f"Self-consumption rate    : {total_self_cons / total_solar * 100:.1f}%")

# %%
summer_dates = [d for d in dates if d.month in [4, 5, 6, 7, 8, 9]]
winter_dates = [d for d in dates if d.month in [10, 11, 12, 1, 2, 3]]

for label, date_subset in [("Summer", summer_dates), ("Winter", winter_dates), ("Annual", dates)]:
    grid_import = sum(all_days[date]["grid_import_kwh"].sum() for date in date_subset)
    demand = sum(all_days[date]["demand_kwh"].sum() for date in date_subset)
    unused_surplus = sum(all_days[date]["surplus_kwh"].sum() for date in date_subset)
    solar = sum(all_days[date]["solar_kwh"].sum() for date in date_subset)

    print(
        f"{label}: grid import {grid_import:.0f} kWh, demand {demand:.0f} kWh, "
        f"solar {solar:.0f} kWh, unused surplus {unused_surplus:.0f} kWh"
    )

# %%
# Check whether any SPs has battery charging and discharging at same time (not physically possible)
results = pd.concat(all_days.values())
simultaneous = results[(results["charge_kwh"] > 1e-6) & (results["discharge_kwh"] > 1e-6)]
print(f"Simultaneous charge/discharge periods: {len(simultaneous)}")

# %%
total_charge = sum(all_days[date]["charge_kwh"].sum() for date in all_days)
total_import = sum(all_days[date]["grid_import_kwh"].sum() for date in all_days)
total_discharge = sum(all_days[date]["discharge_kwh"].sum() for date in all_days)


print(f"Annual charge         : {total_charge:.1f} kWh")
print(f"Annual discharge         : {total_discharge:.1f} kWh")

# %%
# Carbon emissions comparison

# Calculate baseline annual emissions (no battery, full grid import)
baseline_emissions = (
    sum(
        (daily_profile[date]["Electricity use (kWh)"] * daily_grid_intensity[date]["intensity (gCO2/kWh)"].values).sum()
        for date in dates
    )
    / 1000
)  # convert gCO2 to kgCO2

# Calculate new annual emissions with battery (grid import only)
battery_emissions = (
    sum(
        (all_days[date]["grid_import_kwh"] * daily_grid_intensity[date]["intensity (gCO2/kWh)"].values).sum()
        for date in dates
    )
    / 1000
)  # kgCO2

emissions_saving = baseline_emissions - battery_emissions
saving_per_kwh = emissions_saving * 1000 / total_discharge

print(f"Baseline annual emissions            : {baseline_emissions:.1f} kgCO2")
print(f"Battery annual emissions             : {battery_emissions:.1f} kgCO2")
print(
    f"Emissions saving                     : {emissions_saving:.1f} kgCO2 ({emissions_saving / baseline_emissions * 100:.1f}%)"
)
print(f"Carbon saving per kWh shifted        : {saving_per_kwh:.1f} gCO2/kWh")

# %%
# Add solar tracking — plug-in solar + battery, no export
solar_by_period = pd.Series(0.0, index=avg_intensity_by_period.index)
solar_self_cons_by_period = pd.Series(0.0, index=avg_intensity_by_period.index)
surplus_by_period = pd.Series(0.0, index=avg_intensity_by_period.index)

for date in dates:
    day = all_days[date]
    for _, row in day.iterrows():
        ts = row["timestamp"]
        if ts in solar_by_period.index:
            solar_by_period[ts] += row["solar_kwh"]
            solar_self_cons_by_period[ts] += row["solar_self_consumption_kwh"]
            surplus_by_period[ts] += row["surplus_kwh"]

# Summarise
off_peak_solar = solar_by_period[solar_by_period.index.isin(off_peak_times)].sum()
standard_solar = solar_by_period[solar_by_period.index.isin(standard_times)].sum()
peak_solar = solar_by_period[solar_by_period.index.isin(peak_times)].sum()

off_peak_self_cons = solar_self_cons_by_period[solar_self_cons_by_period.index.isin(off_peak_times)].sum()
standard_self_cons = solar_self_cons_by_period[solar_self_cons_by_period.index.isin(standard_times)].sum()
peak_self_cons = solar_self_cons_by_period[solar_self_cons_by_period.index.isin(peak_times)].sum()

off_peak_surplus = surplus_by_period[surplus_by_period.index.isin(off_peak_times)].sum()
standard_surplus = surplus_by_period[surplus_by_period.index.isin(standard_times)].sum()
peak_surplus = surplus_by_period[surplus_by_period.index.isin(peak_times)].sum()

total_solar = sum(all_days[date]["solar_kwh"].sum() for date in all_days)
total_self_cons = sum(all_days[date]["solar_self_consumption_kwh"].sum() for date in all_days)
total_surplus = sum(all_days[date]["surplus_kwh"].sum() for date in all_days)

print(f"\n=== SOLAR GENERATION PROFILE ===")
print(f"Off-peak (00:00-06:00)              : {off_peak_solar:.1f} kWh ({off_peak_solar / total_solar * 100:.1f}%)")
print(f"Standard (06:00-16:00, 19:00-00:00) : {standard_solar:.1f} kWh ({standard_solar / total_solar * 100:.1f}%)")
print(f"Peak     (16:00-19:00)              : {peak_solar:.1f} kWh ({peak_solar / total_solar * 100:.1f}%)")

print(f"\n=== SOLAR SELF-CONSUMPTION PROFILE ===")
print(
    f"Off-peak (00:00-06:00)              : {off_peak_self_cons:.1f} kWh ({off_peak_self_cons / max(total_self_cons, 0.001) * 100:.1f}%)"
)
print(
    f"Standard (06:00-16:00, 19:00-00:00) : {standard_self_cons:.1f} kWh ({standard_self_cons / max(total_self_cons, 0.001) * 100:.1f}%)"
)
print(
    f"Peak     (16:00-19:00)              : {peak_self_cons:.1f} kWh ({peak_self_cons / max(total_self_cons, 0.001) * 100:.1f}%)"
)

print(f"\n=== SURPLUS PROFILE ===")
print(
    f"Off-peak (00:00-06:00)              : {off_peak_surplus:.1f} kWh ({off_peak_surplus / max(total_surplus, 0.001) * 100:.1f}%)"
)
print(
    f"Standard (06:00-16:00, 19:00-00:00) : {standard_surplus:.1f} kWh ({standard_surplus / max(total_surplus, 0.001) * 100:.1f}%)"
)
print(
    f"Peak     (16:00-19:00)              : {peak_surplus:.1f} kWh ({peak_surplus / max(total_surplus, 0.001) * 100:.1f}%)"
)
print(f"Total annual surplus curtailed      : {total_surplus:.1f} kWh")

# %% [markdown]
# ### Octopus Agile import

# %%
all_days = {}

soc_init = 0.5

capacity = 2.5

for date in dates:
    daily_result = run_linear_cost_optimisation_solar_battery_plugin(
        time_series=daily_profile.get(date)["Settlement timestamp"].tolist(),
        demand_profile=daily_profile.get(date)["Electricity use (kWh)"].tolist(),
        solar_profile=daily_radiation_profile.get(date)["generation_kWh"].tolist(),
        import_prices=daily_agile_import_rate.get(date)["value_inc_vat"].tolist(),
        capacity=capacity,
        inverter_power=0.8,
        lifespan=10,  # battery
        max_life_cycles=3_650,  # 1 cycle/day × 10 years × 365 days
        soc_init=soc_init,
        soc_min=0.1,
        soc_max=0.9,
        delta_t=0.5,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
    )

    daily_result = daily_result.copy()
    daily_result["Settlement Date"] = date

    soc_init = daily_result["soc_kwh"].iloc[-1] / capacity

    all_days[date] = daily_result

# %%
# Solar and battery optimised bill
solar_battery_import_bill = sum(all_days[date]["import_cost_p"].sum() for date in dates) / 100
solar_battery_import_bill += (agile_standing_charge * 365) / 100

# Tariff switch only bill
tariff_switch_import_bill = (
    sum(
        sum(
            d * p
            for d, p in zip(
                daily_profile[date]["Electricity use (kWh)"].tolist(),
                daily_agile_import_rate[date]["value_inc_vat"].tolist(),
            )
        )
        for date in dates
    )
    / 100
)
tariff_switch_import_bill += (agile_standing_charge * 365) / 100


print(f"Baseline SVT: £{baseline_bill:.2f}")
print(f"Agile - no battery or solar : £{tariff_switch_import_bill:.2f}")
print(f"Agile + battery + solar bill:  £{solar_battery_import_bill:.2f}")


# %%
total_solar = sum(all_days[date]["solar_kwh"].sum() for date in all_days)
total_surplus = sum(all_days[date]["surplus_kwh"].sum() for date in all_days)
total_self_cons = sum(all_days[date]["solar_self_consumption_kwh"].sum() for date in all_days)

print(f"Annual solar generation  : {total_solar:.1f} kWh")
print(f"Annual surplus : {total_surplus:.1f} kWh")
print(f"Annual self-consumption  : {total_self_cons:.1f} kWh")
print(f"Self-consumption rate    : {total_self_cons / total_solar * 100:.1f}%")

# %%
summer_dates = [d for d in dates if d.month in [4, 5, 6, 7, 8, 9]]
winter_dates = [d for d in dates if d.month in [10, 11, 12, 1, 2, 3]]

for label, date_subset in [("Summer", summer_dates), ("Winter", winter_dates), ("Annual", dates)]:
    grid_import = sum(all_days[date]["grid_import_kwh"].sum() for date in date_subset)
    demand = sum(all_days[date]["demand_kwh"].sum() for date in date_subset)
    unused_surplus = sum(all_days[date]["surplus_kwh"].sum() for date in date_subset)
    solar = sum(all_days[date]["solar_kwh"].sum() for date in date_subset)

    print(
        f"{label}: grid import {grid_import:.0f} kWh, demand {demand:.0f} kWh, "
        f"solar {solar:.0f} kWh, unused surplus {unused_surplus:.0f} kWh"
    )

# %%
# Check whether any SPs has battery charging and discharging at same time (not physically possible)
results = pd.concat(all_days.values())
simultaneous = results[(results["charge_kwh"] > 1e-6) & (results["discharge_kwh"] > 1e-6)]
print(f"Simultaneous charge/discharge periods: {len(simultaneous)}")

# %%
total_charge = sum(all_days[date]["charge_kwh"].sum() for date in all_days)
total_import = sum(all_days[date]["grid_import_kwh"].sum() for date in all_days)
total_discharge = sum(all_days[date]["discharge_kwh"].sum() for date in all_days)


print(f"Annual charge         : {total_charge:.1f} kWh")
print(f"Annual discharge         : {total_discharge:.1f} kWh")

# %%
# Carbon emissions comparison

# Calculate baseline annual emissions (no battery, full grid import)
baseline_emissions = (
    sum(
        (daily_profile[date]["Electricity use (kWh)"] * daily_grid_intensity[date]["intensity (gCO2/kWh)"].values).sum()
        for date in dates
    )
    / 1000
)  # convert gCO2 to kgCO2

# Calculate new annual emissions with battery (grid import only)
battery_emissions = (
    sum(
        (all_days[date]["grid_import_kwh"] * daily_grid_intensity[date]["intensity (gCO2/kWh)"].values).sum()
        for date in dates
    )
    / 1000
)  # kgCO2

emissions_saving = baseline_emissions - battery_emissions
saving_per_kwh = emissions_saving * 1000 / total_discharge

print(f"Baseline annual emissions            : {baseline_emissions:.1f} kgCO2")
print(f"Battery annual emissions             : {battery_emissions:.1f} kgCO2")
print(
    f"Emissions saving                     : {emissions_saving:.1f} kgCO2 ({emissions_saving / baseline_emissions * 100:.1f}%)"
)
print(f"Carbon saving per kWh shifted        : {saving_per_kwh:.1f} gCO2/kWh")

# %%
# Add solar tracking — plug-in solar + battery, no export
solar_by_period = pd.Series(0.0, index=avg_intensity_by_period.index)
solar_self_cons_by_period = pd.Series(0.0, index=avg_intensity_by_period.index)
surplus_by_period = pd.Series(0.0, index=avg_intensity_by_period.index)

for date in dates:
    day = all_days[date]
    for _, row in day.iterrows():
        ts = row["timestamp"]
        if ts in solar_by_period.index:
            solar_by_period[ts] += row["solar_kwh"]
            solar_self_cons_by_period[ts] += row["solar_self_consumption_kwh"]
            surplus_by_period[ts] += row["surplus_kwh"]

# Summarise
off_peak_solar = solar_by_period[solar_by_period.index.isin(off_peak_times)].sum()
standard_solar = solar_by_period[solar_by_period.index.isin(standard_times)].sum()
peak_solar = solar_by_period[solar_by_period.index.isin(peak_times)].sum()

off_peak_self_cons = solar_self_cons_by_period[solar_self_cons_by_period.index.isin(off_peak_times)].sum()
standard_self_cons = solar_self_cons_by_period[solar_self_cons_by_period.index.isin(standard_times)].sum()
peak_self_cons = solar_self_cons_by_period[solar_self_cons_by_period.index.isin(peak_times)].sum()

off_peak_surplus = surplus_by_period[surplus_by_period.index.isin(off_peak_times)].sum()
standard_surplus = surplus_by_period[surplus_by_period.index.isin(standard_times)].sum()
peak_surplus = surplus_by_period[surplus_by_period.index.isin(peak_times)].sum()

total_solar = sum(all_days[date]["solar_kwh"].sum() for date in all_days)
total_self_cons = sum(all_days[date]["solar_self_consumption_kwh"].sum() for date in all_days)
total_surplus = sum(all_days[date]["surplus_kwh"].sum() for date in all_days)

print(f"\n=== SOLAR GENERATION PROFILE ===")
print(f"Off-peak (00:00-06:00)              : {off_peak_solar:.1f} kWh ({off_peak_solar / total_solar * 100:.1f}%)")
print(f"Standard (06:00-16:00, 19:00-00:00) : {standard_solar:.1f} kWh ({standard_solar / total_solar * 100:.1f}%)")
print(f"Peak     (16:00-19:00)              : {peak_solar:.1f} kWh ({peak_solar / total_solar * 100:.1f}%)")

print(f"\n=== SOLAR SELF-CONSUMPTION PROFILE ===")
print(
    f"Off-peak (00:00-06:00)              : {off_peak_self_cons:.1f} kWh ({off_peak_self_cons / max(total_self_cons, 0.001) * 100:.1f}%)"
)
print(
    f"Standard (06:00-16:00, 19:00-00:00) : {standard_self_cons:.1f} kWh ({standard_self_cons / max(total_self_cons, 0.001) * 100:.1f}%)"
)
print(
    f"Peak     (16:00-19:00)              : {peak_self_cons:.1f} kWh ({peak_self_cons / max(total_self_cons, 0.001) * 100:.1f}%)"
)

print(f"\n=== SURPLUS PROFILE ===")
print(
    f"Off-peak (00:00-06:00)              : {off_peak_surplus:.1f} kWh ({off_peak_surplus / max(total_surplus, 0.001) * 100:.1f}%)"
)
print(
    f"Standard (06:00-16:00, 19:00-00:00) : {standard_surplus:.1f} kWh ({standard_surplus / max(total_surplus, 0.001) * 100:.1f}%)"
)
print(
    f"Peak     (16:00-19:00)              : {peak_surplus:.1f} kWh ({peak_surplus / max(total_surplus, 0.001) * 100:.1f}%)"
)
print(f"Total annual surplus curtailed      : {total_surplus:.1f} kWh")

# %% [markdown]
# ### Octopus Go import

# %%
go_price_raw = octopus_tariffs.get_go_rates(region="E", start_date="2025-01-01", end_date="2026-01-01")
go_price = octopus_tariffs.expand_flux_to_half_hourly(go_price_raw)

# Convert UTC valid_from to local time to get correct settlement timestamps
go_price["valid_from_local"] = go_price["valid_from"].dt.tz_convert("Europe/London")

# Derive Settlement Date and timestamp from LOCAL time
go_price["Settlement Date"] = go_price["valid_from_local"].dt.date
go_price["Settlement Date"] = pd.to_datetime(go_price["Settlement Date"])
go_price["Settlement timestamp"] = go_price["valid_from_local"].dt.strftime("%H:%M")

# Settlement Period ID from local time
go_price["Settlement Period ID"] = (
    pd.to_timedelta(go_price["Settlement timestamp"] + ":00").dt.total_seconds() / 1800
).astype(int) + 1

# %%
df = octopus_tariffs.get_go_standing_charge(region="E", start_date="2025-01-01", end_date="2026-01-01")
df["valid_from"] = pd.to_datetime(df["valid_from"])
df["valid_to"] = pd.to_datetime(df["valid_to"])
df["days"] = (df["valid_to"] - df["valid_from"]).dt.total_seconds() / 86400
go_standing_charge = (df["value_inc_vat"] * df["days"]).sum() / df["days"].sum()

# %%
daily_go_rate = {}
for date in dates:
    daily_go_rate[date] = go_price[go_price["Settlement Date"] == date]

# %%
all_days = {}

soc_init = 0.5

capacity = 2.5

for date in dates:
    daily_result = run_linear_cost_optimisation_solar_battery_plugin(
        time_series=daily_profile.get(date)["Settlement timestamp"].tolist(),
        demand_profile=daily_profile.get(date)["Electricity use (kWh)"].tolist(),
        solar_profile=daily_radiation_profile.get(date)["generation_kWh"].tolist(),
        import_prices=daily_go_rate.get(date)["value_inc_vat"].tolist(),
        capacity=capacity,
        inverter_power=0.8,
        lifespan=10,  # battery
        max_life_cycles=3_650,  # 1 cycle/day × 10 years × 365 days
        soc_init=soc_init,
        soc_min=0.1,
        soc_max=0.9,
        delta_t=0.5,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
    )

    daily_result = daily_result.copy()
    daily_result["Settlement Date"] = date

    soc_init = daily_result["soc_kwh"].iloc[-1] / capacity

    all_days[date] = daily_result

# %%
# Solar and battery optimised bill
solar_battery_import_bill = sum(all_days[date]["import_cost_p"].sum() for date in dates) / 100
solar_battery_import_bill += (go_standing_charge * 365) / 100

# Tariff switch only bill
tariff_switch_import_bill = (
    sum(
        sum(
            d * p
            for d, p in zip(
                daily_profile[date]["Electricity use (kWh)"].tolist(),
                daily_go_rate[date]["value_inc_vat"].tolist(),
            )
        )
        for date in dates
    )
    / 100
)
tariff_switch_import_bill += (go_standing_charge * 365) / 100


print(f"Baseline SVT: £{baseline_bill:.2f}")
print(f"Go - no battery or solar : £{tariff_switch_import_bill:.2f}")
print(f"Go + battery + solar bill:  £{solar_battery_import_bill:.2f}")


# %%
total_solar = sum(all_days[date]["solar_kwh"].sum() for date in all_days)
total_surplus = sum(all_days[date]["surplus_kwh"].sum() for date in all_days)
total_self_cons = sum(all_days[date]["solar_self_consumption_kwh"].sum() for date in all_days)

print(f"Annual solar generation  : {total_solar:.1f} kWh")
print(f"Annual surplus : {total_surplus:.1f} kWh")
print(f"Annual self-consumption  : {total_self_cons:.1f} kWh")
print(f"Self-consumption rate    : {total_self_cons / total_solar * 100:.1f}%")

# %%
summer_dates = [d for d in dates if d.month in [4, 5, 6, 7, 8, 9]]
winter_dates = [d for d in dates if d.month in [10, 11, 12, 1, 2, 3]]

for label, date_subset in [("Summer", summer_dates), ("Winter", winter_dates), ("Annual", dates)]:
    grid_import = sum(all_days[date]["grid_import_kwh"].sum() for date in date_subset)
    demand = sum(all_days[date]["demand_kwh"].sum() for date in date_subset)
    unused_surplus = sum(all_days[date]["surplus_kwh"].sum() for date in date_subset)
    solar = sum(all_days[date]["solar_kwh"].sum() for date in date_subset)

    print(
        f"{label}: grid import {grid_import:.0f} kWh, demand {demand:.0f} kWh, "
        f"solar {solar:.0f} kWh, unused surplus {unused_surplus:.0f} kWh"
    )

# %%
# Check whether any SPs has battery charging and discharging at same time (not physically possible)
results = pd.concat(all_days.values())
simultaneous = results[(results["charge_kwh"] > 1e-6) & (results["discharge_kwh"] > 1e-6)]
print(f"Simultaneous charge/discharge periods: {len(simultaneous)}")

# %%
total_charge = sum(all_days[date]["charge_kwh"].sum() for date in all_days)
total_import = sum(all_days[date]["grid_import_kwh"].sum() for date in all_days)
total_discharge = sum(all_days[date]["discharge_kwh"].sum() for date in all_days)


print(f"Annual charge         : {total_charge:.1f} kWh")
print(f"Annual discharge         : {total_discharge:.1f} kWh")

# %%
# On a sunny summer day, does solar ever exceed demand?
date = dates[180]  # summer day
day = all_days[date]
surplus_periods = (day["solar_kwh"] > day["demand_kwh"]).sum()
print(f"Periods where solar > demand: {surplus_periods}")
print(f"Max solar in any period     : {day['solar_kwh'].max():.3f} kWh")
print(f"Min demand in any period    : {day['demand_kwh'].min():.3f} kWh")

# %%
# Carbon emissions comparison

# Calculate baseline annual emissions (no battery, full grid import)
baseline_emissions = (
    sum(
        (daily_profile[date]["Electricity use (kWh)"] * daily_grid_intensity[date]["intensity (gCO2/kWh)"].values).sum()
        for date in dates
    )
    / 1000
)  # convert gCO2 to kgCO2

# Calculate new annual emissions with battery (grid import only)
battery_emissions = (
    sum(
        (all_days[date]["grid_import_kwh"] * daily_grid_intensity[date]["intensity (gCO2/kWh)"].values).sum()
        for date in dates
    )
    / 1000
)  # kgCO2

emissions_saving = baseline_emissions - battery_emissions
saving_per_kwh = emissions_saving * 1000 / total_discharge

print(f"Baseline annual emissions            : {baseline_emissions:.1f} kgCO2")
print(f"Battery annual emissions             : {battery_emissions:.1f} kgCO2")
print(
    f"Emissions saving                     : {emissions_saving:.1f} kgCO2 ({emissions_saving / baseline_emissions * 100:.1f}%)"
)
print(f"Carbon saving per kWh shifted        : {saving_per_kwh:.1f} gCO2/kWh")

# %%
# Add solar tracking — plug-in solar + battery, no export
solar_by_period = pd.Series(0.0, index=avg_intensity_by_period.index)
solar_self_cons_by_period = pd.Series(0.0, index=avg_intensity_by_period.index)
surplus_by_period = pd.Series(0.0, index=avg_intensity_by_period.index)

for date in dates:
    day = all_days[date]
    for _, row in day.iterrows():
        ts = row["timestamp"]
        if ts in solar_by_period.index:
            solar_by_period[ts] += row["solar_kwh"]
            solar_self_cons_by_period[ts] += row["solar_self_consumption_kwh"]
            surplus_by_period[ts] += row["surplus_kwh"]

# Summarise
off_peak_solar = solar_by_period[solar_by_period.index.isin(off_peak_times)].sum()
standard_solar = solar_by_period[solar_by_period.index.isin(standard_times)].sum()
peak_solar = solar_by_period[solar_by_period.index.isin(peak_times)].sum()

off_peak_self_cons = solar_self_cons_by_period[solar_self_cons_by_period.index.isin(off_peak_times)].sum()
standard_self_cons = solar_self_cons_by_period[solar_self_cons_by_period.index.isin(standard_times)].sum()
peak_self_cons = solar_self_cons_by_period[solar_self_cons_by_period.index.isin(peak_times)].sum()

off_peak_surplus = surplus_by_period[surplus_by_period.index.isin(off_peak_times)].sum()
standard_surplus = surplus_by_period[surplus_by_period.index.isin(standard_times)].sum()
peak_surplus = surplus_by_period[surplus_by_period.index.isin(peak_times)].sum()

total_solar = sum(all_days[date]["solar_kwh"].sum() for date in all_days)
total_self_cons = sum(all_days[date]["solar_self_consumption_kwh"].sum() for date in all_days)
total_surplus = sum(all_days[date]["surplus_kwh"].sum() for date in all_days)

print(f"\n=== SOLAR GENERATION PROFILE ===")
print(f"Off-peak (00:00-06:00)              : {off_peak_solar:.1f} kWh ({off_peak_solar / total_solar * 100:.1f}%)")
print(f"Standard (06:00-16:00, 19:00-00:00) : {standard_solar:.1f} kWh ({standard_solar / total_solar * 100:.1f}%)")
print(f"Peak     (16:00-19:00)              : {peak_solar:.1f} kWh ({peak_solar / total_solar * 100:.1f}%)")

print(f"\n=== SOLAR SELF-CONSUMPTION PROFILE ===")
print(
    f"Off-peak (00:00-06:00)              : {off_peak_self_cons:.1f} kWh ({off_peak_self_cons / max(total_self_cons, 0.001) * 100:.1f}%)"
)
print(
    f"Standard (06:00-16:00, 19:00-00:00) : {standard_self_cons:.1f} kWh ({standard_self_cons / max(total_self_cons, 0.001) * 100:.1f}%)"
)
print(
    f"Peak     (16:00-19:00)              : {peak_self_cons:.1f} kWh ({peak_self_cons / max(total_self_cons, 0.001) * 100:.1f}%)"
)

print(f"\n=== SURPLUS PROFILE ===")
print(
    f"Off-peak (00:00-06:00)              : {off_peak_surplus:.1f} kWh ({off_peak_surplus / max(total_surplus, 0.001) * 100:.1f}%)"
)
print(
    f"Standard (06:00-16:00, 19:00-00:00) : {standard_surplus:.1f} kWh ({standard_surplus / max(total_surplus, 0.001) * 100:.1f}%)"
)
print(
    f"Peak     (16:00-19:00)              : {peak_surplus:.1f} kWh ({peak_surplus / max(total_surplus, 0.001) * 100:.1f}%)"
)
print(f"Total annual surplus curtailed      : {total_surplus:.1f} kWh")

# %%

# %%
