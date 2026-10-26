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

from asf_faster_further_electrification.solar import (
    calculate_solar_household_energy_balance,
    calculate_solar_household_energy_balance_no_export,
)
from asf_faster_further_electrification.getters import elexon_demand_profiles, pvgis, octopus_tariffs

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
# ## Rooftop solar panels

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

# %%
print(f"Annual generation: {radiation_profile['generation_kWh'].sum():,.1f} kWh")

# %% [markdown]
# ### Octopus Agile tariff

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

# %% [markdown]
# **Calculate electricity balance for household**

# %%
agile_solar_df = calculate_solar_household_energy_balance(
    demand_profile_df=elexon_profile_class_1,
    radiation_profile_df=radiation_profile,
    import_price_profile_df=agile_import_price,
    export_price_profile_df=agile_export_price,
)

# %%
print(
    f"Overall self-consumption rate: {(agile_solar_df['Self consumption (kWh)'].sum() / agile_solar_df['Solar generation (kWh)'].sum()) * 100:.1f}%"
)

# %%
print(
    f"Bill from import tariff switch only: £{(agile_solar_df['Import cost without solar (p)'].sum() + (agile_standing_charge * 365)) / 100:.2f}"
)
print(
    f"Bill from import tariff switch + solar: £{(agile_solar_df['Import cost with solar (p)'].sum() + (agile_standing_charge * 365)) / 100:.2f}"
)
print(f"Export benefit: £{agile_solar_df['Export revenue (p)'].sum() / 100:.2f}")


# %% [markdown]
# ### Standard variable tariff for import + Fixed SEG export

# %% [markdown]
# **Set up import and export rates time series**

# %%
# Import - standard variable tariff
import_svt_price = elexon_profile_class_1[["Settlement Date", "Settlement timestamp"]].copy()
import_svt_price["value_inc_vat"] = electricity_unit_rate  # p/kWh

# Export - fixed Smart Export Guarantee
# https://www.ecotricity.co.uk/your-green-energy/smart-export-tariff
export_fixed_price = elexon_profile_class_1[["Settlement Date", "Settlement timestamp"]].copy()
export_fixed_price["value_inc_vat"] = 16.0  # p/kWh

# %% [markdown]
# **Calculate electricity balance for household**

# %%
static_solar_df = calculate_solar_household_energy_balance(
    demand_profile_df=elexon_profile_class_1,
    radiation_profile_df=radiation_profile,
    import_price_profile_df=import_svt_price,
    export_price_profile_df=export_fixed_price,
)

# %%
print(
    f"Overall self-consumption rate: {(static_solar_df['Self consumption (kWh)'].sum() / static_solar_df['Solar generation (kWh)'].sum()) * 100:.1f}%"
)

# %%
print(
    f"Bill from import tariff switch only: £{(static_solar_df['Import cost without solar (p)'].sum() + (electricity_standing_charge * 365)) / 100:.2f}"
)
print(
    f"Bill from import tariff switch + solar: £{(static_solar_df['Import cost with solar (p)'].sum() + (electricity_standing_charge * 365)) / 100:.2f}"
)
print(f"Export benefit: £{static_solar_df['Export revenue (p)'].sum() / 100:.2f}")


# %% [markdown]
# ### Octopus Flux

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

# %% [markdown]
# **Calculate electricity balance for household**

# %%
flux_solar_df = calculate_solar_household_energy_balance(
    demand_profile_df=elexon_profile_class_1,
    radiation_profile_df=radiation_profile,
    import_price_profile_df=flux_import_price,
    export_price_profile_df=flux_export_price,
)

# %%
print(
    f"Overall self-consumption rate: {(flux_solar_df['Self consumption (kWh)'].sum() / flux_solar_df['Solar generation (kWh)'].sum()) * 100:.1f}%"
)

# %%
print(
    f"Bill from import tariff switch only: £{(flux_solar_df['Import cost without solar (p)'].sum() + (flux_standing_charge * 365)) / 100:.2f}"
)
print(
    f"Bill from import tariff switch + solar: £{(flux_solar_df['Import cost with solar (p)'].sum() + (flux_standing_charge * 365)) / 100:.2f}"
)
print(f"Export benefit: £{flux_solar_df['Export revenue (p)'].sum() / 100:.2f}")


# %% [markdown]
# ### Emissions analysis (tariff-agnostic)

# %%
solar_energy_balance = agile_solar_df

# Merge grid intensity into solar energy balance df
solar_energy_balance = solar_energy_balance.merge(
    grid_intensity[["Settlement Date", "Settlement timestamp", "intensity (gCO2/kWh)"]],
    on=["Settlement Date", "Settlement timestamp"],
    how="left",
)

# Annual emissions
baseline_emissions_solar = (
    solar_energy_balance["Electricity use (kWh)"] * solar_energy_balance["intensity (gCO2/kWh)"]
).sum() / 1000  # kgCO2

solar_emissions = (
    solar_energy_balance["Grid draw (kWh)"] * solar_energy_balance["intensity (gCO2/kWh)"]
).sum() / 1000  # kgCO2

emissions_saving = baseline_emissions_solar - solar_emissions

print(f"Baseline annual emissions : {baseline_emissions_solar:.1f} kgCO2")
print(f"Solar annual emissions    : {solar_emissions:.1f} kgCO2")
print(
    f"Emissions saving          : {emissions_saving:.1f} kgCO2 ({emissions_saving / baseline_emissions_solar * 100:.1f}%)"
)


# %%
# Period breakdown
periods = [
    ("Off-peak (00:00-06:00)             ", off_peak_times),
    ("Standard (06:00-16:00, 19:00-00:00)", standard_times),
    ("Peak     (16:00-19:00)             ", peak_times),
]

print(f"\n=== DEMAND PROFILE (without solar) ===")
for label, times in periods:
    mask = solar_energy_balance["Settlement timestamp"].isin(times)
    val = solar_energy_balance.loc[mask, "Electricity use (kWh)"].sum()
    total = solar_energy_balance["Electricity use (kWh)"].sum()
    print(f"{label} : {val:.1f} kWh ({val / total * 100:.1f}%)")

print(f"\n=== GRID DRAW PROFILE (with solar) ===")
for label, times in periods:
    mask = solar_energy_balance["Settlement timestamp"].isin(times)
    val = solar_energy_balance.loc[mask, "Grid draw (kWh)"].sum()
    total = solar_energy_balance["Grid draw (kWh)"].sum()
    print(f"{label} : {val:.1f} kWh ({val / total * 100:.1f}%)")

print(f"\n=== SOLAR GENERATION PROFILE ===")
for label, times in periods:
    mask = solar_energy_balance["Settlement timestamp"].isin(times)
    val = solar_energy_balance.loc[mask, "Solar generation (kWh)"].sum()
    total = solar_energy_balance["Solar generation (kWh)"].sum()
    print(f"{label} : {val:.1f} kWh ({val / total * 100:.1f}%)")

print(f"\n=== SELF CONSUMPTION PROFILE ===")
for label, times in periods:
    mask = solar_energy_balance["Settlement timestamp"].isin(times)
    val = solar_energy_balance.loc[mask, "Self consumption (kWh)"].sum()
    total = solar_energy_balance["Self consumption (kWh)"].sum()
    print(f"{label} : {val:.1f} kWh ({val / total * 100:.1f}%)")

print(f"\n=== EXPORT PROFILE ===")
total_export = solar_energy_balance["Export (kWh)"].sum()
for label, times in periods:
    mask = solar_energy_balance["Settlement timestamp"].isin(times)
    val = solar_energy_balance.loc[mask, "Export (kWh)"].sum()
    print(f"{label} : {val:.1f} kWh ({val / max(total_export, 0.001) * 100:.1f}%)")

print(f"\n=== SHIFT FROM BASELINE ===")
for label, times in periods:
    mask = solar_energy_balance["Settlement timestamp"].isin(times)
    demand = solar_energy_balance.loc[mask, "Electricity use (kWh)"].sum()
    grid = solar_energy_balance.loc[mask, "Grid draw (kWh)"].sum()
    print(f"{label} : {grid - demand:+.1f} kWh")

# %% [markdown]
# ### Plotting energy balance (tariff-agnostic)

# %%
# df = agile_solar_df.copy()
# df["Settlement Date"] = pd.to_datetime(df["Settlement Date"])
# df["date_str"] = df["Settlement Date"].dt.strftime("%Y-%m-%d")

# alt.data_transformers.enable("vegafusion")

# dates_available = sorted(df["date_str"].unique().tolist())
# date_select = alt.binding_select(options=dates_available, name="Date: ")
# date_param = alt.param(name="selected_date", bind=date_select, value=dates_available[0])

# # Melt for area layers (self consumption, export, grid draw)
# area_cols = ["Self consumption (kWh)", "Export (kWh)", "Grid draw (kWh)"]
# df_area = df.melt(
#     id_vars=[
#         "date_str",
#         "Settlement Period ID",
#         "Settlement timestamp",
#         "Solar generation (kWh)",
#         "Electricity use (kWh)",
#     ],
#     value_vars=area_cols,
#     var_name="Series",
#     value_name="kWh",
# )

# color_scale = alt.Scale(
#     domain=[
#         "Self consumption (kWh)",
#         "Export (kWh)",
#         "Grid draw (kWh)",
#         "Solar generation (kWh)",
#         "Electricity use (kWh)",
#     ],
#     range=["#2ca02c", "#1f77b4", "#d62728", "#ff7f0e", "#9467bd"],
# )

# base_area = alt.Chart(df_area).transform_filter(alt.datum.date_str == date_param).add_params(date_param)
# base_line = alt.Chart(df).transform_filter(alt.datum.date_str == date_param)

# # Non-stacked areas
# area = base_area.mark_area(opacity=0.5, interpolate="step-after").encode(
#     x=alt.X("Settlement Period ID:O", title="Settlement Period"),
#     y=alt.Y("kWh:Q", title="Energy (kWh)", stack=None),
#     color=alt.Color("Series:N", scale=color_scale, title="Series"),
#     tooltip=["Settlement Period ID:O", "Settlement timestamp:N", "Series:N", "kWh:Q"],
# )

# # Solar generation line
# solar_line = base_line.mark_line(strokeWidth=2, strokeDash=[4, 2], interpolate="step-after").encode(
#     x=alt.X("Settlement Period ID:O"),
#     y=alt.Y("Solar generation (kWh):Q"),
#     color=alt.value("#ff7f0e"),
#     tooltip=["Settlement Period ID:O", "Settlement timestamp:N", "Solar generation (kWh):Q"],
# )

# # Electricity use line
# electricity_line = base_line.mark_line(strokeWidth=2, strokeDash=[4, 2], interpolate="step-after").encode(
#     x=alt.X("Settlement Period ID:O"),
#     y=alt.Y("Electricity use (kWh):Q"),
#     color=alt.value("#9467bd"),
#     tooltip=["Settlement Period ID:O", "Settlement timestamp:N", "Electricity use (kWh):Q"],
# )

# chart = (area + solar_line + electricity_line).properties(
#     width=800,
#     height=400,
#     title="Half-Hourly Solar Generation, Self Consumption, Export and Grid Draw for medium TDCV household",
# )

# chart

# %% [markdown]
# ## Plug-in solar

# %% [markdown]
# Note - no export: https://www.britishgas.co.uk/the-source/news/plug-in-solar-panels-explained.html#:~:text=Plug%E2%80%91in%20solar%20panels%20are%20designed%20mainly%20for%20using%20what%20you%20generate.%20Future%20rules%20may%20evolve%2C%20but%20initially%20the%20focus%20is%20on%20making%20the%20technology%20simple%2C%20safe%2C%20and%20accessible.

# %% [markdown]
# **Generate solar yield profiles**

# %% [markdown]
# Plug-in solar systems capped at 800W, assume same other params as conventional

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

# %%
print(f"Annual generation: {radiation_profile['generation_kWh'].sum():,.1f} kWh")

# %% [markdown]
# ### Octopus Agile tariff

# %%
agile_plugin_df = calculate_solar_household_energy_balance_no_export(
    demand_profile_df=elexon_profile_class_1,
    radiation_profile_df=radiation_profile,
    import_price_profile_df=agile_import_price,
)

# %%
print(
    f"Overall self-consumption rate: {(agile_plugin_df['Self consumption (kWh)'].sum() / agile_plugin_df['Solar generation (kWh)'].sum()) * 100:.1f}%"
)

# %%
print(
    f"Bill from import tariff switch only: £{(agile_plugin_df['Import cost without solar (p)'].sum() + (agile_standing_charge * 365)) / 100:.2f}"
)
print(
    f"Bill from import tariff switch + solar: £{(agile_plugin_df['Import cost with solar (p)'].sum() + (agile_standing_charge * 365)) / 100:.2f}"
)


# %% [markdown]
# ### Standard variable tariff for import + Fixed SEG export

# %%
static_plugin_df = calculate_solar_household_energy_balance_no_export(
    demand_profile_df=elexon_profile_class_1,
    radiation_profile_df=radiation_profile,
    import_price_profile_df=import_svt_price,
)

# %%
print(
    f"Overall self-consumption rate: {(static_plugin_df['Self consumption (kWh)'].sum() / static_plugin_df['Solar generation (kWh)'].sum()) * 100:.1f}%"
)

# %%
print(
    f"Bill from import tariff switch only: £{(static_plugin_df['Import cost without solar (p)'].sum() + (electricity_standing_charge * 365)) / 100:.2f}"
)
print(
    f"Bill from import tariff switch + solar: £{(static_plugin_df['Import cost with solar (p)'].sum() + (electricity_standing_charge * 365)) / 100:.2f}"
)


# %% [markdown]
# ### Octopus Flux

# %%
flux_plugin_df = calculate_solar_household_energy_balance_no_export(
    demand_profile_df=elexon_profile_class_1,
    radiation_profile_df=radiation_profile,
    import_price_profile_df=flux_import_price,
)

# %%
print(
    f"Overall self-consumption rate: {(flux_plugin_df['Self consumption (kWh)'].sum() / flux_plugin_df['Solar generation (kWh)'].sum()) * 100:.1f}%"
)

# %%
print(
    f"Bill from import tariff switch only: £{(flux_plugin_df['Import cost without solar (p)'].sum() + (flux_standing_charge * 365)) / 100:.2f}"
)
print(
    f"Bill from import tariff switch + solar: £{(flux_plugin_df['Import cost with solar (p)'].sum() + (flux_standing_charge * 365)) / 100:.2f}"
)


# %% [markdown]
# ### Emissions analysis (tariff-agnostic)

# %%
solar_energy_balance = agile_plugin_df

# Merge grid intensity into solar energy balance df
solar_energy_balance = solar_energy_balance.merge(
    grid_intensity[["Settlement Date", "Settlement timestamp", "intensity (gCO2/kWh)"]],
    on=["Settlement Date", "Settlement timestamp"],
    how="left",
)

# Annual emissions
baseline_emissions_solar = (
    solar_energy_balance["Electricity use (kWh)"] * solar_energy_balance["intensity (gCO2/kWh)"]
).sum() / 1000  # kgCO2

solar_emissions = (
    solar_energy_balance["Grid draw (kWh)"] * solar_energy_balance["intensity (gCO2/kWh)"]
).sum() / 1000  # kgCO2

emissions_saving = baseline_emissions_solar - solar_emissions

print(f"Baseline annual emissions : {baseline_emissions_solar:.1f} kgCO2")
print(f"Solar annual emissions    : {solar_emissions:.1f} kgCO2")
print(
    f"Emissions saving          : {emissions_saving:.1f} kgCO2 ({emissions_saving / baseline_emissions_solar * 100:.1f}%)"
)


# %%
# Period breakdown
periods = [
    ("Off-peak (00:00-06:00)             ", off_peak_times),
    ("Standard (06:00-16:00, 19:00-00:00)", standard_times),
    ("Peak     (16:00-19:00)             ", peak_times),
]

print(f"\n=== DEMAND PROFILE (without solar) ===")
for label, times in periods:
    mask = solar_energy_balance["Settlement timestamp"].isin(times)
    val = solar_energy_balance.loc[mask, "Electricity use (kWh)"].sum()
    total = solar_energy_balance["Electricity use (kWh)"].sum()
    print(f"{label} : {val:.1f} kWh ({val / total * 100:.1f}%)")

print(f"\n=== GRID DRAW PROFILE (with solar) ===")
for label, times in periods:
    mask = solar_energy_balance["Settlement timestamp"].isin(times)
    val = solar_energy_balance.loc[mask, "Grid draw (kWh)"].sum()
    total = solar_energy_balance["Grid draw (kWh)"].sum()
    print(f"{label} : {val:.1f} kWh ({val / total * 100:.1f}%)")

print(f"\n=== SOLAR GENERATION PROFILE ===")
for label, times in periods:
    mask = solar_energy_balance["Settlement timestamp"].isin(times)
    val = solar_energy_balance.loc[mask, "Solar generation (kWh)"].sum()
    total = solar_energy_balance["Solar generation (kWh)"].sum()
    print(f"{label} : {val:.1f} kWh ({val / total * 100:.1f}%)")

print(f"\n=== SELF CONSUMPTION PROFILE ===")
for label, times in periods:
    mask = solar_energy_balance["Settlement timestamp"].isin(times)
    val = solar_energy_balance.loc[mask, "Self consumption (kWh)"].sum()
    total = solar_energy_balance["Self consumption (kWh)"].sum()
    print(f"{label} : {val:.1f} kWh ({val / total * 100:.1f}%)")

print(f"\n=== SURPLUS PROFILE ===")
total_surplus = solar_energy_balance["Surplus (kWh)"].sum()
for label, times in periods:
    mask = solar_energy_balance["Settlement timestamp"].isin(times)
    val = solar_energy_balance.loc[mask, "Surplus (kWh)"].sum()
    print(f"{label} : {val:.1f} kWh ({val / max(total_surplus, 0.001) * 100:.1f}%)")
print(f"Total annual surplus curtailed : {total_surplus:.1f} kWh")

print(f"\n=== SHIFT FROM BASELINE ===")
for label, times in periods:
    mask = solar_energy_balance["Settlement timestamp"].isin(times)
    demand = solar_energy_balance.loc[mask, "Electricity use (kWh)"].sum()
    grid = solar_energy_balance.loc[mask, "Grid draw (kWh)"].sum()
    print(f"{label} : {grid - demand:+.1f} kWh")

# %%
