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

from asf_faster_further_electrification.getters import elexon_demand_profiles, octopus_tariffs
from asf_faster_further_electrification.batteries import run_linear_cost_optimisation_battery_with_export
from asf_faster_further_electrification.utils.helpers import apply_dst_adjustments

# %%
capacity = 13.5
inverter_power = 11.5

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
# Octopus Go tariff

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

# %% [markdown]
# Grid carbon intensity

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

# %%
# Find the extreme periods
print("=== LOWEST CARBON INTENSITY PERIODS ===")
print(
    grid_intensity.nsmallest(10, "intensity (gCO2/kWh)")[
        ["Settlement Date", "Settlement timestamp", "intensity (gCO2/kWh)"]
    ].to_string()
)

print("\n=== HIGHEST CARBON INTENSITY PERIODS ===")
print(
    grid_intensity.nlargest(10, "intensity (gCO2/kWh)")[
        ["Settlement Date", "Settlement timestamp", "intensity (gCO2/kWh)"]
    ].to_string()
)

# Daily min and max
daily_min = grid_intensity.groupby("Settlement Date")["intensity (gCO2/kWh)"].min()
daily_max = grid_intensity.groupby("Settlement Date")["intensity (gCO2/kWh)"].max()
daily_ratio = daily_max / daily_min

print(f"\n=== DAILY VARIATION ===")
print(f"Average daily min    : {daily_min.mean():.1f} gCO2/kWh")
print(f"Average daily max    : {daily_max.mean():.1f} gCO2/kWh")
print(f"Average daily ratio  : {daily_ratio.mean():.1f}x")
print(f"Max daily ratio      : {daily_ratio.max():.1f}x on {daily_ratio.idxmax().date()}")

# Annual extremes
print(f"\n=== ANNUAL EXTREMES ===")
print(f"Annual minimum       : {grid_intensity['intensity (gCO2/kWh)'].min():.1f} gCO2/kWh")
print(f"Annual maximum       : {grid_intensity['intensity (gCO2/kWh)'].max():.1f} gCO2/kWh")
print(
    f"Annual ratio         : {grid_intensity['intensity (gCO2/kWh)'].max() / grid_intensity['intensity (gCO2/kWh)'].min():.1f}x"
)

# %% [markdown]
# Daily dataframes

# %%
# Create daily lookups
dates = elexon_profile_class_1["Settlement Date"].unique().tolist()

daily_profile = {}  # 2025
daily_agile_import_rate = {}  # 2025 historical
daily_agile_export_rate = {}  # 2025 historical
daily_go_rate = {}  # 2025 historical
daily_grid_intensity = {}  # 2025 historical
for date in dates:
    daily_profile[date] = elexon_profile_class_1[elexon_profile_class_1["Settlement Date"] == date]
    daily_agile_import_rate[date] = agile_import_price[agile_import_price["Settlement Date"] == date]
    daily_agile_export_rate[date] = agile_export_price[agile_export_price["Settlement Date"] == date]
    daily_go_rate[date] = go_price[go_price["Settlement Date"] == date]
    daily_grid_intensity[date] = grid_intensity[grid_intensity["Settlement Date"] == date]

# %% [markdown]
# ## Fixed batteries

# %% [markdown]
# ### Octopus Go import + fixed export

# %%
# fixed export at 16 p/kWh
# Export - fixed Smart Export Guarantee
# https://www.ecotricity.co.uk/your-green-energy/smart-export-tariff
outgoing_fixed_rate = 16.0  # p/kWh

# %%
all_days = {}

soc_init = 0.5

for date in dates:
    daily_result = run_linear_cost_optimisation_battery_with_export(
        time_series=daily_profile.get(date)["Settlement timestamp"].tolist(),
        demand_profile=daily_profile.get(date)["Electricity use (kWh)"].tolist(),
        import_prices=daily_go_rate.get(date)["value_inc_vat"].tolist(),
        export_prices=[outgoing_fixed_rate] * len(daily_profile.get(date)),
        capacity=capacity,
        inverter_power=inverter_power,
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
# Battery optimised bill
battery_import_bill = sum(all_days[date]["import_cost_p"].sum() for date in dates) / 100
battery_import_bill += (go_standing_charge * 365) / 100

battery_export = sum(all_days[date]["export_cost_p"].sum() for date in dates) / 100

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
print(f"Go import - no battery: £{tariff_switch_import_bill:.2f}")
print(f"Go import + battery bill:  £{battery_import_bill:.2f}")
print(f"Fixed export + battery gain:  £{battery_export:.2f}")


# %%
total_charge = sum(all_days[date]["charge_kwh"].sum() for date in all_days)
total_export = sum(all_days[date]["grid_export_kwh"].sum() for date in all_days)
total_import = sum(all_days[date]["grid_import_kwh"].sum() for date in all_days)
total_discharge = sum(all_days[date]["discharge_kwh"].sum() for date in all_days)

print(f"Total charge    : {total_charge:.1f} kWh")
print(f"Total discharge : {total_discharge:.1f} kWh")
print(f"Total import    : {total_import:.1f} kWh")
print(f"Total export    : {total_export:.1f} kWh")

print(f"Export / charge ratio: {total_export / total_charge:.2f}")

# %%
# Check whether any SPs has battery charging and discharging at same time (not physically possible)
results = pd.concat(all_days.values())
simultaneous = results[(results["charge_kwh"] > 1e-6) & (results["discharge_kwh"] > 1e-6)]
print(f"Simultaneous charge/discharge periods: {len(simultaneous)}")

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

# %% [markdown]
# Using consumption-based accounting (only account for scope 2 emissions of the household = grid imports x average grid intensity) - export is not counted (i.e. no carbon credit for exporting because the household didn't generate the electricity, it came from the grid). But on a grid- or system-level, impact may be neutral or positive (exports at peak may be able to displace marginal sources, i.e. gas)

# %%
# Breakdown of original demand by time period
demand_by_period = pd.Series(0.0, index=avg_intensity_by_period.index)
grid_import_by_period = pd.Series(0.0, index=avg_intensity_by_period.index)
grid_export_by_period = pd.Series(0.0, index=avg_intensity_by_period.index)

for date in dates:
    day = all_days[date]
    for _, row in day.iterrows():
        ts = row["timestamp"]
        if ts in demand_by_period.index:
            demand_by_period[ts] += row["demand_kwh"]
            grid_import_by_period[ts] += row["grid_import_kwh"]
            grid_export_by_period[ts] += row["grid_export_kwh"]

# Summarise by period
off_peak_demand = demand_by_period[demand_by_period.index.isin(off_peak_times)].sum()
standard_demand = demand_by_period[demand_by_period.index.isin(standard_times)].sum()
peak_demand = demand_by_period[demand_by_period.index.isin(peak_times)].sum()

off_peak_import = grid_import_by_period[grid_import_by_period.index.isin(off_peak_times)].sum()
standard_import = grid_import_by_period[grid_import_by_period.index.isin(standard_times)].sum()
peak_import = grid_import_by_period[grid_import_by_period.index.isin(peak_times)].sum()

off_peak_export = grid_export_by_period[grid_export_by_period.index.isin(off_peak_times)].sum()
standard_export = grid_export_by_period[grid_export_by_period.index.isin(standard_times)].sum()
peak_export = grid_export_by_period[grid_export_by_period.index.isin(peak_times)].sum()

total_demand = sum(all_days[date]["demand_kwh"].sum() for date in dates)
total_export = sum(all_days[date]["grid_export_kwh"].sum() for date in dates)

print(f"=== DEMAND PROFILE (without battery) ===")
print(f"Off-peak (00:00-06:00)              : {off_peak_demand:.1f} kWh ({off_peak_demand / total_demand * 100:.1f}%)")
print(f"Standard (06:00-16:00, 19:00-00:00) : {standard_demand:.1f} kWh ({standard_demand / total_demand * 100:.1f}%)")
print(f"Peak     (16:00-19:00)              : {peak_demand:.1f} kWh ({peak_demand / total_demand * 100:.1f}%)")

print(f"\n=== GRID IMPORT PROFILE (with battery) ===")
print(f"Off-peak (00:00-06:00)              : {off_peak_import:.1f} kWh ({off_peak_import / total_import * 100:.1f}%)")
print(f"Standard (06:00-16:00, 19:00-00:00) : {standard_import:.1f} kWh ({standard_import / total_import * 100:.1f}%)")
print(f"Peak     (16:00-19:00)              : {peak_import:.1f} kWh ({peak_import / total_import * 100:.1f}%)")

print(f"\n=== GRID EXPORT PROFILE (with battery) ===")
print(f"Off-peak (00:00-06:00)              : {off_peak_export:.1f} kWh ({off_peak_export / total_export * 100:.1f}%)")
print(f"Standard (06:00-16:00, 19:00-00:00) : {standard_export:.1f} kWh ({standard_export / total_export * 100:.1f}%)")
print(f"Peak     (16:00-19:00)              : {peak_export:.1f} kWh ({peak_export / total_export * 100:.1f}%)")

print(f"\n=== NET GRID FLOW (import - export) ===")
print(f"Off-peak (00:00-06:00)              : {off_peak_import - off_peak_export:+.1f} kWh")
print(f"Standard (06:00-16:00, 19:00-00:00) : {standard_import - standard_export:+.1f} kWh")
print(f"Peak     (16:00-19:00)              : {peak_import - peak_export:+.1f} kWh")

print(f"\n=== SHIFT FROM BASELINE ===")
print(f"Off-peak (00:00-06:00)              : {(off_peak_import - off_peak_export) - off_peak_demand:+.1f} kWh")
print(f"Standard (06:00-16:00, 19:00-00:00) : {(standard_import - standard_export) - standard_demand:+.1f} kWh")
print(f"Peak     (16:00-19:00)              : {(peak_import - peak_export) - peak_demand:+.1f} kWh")

# %% [markdown]
# ### Octopus Agile import + export

# %%
all_days = {}

soc_init = 0.5

for date in dates:
    daily_result = run_linear_cost_optimisation_battery_with_export(
        time_series=daily_profile.get(date)["Settlement timestamp"].tolist(),
        demand_profile=daily_profile.get(date)["Electricity use (kWh)"].tolist(),
        import_prices=daily_agile_import_rate.get(date)["value_inc_vat"].tolist(),
        export_prices=daily_agile_export_rate.get(date)["value_inc_vat"].tolist(),
        capacity=capacity,
        inverter_power=inverter_power,
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
# Battery optimised bill
battery_import_bill = sum(all_days[date]["import_cost_p"].sum() for date in dates) / 100
battery_import_bill += (agile_standing_charge * 365) / 100

battery_export = sum(all_days[date]["export_cost_p"].sum() for date in dates) / 100

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
print(f"Agile import - no battery: £{tariff_switch_import_bill:.2f}")
print(f"Agile import + battery bill:  £{battery_import_bill:.2f}")
print(f"Agile export + battery gain:  £{battery_export:.2f}")


# %%
total_charge = sum(all_days[date]["charge_kwh"].sum() for date in all_days)
total_export = sum(all_days[date]["grid_export_kwh"].sum() for date in all_days)
total_import = sum(all_days[date]["grid_import_kwh"].sum() for date in all_days)
total_discharge = sum(all_days[date]["discharge_kwh"].sum() for date in all_days)

print(f"Total charge    : {total_charge:.1f} kWh")
print(f"Total discharge : {total_discharge:.1f} kWh")
print(f"Total import    : {total_import:.1f} kWh")
print(f"Total export    : {total_export:.1f} kWh")

print(f"Export / charge ratio: {total_export / total_charge:.2f}")

# %%
# Check whether any SPs has battery charging and discharging at same time (not physically possible)
results = pd.concat(all_days.values())
simultaneous = results[(results["charge_kwh"] > 1e-6) & (results["discharge_kwh"] > 1e-6)]
print(f"Simultaneous charge/discharge periods: {len(simultaneous)}")

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
# Breakdown of original demand by time period
demand_by_period = pd.Series(0.0, index=avg_intensity_by_period.index)
grid_import_by_period = pd.Series(0.0, index=avg_intensity_by_period.index)
grid_export_by_period = pd.Series(0.0, index=avg_intensity_by_period.index)

for date in dates:
    day = all_days[date]
    for _, row in day.iterrows():
        ts = row["timestamp"]
        if ts in demand_by_period.index:
            demand_by_period[ts] += row["demand_kwh"]
            grid_import_by_period[ts] += row["grid_import_kwh"]
            grid_export_by_period[ts] += row["grid_export_kwh"]

# Summarise by period
off_peak_demand = demand_by_period[demand_by_period.index.isin(off_peak_times)].sum()
standard_demand = demand_by_period[demand_by_period.index.isin(standard_times)].sum()
peak_demand = demand_by_period[demand_by_period.index.isin(peak_times)].sum()

off_peak_import = grid_import_by_period[grid_import_by_period.index.isin(off_peak_times)].sum()
standard_import = grid_import_by_period[grid_import_by_period.index.isin(standard_times)].sum()
peak_import = grid_import_by_period[grid_import_by_period.index.isin(peak_times)].sum()

off_peak_export = grid_export_by_period[grid_export_by_period.index.isin(off_peak_times)].sum()
standard_export = grid_export_by_period[grid_export_by_period.index.isin(standard_times)].sum()
peak_export = grid_export_by_period[grid_export_by_period.index.isin(peak_times)].sum()

total_demand = sum(all_days[date]["demand_kwh"].sum() for date in dates)
total_export = sum(all_days[date]["grid_export_kwh"].sum() for date in dates)

print(f"=== DEMAND PROFILE (without battery) ===")
print(f"Off-peak (00:00-06:00)              : {off_peak_demand:.1f} kWh ({off_peak_demand / total_demand * 100:.1f}%)")
print(f"Standard (06:00-16:00, 19:00-00:00) : {standard_demand:.1f} kWh ({standard_demand / total_demand * 100:.1f}%)")
print(f"Peak     (16:00-19:00)              : {peak_demand:.1f} kWh ({peak_demand / total_demand * 100:.1f}%)")

print(f"\n=== GRID IMPORT PROFILE (with battery) ===")
print(f"Off-peak (00:00-06:00)              : {off_peak_import:.1f} kWh ({off_peak_import / total_import * 100:.1f}%)")
print(f"Standard (06:00-16:00, 19:00-00:00) : {standard_import:.1f} kWh ({standard_import / total_import * 100:.1f}%)")
print(f"Peak     (16:00-19:00)              : {peak_import:.1f} kWh ({peak_import / total_import * 100:.1f}%)")

print(f"\n=== GRID EXPORT PROFILE (with battery) ===")
print(f"Off-peak (00:00-06:00)              : {off_peak_export:.1f} kWh ({off_peak_export / total_export * 100:.1f}%)")
print(f"Standard (06:00-16:00, 19:00-00:00) : {standard_export:.1f} kWh ({standard_export / total_export * 100:.1f}%)")
print(f"Peak     (16:00-19:00)              : {peak_export:.1f} kWh ({peak_export / total_export * 100:.1f}%)")

print(f"\n=== NET GRID FLOW (import - export) ===")
print(f"Off-peak (00:00-06:00)              : {off_peak_import - off_peak_export:+.1f} kWh")
print(f"Standard (06:00-16:00, 19:00-00:00) : {standard_import - standard_export:+.1f} kWh")
print(f"Peak     (16:00-19:00)              : {peak_import - peak_export:+.1f} kWh")

print(f"\n=== SHIFT FROM BASELINE ===")
print(f"Off-peak (00:00-06:00)              : {(off_peak_import - off_peak_export) - off_peak_demand:+.1f} kWh")
print(f"Standard (06:00-16:00, 19:00-00:00) : {(standard_import - standard_export) - standard_demand:+.1f} kWh")
print(f"Peak     (16:00-19:00)              : {(peak_import - peak_export) - peak_demand:+.1f} kWh")
