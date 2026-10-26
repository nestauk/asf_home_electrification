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

from asf_faster_further_electrification.batteries import run_linear_cost_optimisation_battery
from asf_faster_further_electrification.getters import elexon_demand_profiles, carbon_intensity, octopus_tariffs
from asf_faster_further_electrification.utils import batteries_utils

# %% [markdown]
# ### **Modelling over a whole year**

# %% [markdown]
# ### Elexon demand profile class 1

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


# %%
peak_coeff = elexon_profile_class_1[elexon_profile_class_1["Settlement timestamp"].between("16:00", "19:30")][
    "Default Period Profile Class Coefficient"
].sum()

total_coeff = elexon_profile_class_1["Default Period Profile Class Coefficient"].sum()

print(f"Peak (16:00-20:00): {peak_coeff / total_coeff * 100:.1f}%")

# %%
# Total demand per day
daily_total = elexon_profile_class_1.groupby("Settlement Date")["Electricity use (kWh)"].sum()

print(f"Average daily demand : {daily_total.mean():.3f} kWh")
print(f"Min daily demand     : {daily_total.min():.3f} kWh")
print(f"Max daily demand     : {daily_total.max():.3f} kWh")

# %%
# Evening and early morning demand (16:00-00:00 and 00:00-08:00)
evening_morning_times = (
    [f"{h:02d}:{m:02d}" for h in range(16, 24) for m in [0, 30]]  # 16:00-23:30
    + [f"{h:02d}:{m:02d}" for h in range(0, 8) for m in [0, 30]]  # 00:00-07:30
)

# Average daily demand during these periods
avg_daily_profile = elexon_profile_class_1.groupby("Settlement timestamp")["Electricity use (kWh)"].mean()

evening_morning_demand = avg_daily_profile[avg_daily_profile.index.isin(evening_morning_times)].sum()
total_daily_demand = avg_daily_profile.sum()
daytime_demand = total_daily_demand - evening_morning_demand

print(f"Average daily demand total          : {total_daily_demand:.2f} kWh")
print(f"Evening + early morning (16:00-08:00): {evening_morning_demand:.2f} kWh")
print(f"Daytime (08:00-16:00)               : {daytime_demand:.2f} kWh")
print(f"Evening/morning as % of daily       : {evening_morning_demand / total_daily_demand * 100:.1f}%")
print(f"\nSuggested battery size              : {evening_morning_demand:.1f} kWh")

# %% [markdown]
# **Calculate baseline bill**

# %%
# April to June 2026 price cap
electricity_unit_rate = 24.6744  # p/kWh
electricity_standing_charge = 57.2061  # p/day

baseline_bill = ((annual_electricity_use * electricity_unit_rate) + (electricity_standing_charge * 365)) / 100  # £/year
baseline_bill

# %%
required_saving = 5_100 / 12
required_bill = baseline_bill - required_saving


# %% [markdown]
# **Get tariff prices**

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
# Octopus Go cheap rate: 00:30 – 05:30 (settlement periods 2-11)
off_peak_mask = go_price["Settlement Period ID"].between(2, 11)
peak_mask = ~off_peak_mask

avg_off_peak = go_price.loc[off_peak_mask, "value_inc_vat"].mean()
avg_peak = go_price.loc[peak_mask, "value_inc_vat"].mean()

print(f"Average off-peak price (00:30–05:30): {avg_off_peak:.2f} p/kWh")
print(f"Average peak price    (05:30–00:30): {avg_peak:.2f} p/kWh")

# %%
df = octopus_tariffs.get_go_standing_charge(region="E", start_date="2025-01-01", end_date="2026-01-01")
df["valid_from"] = pd.to_datetime(df["valid_from"])
df["valid_to"] = pd.to_datetime(df["valid_to"])
df["days"] = (df["valid_to"] - df["valid_from"]).dt.total_seconds() / 86400
go_standing_charge = (df["value_inc_vat"] * df["days"]).sum() / df["days"].sum()

# %% [markdown]
# Octopus Agile

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

# %%
agile_standing_charge_df = octopus_tariffs.get_agile_standing_charge(
    region="E", start_date="2025-01-01", end_date="2026-01-01"
)
agile_standing_charge = agile_standing_charge_df["value_inc_vat"][0]  # p/day

# %%
# Convert UTC to local time
agile_import_price["valid_from_local"] = agile_import_price["valid_from"].dt.tz_convert("Europe/London")

# Derive Settlement Date and timestamp from local time
agile_import_price["Settlement Date"] = pd.to_datetime(agile_import_price["valid_from_local"].dt.date)
agile_import_price["Settlement timestamp"] = agile_import_price["valid_from_local"].dt.strftime("%H:%M")

# Settlement Period ID from local time
agile_import_price["Settlement Period ID"] = (
    pd.to_timedelta(agile_import_price["Settlement timestamp"] + ":00").dt.total_seconds() / 1800
).astype(int) + 1

agile_price = agile_import_price

# %%
# Agile peak: 16:00 – 19:00 (settlement periods 33-38)
# Agile off-peak: 00:00 – 06:00 (settlement periods 1-12) — cheapest overnight window
# Standard: everything else

peak_mask = agile_price["Settlement Period ID"].between(33, 38)
off_peak_mask = agile_price["Settlement Period ID"].between(1, 12)
standard_mask = ~peak_mask & ~off_peak_mask

avg_peak = agile_price.loc[peak_mask, "value_inc_vat"].mean()
avg_off_peak = agile_price.loc[off_peak_mask, "value_inc_vat"].mean()
avg_standard = agile_price.loc[standard_mask, "value_inc_vat"].mean()
avg_overall = agile_price["value_inc_vat"].mean()

print(f"Average peak     (16:00–19:00): {avg_peak:.2f} p/kWh")
print(f"Average off-peak (00:00–06:00): {avg_off_peak:.2f} p/kWh")
print(f"Average standard (rest of day): {avg_standard:.2f} p/kWh")
print(f"Average overall               : {avg_overall:.2f} p/kWh")

# %% [markdown]
# Grid carbon intensity

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

# Verify
by_date = grid_intensity.groupby("Settlement Date").size()
print(f"30 Mar: {by_date.get(pd.Timestamp('2025-03-30'), 'missing')}")
print(f"26 Oct: {by_date.get(pd.Timestamp('2025-10-26'), 'missing')}")
print(f"Unusual days: {(by_date != 48).sum()}")

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
print(f"Standard (06:00-16:00, 19:00-23:30)  : {standard_intensity:.1f} gCO2/kWh")
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

# %%
# Average grid intensity by settlement timestamp across 2025
avg_intensity_by_period = grid_intensity.groupby("Settlement timestamp")["intensity (gCO2/kWh)"].mean()

# Find 3-hour window with lowest average carbon intensity
window_hours = 3
window_periods = window_hours * 2  # 6 half-hourly periods

intensities = avg_intensity_by_period.values
timestamps = avg_intensity_by_period.index.tolist()

results = []
for i in range(48):
    indices = [(i + j) % 48 for j in range(window_periods)]
    window_ts = [timestamps[idx] for idx in indices]
    avg = sum(intensities[idx] for idx in indices) / window_periods
    results.append(
        {
            "start": timestamps[i],
            "end": timestamps[(i + window_periods) % 48],
            "timestamps": window_ts,
            "avg_intensity": round(avg, 1),
        }
    )

results_df = pd.DataFrame(results).sort_values("avg_intensity")

print("=== TOP 5 CLEANEST 3-HOUR WINDOWS ===")
print(results_df[["start", "end", "avg_intensity"]].head(5).to_string(index=False))

# %% [markdown]
# Harmonising demand, price and carbon intensity profiles

# %%
# Create daily lookups
dates = elexon_profile_class_1["Settlement Date"].unique().tolist()

daily_profile = {}  # 2025
daily_go_rate = {}  # 2025 historical
daily_agile_rate = {}  # 2025 historical
daily_grid_intensity = {}  # 2025 historical
for date in dates:
    daily_profile[date] = elexon_profile_class_1[elexon_profile_class_1["Settlement Date"] == date]
    daily_go_rate[date] = go_price[go_price["Settlement Date"] == date]
    daily_agile_rate[date] = agile_price[agile_price["Settlement Date"] == date]
    daily_grid_intensity[date] = grid_intensity[grid_intensity["Settlement Date"] == date]

# %% [markdown]
# ## Conventional installed battery

# %%
capacity = 10.0
inverter_power = 5.0

# %% [markdown]
# ### Elexon Profile 1 - Every day in 2025 - Two-rate tariff - Perfect foresight day-ahead optimisation battery model

# %%
all_days = {}

soc_init = 0.5

for date in dates:
    daily_result = run_linear_cost_optimisation_battery(
        time_series=daily_profile.get(date)["Settlement timestamp"].tolist(),
        demand_profile=daily_profile.get(date)["Electricity use (kWh)"].tolist(),
        prices=daily_go_rate.get(date)["value_inc_vat"].tolist(),
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
battery_annual = sum(all_days[date]["cost_p"].sum() for date in dates) / 100
battery_annual += (go_standing_charge * 365) / 100

# No battery on Go
no_battery_annual = (
    sum(
        sum(
            d * p
            for d, p in zip(
                daily_profile[date]["Electricity use (kWh)"].tolist(), daily_go_rate[date]["value_inc_vat"].tolist()
            )
        )
        for date in dates
    )
    / 100
)
no_battery_annual += (go_standing_charge * 365) / 100


print(f"Baseline SVT: £{baseline_bill:.2f}")
print(f"Go no battery: £{no_battery_annual:.2f}")
print(f"Go + battery:  £{battery_annual:.2f}")

# %%
total_charge = sum(all_days[date]["charge_kwh"].sum() for date in all_days)
total_import = sum(all_days[date]["grid_kwh"].sum() for date in all_days)
total_discharge = sum(all_days[date]["discharge_kwh"].sum() for date in all_days)

print(f"Total charge    : {total_charge:.1f} kWh")
print(f"Total discharge : {total_discharge:.1f} kWh")
print(f"Total import    : {total_import:.1f} kWh")

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
        (all_days[date]["grid_kwh"] * daily_grid_intensity[date]["intensity (gCO2/kWh)"].values).sum() for date in dates
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
print(f"Avg peak vs off-peak intensity diff  : {peak_intensity - off_peak_intensity:.1f} gCO2/kWh")

# %%
# Breakdown of original demand by time period
demand_by_period = pd.Series(0.0, index=avg_intensity_by_period.index)
grid_by_period = pd.Series(0.0, index=avg_intensity_by_period.index)

for date in dates:
    day = all_days[date]
    for _, row in day.iterrows():
        ts = row["timestamp"]
        if ts in demand_by_period.index:
            demand_by_period[ts] += row["demand_kwh"]
            grid_by_period[ts] += row["grid_kwh"]

# Summarise by period
off_peak_demand = demand_by_period[demand_by_period.index.isin(off_peak_times)].sum()
standard_demand = demand_by_period[demand_by_period.index.isin(standard_times)].sum()
peak_demand = demand_by_period[demand_by_period.index.isin(peak_times)].sum()

off_peak_grid = grid_by_period[grid_by_period.index.isin(off_peak_times)].sum()
standard_grid = grid_by_period[grid_by_period.index.isin(standard_times)].sum()
peak_grid = grid_by_period[grid_by_period.index.isin(peak_times)].sum()

total_demand = sum(all_days[date]["demand_kwh"].sum() for date in dates)

print(f"=== DEMAND PROFILE (without battery) ===")
print(f"Off-peak (00:00-06:00) : {off_peak_demand:.1f} kWh ({off_peak_demand / total_demand * 100:.1f}%)")
print(f"Standard (06:00-16:00, 19:00-00:00) : {standard_demand:.1f} kWh ({standard_demand / total_demand * 100:.1f}%)")
print(f"Peak     (16:00-19:00) : {peak_demand:.1f} kWh ({peak_demand / total_demand * 100:.1f}%)")

print(f"\n=== GRID DRAW PROFILE (with battery) ===")
print(f"Off-peak (00:00-06:00) : {off_peak_grid:.1f} kWh ({off_peak_grid / total_import * 100:.1f}%)")
print(f"Standard (06:00-16:00, 19:00-00:00) : {standard_grid:.1f} kWh ({standard_grid / total_import * 100:.1f}%)")
print(f"Peak     (16:00-19:00) : {peak_grid:.1f} kWh ({peak_grid / total_import * 100:.1f}%)")

print(f"\n=== SHIFT FROM BATTERY ===")
print(f"Off-peak change        : {off_peak_grid - off_peak_demand:+.1f} kWh (battery charging adds to off-peak import)")
print(f"Standard change        : {standard_grid - standard_demand:+.1f} kWh")
print(f"Peak change            : {peak_grid - peak_demand:+.1f} kWh (battery discharge reduces peak import)")

# %%
off_peak_extra = off_peak_grid - off_peak_demand
standard_change = standard_grid - standard_demand
peak_change = peak_grid - peak_demand

# Percentage eliminated
peak_eliminated = abs(peak_change) / peak_demand * 100
standard_eliminated = abs(standard_change) / standard_demand * 100
off_peak_extra_pct = off_peak_extra / off_peak_demand * 100

print(f"Peak eliminated              : {peak_eliminated:.1f}%  ({abs(peak_change):.1f} / {peak_demand:.1f} kWh)")
print(
    f"Standard eliminated          : {standard_eliminated:.1f}% ({abs(standard_change):.1f} / {standard_demand:.1f} kWh)"
)
print(
    f"Off-peak extra import        : +{off_peak_extra:.1f} kWh (+{off_peak_extra_pct:.1f}% of baseline off-peak demand)"
)

# %%
# batteries_utils.create_battery_modelling_results_chart(
#     profile_label="Medium TDCV, Elexon Profile Class 1",
#     baseline_bill=baseline_bill,
#     new_bill=battery_annual,
#     all_days=all_days,
#     price_label="Octopus Go tariff",
# )

# %%
# Classify battery behaviour by settlement period across the year
behaviour_summary = []

for date in dates:
    day = all_days[date]
    for _, row in day.iterrows():
        charge = row["charge_kwh"]
        discharge = row["discharge_kwh"]
        soc = row["soc_kwh"]
        price = row["price_p_per_kwh"]
        demand = row["demand_kwh"]
        grid = row["grid_kwh"]
        ts = row["timestamp"]

        # Classify behaviour
        if charge > 0.001:
            behaviour = "charging"
        elif discharge > 0.001:
            if grid < 0.001:
                behaviour = "fully offsetting demand"  # no grid import needed
            elif discharge >= demand * 0.9:
                behaviour = "mostly offsetting demand"
            else:
                behaviour = "partially offsetting demand"
        else:
            behaviour = "idle"

        behaviour_summary.append(
            {
                "Settlement Date": date,
                "timestamp": ts,
                "charge_kwh": charge,
                "discharge_kwh": discharge,
                "soc_kwh": soc,
                "price_p_per_kwh": price,
                "demand_kwh": demand,
                "grid_kwh": grid,
                "behaviour": behaviour,
            }
        )

behaviour_df = pd.DataFrame(behaviour_summary)

# Summary statistics
print("=== BATTERY BEHAVIOUR SUMMARY ===")
behaviour_counts = behaviour_df["behaviour"].value_counts()
total_periods = len(behaviour_df)
for behaviour, count in behaviour_counts.items():
    print(f"{behaviour:35s} : {count:6d} periods ({count / total_periods * 100:.1f}%)")

print(f"\n=== ENERGY BY BEHAVIOUR ===")
for behaviour in behaviour_df["behaviour"].unique():
    mask = behaviour_df["behaviour"] == behaviour
    charge = behaviour_df.loc[mask, "charge_kwh"].sum()
    discharge = behaviour_df.loc[mask, "discharge_kwh"].sum()
    print(f"{behaviour:35s} : charge {charge:.1f} kWh, discharge {discharge:.1f} kWh")

print(f"\n=== AVERAGE PRICE BY BEHAVIOUR ===")
avg_price = behaviour_df.groupby("behaviour")["price_p_per_kwh"].mean()
for behaviour, price in avg_price.items():
    print(f"{behaviour:35s} : {price:.2f} p/kWh")

# %% [markdown]
# Free window

# %%
# Use the best overnight 3-hour window
free_window_times = ["01:30", "02:00", "02:30", "03:00", "03:30", "04:00"]  # 01:30-04:30

# Modify Go prices — set free window to 0p
daily_go_rate_free = {}
for date in dates:
    day_prices = daily_go_rate[date].copy()
    mask = day_prices["Settlement timestamp"].isin(free_window_times)
    day_prices.loc[mask, "value_inc_vat"] = 0.0
    daily_go_rate_free[date] = day_prices

# %%
all_days_free = {}
soc_init = 0.5

for date in dates:
    daily_result = run_linear_cost_optimisation_battery(
        time_series=daily_profile.get(date)["Settlement timestamp"].tolist(),
        demand_profile=daily_profile.get(date)["Electricity use (kWh)"].tolist(),
        prices=daily_go_rate_free.get(date)["value_inc_vat"].tolist(),
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
    all_days_free[date] = daily_result

# %%
# Battery optimised bill
battery_annual = sum(all_days_free[date]["cost_p"].sum() for date in dates) / 100
battery_annual += (go_standing_charge * 365) / 100

# No battery on Go with free window
no_battery_annual = (
    sum(
        sum(
            d * p
            for d, p in zip(
                daily_profile[date]["Electricity use (kWh)"].tolist(),
                daily_go_rate_free[date]["value_inc_vat"].tolist(),
            )
        )
        for date in dates
    )
    / 100
)
no_battery_annual += (go_standing_charge * 365) / 100


print(f"Baseline SVT: £{baseline_bill:.2f}")
print(f"Go no battery: £{no_battery_annual:.2f}")
print(f"Go + battery:  £{battery_annual:.2f}")

# %%
# batteries_utils.create_battery_modelling_results_chart(
#     profile_label="Medium TDCV, Elexon Profile Class 1",
#     baseline_bill=baseline_bill,
#     new_bill=battery_annual,
#     all_days=all_days_free,
#     price_label="Octopus Go tariff - free 3h window",
# )

# %%
# Compare grid import by time period — base vs free window
periods = [
    ("00:00-01:30", [f"{h:02d}:{m:02d}" for h in range(0, 2) for m in [0, 30]] + ["01:00"]),
    ("01:30-04:30 (free)", free_window_times),
    ("04:30-05:30", ["04:30", "05:00", "05:30"]),
    ("05:30-16:00", [f"{h:02d}:{m:02d}" for h in range(6, 16) for m in [0, 30]] + ["05:30"]),
    ("16:00-19:00", [f"{h:02d}:{m:02d}" for h in range(16, 19) for m in [0, 30]]),
    ("19:00-00:00", [f"{h:02d}:{m:02d}" for h in range(19, 24) for m in [0, 30]]),
]

print(f"{'Period':25} {'Base import':12} {'Free import':12} {'Diff':10} {'Avg price base':15}")
print("-" * 75)
for label, times in periods:
    base = sum(all_days[date][all_days[date]["timestamp"].isin(times)]["grid_kwh"].sum() for date in dates)
    free = sum(all_days_free[date][all_days_free[date]["timestamp"].isin(times)]["grid_kwh"].sum() for date in dates)

    # Average price during these periods (base)
    base_cost = sum(all_days[date][all_days[date]["timestamp"].isin(times)]["cost_p"].sum() for date in dates)
    avg_price = base_cost / max(base, 0.001)

    print(f"{label:25} {base:12.1f} {free:12.1f} {free - base:+10.1f} {avg_price:15.2f}p")

# %% [markdown]
# ---

# %% [markdown]
# ### Elexon Profile 1 - Every day in 2025 - Dynamic tariff - Perfect foresight day-ahead optimisation battery model

# %%
all_days = {}

soc_init = 0.5  # fraction

for date in dates:
    daily_result = run_linear_cost_optimisation_battery(
        time_series=daily_profile.get(date)["Settlement timestamp"].tolist(),
        demand_profile=daily_profile.get(date)["Electricity use (kWh)"].tolist(),
        prices=daily_agile_rate.get(date)["value_inc_vat"].tolist(),
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

    soc_init = daily_result["soc_kwh"].iloc[-1] / capacity  # fraction

    all_days[date] = daily_result

# %%
# Battery optimised bill
battery_annual = sum(all_days[date]["cost_p"].sum() for date in dates) / 100
battery_annual += (agile_standing_charge * 365) / 100

# No battery on Agile
no_battery_annual = (
    sum(
        sum(
            d * p
            for d, p in zip(
                daily_profile[date]["Electricity use (kWh)"].tolist(), daily_agile_rate[date]["value_inc_vat"].tolist()
            )
        )
        for date in dates
    )
    / 100
)
no_battery_annual += (agile_standing_charge * 365) / 100


print(f"Baseline SVT: £{baseline_bill:.2f}")
print(f"Agile no battery: £{no_battery_annual:.2f}")
print(f"Agile + battery:  £{battery_annual:.2f}")

# %%
total_charge = sum(all_days[date]["charge_kwh"].sum() for date in all_days)
total_import = sum(all_days[date]["grid_kwh"].sum() for date in all_days)
total_discharge = sum(all_days[date]["discharge_kwh"].sum() for date in all_days)

print(f"Total charge    : {total_charge:.1f} kWh")
print(f"Total discharge : {total_discharge:.1f} kWh")
print(f"Total import    : {total_import:.1f} kWh")

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
        (all_days[date]["grid_kwh"] * daily_grid_intensity[date]["intensity (gCO2/kWh)"].values).sum() for date in dates
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
print(f"Avg peak vs off-peak intensity diff  : {peak_intensity - off_peak_intensity:.1f} gCO2/kWh")

# %%
# Breakdown of original demand by time period
demand_by_period = pd.Series(0.0, index=avg_intensity_by_period.index)
grid_by_period = pd.Series(0.0, index=avg_intensity_by_period.index)

for date in dates:
    day = all_days[date]
    for _, row in day.iterrows():
        ts = row["timestamp"]
        if ts in demand_by_period.index:
            demand_by_period[ts] += row["demand_kwh"]
            grid_by_period[ts] += row["grid_kwh"]

# Summarise by period
off_peak_demand = demand_by_period[demand_by_period.index.isin(off_peak_times)].sum()
standard_demand = demand_by_period[demand_by_period.index.isin(standard_times)].sum()
peak_demand = demand_by_period[demand_by_period.index.isin(peak_times)].sum()

off_peak_grid = grid_by_period[grid_by_period.index.isin(off_peak_times)].sum()
standard_grid = grid_by_period[grid_by_period.index.isin(standard_times)].sum()
peak_grid = grid_by_period[grid_by_period.index.isin(peak_times)].sum()

total_demand = sum(all_days[date]["demand_kwh"].sum() for date in dates)

print(f"=== DEMAND PROFILE (without battery) ===")
print(f"Off-peak (00:00-06:00) : {off_peak_demand:.1f} kWh ({off_peak_demand / total_demand * 100:.1f}%)")
print(f"Standard (06:00-16:00, 19:00-00:00) : {standard_demand:.1f} kWh ({standard_demand / total_demand * 100:.1f}%)")
print(f"Peak     (16:00-19:00) : {peak_demand:.1f} kWh ({peak_demand / total_demand * 100:.1f}%)")

print(f"\n=== GRID DRAW PROFILE (with battery) ===")
print(f"Off-peak (00:00-06:00) : {off_peak_grid:.1f} kWh ({off_peak_grid / total_import * 100:.1f}%)")
print(f"Standard (06:00-16:00, 19:00-00:00) : {standard_grid:.1f} kWh ({standard_grid / total_import * 100:.1f}%)")
print(f"Peak     (16:00-19:00) : {peak_grid:.1f} kWh ({peak_grid / total_import * 100:.1f}%)")

print(f"\n=== SHIFT FROM BATTERY ===")
print(f"Off-peak change        : {off_peak_grid - off_peak_demand:+.1f} kWh (battery charging adds to off-peak import)")
print(f"Standard change        : {standard_grid - standard_demand:+.1f} kWh")
print(f"Peak change            : {peak_grid - peak_demand:+.1f} kWh (battery discharge reduces peak import)")

# %%
# batteries_utils.create_battery_modelling_results_chart(
#     profile_label="Medium TDCV, Elexon Profile Class 1",
#     baseline_bill=baseline_bill,
#     new_bill=battery_annual,
#     all_days=all_days,
#     price_label="Octopus Agile tariff",
# )

# %%
# Classify battery behaviour by settlement period across the year
behaviour_summary = []

for date in dates:
    day = all_days[date]
    for _, row in day.iterrows():
        charge = row["charge_kwh"]
        discharge = row["discharge_kwh"]
        soc = row["soc_kwh"]
        price = row["price_p_per_kwh"]
        demand = row["demand_kwh"]
        grid = row["grid_kwh"]
        ts = row["timestamp"]

        # Classify behaviour
        if charge > 0.001:
            behaviour = "charging"
        elif discharge > 0.001:
            if grid < 0.001:
                behaviour = "fully offsetting demand"  # no grid import needed
            elif discharge >= demand * 0.9:
                behaviour = "mostly offsetting demand"
            else:
                behaviour = "partially offsetting demand"
        else:
            behaviour = "idle"

        behaviour_summary.append(
            {
                "Settlement Date": date,
                "timestamp": ts,
                "charge_kwh": charge,
                "discharge_kwh": discharge,
                "soc_kwh": soc,
                "price_p_per_kwh": price,
                "demand_kwh": demand,
                "grid_kwh": grid,
                "behaviour": behaviour,
            }
        )

behaviour_df = pd.DataFrame(behaviour_summary)

# Summary statistics
print("=== BATTERY BEHAVIOUR SUMMARY ===")
behaviour_counts = behaviour_df["behaviour"].value_counts()
total_periods = len(behaviour_df)
for behaviour, count in behaviour_counts.items():
    print(f"{behaviour:35s} : {count:6d} periods ({count / total_periods * 100:.1f}%)")

print(f"\n=== ENERGY BY BEHAVIOUR ===")
for behaviour in behaviour_df["behaviour"].unique():
    mask = behaviour_df["behaviour"] == behaviour
    charge = behaviour_df.loc[mask, "charge_kwh"].sum()
    discharge = behaviour_df.loc[mask, "discharge_kwh"].sum()
    print(f"{behaviour:35s} : charge {charge:.1f} kWh, discharge {discharge:.1f} kWh")

print(f"\n=== AVERAGE PRICE BY BEHAVIOUR ===")
avg_price = behaviour_df.groupby("behaviour")["price_p_per_kwh"].mean()
for behaviour, price in avg_price.items():
    print(f"{behaviour:35s} : {price:.2f} p/kWh")

# %%
# From your shift from battery results
off_peak_extra = off_peak_grid - off_peak_demand
standard_change = standard_grid - standard_demand
peak_change = peak_grid - peak_demand

# Percentage eliminated
peak_eliminated = abs(peak_change) / peak_demand * 100
standard_eliminated = abs(standard_change) / standard_demand * 100
off_peak_extra_pct = off_peak_extra / off_peak_demand * 100

print(f"Peak eliminated              : {peak_eliminated:.1f}%  ({abs(peak_change):.1f} / {peak_demand:.1f} kWh)")
print(
    f"Standard eliminated          : {standard_eliminated:.1f}% ({abs(standard_change):.1f} / {standard_demand:.1f} kWh)"
)
print(
    f"Off-peak extra import        : +{off_peak_extra:.1f} kWh (+{off_peak_extra_pct:.1f}% of baseline off-peak demand)"
)

# %% [markdown]
# Free window

# %%
# Use the best overnight 3-hour window
free_window_times = ["01:30", "02:00", "02:30", "03:00", "03:30", "04:00"]  # 01:30-04:30

# Modify Agile prices — set free window to 0p (override any existing price including negatives)
daily_agile_rate_free = {}
for date in dates:
    day_prices = daily_agile_rate[date].copy()
    mask = day_prices["Settlement timestamp"].isin(free_window_times)
    day_prices.loc[mask, "value_inc_vat"] = 0.0
    daily_agile_rate_free[date] = day_prices


# %%
avg_agile_free_window = agile_import_price[agile_import_price["Settlement timestamp"].isin(free_window_times)][
    "value_inc_vat"
].mean()

print(f"Average Agile price during free window : {avg_agile_free_window:.2f} p/kWh")

# %%
all_days_free = {}
soc_init = 0.5

for date in dates:
    daily_result = run_linear_cost_optimisation_battery(
        time_series=daily_profile.get(date)["Settlement timestamp"].tolist(),
        demand_profile=daily_profile.get(date)["Electricity use (kWh)"].tolist(),
        prices=daily_agile_rate_free.get(date)["value_inc_vat"].tolist(),
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
    all_days_free[date] = daily_result

# %%
# Battery optimised bill
battery_annual = sum(all_days_free[date]["cost_p"].sum() for date in dates) / 100
battery_annual += (agile_standing_charge * 365) / 100

# No battery on Agile
no_battery_annual = (
    sum(
        sum(
            d * p
            for d, p in zip(
                daily_profile[date]["Electricity use (kWh)"].tolist(),
                daily_agile_rate_free[date]["value_inc_vat"].tolist(),
            )
        )
        for date in dates
    )
    / 100
)
no_battery_annual += (agile_standing_charge * 365) / 100


print(f"Baseline SVT: £{baseline_bill:.2f}")
print(f"Agile no battery: £{no_battery_annual:.2f}")
print(f"Agile + battery:  £{battery_annual:.2f}")

# %%
# Compare grid import by time period — base vs free window
periods = [
    ("00:00-01:30", [f"{h:02d}:{m:02d}" for h in range(0, 2) for m in [0, 30]] + ["01:00"]),
    ("01:30-04:30 (free)", free_window_times),
    ("04:30-05:30", ["04:30", "05:00", "05:30"]),
    ("05:30-16:00", [f"{h:02d}:{m:02d}" for h in range(6, 16) for m in [0, 30]] + ["05:30"]),
    ("16:00-19:00", [f"{h:02d}:{m:02d}" for h in range(16, 19) for m in [0, 30]]),
    ("19:00-00:00", [f"{h:02d}:{m:02d}" for h in range(19, 24) for m in [0, 30]]),
]

print(f"{'Period':25} {'Base import':12} {'Free import':12} {'Diff':10} {'Avg price base':15}")
print("-" * 75)
for label, times in periods:
    base = sum(all_days[date][all_days[date]["timestamp"].isin(times)]["grid_kwh"].sum() for date in dates)
    free = sum(all_days_free[date][all_days_free[date]["timestamp"].isin(times)]["grid_kwh"].sum() for date in dates)

    # Average price during these periods (base)
    base_cost = sum(all_days[date][all_days[date]["timestamp"].isin(times)]["cost_p"].sum() for date in dates)
    avg_price = base_cost / max(base, 0.001)

    print(f"{label:25} {base:12.1f} {free:12.1f} {free - base:+10.1f} {avg_price:15.2f}p")

# %%
# batteries_utils.create_battery_modelling_results_chart(
#     profile_label="Medium TDCV, Elexon Profile Class 1",
#     baseline_bill=baseline_bill,
#     new_bill=battery_annual,
#     all_days=all_days_free,
#     price_label="Octopus Agile tariff - free 3h window",
# )

# %% [markdown]
# ## Plug-in battery

# %%
plugin_capacity = 5.0
plugin_inverter_power = 3.68

# %% [markdown]
# ### Elexon Profile 1 - Every day in 2025 - Two-rate tariff - Perfect foresight day-ahead optimisation battery model

# %%
all_days = {}

soc_init = 0.5

for date in dates:
    daily_result = run_linear_cost_optimisation_battery(
        time_series=daily_profile.get(date)["Settlement timestamp"].tolist(),
        demand_profile=daily_profile.get(date)["Electricity use (kWh)"].tolist(),
        prices=daily_go_rate.get(date)["value_inc_vat"].tolist(),
        capacity=plugin_capacity,
        inverter_power=plugin_inverter_power,
        lifespan=10,
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

    soc_init = daily_result["soc_kwh"].iloc[-1] / plugin_capacity

    all_days[date] = daily_result

# %%
# Battery optimised bill
battery_annual = sum(all_days[date]["cost_p"].sum() for date in dates) / 100
battery_annual += (go_standing_charge * 365) / 100

# No battery on Agile
no_battery_annual = (
    sum(
        sum(
            d * p
            for d, p in zip(
                daily_profile[date]["Electricity use (kWh)"].tolist(), daily_go_rate[date]["value_inc_vat"].tolist()
            )
        )
        for date in dates
    )
    / 100
)
no_battery_annual += (go_standing_charge * 365) / 100


print(f"Baseline SVT: £{baseline_bill:.2f}")
print(f"Go no battery: £{no_battery_annual:.2f}")
print(f"Go + battery:  £{battery_annual:.2f}")
print(f"Saving:           £{no_battery_annual - battery_annual:.2f}")

# %%
total_charge = sum(all_days[date]["charge_kwh"].sum() for date in all_days)
total_import = sum(all_days[date]["grid_kwh"].sum() for date in all_days)
total_discharge = sum(all_days[date]["discharge_kwh"].sum() for date in all_days)

print(f"Total charge    : {total_charge:.1f} kWh")
print(f"Total discharge : {total_discharge:.1f} kWh")
print(f"Total import    : {total_import:.1f} kWh")

# %% [markdown]
# - Go charges 33.54p during peak, vs Ofgem's flat 24.67p
# - A 2.5 kWh battery can only shift ~2.0kWh/day of demand to off-peak
# - The rest of your peak demand is now more expensive
# - The battery isn't big enough to overcome the tariff penalty

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
        (all_days[date]["grid_kwh"] * daily_grid_intensity[date]["intensity (gCO2/kWh)"].values).sum() for date in dates
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
print(f"Avg peak vs off-peak intensity diff  : {peak_intensity - off_peak_intensity:.1f} gCO2/kWh")

# %%
# Breakdown of original demand by time period
demand_by_period = pd.Series(0.0, index=avg_intensity_by_period.index)
grid_by_period = pd.Series(0.0, index=avg_intensity_by_period.index)

for date in dates:
    day = all_days[date]
    for _, row in day.iterrows():
        ts = row["timestamp"]
        if ts in demand_by_period.index:
            demand_by_period[ts] += row["demand_kwh"]
            grid_by_period[ts] += row["grid_kwh"]

# Summarise by period
off_peak_demand = demand_by_period[demand_by_period.index.isin(off_peak_times)].sum()
standard_demand = demand_by_period[demand_by_period.index.isin(standard_times)].sum()
peak_demand = demand_by_period[demand_by_period.index.isin(peak_times)].sum()

off_peak_grid = grid_by_period[grid_by_period.index.isin(off_peak_times)].sum()
standard_grid = grid_by_period[grid_by_period.index.isin(standard_times)].sum()
peak_grid = grid_by_period[grid_by_period.index.isin(peak_times)].sum()

total_demand = sum(all_days[date]["demand_kwh"].sum() for date in dates)

print(f"=== DEMAND PROFILE (without battery) ===")
print(f"Off-peak (00:00-06:00) : {off_peak_demand:.1f} kWh ({off_peak_demand / total_demand * 100:.1f}%)")
print(f"Standard (06:00-16:00, 19:00-00:00) : {standard_demand:.1f} kWh ({standard_demand / total_demand * 100:.1f}%)")
print(f"Peak     (16:00-19:00) : {peak_demand:.1f} kWh ({peak_demand / total_demand * 100:.1f}%)")

print(f"\n=== GRID DRAW PROFILE (with battery) ===")
print(f"Off-peak (00:00-06:00) : {off_peak_grid:.1f} kWh ({off_peak_grid / total_import * 100:.1f}%)")
print(f"Standard (06:00-16:00, 19:00-00:00) : {standard_grid:.1f} kWh ({standard_grid / total_import * 100:.1f}%)")
print(f"Peak     (16:00-19:00) : {peak_grid:.1f} kWh ({peak_grid / total_import * 100:.1f}%)")

print(f"\n=== SHIFT FROM BATTERY ===")
print(f"Off-peak change        : {off_peak_grid - off_peak_demand:+.1f} kWh (battery charging adds to off-peak import)")
print(f"Standard change        : {standard_grid - standard_demand:+.1f} kWh")
print(f"Peak change            : {peak_grid - peak_demand:+.1f} kWh (battery discharge reduces peak import)")

# %%
# batteries_utils.create_battery_modelling_results_chart(
#     profile_label="Plug-in battery - Elexon Profile 1 - Static tariff - Linear cost optimisation battery model",
#     baseline_bill=baseline_bill,
#     new_bill=battery_annual,
#     all_days=all_days,
#     price_label="Octopus Go tariff (London, May 2026)",
# )

# %% [markdown]
# ### Elexon Profile 1 - Every day in 2026 - Dynamic tariff - Perfect foresight day-ahead optimisation battery model

# %%
all_days = {}

soc_init = 0.5

for date in dates:
    daily_result = run_linear_cost_optimisation_battery(
        time_series=daily_profile.get(date)["Settlement timestamp"].tolist(),
        demand_profile=daily_profile.get(date)["Electricity use (kWh)"].tolist(),
        prices=daily_agile_rate.get(date)["value_inc_vat"].tolist(),
        capacity=plugin_capacity,
        inverter_power=plugin_inverter_power,
        lifespan=10,
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

    soc_init = daily_result["soc_kwh"].iloc[-1] / plugin_capacity

    all_days[date] = daily_result

# %%
# Battery optimised bill
battery_annual = sum(all_days[date]["cost_p"].sum() for date in dates) / 100
battery_annual += (agile_standing_charge * 365) / 100

# No battery on Agile
no_battery_annual = (
    sum(
        sum(
            d * p
            for d, p in zip(
                daily_profile[date]["Electricity use (kWh)"].tolist(), daily_agile_rate[date]["value_inc_vat"].tolist()
            )
        )
        for date in dates
    )
    / 100
)
no_battery_annual += (agile_standing_charge * 365) / 100


print(f"Baseline SVT: £{baseline_bill:.2f}")
print(f"Agile no battery: £{no_battery_annual:.2f}")
print(f"Agile + battery:  £{battery_annual:.2f}")
print(f"Saving:           £{no_battery_annual - battery_annual:.2f}")

# %%
total_charge = sum(all_days[date]["charge_kwh"].sum() for date in all_days)
total_import = sum(all_days[date]["grid_kwh"].sum() for date in all_days)
total_discharge = sum(all_days[date]["discharge_kwh"].sum() for date in all_days)

print(f"Total charge    : {total_charge:.1f} kWh")
print(f"Total discharge : {total_discharge:.1f} kWh")
print(f"Total import    : {total_import:.1f} kWh")

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
        (all_days[date]["grid_kwh"] * daily_grid_intensity[date]["intensity (gCO2/kWh)"].values).sum() for date in dates
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
print(f"Avg peak vs off-peak intensity diff  : {peak_intensity - off_peak_intensity:.1f} gCO2/kWh")

# %%
# Breakdown of original demand by time period
demand_by_period = pd.Series(0.0, index=avg_intensity_by_period.index)
grid_by_period = pd.Series(0.0, index=avg_intensity_by_period.index)

for date in dates:
    day = all_days[date]
    for _, row in day.iterrows():
        ts = row["timestamp"]
        if ts in demand_by_period.index:
            demand_by_period[ts] += row["demand_kwh"]
            grid_by_period[ts] += row["grid_kwh"]

# Summarise by period
off_peak_demand = demand_by_period[demand_by_period.index.isin(off_peak_times)].sum()
standard_demand = demand_by_period[demand_by_period.index.isin(standard_times)].sum()
peak_demand = demand_by_period[demand_by_period.index.isin(peak_times)].sum()

off_peak_grid = grid_by_period[grid_by_period.index.isin(off_peak_times)].sum()
standard_grid = grid_by_period[grid_by_period.index.isin(standard_times)].sum()
peak_grid = grid_by_period[grid_by_period.index.isin(peak_times)].sum()

total_demand = sum(all_days[date]["demand_kwh"].sum() for date in dates)

print(f"=== DEMAND PROFILE (without battery) ===")
print(f"Off-peak (00:00-06:00) : {off_peak_demand:.1f} kWh ({off_peak_demand / total_demand * 100:.1f}%)")
print(f"Standard (06:00-16:00, 19:00-00:00) : {standard_demand:.1f} kWh ({standard_demand / total_demand * 100:.1f}%)")
print(f"Peak     (16:00-19:00) : {peak_demand:.1f} kWh ({peak_demand / total_demand * 100:.1f}%)")

print(f"\n=== GRID DRAW PROFILE (with battery) ===")
print(f"Off-peak (00:00-06:00) : {off_peak_grid:.1f} kWh ({off_peak_grid / total_import * 100:.1f}%)")
print(f"Standard (06:00-16:00, 19:00-00:00) : {standard_grid:.1f} kWh ({standard_grid / total_import * 100:.1f}%)")
print(f"Peak     (16:00-19:00) : {peak_grid:.1f} kWh ({peak_grid / total_import * 100:.1f}%)")

print(f"\n=== SHIFT FROM BATTERY ===")
print(f"Off-peak change        : {off_peak_grid - off_peak_demand:+.1f} kWh (battery charging adds to off-peak import)")
print(f"Standard change        : {standard_grid - standard_demand:+.1f} kWh")
print(f"Peak change            : {peak_grid - peak_demand:+.1f} kWh (battery discharge reduces peak import)")


# %%
# batteries_utils.create_battery_modelling_results_chart(
#     profile_label="Plug-in Battery - Elexon Profile 1 - Dynamic tariff - Linear cost optimisation battery model",
#     baseline_bill=baseline_bill,
#     new_bill=battery_annual,
#     all_days=all_days,
#     price_label="Octopus Agile tariff (London, Historical 2025)",
# )

# %% [markdown]
# ### Sensitivity analysis: Testing what happens if high-draw appliances run

# %% [markdown]
# #### Resample demand data to 5 minutes

# %%
# Convert to 5 minute resolution
def resample_to_5min(demand_half_hourly_df: pd.DataFrame) -> pd.DataFrame:

    rows = []
    for _, row in demand_half_hourly_df.iterrows():
        base_time = pd.Timestamp(row["Settlement Date"]) + pd.Timedelta(minutes=(row["Settlement Period ID"] - 1) * 30)

        for i in range(6):
            rows.append(
                {
                    "Profile Class ID": row["Profile Class ID"],
                    "Settlement Date": row["Settlement Date"],
                    "Settlement Period ID": row["Settlement Period ID"],
                    "Default Period Profile Class Coefficient": row["Default Period Profile Class Coefficient"] / 6,
                    "Settlement timestamp": row["Settlement timestamp"],
                    "time_5min": base_time + pd.Timedelta(minutes=i * 5),
                    "Electricity use (kWh)": row["Electricity use (kWh)"] / 6,
                }
            )

    return pd.DataFrame(rows)


# %%
elexon_profile_class_1_5min = resample_to_5min(elexon_profile_class_1)

# %% [markdown]
# Appliance schedule

# %%
appliances_low_tdcv = [
    # Low demand household — likely 1-2 occupants, may not have electric shower
    # No electric shower assumed — gas shower or low-demand household
    {"time": "07:00", "power_kw": 2.5, "duration_mins": 5},  # kettle morning
    {"time": "18:00", "power_kw": 2.5, "duration_mins": 5},  # kettle evening
    {"time": "18:00", "power_kw": 2.5, "duration_mins": 30},  # oven first half
    {"time": "18:30", "power_kw": 2.5, "duration_mins": 30},  # oven second half
]

appliances_medium_tdcv = [
    # Medium demand household — typical 2-3 occupants, electric shower assumed
    {"time": "07:00", "power_kw": 8.5, "duration_mins": 10},  # electric shower
    {"time": "07:00", "power_kw": 2.5, "duration_mins": 5},  # kettle morning
    {"time": "18:00", "power_kw": 2.5, "duration_mins": 5},  # kettle evening
    {"time": "18:00", "power_kw": 2.5, "duration_mins": 30},  # oven first half
    {"time": "18:30", "power_kw": 2.5, "duration_mins": 30},  # oven second half
]

appliances_high_tdcv = [
    # High demand household — likely 3-4+ occupants, electric shower, more cooking
    {"time": "07:00", "power_kw": 8.5, "duration_mins": 10},  # electric shower
    {"time": "07:00", "power_kw": 2.5, "duration_mins": 5},  # kettle morning
    {"time": "18:00", "power_kw": 2.5, "duration_mins": 5},  # kettle evening
    {"time": "18:00", "power_kw": 3.0, "duration_mins": 30},  # oven first half — larger oven
    {"time": "18:30", "power_kw": 3.0, "duration_mins": 30},  # oven second half
    {"time": "18:00", "power_kw": 2.5, "duration_mins": 5},  # second kettle use (more occupants)
]

# %%
appliances = appliances_high_tdcv


# %%
# appliances = [
#     # Electric shower: 8.5 kW is the most common UK rating (Triton, Mira)
#     # Duration: UK average shower = 8 mins (Waterwise, 2021), rounded to 10 mins
#     # Source: https://www.waterwise.org.uk/save-water/
#     {"time": "07:00", "power_kw": 8.5, "duration_mins": 10},  # electric shower
#     # Kettle: 2.5 kW typical UK kettle (Which?, 2023)
#     # Duration: ~1.5 mins to boil 1 cup, rounded to 5 mins for full kettle
#     # Source: https://www.which.co.uk/reviews/kettles
#     {"time": "07:00", "power_kw": 2.5, "duration_mins": 5},  # kettle morning
#     # Kettle: same as above, evening use
#     {"time": "18:00", "power_kw": 2.5, "duration_mins": 5},  # kettle evening
#     # Electric oven: 2.0–3.0 kW typical range (Energy Saving Trust, 2023)
#     # Using 2.5 kW as midpoint; 60 min total cook time split across two settlement periods
#     # Source: https://energysavingtrust.org.uk/hub/quick-tips-to-save-energy/
#     {"time": "18:00", "power_kw": 2.5, "duration_mins": 30},  # oven first half
#     {"time": "18:30", "power_kw": 2.5, "duration_mins": 30},  # oven second half
# ]

# %%
def calculate_inverter_coverage_with_spikes(demand_5min_df, appliances, inverter_power, delta_t_5=5 / 60):
    max_discharge_per_5min = inverter_power * delta_t_5

    # Baseline demand without spikes — average 5-min interval energy across the year
    baseline_per_interval = demand_5min_df["Electricity use (kWh)"].mean()

    total_unmet_per_day = 0
    total_spike_energy_per_day = 0

    for a in appliances:
        # Energy drawn by this appliance in a single 5-min interval (kW × hours)
        spike_energy_per_interval = a["power_kw"] * delta_t_5

        # Number of 5-min intervals this appliance runs for
        n_intervals = a["duration_mins"] // 5

        # Total demand during spike = appliance draw + background baseload happening simultaneously
        total_demand_during_spike = spike_energy_per_interval + baseline_per_interval

        # Unmet energy per interval = amount exceeding inverter physical limit
        # If total demand <= inverter limit, unmet = 0 (battery can keep up)
        # If total demand > inverter limit, unmet = excess the battery cannot discharge fast enough
        unmet = max(0, total_demand_during_spike - max_discharge_per_5min)

        # Accumulate across all intervals this appliance runs (e.g. 10-min shower = 2 intervals)
        total_unmet_per_day += unmet * n_intervals

        # Track total spike energy for reference
        total_spike_energy_per_day += spike_energy_per_interval * n_intervals

    # Scale daily unmet energy to annual
    unmet_annual = total_unmet_per_day * 365
    total_annual_demand = demand_5min_df["Electricity use (kWh)"].sum()

    # Coverage = fraction of annual demand the inverter can physically serve
    coverage_pct = 1 - (unmet_annual / total_annual_demand)

    return coverage_pct, max_discharge_per_5min, unmet_annual


# %%
coverage_5kwh, limit_5kwh, unmet_5kwh = calculate_inverter_coverage_with_spikes(
    elexon_profile_class_1_5min, appliances, inverter_power=3.68
)

coverage_plugin, limit_plugin, unmet_plugin = calculate_inverter_coverage_with_spikes(
    elexon_profile_class_1_5min, appliances, inverter_power=0.8
)

print(f"5kWh/3.68kW  — coverage: {coverage_5kwh * 100:.1f}%, unmet annual: {unmet_5kwh:.1f} kWh")
print(f"2.5kWh/0.8kW — coverage: {coverage_plugin * 100:.1f}%, unmet annual: {unmet_plugin:.1f} kWh")

# %%
coverage_10kwh, limit_10kwh, unmet_10kwh = calculate_inverter_coverage_with_spikes(
    elexon_profile_class_1_5min, appliances, inverter_power=5.0
)

print(f"10kWh/5.0kW  — coverage: {coverage_10kwh * 100:.1f}%, unmet annual: {unmet_10kwh:.1f} kWh")

# %%
coverage_13_5kwh, limit_13_5kwh, unmet_13_5kwh = calculate_inverter_coverage_with_spikes(
    elexon_profile_class_1_5min, appliances, inverter_power=11.5
)

print(f"13.5kWh/11.5kW  — coverage: {coverage_13_5kwh * 100:.1f}%, unmet annual: {unmet_13_5kwh:.1f} kWh")
