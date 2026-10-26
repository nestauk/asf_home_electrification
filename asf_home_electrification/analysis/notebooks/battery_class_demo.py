# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: asf-home-electrification (3.13.2)
#     language: python
#     name: python3
# ---

# %%
import altair as alt
import pandas as pd

from asf_home_electrification.getters import elexon_demand_profiles, octopus_tariffs
from asf_home_electrification.pipeline.battery import Battery

# %% [markdown]
# ## Setting up demand and price profiles

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

baseline_annual_bill = (annual_electricity_use_kwh * price_cap_rate + n_days * price_cap_standing_charge) / 100  # £
baseline_annual_bill

# %% [markdown]
# Octopus Go

# %%
# Octopus Go tariff as an example
# Cheap rate from 00:30 to 05:30

# Unit rates
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

# Standing charge rates
df = octopus_tariffs.get_go_standing_charge(region="E", start_date="2025-01-01", end_date="2026-01-01")
df["valid_from"] = pd.to_datetime(df["valid_from"])
df["valid_to"] = pd.to_datetime(df["valid_to"])
df["days"] = (df["valid_to"] - df["valid_from"]).dt.total_seconds() / 86400
go_standing_charge = (df["value_inc_vat"] * df["days"]).sum() / df["days"].sum()

# %%
# Create price_profile arg as pd.Series
go_import_price_profile = (
    pd.Series(
        go_price["value_inc_vat"].to_numpy(),
        index=pd.DatetimeIndex(go_price["valid_from"].dt.tz_convert("Europe/London"), name="timestamp"),
    )
    .sort_index()
    .loc["2025"]
)

# %% [markdown]
# Octopus Agile

# %%
# # Run this first time only
# df = octopus_tariffs.get_agile_rates(region="E", start_date="2025-01-01", end_date="2026-01-01")
# df.to_pickle("agile_region_E_2025.pkl")

# %%
# Unit rates
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

# Standing charges
agile_standing_charge_df = octopus_tariffs.get_agile_standing_charge(
    region="E", start_date="2025-01-01", end_date="2026-01-01"
)
agile_standing_charge = agile_standing_charge_df["value_inc_vat"][0]  # p/day

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
# Create price_profile arg as pd.Series
agile_import_price_profile = pd.Series(
    agile_import_price["value_inc_vat"].to_numpy(),  # p/kWh including VAT
    index=pd.DatetimeIndex(agile_import_price["valid_from_local"], name="timestamp"),
).sort_index()

# %% [markdown]
# Index checks (BST is tricky!)

# %%
# Checks: Period IDs should reach 46 and 50 on the clock-change days
print(elexon_profile_class_1.groupby("Settlement Date")["Settlement Period ID"].max().loc[["2025-03-30", "2025-10-26"]])

# Demand profile: regular, unique, half-hourly
assert demand_profile.index.is_unique
assert len(demand_profile) == 17520
assert demand_profile.index.to_series().diff().dropna().eq(pd.Timedelta("30min")).all()

# Price profiles: each must match the demand index exactly
for name, prices in {"go": go_import_price_profile, "agile": agile_import_price_profile}.items():
    assert prices.index.is_unique, f"{name}: duplicated timestamps"
    assert len(prices) == len(demand_profile), f"{name}: {len(prices)} rows, expected {len(demand_profile)}"
    assert prices.index.equals(demand_profile.index), f"{name}: index differs from demand"
    assert prices.notna().all(), f"{name}: {prices.isna().sum()} missing prices"

# %% [markdown]
# ## Instantiate the Battery

# %%
my_battery = Battery(
    capacity_kwh=5.0,
    max_power_kw=3.68,
    lifespan_years=12,
    cycle_life=6_000,
    charge_efficiency=0.95,
    discharge_efficiency=0.95,
    state_of_charge_min_frac=0.1,
    state_of_charge_max_frac=0.9,
    state_of_charge_init_frac=0.1,
)
my_battery

# %% [markdown]
# ## 1. Running the battery on a fixed schedule

# %%
# Assume the battery is set on a fixed charging schedule
# Charge between 00:30 to 05:30, discharge any time outside this window

charge_schedule = pd.Series(-1, index=demand_profile.index)
charge_schedule.iloc[demand_profile.index.indexer_between_time("00:30", "05:00")] = 1

# %%
fixed_schedule_run = my_battery.run_fixed_schedule(
    demand_profile=demand_profile,
    import_price_profile=go_import_price_profile,
    charge_schedule=charge_schedule,
    delta_t=0.5,
)

# %%
# Bills summary
grid_import_kwh = fixed_schedule_run["grid_import_kwh"]
bill_go_no_battery = ((demand_profile * go_import_price_profile).sum() + go_standing_charge * n_days) / 100  # £
bill_price_cap_with_battery = ((grid_import_kwh * price_cap_rate).sum() + price_cap_standing_charge * n_days) / 100  # £
bill_go_with_battery = (fixed_schedule_run["cost_p"].sum() + go_standing_charge * n_days) / 100  # £
saving_go_no_battery = baseline_annual_bill - bill_go_no_battery
saving_price_cap_with_battery = baseline_annual_bill - bill_price_cap_with_battery
saving_go_with_battery = baseline_annual_bill - bill_go_with_battery

print(f"Price cap, no battery:                 £{baseline_annual_bill:,.2f}")
print(
    f"Octopus Go, no battery:                £{bill_go_no_battery:,.2f}   saving £{saving_go_no_battery:,.2f} ({saving_go_no_battery / baseline_annual_bill:.0%})"
)
print(
    f"Price cap, battery on Go schedule:     £{bill_price_cap_with_battery:,.2f}   saving £{saving_price_cap_with_battery:,.2f} ({saving_price_cap_with_battery / baseline_annual_bill:.0%})"
)
print(
    f"Octopus Go, battery on Go schedule:    £{bill_go_with_battery:,.2f}   saving £{saving_go_with_battery:,.2f} ({saving_go_with_battery / baseline_annual_bill:.0%})"
)

# %% [markdown]
# - Tariff switch alone (no battery) saves ~£30
# - Running the charging schedule on the flat price cap tariff costs more than having no battery because the battery's round-trip losses mean you import extra energy.

# %%
# Simple average across all half-hours, and weighted by when the household actually uses energy
go_mean_price = go_import_price_profile.mean()
go_demand_weighted_price = (demand_profile * go_import_price_profile).sum() / demand_profile.sum()

print(f"Go average import price (time-weighted):   {go_mean_price:.2f} p/kWh")
print(f"Go average import price (demand-weighted): {go_demand_weighted_price:.2f} p/kWh")
print(f"Go standing charge:                        {go_standing_charge:.2f} p/day")


# %%
def create_battery_modelling_results_chart(
    profile_label: str,
    baseline_bill: float,
    new_bill: float,
    run: pd.DataFrame,
    price_label: str,
    date: str | pd.Timestamp,
) -> alt.Chart:
    """Build a chart of battery behaviour, demand and price for a single day.

    Args:
        profile_label: Description of the household, shown in the chart title.
        baseline_bill: Annual bill without the battery, in pounds.
        new_bill: Annual bill with the battery, in pounds.
        run: Output of a battery simulation, indexed by timestamp, with columns
            soc_kwh, demand_kwh, grid_import_kwh and import_price_p_per_kwh.
        price_label: Name of the tariff, shown on the price axis.
        date: Day to plot, e.g. "2025-01-01". Matched against the local date
            of the index.

    Returns:
        Layered Altair chart for the selected day.

    Raises:
        ValueError: If the run has no data for the given date.
    """
    date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
    day = run.loc[run.index.strftime("%Y-%m-%d") == date_str]
    if day.empty:
        raise ValueError(f"No data for {date_str}.")

    # Rename to the column names the chart uses, and move the index into columns
    df = (
        day[["soc_kwh", "demand_kwh", "grid_import_kwh", "import_price_p_per_kwh"]]
        .rename(columns={"grid_import_kwh": "grid_kwh", "import_price_p_per_kwh": "price_p_per_kwh"})
        .reset_index(drop=True)
    )
    df["timestamp"] = day.index.strftime("%H:%M")

    power_vars = ["soc_kwh", "demand_kwh", "grid_kwh"]
    df_power = df.melt(
        id_vars=["timestamp"],
        value_vars=power_vars,
        var_name="variable",
        value_name="value",
    )

    color_scale = alt.Scale(
        domain=["soc_kwh", "demand_kwh", "grid_kwh", "price"],
        range=["#1f77b4", "#2ca02c", "#9467bd", "#000000"],
    )

    label_expr = (
        "datum.label === 'soc_kwh' ? 'State of charge' : "
        "datum.label === 'demand_kwh' ? 'Demand' : "
        "datum.label === 'grid_kwh' ? 'Grid import' : "
        "datum.label === 'price' ? 'Price (p/kWh)' : datum.label"
    )

    power_lines = (
        alt.Chart(df_power)
        .mark_line(interpolate="step-after")
        .encode(
            x=alt.X(
                "timestamp:O",
                title="Time of day",
                axis=alt.Axis(labelAngle=-45, labelOverlap=False),
            ),
            y=alt.Y("value:Q", title="Energy (kWh)"),
            color=alt.Color(
                "variable:N",
                scale=color_scale,
                legend=alt.Legend(title="variable", labelExpr=label_expr),
            ),
            tooltip=[
                alt.Tooltip("timestamp:O", title="Time of day"),
                alt.Tooltip("variable:N", title="Series"),
                alt.Tooltip("value:Q", title="Value", format=".2f"),
            ],
        )
    )

    price_line = (
        alt.Chart(df)
        .mark_line(interpolate="step-after")
        .encode(
            x=alt.X("timestamp:O"),
            y=alt.Y(
                "price_p_per_kwh:Q",
                title=f"Price (p/kWh) ({price_label})",
                axis=alt.Axis(orient="right"),
            ),
            color=alt.Color(
                "variable:N",
                scale=alt.Scale(domain=["price"], range=["#000000"]),
                legend=alt.Legend(title="variable", labelExpr=label_expr),
            ),
            tooltip=[
                alt.Tooltip("timestamp:O", title="Time of day"),
                alt.Tooltip("price_p_per_kwh:Q", title="Price (p/kWh)", format=".2f"),
            ],
        )
        .transform_calculate(variable="'price'")
    )

    saving_pct = (baseline_bill - new_bill) / baseline_bill * 100

    chart = (
        alt.layer(power_lines, price_line)
        .resolve_scale(y="independent", color="shared")
        .properties(
            width=800,
            height=400,
            title=[
                f"Household: {profile_label} ({date_str})",
                f"Annual baseline bill: £{baseline_bill:.2f}, new bill: £{new_bill:.2f}, savings: {saving_pct:.0f}%",
            ],
        )
    )

    return chart


# %%
winter_chart = create_battery_modelling_results_chart(
    profile_label="Elexon Profile Class 1, 2,500 kWh annual",
    baseline_bill=bill_go_no_battery,
    new_bill=bill_go_with_battery,
    run=fixed_schedule_run,
    price_label="Octopus Go",
    date="2025-01-01",
)

# %%
winter_chart

# %%
summer_chart = create_battery_modelling_results_chart(
    profile_label="Elexon Profile Class 1, 2,500 kWh annual",
    baseline_bill=bill_go_no_battery,
    new_bill=bill_go_with_battery,
    run=fixed_schedule_run,
    price_label="Octopus Go",
    date="2025-07-01",
)

# %%
summer_chart

# %% [markdown]
# ## 2. Running the battery controlled with cost optimiser with perfect foresight

# %% [markdown]
# ### Octopus Go - we expect this to be similar to the fixed charging schedule run

# %%
octopus_go_optimised_run = my_battery.run_cost_optimisation(
    demand_profile=demand_profile,
    import_price_profile=go_import_price_profile,
    delta_t=0.5,
)

# %%
bill_go_optimised = (octopus_go_optimised_run["cost_p"].sum() + go_standing_charge * n_days) / 100

print(f"Go, fixed schedule: £{bill_go_with_battery:,.2f}")
print(f"Go, optimised:      £{bill_go_optimised:,.2f}")

# %%
create_battery_modelling_results_chart(
    profile_label="Elexon Profile Class 1, 2,500 kWh annual",
    baseline_bill=bill_go_no_battery,
    new_bill=bill_go_optimised,
    run=octopus_go_optimised_run,
    price_label="Octopus Go",
    date="2025-01-01",
)

# %% [markdown]
# ### Octopus Agile

# %%
octopus_agile_optimised_run = my_battery.run_cost_optimisation(
    demand_profile=demand_profile,
    import_price_profile=agile_import_price_profile,
    delta_t=0.5,
)

# %%
run = octopus_agile_optimised_run
simultaneous = (run["charge_kwh"] > 1e-6) & (run["discharge_kwh"] > 1e-6)
sim = run.loc[simultaneous]

cycled_kwh = sim["discharge_kwh"].sum()
# Upper bound on the bill benefit of simultaneous charge and discharge
# During negative import price intervals, discharging while charging lets the battery import slightly more.
# The most this could earn is the energy discharged in each interval times the size of the negative
# price, summed over the year and converted from pence to pounds. The real benefit is smaller, since
# only the round-trip loss on that energy is actually gained.
benefit_bound_gbp = (sim["discharge_kwh"] * sim["import_price_p_per_kwh"].abs()).sum() / 100

print(f"{simultaneous.sum()} intervals, all negative price: {(sim['import_price_p_per_kwh'] < 0).all()}")
print(f"Energy discharged in those intervals: {cycled_kwh:.1f} kWh")
print(f"Bill benefit upper bound: £{benefit_bound_gbp:.2f}")


# %%
# Bills summary (all bills include the relevant standing charge, in £)
grid_import_agile_kwh = octopus_agile_optimised_run["grid_import_kwh"]

bill_agile_no_battery = ((demand_profile * agile_import_price_profile).sum() + agile_standing_charge * n_days) / 100
bill_agile_with_battery = (octopus_agile_optimised_run["cost_p"].sum() + agile_standing_charge * n_days) / 100
bill_price_cap_with_agile_battery = (
    (grid_import_agile_kwh * price_cap_rate).sum() + price_cap_standing_charge * n_days
) / 100  # same battery behaviour, priced at the flat price cap rate

saving_agile_no_battery = baseline_annual_bill - bill_agile_no_battery
saving_agile_with_battery = baseline_annual_bill - bill_agile_with_battery
saving_price_cap_with_agile_battery = baseline_annual_bill - bill_price_cap_with_agile_battery

print(f"Price cap, no battery:                    £{baseline_annual_bill:,.2f}")
print(
    f"Octopus Agile, no battery:                £{bill_agile_no_battery:,.2f}   "
    f"saving £{saving_agile_no_battery:,.2f} ({saving_agile_no_battery / baseline_annual_bill:.0%})"
)
print(
    f"Octopus Agile, optimised battery:         £{bill_agile_with_battery:,.2f}   "
    f"saving £{saving_agile_with_battery:,.2f} ({saving_agile_with_battery / baseline_annual_bill:.0%})"
)
print(
    f"Price cap, battery on Agile dispatch:     £{bill_price_cap_with_agile_battery:,.2f}   "
    f"saving £{saving_price_cap_with_agile_battery:,.2f} ({saving_price_cap_with_agile_battery / baseline_annual_bill:.0%})"
)
print(f"\nBattery saving on Agile alone:            £{bill_agile_no_battery - bill_agile_with_battery:,.2f}")

# %%
create_battery_modelling_results_chart(
    profile_label="Elexon Profile Class 1, 2,500 kWh annual",
    baseline_bill=baseline_annual_bill,
    new_bill=bill_agile_with_battery,
    run=octopus_agile_optimised_run,
    price_label="Octopus Agile",
    date="2025-01-01",
)

# %% [markdown]
# ## 2. Running the battery controlled with cost optimiser with perfect foresight, with exports enabled

# %% [markdown]
# Octopus Agile - export

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
agile_export_price_profile = pd.Series(
    agile_export_price["value_exc_vat"].to_numpy(),  # p/kWh
    index=pd.DatetimeIndex(agile_export_price["valid_from_local"], name="timestamp"),
).sort_index()

# %%
octopus_agile_export_run = my_battery.run_cost_optimisation_with_export(
    demand_profile=demand_profile,
    import_price_profile=agile_import_price_profile,
    export_price_profile=agile_export_price_profile,
    delta_t=0.5,
)

# %%
run = octopus_agile_export_run
pass_through = (
    (run["charge_kwh"] > 1e-6)
    & (run["grid_export_kwh"] > 1e-6)
    & (run["export_price_p_per_kwh"] > run["import_price_p_per_kwh"])
)
simultaneous = (run["charge_kwh"] > 1e-6) & (run["discharge_kwh"] > 1e-6)
print(
    "Same intervals:", (pass_through == simultaneous).all() or pass_through.sum() == (pass_through & simultaneous).sum()
)

pt = run.loc[pass_through]
print(pt[["import_price_p_per_kwh", "export_price_p_per_kwh", "charge_kwh", "grid_import_kwh", "grid_export_kwh"]])

# Upper bound: energy that goes in and straight out, times the price spread, in pounds
looped_kwh = pt[["charge_kwh", "grid_export_kwh"]].min(axis=1)
spread_p = pt["export_price_p_per_kwh"] - pt["import_price_p_per_kwh"]
benefit_bound_gbp = (looped_kwh * spread_p).sum() / 100
print(f"Energy looped through the battery: {looped_kwh.sum():.1f} kWh")
print(f"Bill benefit upper bound: £{benefit_bound_gbp:.2f}")

# %%
# Bills summary with export (all bills include the relevant standing charge, in £)
export_run = octopus_agile_export_run

import_cost_gbp = export_run["import_cost_p"].sum() / 100
export_revenue_gbp = export_run["export_revenue_p"].sum() / 100

# cost_p is already net: import cost minus export revenue
bill_agile_with_export_battery = (export_run["cost_p"].sum() + agile_standing_charge * n_days) / 100

# Same battery behaviour, with imports priced at the flat price cap rate
# (export revenue is still earned at the Agile Outgoing rates)
bill_price_cap_with_export_battery = (
    (export_run["grid_import_kwh"] * price_cap_rate).sum()
    + price_cap_standing_charge * n_days
    - export_run["export_revenue_p"].sum()
) / 100

saving_agile_with_export_battery = baseline_annual_bill - bill_agile_with_export_battery
saving_price_cap_with_export_battery = baseline_annual_bill - bill_price_cap_with_export_battery

print(f"Price cap, no battery:                    £{baseline_annual_bill:,.2f}")
print(
    f"Octopus Agile, no battery:                £{bill_agile_no_battery:,.2f}   saving £{saving_agile_no_battery:,.2f} ({saving_agile_no_battery / baseline_annual_bill:.0%})"
)
print(
    f"Octopus Agile, battery, no export:        £{bill_agile_with_battery:,.2f}   "
    f"saving £{saving_agile_with_battery:,.2f} ({saving_agile_with_battery / baseline_annual_bill:.0%})"
)
print(
    f"Octopus Agile, battery, with export:      £{bill_agile_with_export_battery:,.2f}   "
    f"saving £{saving_agile_with_export_battery:,.2f} ({saving_agile_with_export_battery / baseline_annual_bill:.0%})"
)

print(
    f"\nWith export: import cost £{import_cost_gbp:,.2f}, export revenue £{export_revenue_gbp:,.2f}, standing charge £{agile_standing_charge * n_days / 100:,.2f}"
)
print(f"Battery saving on Agile alone (with export):  £{bill_agile_no_battery - bill_agile_with_export_battery:,.2f}")
print(f"Extra saving from allowing export:            £{bill_agile_with_battery - bill_agile_with_export_battery:,.2f}")

# %% [markdown]
# Export adds a modest amount of savings because the battery is too small to have energy to spare after covering household demand

# %%
run = octopus_agile_export_run
exported_share = run["grid_export_kwh"].sum() / run["discharge_kwh"].sum()
avg_export_price = run["export_revenue_p"].sum() / run["grid_export_kwh"].sum()
print(f"Share of discharge that is exported: {exported_share:.1%}")
print(f"Average price achieved on exports: {avg_export_price:.1f} p/kWh")
print(f"Average Agile Outgoing price over the year: {run['export_price_p_per_kwh'].mean():.1f} p/kWh")

# %%
