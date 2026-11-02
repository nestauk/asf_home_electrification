# %%
import pandas as pd

from asf_home_electrification.getters import elexon_demand_profiles
from asf_home_electrification.pipeline.battery import Battery
from asf_home_electrification.pipeline.solar import SolarPV
from asf_home_electrification.pipeline.solar_battery import SolarBatterySystem

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

baseline_annual_bill = (demand_profile.values.sum() * price_cap_rate + n_days * price_cap_standing_charge) / 100  # £
baseline_annual_bill

# %% [markdown]
# Octopus Flux

# %%
# Off-peak (02:00-05:00), Standard, Peak (16:00-19:00)

# Unit rates
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

# Standing charge
flux_standing_charge_df = octopus_tariffs.get_flux_standing_charge(
    region="E", start_date="2025-01-01", end_date="2026-01-01"
)
df = flux_standing_charge_df.copy()
df["valid_from"] = pd.to_datetime(df["valid_from"])
df["valid_to"] = pd.to_datetime(df["valid_to"])
df["days"] = (df["valid_to"] - df["valid_from"]).dt.total_seconds() / 86400

flux_standing_charge = (df["value_inc_vat"] * df["days"]).sum() / df["days"].sum()  # p/day, weighted average

# %%
# Export rates
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
# Create price_profile arg as pd.Series
flux_import_price_profile = (
    pd.Series(
        flux_import_price["value_inc_vat"].to_numpy(),
        index=pd.DatetimeIndex(flux_import_price["valid_from"].dt.tz_convert("Europe/London"), name="timestamp"),
    )
    .sort_index()
    .loc["2025"]
)

flux_export_price_profile = (
    pd.Series(
        flux_export_price["value_exc_vat"].to_numpy(),
        index=pd.DatetimeIndex(flux_export_price["valid_from"].dt.tz_convert("Europe/London"), name="timestamp"),
    )
    .sort_index()
    .loc["2025"]
)

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
for name, prices in {"flux import": flux_import_price_profile, "flux export": flux_export_price_profile}.items():
    assert prices.index.is_unique, f"{name}: duplicated timestamps"
    assert len(prices) == len(demand_profile), f"{name}: {len(prices)} rows, expected {len(demand_profile)}"
    assert prices.index.equals(demand_profile.index), f"{name}: index differs from demand"
    assert prices.notna().all(), f"{name}: {prices.isna().sum()} missing prices"

# %% [markdown]
# ## Instantiate the Battery and SolarPV

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

# %%
lat = 52.5  # West Midlands region (typical central England suburban/semi-rural conditions)
lon = -1.5
peakpower = 4.0  # kWp (typical, semi-detached)
loss = 14  # % (standard PVGIS default, realistic)
mountingplace = "building"
aspect = 0  # south-facing
angle = 35  # degrees (typical UK roof pitch)
my_solar = SolarPV(
    peak_power_kw=peakpower, lat=lat, lon=lon, loss=loss, mounting_place=mountingplace, aspect=aspect, angle=angle
)

# %%
my_system = SolarBatterySystem(battery=my_battery, solar=my_solar)

# %% [markdown]
# ## 1. Running the solar and battery system on self consumption mode only
# Battery only charges from surplus solar, and any generation that the battery can't take is exported.

# %%
self_consumption_run = my_system.run_self_consumption_only(
    demand_profile=demand_profile,
    import_price_profile=flux_import_price_profile,
    export_price_profile=flux_export_price_profile,
)

# %%
run = self_consumption_run
battery = my_system.battery

# Annual energy totals (kWh)
demand_kwh = run["demand_kwh"].sum()
generation_kwh = run["solar_generation_kwh"].sum()
direct_use_kwh = run["self_consumption_kwh"].sum()  # solar used in the home straight away
charged_kwh = run["charge_kwh"].sum()  # surplus solar stored (never charges from the grid)
discharged_kwh = run["discharge_kwh"].sum()  # delivered from the battery to the home
exported_kwh = run["grid_export_kwh"].sum()
curtailed_kwh = run["curtailed_kwh"].sum()
imported_kwh = run["grid_import_kwh"].sum()

# Battery losses: energy in = energy out + losses + change in stored energy
soc_change_kwh = run["soc_kwh"].iloc[-1] - battery.starting_charge_kwh
battery_losses_kwh = charged_kwh - discharged_kwh - soc_change_kwh

# Ratios
self_consumption_rate = (direct_use_kwh + charged_kwh) / generation_kwh  # share of generation kept on site
self_sufficiency = 1 - imported_kwh / demand_kwh  # share of demand met without the grid

# Battery use
n_days = run.index.normalize().nunique()
equivalent_full_cycles = (discharged_kwh / battery.discharge_efficiency) / battery.usable_capacity_kwh

# Bills (£, each including its own tariff's standing charge)
flux_standing_charge_gbp = flux_standing_charge * n_days / 100
svt_standing_charge_gbp = price_cap_standing_charge * n_days / 100

# SVT baseline is built from this run's demand, so the comparison isn't distorted by a different demand total
bill_svt_no_solar = demand_kwh * price_cap_rate / 100 + svt_standing_charge_gbp
bill_flux_no_solar = (run["demand_kwh"] * run["import_price_p_per_kwh"]).sum() / 100 + flux_standing_charge_gbp
bill_solar_battery = run["cost_p"].sum() / 100 + flux_standing_charge_gbp

saving_tariff_only = bill_svt_no_solar - bill_flux_no_solar  # from switching tariff alone
saving_vs_svt = bill_svt_no_solar - bill_solar_battery  # tariff + solar + battery
saving_vs_flux = bill_flux_no_solar - bill_solar_battery  # solar + battery on top of Flux

print(f"Demand:                  {demand_kwh:,.0f} kWh")
print(f"Solar generation:        {generation_kwh:,.0f} kWh")
print(f"  Used directly:         {direct_use_kwh:,.0f} kWh")
print(f"  Stored in battery:     {charged_kwh:,.0f} kWh")
print(f"  Exported:              {exported_kwh:,.0f} kWh")
print(f"  Curtailed:             {curtailed_kwh:,.0f} kWh")
print(
    f"Battery discharged:      {discharged_kwh:,.0f} kWh "
    f"({equivalent_full_cycles:.0f} full cycles, {equivalent_full_cycles / n_days:.2f} per day)"
)
print(f"Battery losses:          {battery_losses_kwh:,.0f} kWh")
print(f"Grid import:             {imported_kwh:,.0f} kWh")
print(f"Self-consumption rate:   {self_consumption_rate:.0%} of generation kept on site")
print(f"Self-sufficiency:        {self_sufficiency:.0%} of demand met without the grid")
print()
print(f"Import cost:             £{run['import_cost_p'].sum() / 100:,.2f}")
print(f"Export revenue:          £{run['export_revenue_p'].sum() / 100:,.2f}")
print(f"Standing charge (Flux):  £{flux_standing_charge_gbp:,.2f}")
print()
print(f"Bill, no solar or battery (SVT):   £{bill_svt_no_solar:,.2f}")
print(
    f"Bill, no solar or battery (Flux):  £{bill_flux_no_solar:,.2f}   "
    f"saving £{saving_tariff_only:,.2f} ({saving_tariff_only / bill_svt_no_solar:.0%}) from the tariff change alone"
)
print(
    f"Bill, solar + battery (Flux):      £{bill_solar_battery:,.2f}   "
    f"saving £{saving_vs_svt:,.2f} ({saving_vs_svt / bill_svt_no_solar:.0%}) vs SVT, "
    f"£{saving_vs_flux:,.2f} ({saving_vs_flux / bill_flux_no_solar:.0%}) vs Flux"
)

# %% [markdown]
# ## 2. Running the solar and battery system on self consumption mode, but with a specified grid charging window
# Battery charges from surplus solar - or, if in specified window, from the grid - and any generation that the battery can't take is exported.

# %%
grid_charge_window = pd.Series(False, index=demand_profile.index)
grid_charge_window.iloc[demand_profile.index.indexer_between_time("02:00", "05:00", include_end=False)] = True

self_consumption_grid_charge_run = my_system.run_self_consumption_with_grid_charge(
    demand_profile=demand_profile,
    import_price_profile=flux_import_price_profile,
    export_price_profile=flux_export_price_profile,
    grid_charge_window=grid_charge_window,
    grid_charge_target_frac=0.4,
)

# %%
run = self_consumption_grid_charge_run
battery = my_system.battery

# Annual energy totals (kWh)
demand_kwh = run["demand_kwh"].sum()
generation_kwh = run["solar_generation_kwh"].sum()
direct_use_kwh = run["self_consumption_kwh"].sum()  # solar used in the home straight away
charged_solar_kwh = run["charge_from_solar_kwh"].sum()  # surplus solar stored
charged_grid_kwh = run["charge_from_grid_kwh"].sum()  # bought from the grid to charge the battery
charged_kwh = run["charge_kwh"].sum()  # total drawn into the battery
discharged_kwh = run["discharge_kwh"].sum()  # delivered from the battery to the home
exported_kwh = run["grid_export_kwh"].sum()
curtailed_kwh = run["curtailed_kwh"].sum()
imported_kwh = run["grid_import_kwh"].sum()  # includes the energy bought to charge the battery

# Battery losses: energy in = energy out + losses + change in stored energy
soc_change_kwh = run["soc_kwh"].iloc[-1] - battery.starting_charge_kwh
battery_losses_kwh = charged_kwh - discharged_kwh - soc_change_kwh

# Ratios
self_consumption_rate = (direct_use_kwh + charged_solar_kwh) / generation_kwh  # share of generation kept on site
# Demand not served by the grid at the time, i.e. met by solar and the battery
# (the battery energy can include grid-charged energy, so this is not purely solar)
self_sufficiency = 1 - (imported_kwh - charged_grid_kwh) / demand_kwh

# Battery use
n_days = run.index.normalize().nunique()
equivalent_full_cycles = (discharged_kwh / battery.discharge_efficiency) / battery.usable_capacity_kwh

# Grid charging
grid_charge_cost_gbp = (run["charge_from_grid_kwh"] * run["import_price_p_per_kwh"]).sum() / 100
avg_grid_charge_price_p = grid_charge_cost_gbp * 100 / charged_grid_kwh if charged_grid_kwh > 0 else float("nan")

# Bills (£, each including its own tariff's standing charge)
flux_standing_charge_gbp = flux_standing_charge * n_days / 100
svt_standing_charge_gbp = price_cap_standing_charge * n_days / 100

bill_svt_no_solar = demand_kwh * price_cap_rate / 100 + svt_standing_charge_gbp
bill_flux_no_solar = (run["demand_kwh"] * run["import_price_p_per_kwh"]).sum() / 100 + flux_standing_charge_gbp
bill_solar_battery = run["cost_p"].sum() / 100 + flux_standing_charge_gbp

saving_tariff_only = bill_svt_no_solar - bill_flux_no_solar
saving_vs_svt = bill_svt_no_solar - bill_solar_battery
saving_vs_flux = bill_flux_no_solar - bill_solar_battery

print(f"Demand:                  {demand_kwh:,.0f} kWh")
print(f"Solar generation:        {generation_kwh:,.0f} kWh")
print(f"  Used directly:         {direct_use_kwh:,.0f} kWh")
print(f"  Stored in battery:     {charged_solar_kwh:,.0f} kWh")
print(f"  Exported:              {exported_kwh:,.0f} kWh")
print(f"  Curtailed:             {curtailed_kwh:,.0f} kWh")
print(
    f"Battery charged from grid: {charged_grid_kwh:,.0f} kWh "
    f"(avg {avg_grid_charge_price_p:.1f} p/kWh, £{grid_charge_cost_gbp:,.2f})"
)
print(
    f"Battery discharged:      {discharged_kwh:,.0f} kWh "
    f"({equivalent_full_cycles:.0f} full cycles, {equivalent_full_cycles / n_days:.2f} per day)"
)
print(f"Battery losses:          {battery_losses_kwh:,.0f} kWh")
print(f"Grid import:             {imported_kwh:,.0f} kWh (of which {charged_grid_kwh:,.0f} kWh charges the battery)")
print(f"Self-consumption rate:   {self_consumption_rate:.0%} of generation kept on site")
print(f"Self-sufficiency:        {self_sufficiency:.0%} of demand met by solar and the battery")
print()
print(f"Import cost:             £{run['import_cost_p'].sum() / 100:,.2f}")
print(f"Export revenue:          £{run['export_revenue_p'].sum() / 100:,.2f}")
print(f"Standing charge (Flux):  £{flux_standing_charge_gbp:,.2f}")
print()
print(f"Bill, no solar or battery (SVT):   £{bill_svt_no_solar:,.2f}")
print(
    f"Bill, no solar or battery (Flux):  £{bill_flux_no_solar:,.2f}   "
    f"saving £{saving_tariff_only:,.2f} ({saving_tariff_only / bill_svt_no_solar:.0%}) from the tariff change alone"
)
print(
    f"Bill, solar + battery (Flux):      £{bill_solar_battery:,.2f}   "
    f"saving £{saving_vs_svt:,.2f} ({saving_vs_svt / bill_svt_no_solar:.0%}) vs SVT, "
    f"£{saving_vs_flux:,.2f} ({saving_vs_flux / bill_flux_no_solar:.0%}) vs Flux"
)

# %% [markdown]
# ## 3. Running the solar and battery system with cost optimiser, with export enabled

# %%
optimisation_export_run = my_system.run_solar_cost_optimisation_with_export(
    demand_profile=demand_profile,
    import_price_profile=flux_import_price_profile,
    export_price_profile=flux_export_price_profile,
)

# %%
run = optimisation_export_run
battery = my_system.battery

# Annual energy totals (kWh)
demand_kwh = run["demand_kwh"].sum()
generation_kwh = run["solar_generation_kwh"].sum()
charged_kwh = run["charge_kwh"].sum()  # drawn into the battery, from solar and grid combined
discharged_kwh = run["discharge_kwh"].sum()  # delivered from the battery (to the home or the grid)
exported_kwh = run["grid_export_kwh"].sum()
curtailed_kwh = run["curtailed_kwh"].sum()
imported_kwh = run["grid_import_kwh"].sum()  # includes any energy bought to charge the battery
net_import_kwh = imported_kwh - exported_kwh

# Battery losses: energy in = energy out + losses + change in stored energy
soc_change_kwh = run["soc_kwh"].iloc[-1] - battery.starting_charge_kwh
battery_losses_kwh = charged_kwh - discharged_kwh - soc_change_kwh

# Battery use
n_days = run.index.normalize().nunique()
equivalent_full_cycles = (discharged_kwh / battery.discharge_efficiency) / battery.usable_capacity_kwh
daily_throughput_kwh = (run["discharge_kwh"] / battery.discharge_efficiency).groupby(run.index.normalize()).sum()
days_at_throughput_limit = (daily_throughput_kwh >= battery.max_daily_throughput_kwh - 1e-6).mean()

# Bills (£, each including its own tariff's standing charge)
flux_standing_charge_gbp = flux_standing_charge * n_days / 100
svt_standing_charge_gbp = price_cap_standing_charge * n_days / 100

bill_svt_no_solar = demand_kwh * price_cap_rate / 100 + svt_standing_charge_gbp
bill_flux_no_solar = (run["demand_kwh"] * run["import_price_p_per_kwh"]).sum() / 100 + flux_standing_charge_gbp
bill_solar_battery = run["cost_p"].sum() / 100 + flux_standing_charge_gbp

saving_tariff_only = bill_svt_no_solar - bill_flux_no_solar
saving_vs_svt = bill_svt_no_solar - bill_solar_battery
saving_vs_flux = bill_flux_no_solar - bill_solar_battery

print(f"Demand:                  {demand_kwh:,.0f} kWh")
print(f"Solar generation:        {generation_kwh:,.0f} kWh")
print(f"  Curtailed:             {curtailed_kwh:,.0f} kWh")
print(f"Battery charged:         {charged_kwh:,.0f} kWh")
print(
    f"Battery discharged:      {discharged_kwh:,.0f} kWh "
    f"({equivalent_full_cycles:.0f} full cycles, {equivalent_full_cycles / n_days:.2f} per day)"
)
print(f"Battery losses:          {battery_losses_kwh:,.0f} kWh")
print(f"Days at throughput limit: {days_at_throughput_limit:.0%}")
print(f"Grid import:             {imported_kwh:,.0f} kWh")
print(f"Grid export:             {exported_kwh:,.0f} kWh")
print(f"Net import:              {net_import_kwh:,.0f} kWh")
print()
print(f"Import cost:             £{run['import_cost_p'].sum() / 100:,.2f}")
print(f"Export revenue:          £{run['export_revenue_p'].sum() / 100:,.2f}")
print(f"Standing charge (Flux):  £{flux_standing_charge_gbp:,.2f}")
print()
print(f"Bill, no solar or battery (SVT):   £{bill_svt_no_solar:,.2f}")
print(
    f"Bill, no solar or battery (Flux):  £{bill_flux_no_solar:,.2f}   "
    f"saving £{saving_tariff_only:,.2f} ({saving_tariff_only / bill_svt_no_solar:.0%}) from the tariff change alone"
)
print(
    f"Bill, solar + battery (Flux):      £{bill_solar_battery:,.2f}   "
    f"saving £{saving_vs_svt:,.2f} ({saving_vs_svt / bill_svt_no_solar:.0%}) vs SVT, "
    f"£{saving_vs_flux:,.2f} ({saving_vs_flux / bill_flux_no_solar:.0%}) vs Flux"
)

# %% [markdown]
# ## 4. Running the solar and battery system with cost optimiser, without exports

# %%
optimisation_no_export_run = my_system.run_solar_cost_optimisation_no_export(
    demand_profile=demand_profile,
    import_price_profile=flux_import_price_profile,
)

# %%
run = optimisation_no_export_run
battery = my_system.battery

# Annual energy totals (kWh)
demand_kwh = run["demand_kwh"].sum()
generation_kwh = run["solar_generation_kwh"].sum()
charged_kwh = run["charge_kwh"].sum()  # drawn into the battery, from solar and grid combined
discharged_kwh = run["discharge_kwh"].sum()  # delivered from the battery to the home
curtailed_kwh = run["curtailed_kwh"].sum()  # surplus solar that is lost, as export is not allowed
imported_kwh = run["grid_import_kwh"].sum()  # includes any energy bought to charge the battery

# With no export, all solar that is not curtailed is used on site, directly or via the battery
solar_used_kwh = generation_kwh - curtailed_kwh
solar_used_rate = solar_used_kwh / generation_kwh

# Battery losses: energy in = energy out + losses + change in stored energy
soc_change_kwh = run["soc_kwh"].iloc[-1] - battery.starting_charge_kwh
battery_losses_kwh = charged_kwh - discharged_kwh - soc_change_kwh

# Battery use
n_days = run.index.normalize().nunique()
equivalent_full_cycles = (discharged_kwh / battery.discharge_efficiency) / battery.usable_capacity_kwh
daily_throughput_kwh = (run["discharge_kwh"] / battery.discharge_efficiency).groupby(run.index.normalize()).sum()
days_at_throughput_limit = (daily_throughput_kwh >= battery.max_daily_throughput_kwh - 1e-6).mean()

# Bills (£, each including its own tariff's standing charge)
flux_standing_charge_gbp = flux_standing_charge * n_days / 100
svt_standing_charge_gbp = price_cap_standing_charge * n_days / 100

bill_svt_no_solar = demand_kwh * price_cap_rate / 100 + svt_standing_charge_gbp
bill_flux_no_solar = (run["demand_kwh"] * run["import_price_p_per_kwh"]).sum() / 100 + flux_standing_charge_gbp
bill_solar_battery = run["cost_p"].sum() / 100 + flux_standing_charge_gbp

saving_tariff_only = bill_svt_no_solar - bill_flux_no_solar
saving_vs_svt = bill_svt_no_solar - bill_solar_battery
saving_vs_flux = bill_flux_no_solar - bill_solar_battery

print(f"Demand:                  {demand_kwh:,.0f} kWh")
print(f"Solar generation:        {generation_kwh:,.0f} kWh")
print(f"  Used on site:          {solar_used_kwh:,.0f} kWh ({solar_used_rate:.0%} of generation)")
print(f"  Curtailed:             {curtailed_kwh:,.0f} kWh ({1 - solar_used_rate:.0%} of generation)")
print(f"Battery charged:         {charged_kwh:,.0f} kWh")
print(
    f"Battery discharged:      {discharged_kwh:,.0f} kWh "
    f"({equivalent_full_cycles:.0f} full cycles, {equivalent_full_cycles / n_days:.2f} per day)"
)
print(f"Battery losses:          {battery_losses_kwh:,.0f} kWh")
print(f"Days at throughput limit: {days_at_throughput_limit:.0%}")
print(f"Grid import:             {imported_kwh:,.0f} kWh")
print()
print(f"Import cost:             £{run['import_cost_p'].sum() / 100:,.2f}")
print(f"Standing charge (Flux):  £{flux_standing_charge_gbp:,.2f}")
print()
print(f"Bill, no solar or battery (SVT):   £{bill_svt_no_solar:,.2f}")
print(
    f"Bill, no solar or battery (Flux):  £{bill_flux_no_solar:,.2f}   "
    f"saving £{saving_tariff_only:,.2f} ({saving_tariff_only / bill_svt_no_solar:.0%}) from the tariff change alone"
)
print(
    f"Bill, solar + battery (Flux):      £{bill_solar_battery:,.2f}   "
    f"saving £{saving_vs_svt:,.2f} ({saving_vs_svt / bill_svt_no_solar:.0%}) vs SVT, "
    f"£{saving_vs_flux:,.2f} ({saving_vs_flux / bill_flux_no_solar:.0%}) vs Flux"
)

# %%
