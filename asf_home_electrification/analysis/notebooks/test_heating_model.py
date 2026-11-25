# %%
import pandas as pd

from asf_home_electrification.core_model.dwelling import Dwelling
from asf_home_electrification.core_model.heating import HeatingSystem
from asf_home_electrification.core_model.hot_water import HotWaterTank
from asf_home_electrification.core_model.thermostat import Thermostat

# %% [markdown]
# Developing the model:
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

# %% [markdown]
# ## Instantiate the physical dwelling

# %%
my_dwelling = Dwelling(lat=52.5, lon=-1.5, archetype_name="gb.eng.sfh.05.gen.reex.001.003")
my_dwelling

# %%
# these are the heat loss coefficients and other relevant parmaters used in the heating model
my_dwelling.archetype_params

# %%
# outside temperature at this location
my_dwelling.outside_temperature

# %%
len(my_dwelling.outside_temperature.loc["2023-03-26"])  # clocks went forward

# %%
len(my_dwelling.outside_temperature.loc["2023-10-29"])  # clocks went back

# %%
# solar irradiance on windows of this building
my_dwelling.solar_irradiance

# %% [markdown]
# ## Instantiate a heating system

# %%
heat_pump = HeatingSystem(type="heat pump", system_size_kw=8.0, fuel="electricity", efficiency=3.0)
heat_pump

# %% [markdown]
# Run the heating balance for a single half-hour period

# %%
# Run heat balance for one half-hour step
half_hour = heat_pump.run_heat_balance_half_hour(
    dwelling=my_dwelling, t_start=16.0, t_target=21.0, step_start="2026-01-01 00:00:00"
)
half_hour

# %% [markdown]
# Run the heating balance for a whole day (48 half-hour periods)

# %% [markdown]
# 1. With constant setpoint temperature and no pre-heating (t_target == t_setpoint)

# %%
start, end = "2026-01-01 00:00:00", "2026-01-02 00:00:00"

thermostat = Thermostat(
    weekday={"00:00": 21.0},
    weekend={"00:00": 21.0},
)

t_setpoint = thermostat.setpoints(start, end)  # 21.0 for every step
t_target = thermostat.targets(t_setpoint)  # identical, no preheating

# %%
# Dummy t_target and t_setpoint series
start, end = "2026-01-01 00:00:00", "2026-01-02 00:00:00"
index = pd.date_range(start, end, freq="30min", inclusive="left", tz="Europe/London")

# Simplest: constant 21 degC all day
t_setpoint = pd.Series(21.0, index=index, name="t_setpoint_c")
t_target = t_setpoint.copy()

# %%
# Run heat balance for a full day
day = heat_pump.run_heat_balance_time_period(
    dwelling=my_dwelling,
    start=start,
    end=end,
    t_initial=16.0,
    t_target=t_target,
    t_setpoint=t_setpoint,
)

# %%
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

t_out = day["t_out_c"]  # outside temperature for each step, already on day.index

step = pd.Timedelta("30min")
capacity_kwh = heat_pump.system_size_kw * 0.5  # max thermal energy per half hour

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

# --- Panel 1: temperatures ---
ax1.step(day.index, day["t_setpoint_c"], where="post", label="Setpoint (comfort)", color="tab:green", ls="--")
ax1.step(day.index, day["t_target_c"], where="post", label="Heating target", color="tab:orange")
ax1.plot(day.index, day["t_start_c"], label="Indoor temperature", color="tab:red", lw=2)
ax1.plot(day.index, t_out, label="Outside temperature", color="tab:blue", alpha=0.7)
ax1.set_ylabel("Temperature (°C)")
ax1.legend(loc="center right")
ax1.grid(alpha=0.3)

# --- Panel 2: heat delivered per half hour ---
ax2.bar(day.index, day["q_delivered_kwh"], width=step, align="edge", label="Delivered", color="tab:red", alpha=0.8)
ax2.bar(
    day.index,
    day["q_unmet_kwh"],
    width=step,
    align="edge",
    bottom=day["q_delivered_kwh"],
    label="Unmet demand",
    color="lightgrey",
)
ax2.axhline(capacity_kwh, color="black", ls=":", label=f"Heat pump max ({capacity_kwh:.1f} kWh)")
ax2.set_ylabel("Energy per half hour (kWh)")
ax2.legend(loc="upper right")
ax2.grid(alpha=0.3)

# tz= makes the axis show London clock time (including BST)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz="Europe/London"))
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2, tz="Europe/London"))
fig.suptitle(f"Heat balance, {day.index[0]:%d %b %Y} ({day['q_delivered_kwh'].sum():.1f} kWh delivered)")
fig.tight_layout()
plt.show()

# %% [markdown]
# 2. With a realistic thermostat schedule (t_setpoint on a schedule)

# %%
start, end = "2026-01-05 00:00:00", "2026-01-12 00:00:00"

thermostat = Thermostat(
    weekday={"00:00": 16.0, "06:30": 21.0, "08:30": 16.0, "17:00": 21.0, "22:30": 16.0},
    weekend={"00:00": 16.0, "07:30": 21.0, "22:30": 16.0},
)

t_setpoint = thermostat.setpoints(start, end)
t_target = thermostat.targets(t_setpoint)  # identical, no preheating

# %%
# Run heat balance for a full week
week_on_setpoint_schedule = heat_pump.run_heat_balance_time_period(
    dwelling=my_dwelling,
    start=start,
    end=end,
    t_initial=16.0,
    t_target=t_target,
    t_setpoint=t_setpoint,
)


# %%
def plot_heat_balance(df: pd.DataFrame, heat_pump, title: str | None = None):
    """Plot temperatures (top) and heat delivered (bottom) for any period."""
    step = pd.Timedelta("30min")
    capacity_kwh = heat_pump.system_size_kw * 0.5  # max thermal energy per half hour
    n_days = (df.index[-1] - df.index[0]).days + 1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    # --- Panel 1: temperatures ---
    ax1.step(df.index, df["t_setpoint_c"], where="post", label="Setpoint (comfort)", color="tab:green", ls="--")
    ax1.plot(df.index, df["t_start_c"], label="Indoor temperature", color="tab:red", lw=1.8)
    ax1.plot(df.index, df["t_out_c"], label="Outside temperature", color="tab:blue", alpha=0.7)
    if not df["t_target_c"].equals(df["t_setpoint_c"]):  # only draw when preheating differs
        ax1.step(df.index, df["t_target_c"], where="post", label="Heating target", color="tab:orange", alpha=0.8)
    # shade where the house is below the comfort setpoint
    below = df["t_start_c"] < df["t_setpoint_c"] - 0.1
    ax1.fill_between(
        df.index,
        df["t_start_c"],
        df["t_setpoint_c"],
        where=below,
        step="post",
        color="tab:red",
        alpha=0.2,
        label="Below setpoint",
    )
    ax1.set_ylabel("Temperature (°C)")
    ax1.legend(loc="upper right", ncol=2, fontsize=9)
    ax1.grid(alpha=0.3)

    # --- Panel 2: heat delivered per half hour ---
    ax2.bar(df.index, df["q_delivered_kwh"], width=step, align="edge", label="Delivered", color="tab:red", alpha=0.8)
    ax2.axhline(capacity_kwh, color="black", ls=":", label=f"Heat pump max ({capacity_kwh:.1f} kWh)")
    ax2.set_ylabel("Energy per half hour (kWh)")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(alpha=0.3)

    # --- Time axis in London clock time ---
    if n_days <= 2:
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2, tz="Europe/London"))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz="Europe/London"))
    else:
        ax2.xaxis.set_major_locator(mdates.DayLocator(tz="Europe/London"))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b", tz="Europe/London"))

    total = df["q_delivered_kwh"].sum()
    fig.suptitle(title or f"Heat balance, {df.index[0]:%d %b} to {df.index[-1]:%d %b %Y} ({total:.0f} kWh delivered)")
    fig.tight_layout()
    plt.show()


plot_heat_balance(week_on_setpoint_schedule, heat_pump)

# %% [markdown]
# 3. With a realistic thermostat schedule (t_setpoint on a schedule) AND preheating behaviour

# %%
start, end = "2026-01-05 00:00:00", "2026-01-12 00:00:00"

thermostat = Thermostat(
    weekday={"00:00": 16.0, "06:30": 21.0, "08:30": 16.0, "17:00": 21.0, "22:30": 16.0},
    weekend={"00:00": 16.0, "07:30": 21.0, "22:30": 16.0},
    preheat_minutes=60,
)

t_setpoint = thermostat.setpoints(start, end)
t_target = thermostat.targets(t_setpoint)  # 1h preheating

# %%
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

LONDON = "Europe/London"


def plot_setpoint_and_target(t_setpoint: pd.Series, t_target: pd.Series, title: str | None = None):
    """Plot the comfort setpoint against the heating target, shading the preheat periods."""
    fig, ax = plt.subplots(figsize=(14, 4))

    ax.step(t_setpoint.index, t_setpoint, where="post", label="Setpoint (comfort)", color="tab:green", ls="--", lw=2)
    ax.step(t_target.index, t_target, where="post", label="Heating target", color="tab:orange", lw=1.8)

    # shade the preheat windows (target above setpoint)
    preheating = t_target > t_setpoint
    # fill_between stops at the x of the last True point, so include the step after
    # each window (target equals setpoint there, so the fill has zero height)
    fill_mask = preheating | preheating.shift(1, fill_value=False)
    ax.fill_between(
        t_setpoint.index,
        t_setpoint,
        t_target,
        where=fill_mask,
        step="post",
        color="tab:orange",
        alpha=0.3,
        label="Preheat",
    )

    span = t_setpoint.index[-1] - t_setpoint.index[0]
    if span <= pd.Timedelta("2D"):
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2, tz=LONDON))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=LONDON))
    else:
        ax.xaxis.set_major_locator(mdates.DayLocator(tz=LONDON))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b", tz=LONDON))

    # headroom above the lines so the legend doesn't sit on top of them
    low = min(t_setpoint.min(), t_target.min())
    high = max(t_setpoint.max(), t_target.max())
    ax.set_ylim(low - 0.5, high + 1.2)

    ax.set_ylabel("Temperature (°C)")
    ax.set_title(title or "Setpoint vs heating target")
    ax.legend(loc="upper right", ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()


plot_setpoint_and_target(t_setpoint, t_target)  # whole week
plot_setpoint_and_target(t_setpoint.loc["2026-01-06"], t_target.loc["2026-01-06"])  # one weekday

# %%
# Run heat balance for a full week
week_on_setpoint_schedule_preheating = heat_pump.run_heat_balance_time_period(
    dwelling=my_dwelling,
    start=start,
    end=end,
    t_initial=16.0,
    t_target=t_target,
    t_setpoint=t_setpoint,
)

# %%
plot_heat_balance(week_on_setpoint_schedule_preheating, heat_pump)

# %% [markdown]
# 4. For the whole winter

# %%
# Run heat balance for the whole winter
start, end = "2026-10-01", "2027-04-01"  # UK local time, end exclusive

thermostat = Thermostat(
    weekday={"00:00": 16.0, "06:30": 21.0, "08:30": 16.0, "17:00": 21.0, "22:30": 16.0},
    weekend={"00:00": 16.0, "07:30": 21.0, "22:30": 16.0},
    preheat_minutes=60,
)

t_setpoint = thermostat.setpoints(start, end)
t_target = thermostat.targets(t_setpoint)  # 1h preheating

winter_preheating = heat_pump.run_heat_balance_time_period(
    dwelling=my_dwelling,
    start=start,
    end=end,
    t_initial=18.0,  # closer to typical than 16, so the first morning isn't distorted
    t_target=t_target,
    t_setpoint=t_setpoint,
)

# %%
total_kwh = winter_preheating["q_delivered_kwh"].sum()
print(f"{total_kwh:.0f} kWh")

# %% [markdown]
# # Including domestic hot water demand

# %% [markdown]
# Instantiate the hot water tank as a separate object

# %%
tank = HotWaterTank(
    volume_litres=210,
    t_cold_feed=10.0,
    t_hot_water=60.0,
    t_room=20.0,  # degC, cylinder in an airing cupboard or utility room
    ref_standing_loss_kwh_per_day=1.0,
    ref_delta_t=45.0,  # K, assumed 65 degC tank, 20 degC room
)
tank

# %%
# Tank at target, no draw: only replaces standing loss (about 0.018 kWh)
tank.run_half_hour_step(t_start=60.0, draw_litres=0, heating_system_size_kw=8.0)

# %%
# Cold tank, plenty of power: should end at exactly 60.0
tank.run_half_hour_step(t_start=40.0, draw_litres=0, heating_system_size_kw=20.0)["t_tank_end_c"]

# %% [markdown]
# Add a hot water tank to the instantiated heating system

# %%
# test type validation
heat_pump.tank = 5

# %%
heat_pump.tank = tank
heat_pump.efficiency_dhw = 2.2
heat_pump

# %% [markdown]
# 1. Example of combined space and hot water heating in one half hour timestep

# %%
heat_pump.run_heat_balance_half_hour(
    dwelling=my_dwelling, t_start=16.0, t_target=21.0, step_start="2026-01-01 00:00:00"
)

# %% [markdown]
# 2. Full week combined space and domestic hot water
# - Space heating: With a realistic thermostat schedule (t_setpoint on a schedule) preheating behaviour
# - Domestic hot water heating: With a realistic draw schedule

# %%
# Thermostat settings
start, end = "2026-01-05 00:00:00", "2026-01-12 00:00:00"

thermostat = Thermostat(
    weekday={"00:00": 16.0, "06:30": 21.0, "08:30": 16.0, "17:00": 21.0, "22:30": 16.0},
    weekend={"00:00": 16.0, "07:30": 21.0, "22:30": 16.0},
    preheat_minutes=60,
)

t_setpoint = thermostat.setpoints(start, end)
t_target = thermostat.targets(t_setpoint)  # 1h preheating

# %%
# DHW draws
DRAW_LITRES = {
    "06:30": 12,
    "07:00": 18,
    "07:30": 10,  # morning showers: 40 L
    "12:30": 8,
    "13:00": 4,  # lunchtime: 12 L
    "18:00": 12,
    "18:30": 10,
    "19:00": 6,  # cooking, washing up: 28 L
    "20:30": 10,
    "21:00": 14,
    "21:30": 4,  # evening: 28 L
}

draws = pd.Series(
    [DRAW_LITRES.get(ts.strftime("%H:%M"), 0.0) for ts in t_setpoint.index],
    index=t_setpoint.index,
    name="dhw_draw_litres",
)

# %%
# Run heat balance for a full week
week_on_setpoint_schedule_preheating_with_dhw = heat_pump.run_heat_balance_time_period(
    dwelling=my_dwelling,
    start=start,
    end=end,
    t_initial=16.0,
    t_target=t_target,
    t_setpoint=t_setpoint,
    dhw_draw_litres=draws,
)

# %%
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

LONDON = "Europe/London"
STEP = pd.Timedelta("30min")


def plot_heat_balance_with_dhw(df: pd.DataFrame, heat_pump, draws: pd.Series | None = None, title: str | None = None):
    """Temperatures, space + hot water energy, and tank state for any period of results."""
    capacity_kwh = heat_pump.system_size_kw * STEP.total_seconds() / 3600
    fig, (ax_temp, ax_energy, ax_tank) = plt.subplots(
        3, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1.3, 1]}
    )

    # --- Panel 1: room temperatures ---
    ax_temp.step(df.index, df["t_setpoint_c"], where="post", label="Setpoint (comfort)", color="tab:green", ls="--")
    if not df["t_target_c"].equals(df["t_setpoint_c"]):
        ax_temp.step(df.index, df["t_target_c"], where="post", label="Heating target", color="tab:orange", alpha=0.8)
    ax_temp.plot(df.index, df["t_start_c"], label="Indoor temperature", color="tab:red", lw=1.8)
    ax_temp.plot(df.index, df["t_out_c"], label="Outside temperature", color="tab:blue", alpha=0.7)
    ax_temp.set_ylabel("Temperature (°C)")
    ax_temp.set_ylim(top=df["t_setpoint_c"].max() + 3)  # headroom for the legend
    ax_temp.legend(loc="upper right", ncol=2, fontsize=9)

    # --- Panel 2: heat delivered, space heating and hot water stacked ---
    ax_energy.bar(
        df.index, df["q_delivered_kwh"], width=STEP, align="edge", label="Space heating", color="tab:red", alpha=0.8
    )
    ax_energy.bar(
        df.index,
        df["q_dhw_kwh"],
        width=STEP,
        align="edge",
        bottom=df["q_delivered_kwh"],
        label="Hot water",
        color="tab:purple",
        alpha=0.8,
    )
    ax_energy.axhline(capacity_kwh, color="black", ls=":", label=f"Heat pump max ({capacity_kwh:.1f} kWh)")
    ax_energy.set_ylabel("Heat per half hour (kWh)")
    ax_energy.legend(loc="upper right", ncol=3, fontsize=9)

    # --- Panel 3: hot water tank ---
    ax_tank.plot(df.index, df["t_tank_end_c"], color="tab:purple", lw=1.8, label="Tank temperature (end of step)")
    ax_tank.set_ylabel("Tank (°C)")
    ax_tank.legend(loc="lower left", fontsize=9)
    if draws is not None:
        ax_draw = ax_tank.twinx()
        d = draws.reindex(df.index)
        ax_draw.bar(d.index, d, width=STEP, align="edge", color="tab:cyan", alpha=0.5, label="Hot water drawn")
        ax_draw.set_ylabel("Drawn (L per half hour)")
        ax_draw.legend(loc="lower right", fontsize=9)

    for ax in (ax_temp, ax_energy, ax_tank):
        ax.grid(alpha=0.3)

    if df.index[-1] - df.index[0] <= pd.Timedelta("2D"):
        ax_tank.xaxis.set_major_locator(mdates.HourLocator(interval=2, tz=LONDON))
        ax_tank.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=LONDON))
    else:
        ax_tank.xaxis.set_major_locator(mdates.DayLocator(tz=LONDON))
        ax_tank.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b", tz=LONDON))

    total = df["q_delivered_kwh"].sum() + df["q_dhw_kwh"].sum()
    fig.suptitle(title or f"{df.index[0]:%d %b} to {df.index[-1]:%d %b %Y}: {total:.0f} kWh heat delivered")
    fig.tight_layout()
    plt.show()


plot_heat_balance_with_dhw(week_on_setpoint_schedule_preheating_with_dhw, heat_pump, draws)  # whole week

# %% [markdown]
# 3. Domestic hot water on its own schedule too

# %%
dhw_thermostat = Thermostat(
    weekday={"00:00": 45.0, "01:00": 60.0, "03:00": 45.0},  # reheat to 60 degC overnight, floor of 45 otherwise
    weekend={"00:00": 45.0, "01:00": 60.0, "03:00": 45.0},
)
t_dhw_target = dhw_thermostat.setpoints(start, end)

week_on_setpoint_schedule_preheating_with_dhw_scheduled = heat_pump.run_heat_balance_time_period(
    dwelling=my_dwelling,
    start=start,
    end=end,
    t_initial=16.0,
    t_target=t_target,
    t_setpoint=t_setpoint,
    dhw_draw_litres=draws,
    t_dhw_target=t_dhw_target,  # this is now a time series, before it was constant at 60C
    t_tank_initial=60.0,
)

# %%
plot_heat_balance_with_dhw(week_on_setpoint_schedule_preheating_with_dhw_scheduled, heat_pump, draws)  # whole week
