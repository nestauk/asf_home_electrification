# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: asf-home-electrification (3.13.x)
#     language: python
#     name: python3
# ---

# %%
import pandas

from asf_home_electrification.tariffs.schedule import SimplePeakSchedule

# %% [markdown]
# ## Functionality

# %%
# Create a simple schedule
schedule = SimplePeakSchedule(peak_start_time="16:00", peak_end_time="19:00")

# %%
# generate weights for a given period
day = schedule.generate_weights(start_date='2026-11-03', end_date='2026-11-03')

# %%
# Plot the schedule
day.plot()

# %%
# Create a weekday only schedule
schedule = SimplePeakSchedule(peak_start_time="16:00", peak_end_time="19:00", weekdays_only=True)

# %%
# generate weights for a given period
days = schedule.generate_weights(start_date='2026-11-01', end_date='2026-11-02')

# %%
days.plot()

# %%
# Create a winter only schedule
schedule = SimplePeakSchedule(peak_start_time="16:00", peak_end_time="19:00", winter_only=True)

# %%
winter = schedule.generate_weights(start_date= "2026-10-30", end_date= "2026-11-01")

# %%
winter.plot()

# %%
# whole year with winter
winter = schedule.generate_weights(start_date= "2026-04-01", end_date= "2027-03-31")

# %%
# Plot hourly summaries
winter.groupby(winter.index.hour).mean().plot()
winter.groupby(winter.index.hour).max().plot()
winter.groupby(winter.index.hour).min().plot()

# %% [markdown]
# ### Validation
#
# As a simple check that the schedule logic produces the same output as directly filtering some test data for electricity usage profiles. 

# %%
url = "https://assets.elexon.co.uk/wp-content/uploads/sites/11/2012/01/28163748/Average_Profiling_data_201314_evaluated@10yearNET_v1.0.xlsx"

profile1 = pandas.read_excel(url, engine='calamine', skiprows=2, nrows=48)

# %%
# NB for our usecase the basecase is spring
season_lookup = {'Smr': ['2026-05-16', '2026-07-24'],
                 'Hsr': ['2026-07-25', '2026-09-06'],
                 'Aut': ['2026-09-07', '2026-10-24'],
                 'Wtr': ['2026-10-25', '2027-03-27']}

# %%
# generate a schedule
schedule = SimplePeakSchedule(peak_start_time="16:00", peak_end_time="19:00").generate_weights(start_date='2026-04-01', end_date='2027-03-31').reset_index()


# %%
# NB this is very inefficient, but it doesn't really need to be any better.
def get_coefficient(row):
    season = 'Spr'
    for name, date_range in season_lookup.items():
        if row['index'] >= pandas.to_datetime(date_range[0]).tz_localize('Europe/London') and row['index'] <= pandas.to_datetime(date_range[1]).tz_localize('Europe/London'):
            season = name

    # Technically bank holidays aren't Wd, but I'm ignroing that here.
    day = {0:'Wd', 1: 'Wd', 2: 'Wd', 3: 'Wd', 4: 'Wd', 5: 'Sat', 6: 'Sun'}.get(row['index'].day_of_week)

    col = season + " " + day

    index = row['index'].time()
    
    return profile1.loc[lambda df: df['Settlement period'] == index, col].values[0]

schedule['coefficients'] = schedule.apply(get_coefficient, axis=1)
# normalise
schedule['coefficients'] = schedule['coefficients'] / schedule['coefficients'].sum()

# %%
# Using the weights, we think 16.922123565400846% of consumption is at peak
(schedule['coefficients'] * schedule['weight']).sum()

# %%
# Manual filter agrees with weights
schedule.loc[lambda df: (df['index'].dt.time >= pandas.to_datetime("16:00", format="%H:%M").time()) & (df['index'].dt.time < pandas.to_datetime("19:00", format="%H:%M").time()), 'coefficients'].sum() / schedule['coefficients'].sum()
