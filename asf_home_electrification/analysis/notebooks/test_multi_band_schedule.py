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
#     display_name: asf-home-electrification (3.13.16)
#     language: python
#     name: python3
# ---

# %%
from asf_home_electrification.tariffs.schedule import BandWindow, MultiBandSchedule

# %% [markdown]
# ## London DUOS Schedule Example

# %%
# As presented in Annex 2
schedule = MultiBandSchedule(
        default_band_name="Green",
        default_weight=0.0,
        windows=[
            BandWindow("Red", "11:00", "14:00", weight=1.0, days_of_week=(0, 1, 2, 3, 4)),
            BandWindow("Red", "16:00", "19:00", weight=1.0, days_of_week=(0, 1, 2, 3, 4)),
            BandWindow("Amber", "07:00", "11:00", weight=0.5, days_of_week=(0, 1, 2, 3, 4)),
            BandWindow("Amber", "14:00", "16:00", weight=0.5, days_of_week=(0, 1, 2, 3, 4)),
            BandWindow("Amber", "19:00", "23:00", weight=0.5, days_of_week=(0, 1, 2, 3, 4)),
        ],
    )

# %%
weights = schedule.generate_weights("2026-11-25", "2026-11-25")

# %%
weights.plot()

# %%
# As simplified
schedule = MultiBandSchedule(
        default_band_name="Green",
        default_weight=0.0,
        windows=[
            BandWindow("Red", "11:00", "14:00", weight=1.0, days_of_week=(0, 1, 2, 3, 4)),
            BandWindow("Red", "16:00", "19:00", weight=1.0, days_of_week=(0, 1, 2, 3, 4)),
            BandWindow("Amber", "07:00", "23:00", weight=0.5, days_of_week=(0, 1, 2, 3, 4)),
        ],
    )

# %%
weights = schedule.generate_weights("2026-11-25", "2026-11-25")

# %%
weights.plot()

# %% [markdown]
# ## North Wales and Mersey Schedule Example

# %%
# As presented in Annex 2
schedule = MultiBandSchedule(
        default_band_name="Green",
        default_weight=0.25,
        windows=[
            BandWindow("Red", "16:30", "19:30", weight=1.0, days_of_week=(0, 1, 2, 3, 4)),
            BandWindow("Amber", "08:00", "16:30", weight=0.5, days_of_week=(0, 1, 2, 3, 4)),
            BandWindow("Amber", "19:30", "22:30", weight=0.5, days_of_week=(0, 1, 2, 3, 4)),
            BandWindow("Amber", "16:00", "20:00", weight=0.5, days_of_week=(5, 6)),
        ],
    )

# %%
weights = schedule.generate_weights("2026-11-25", "2026-11-25")

# %%
weights.plot()

# %%
weights = schedule.generate_weights("2026-11-22", "2026-11-22")

# %%
weights.plot()
