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
import io
import pandas
import matplotlib.pyplot as pyplot

from asf_home_electrification.tariffs.schedule import SimplePeakSchedule
from asf_home_electrification.tariffs.cost_curve import TransmissionCostCurve


# %%
def settlement_datetime_from_ids(df: pandas.DataFrame, date_col: str='Settlement Date', settlement_id_col: str='Settlement Period ID') -> pandas.Series:
    """Helper to generate datetime series from date and settlement period IDs."""

    # Create Europe/London midnights as starting points to allow for GMT/BST
    midnights = df[date_col].dt.tz_localize('Europe/London')

    # Workout time deltas to add to midnights
    offset = pandas.to_timedelta((df[settlement_id_col] - 1) * 30, unit='min')

    timestamps = (midnights.dt.tz_convert('UTC') + offset).dt.tz_convert("Europe/London")

    return pandas.Series(timestamps)


# %%
nhh_text = """zone,gsp_group,NHH Demand Tariff (p/kWh),domestic_standard_meters,domestic_e7_meters,domestic_standard_consumption_GWh,domestic_e7_consumption_GWh,peak_domestic_standard_consumption_GWh,peak_domestic_e7_consumption_GWh,revenue_£
Northern Scotland,_P,0.0,669536.060304745,123352.89766417182,2194.3551902094696,732.1801466290792,382.69093702663207,104.68784954517237,0.0
Southern Scotland,_N,0.0,1830609.417158052,185029.61085447716,5104.995400481857,787.2827240813713,890.3004773536949,112.56647117187855,0.0
Northern,_F,0.0,1486801.1375779673,80237.36130160675,4230.465318176676,400.948833849125,737.7842675128442,57.32806521258174,0.0
North West,_G,0.0,2174695.6769089764,157475.84569761934,6677.692268581517,780.9033455264454,1164.5755084868524,111.65434124671668,0.0
Yorkshire,_M,0.0,2094539.85826202,122081.02404815907,6238.533872747548,565.009516021211,1087.9872064860397,80.78562561022876,0.0
N Wales & Mersey,_D,0.0,1379456.2637035327,83797.920664326,4208.956561383604,432.775290407946,734.0331854965216,61.878643797818526,0.0
East Midlands,_B,0.0,2039356.2223405784,653769.5183578833,6384.305714226201,2584.288254862772,1113.4095095190496,369.50411896853404,0.0
Midlands,_E,0.358668,2192916.2211775864,248533.22033639607,7060.7059879059125,1193.528661934709,1231.3722968082166,170.65192161208662,5028612.223723733
Eastern,_A,0.00914,2879732.9344088654,756899.1899809596,9647.747827315896,3405.1259033470737,1682.546960813455,486.86830678646794,198284.55545863297
South Wales,_K,0.739032,1025369.935682933,49277.45200205313,3040.8956078141305,254.88374750512165,530.325808122008,36.4435331020298,4188606.7978348304
South East,_J,0.63342,1878614.9705650045,377232.7347682088,6337.881988875788,1673.8817503094529,1105.3133093077608,239.33328654099589,8517260.467425196
London,_C,0.700677,2084119.8751033207,167884.8113923377,6272.7236317455945,800.1973183035537,1093.949828656805,114.41301276836042,8466720.506412607
Southern,_H,1.00812,2745512.7878788644,286763.8212275142,9385.654716396231,1605.1744809348122,1636.8384726645982,229.50945245854038,18815026.702751383
South Western,_L,2.071095,1360319.6389275542,192272.5917042872,4406.482141047758,1037.2479349505106,768.4812317862328,148.30674698715896,18987549.988976777"""

# Revenue estimates for 2026/27 TNUoS tariff
nhh_demand_charge = pandas.read_csv(io.StringIO(nhh_text))

# %%
# Get profiles 1 and 2 which cover 
# tnuos years are financial year.
profile1 = (pandas.read_csv("../../../inputs/Default_Period_Profile_Class_Coefficient_380.csv", parse_dates=['Settlement Date'], date_format='%d/%m/%Y')
            .loc[lambda df: (df['Profile Class ID'] == 1) & (df['Settlement Date'] >= '2026-04-01'), :])

profile2 = (pandas.read_csv("../../../inputs/Default_Period_Profile_Class_Coefficient_380.csv", parse_dates=['Settlement Date'], date_format='%d/%m/%Y')
            .loc[lambda df: (df['Profile Class ID'] == 2) & (df['Settlement Date'] >= '2026-04-01'), :])

# %% [markdown]
# ### Current TNUoS Charges
#
# Obviously here we're basically calculating the values and then checking that the model produces the same results. However, it does allow us to then manipulate transmisison costs from that baseline.

# %%
ccs = []

for row in nhh_demand_charge.itertuples(index=False):
    # TNUoS revenues by DNO licence area
    loc_rev = row[9]
    # Total residual revenue
    residual_rev = 0.222566 * 365 * nhh_demand_charge[['domestic_standard_meters', 'domestic_e7_meters']].sum().sum()

    # Consumption accounting for profile 1 and 2 sources.
    consumption = (
        profile1
        .assign(datetime = settlement_datetime_from_ids(profile1))
        .set_index('datetime')
        .loc[:, 'Default Period Profile Class Coefficient'] * row.domestic_standard_consumption_GWh * 1_000_000
        + profile2
        .assign(datetime = settlement_datetime_from_ids(profile2)).set_index('datetime')
        .loc[:, 'Default Period Profile Class Coefficient'] * row.domestic_e7_consumption_GWh * 1_000_000
        )

    cc = TransmissionCostCurve((loc_rev + residual_rev), (residual_rev / (loc_rev + residual_rev)), 1, SimplePeakSchedule("16:00", "19:00"))

    ccs.append(cc.calculate_cost_curve(consumption))
else:
    standing_charge = cc.calculate_standing_charge(nhh_demand_charge[['domestic_standard_meters', 'domestic_e7_meters']].sum().sum())

# %%
f, ax = pyplot.subplots()

for cc in ccs:
    cc.groupby(cc.index.hour).mean()["total"].plot(ax=ax)

ax.set_ylabel("Transmission Costs (p/kWh)")
ax.set_xlabel("Time of Day")

ax.set_title(f"Current TNUoS, Standing charge: {standing_charge}p/meter/day")

# %% [markdown]
# ### Transmission costs with shifted peak and partial volumetric costs

# %%
ccs = []

for row in nhh_demand_charge.itertuples(index=False):
    # TNUoS revenues by DNO licence area
    loc_rev = row[9]
    # Total residual revenue
    residual_rev = 0.222566 * 365 * nhh_demand_charge[['domestic_standard_meters', 'domestic_e7_meters']].sum().sum()

    # Consumption accounting for profile 1 and 2 sources.
    consumption = (
        profile1
        .assign(datetime = settlement_datetime_from_ids(profile1))
        .set_index('datetime')
        .loc[:, 'Default Period Profile Class Coefficient'] * row.domestic_standard_consumption_GWh * 1_000_000
        + profile2
        .assign(datetime = settlement_datetime_from_ids(profile2)).set_index('datetime')
        .loc[:, 'Default Period Profile Class Coefficient'] * row.domestic_e7_consumption_GWh * 1_000_000
        )

    cc = TransmissionCostCurve((loc_rev + residual_rev), (residual_rev / (loc_rev + residual_rev)), 0.5, SimplePeakSchedule("10:00", "16:00"))

    ccs.append(cc.calculate_cost_curve(consumption, nhh_demand_charge[['domestic_standard_consumption_GWh', 'domestic_e7_consumption_GWh']].sum().sum() * 1_000_000))
else:
    standing_charge = cc.calculate_standing_charge(nhh_demand_charge[['domestic_standard_meters', 'domestic_e7_meters']].sum().sum())

# %%
f, ax = pyplot.subplots()

for cc in ccs:
    cc.groupby(cc.index.hour).mean()["total"].plot(ax=ax)

ax.set_ylabel("Transmission Costs (p/kWh)")
ax.set_xlabel("Time of Day")

ax.set_title(f"Current TNUoS, Standing charge: {standing_charge}p/meter/day")
