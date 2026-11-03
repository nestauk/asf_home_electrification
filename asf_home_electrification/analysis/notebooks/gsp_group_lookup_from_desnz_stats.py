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
import io

import geopandas
import pandas
import requests

# %% [markdown]
# ### DNO Licence Areas

# %%
url = "https://api.neso.energy/dataset/0e377f16-95e9-4c15-a1fc-49e06a39cfa0/resource/668df251-b8a5-4ac3-8427-20b93b2f5f67/download/gb-dno-license-areas-20240503-as-esri-shape-file.zip"

res = requests.get(url)

if res.status_code == 200:
    dno = geopandas.read_file(io.BytesIO(res.content))

# %% [markdown]
# ### DESNZ LSOA and LA Domestic Electricity Consumption

# %%
url = "https://assets.publishing.service.gov.uk/media/69427b7bd8156a816c419351/LSOA_domestic_elec_2010-2024.xlsx"

lsoa_elec = pandas.read_excel(url, engine='calamine', sheet_name="2024", skiprows=4).rename(columns=lambda col: col.replace('\n', ' ').replace("  ", " "))

# %%
# get LA consumption stats
url = "https://assets.publishing.service.gov.uk/media/69426e905431f4f94d7f0b5e/Subnational_electricity_consumption_statistics_2005-2024.xlsx"

la_stats = (pandas.read_excel(url, engine='calamine', sheet_name='2024', skiprows=4)
            .rename(columns=lambda col: col.replace('\n', ' ').replace("  ", " "))
            .loc[lambda df: ~(df['Code'].str.startswith('K') | df['Code'].str.startswith('E1')
                              | df['Code'].str.startswith('E9') | df['Code'].str.startswith('W9') | df['Code'].str.startswith('S9')
                              | df['Code'].str.startswith('Unallocated E') | df['Code'].str.startswith('Unallocated S'))]).reset_index(drop=True)

la_stats.loc[lambda df: df['Code'].str.startswith("Unallocated"), 'Code'] = "Unallocated"

# %% [markdown]
# ### LSOAs and DataZones

# %%
# lsoas in England and Wales
url = "https://geoportal.statistics.gov.uk/api/download/v1/items/b8263c2364e9452483a0e5783c6fdb53/shapefile?layers=0"

res = requests.get(url)

if res.status_code == 200:
    lsoas = geopandas.read_file(io.BytesIO(res.content)).loc[:,['LSOA21CD', 'LSOA21NM', 'geometry']].rename(columns={'LSOA21CD':'zonecode', 'LSOA21NM':'zonename'})

# %%
# Datazones in Scotland
url = "https://maps.gov.scot/ATOM/shapefiles/SG_DataZoneBdry_2022.zip"

res = requests.get(url)

if res.status_code == 200:
    dz = geopandas.read_file(io.BytesIO(res.content)).loc[:,['dzcode', 'dzname', 'geometry']].rename(columns={'dzcode': 'zonecode', 'dzname': 'zonename'})

# %%
# concat lsoas and dzs
zones = pandas.concat([lsoas, dz], ignore_index=True)
del lsoas, dz

# %% [markdown]
# ### Create LSOA/DZ-level apportionment weights
#
# NB I'm implementing some fairly arbitrary rules such that very small lsoa divisions are ignored.
#
# Specifically, if an LSOA is split between 2 DNO areas, for both to count the smaller sub-division must be at least 10% of the total LSOA area; and, if an lsoa is split into 3 DNO areas the areas much be at least 5% of the total lsoa area to count.

# %%
# Polygon overlay intersection of zones with dno areas
zones_overlay = zones.overlay(dno)


# %%
def dno_area_selector(df):
    if len(df) == 1:
        return {df['Name'].values[0]: 1}
    elif len(df) == 2:
        areas = df.area / df.area.sum()
        if areas.max() > 0.90:
            return {df.loc[lambda df: df.area == df.area.max(), 'Name'].values[0]: 1}
        return {df['Name'].values[0]: areas.values[0], df['Name'].values[1]: areas.values[1]}
    elif len(df) == 3:
        areas = df.area / df.area.sum()
        areas = areas.loc[lambda df: df > 0.05]
        if len(areas) == 3:
            return {df['Name'].values[0]: areas.values[0], df['Name'].values[1]: areas.values[1], df['Name'].values[2]: areas.values[2]}
        elif len(areas)  == 2:
            remainder = 1 - areas.sum()
            areas += remainder/2
            return {df['Name'].values[0]: areas.values[0], df['Name'].values[1]: areas.values[1]}
        elif len(areas) == 1:
            return {df['Name'].values[0]: 1}
        else:
            raise ValueError()
        

lookup = zones_overlay.groupby('zonecode').apply(dno_area_selector)

# For some reason the generalisation of DNO shapefile excludes two lsoas on Walney Island in Barrow-in-Furness
missing = pandas.Series(data = [{'_G':1},{'_G':1}], index = ['E01019185', 'E01033160'])
lookup = pandas.concat([lookup, missing]).apply(pandas.Series).loc[:, lambda df: df.columns.sort_values()].fillna(0)

# %% [markdown]
# ### Create domestic consumption totals (for 2024)

# %%
# merge lsoa electricity consumption with lookup
lsoa_elec = lsoa_elec.merge(lookup, left_on='LSOA code', right_index=True, how='left')

# %%
# Calculate domestic consumption totals (for 2024)
for gsp_group in ['_A', '_B', '_C', '_D', '_E', '_F', '_G', '_H', '_J', '_K', '_L', '_M', '_N', '_P']:
    lsoa_elec = lsoa_elec.assign(**{f'consumption{gsp_group}': lambda df: df[gsp_group] * df['Total consumption (kWh)']})

# %%
# Domestic consumption in kWh by GSP group
gsp_consumption = lsoa_elec[lsoa_elec.columns[lsoa_elec.columns.str.contains("consumption_")]].sum()


# %% [markdown]
# ### LA lookup
#
# We define the LA weights from the the lsoa weights.

# %%
def la_weights(df):
    out = {}
    for gsp_group in ['_A', '_B', '_C', '_D', '_E', '_F', '_G', '_H', '_J', '_K', '_L', '_M', '_N', '_P']:
        weight = (df['Total consumption (kWh)'] * df[gsp_group]).sum() / df['Total consumption (kWh)'].sum()
        out[gsp_group] = weight
    return out


la_elec = lsoa_elec.groupby('Local authority code').apply(la_weights).apply(pandas.Series)

# %%
la_elec = la_stats.merge(la_elec, how='left', left_on='Code', right_index=True)

# %%
# Calculate domestic consumption totals (for 2024)
for gsp_group in ['_A', '_B', '_C', '_D', '_E', '_F', '_G', '_H', '_J', '_K', '_L', '_M', '_N', '_P']:
    la_elec = la_elec.assign(**{f'domestic_standard_meters{gsp_group}': lambda df: df[gsp_group] * df['Number of meters (thousands): Domestic Standard'],
                                f'domestic_e7_meters{gsp_group}': lambda df: df[gsp_group] * df['Number of meters (thousands): Domestic E7'],
                                f'domestic_standard_consumption{gsp_group}': lambda df: df[gsp_group] * df['Total consumption (GWh): Domestic Standard'],
                                f'domestic_e7_consumption{gsp_group}': lambda df: df[gsp_group] * df['Total consumption (GWh): Domestic E7'],
                                })

# %%
gsp_group_stats = pandas.concat([
    (la_elec[la_elec.columns[la_elec.columns.str.contains("domestic_standard_meters")]].sum() * 1000).rename(index = lambda index: index[-2:]).rename("domestic_standard_meters"),
    (la_elec[la_elec.columns[la_elec.columns.str.contains("domestic_e7_meters")]].sum() * 1000).rename(index = lambda index: index[-2:]).rename("domestic_e7_meters"),
    (la_elec[la_elec.columns[la_elec.columns.str.contains("domestic_standard_consumption")]].sum()).rename(index = lambda index: index[-2:]).rename("domestic_standard_consumption_GWh"),
    (la_elec[la_elec.columns[la_elec.columns.str.contains("domestic_e7_consumption")]].sum()).rename(index = lambda index: index[-2:]).rename("domestic_e7_consumption_GWh"),
], axis=1).reset_index(names="gsp_group")

# %%
gsp_group_stats.to_csv("../../outputs/GSP_Group_consumption_estimates.csv")
