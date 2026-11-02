"""
Getters for solar energy radiation and PV performance data from PVGIS.

Documentation: https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/using-pvgis-5/api-non-interactive-service_en
"""

import pandas as pd
import requests

# Years covered by the PVGIS data version in use
PVGIS_FIRST_YEAR = 2005
PVGIS_LAST_YEAR = 2023


def get_hourly_radiation_data(
    lat: float,
    lon: float,
    peakpower: float,
    loss: float = 14,
    mountingplace: str = "building",
    aspect: int = 0,  # 0 = south facing
    angle: int = 35,  # optimal tilt
) -> pd.DataFrame:

    params = {
        "lat": lat,
        "lon": lon,
        "peakpower": peakpower,  # kWp set system size here
        "loss": loss,  # typical system losses %
        "outputformat": "json",
        "hourly": 1,
        "pvcalculation": 1,
        "mountingplace": mountingplace,
        "aspect": aspect,
        "angle": angle,
    }

    url = "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc"

    response = requests.get(url, params=params)
    data = response.json()

    return pd.DataFrame(data["outputs"]["hourly"])
