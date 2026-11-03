"""
Getters for solar energy radiation and PV performance data from PVGIS.

Documentation: https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/using-pvgis-5/api-non-interactive-service_en
"""

import pandas as pd
import requests

# Years covered by the PVGIS data version in use
PVGIS_FIRST_YEAR = 2005
PVGIS_LAST_YEAR = 2023


def get_hourly_pvgis_data(
    lat: float,
    lon: float,
    peakpower: float,
    loss: float,
    mountingplace: str,
    aspect: int,
    angle: int,
) -> pd.DataFrame:
    """Get hourly irradiance, temperature and estimated PV output for a fixed PV system from PVGIS.

    Calls the PVGIS 5.3 hourly radiation tool (`seriescalc`) with PV calculation switched on.

    Args:
        lat (float): Latitude in decimal degrees. South is negative.
        lon (float): Longitude in decimal degrees. West is negative.
        peakpower (float): Nominal power of the PV system in kW.
        loss (float): Sum of system losses in percent.
        mountingplace (str): Type of mounting of the PV modules. Choices are "free" for free-standing and
            "building" for building integrated.
        aspect (int): Orientation (azimuth) angle of the fixed PV system, in degrees, where 0 = south,
            90 = west and -90 = east.
        angle (int): Inclination angle of the fixed PV system from the horizontal plane, in degrees
            (0 = flat, 90 = vertical).

    Returns:
        pd.DataFrame: One row per hour, covering every year in the PVGIS database (full series, 2005-2023
        for this version). Columns:

            - time (str): time stamp, in UTC, formatted like "20230101:0011".
            - P (float): estimated PV system power output (W).
            - G(i) (float): global irradiance on the inclined plane (W/m2).
            - H_sun (float): sun height above the horizon (degrees).
            - T2m (float): air temperature at 2 m above ground (degC).
            - WS10m (float): wind speed at 10 m (m/s).
            - Int (int): 1 if the radiation value was reconstructed (interpolated), otherwise 0.

    Raises:
        KeyError: If PVGIS rejects the inputs. It then returns an error message instead of data, so the
            response has no "outputs" key. The HTTP status is not checked.
    """

    params = {
        "lat": lat,
        "lon": lon,
        "peakpower": peakpower,
        "loss": loss,
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
