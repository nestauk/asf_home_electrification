"""A dwelling: its location, TABULA archetype parameters and weather data.

A Dwelling loads the heat loss parameters for a TABULA archetype from a YAML file and fetches
the matching weather from PVGIS once, at construction. The heat balance model reads these
parameters and weather series.

Conventions:
    - Weather is half-hourly, on a tz-aware Europe/London index. PVGIS gives hourly UTC
      values, which are repeated across two half hours and then converted to London time.
    - PVGIS returns the latest year in its dataset (2023), so the weather index is in 2023.
      The model maps each timestamp it looks up onto that year.
    - Temperatures are in degC and irradiance is in W/m2.
"""

import fsspec
import pandas as pd
import yaml

from asf_home_electrification.getters import pvgis

# PVGIS surface orientation for each window direction. aspect is the azimuth in degrees
# (0 = south, -90 = east, 90 = west, 180 = north), angle is the tilt from horizontal.
ORIENTATIONS = {
    "horizontal": {"aspect": 0, "angle": 0},
    "east": {"aspect": -90, "angle": 90},
    "west": {"aspect": 90, "angle": 90},
    "south": {"aspect": 0, "angle": 90},
    "north": {"aspect": 180, "angle": 90},
}


# move this helper somewhere and also refactor in solar.py
def _hourly_to_half_hourly(pvgis_data: pd.DataFrame) -> pd.DataFrame:
    """Turn hourly PVGIS rows into half-hourly rows, each hour repeated twice."""
    pvgis_data["time"] = pd.to_datetime(pvgis_data["time"], format="%Y%m%d:%H%M")
    half_hourly = pvgis_data.loc[pvgis_data.index.repeat(2)].reset_index(drop=True)
    offsets = pd.to_timedelta([0, 30] * len(pvgis_data), unit="min")
    half_hourly["time"] = half_hourly["time"].dt.floor("h") + offsets
    half_hourly["time"] = half_hourly["time"].dt.tz_localize("UTC").dt.tz_convert("Europe/London")
    return half_hourly


class Dwelling:
    """A building at a location, with TABULA heat loss parameters and weather data.

    Loads the archetype's YAML file from S3 and fetches PVGIS weather when constructed, so
    creating a Dwelling makes one PVGIS request per window orientation.
    The lat and lon are fixed at construction, so treat them as read-only, as the weather
    is not refetched if they change.

    Args:
        lat (float): Latitude, in degrees.
        lon (float): Longitude, in degrees.
        archetype_name (str): TABULA archetype code, e.g. "GB.ENG.SFH.05.Gen.ReEx.001.003".
            Used to find the archetype's YAML file.

    Attributes:
        archetype_params (dict): Parameters used by the heat balance model:
            - a_ref: reference floor area (m2).
            - h_tr: transmission heat transfer coefficient, including thermal bridging (W/K).
            - f_nu: temperature reduction factor for non-uniform heating (-).
            - n_air_infiltr: infiltration air change rate (1/h).
            - window_area: window area by orientation (m2), e.g. {"east": 15.3, "west": 15.2}.
            - g_gl: glazing solar transmittance (-).
        archetype_description (str): Description of the archetype from the YAML file.
        outside_temperature (pd.Series): Half-hourly outside air temperature (degC).
        solar_irradiance (pd.DataFrame): Half-hourly irradiance on each window orientation
            (W/m2), one column per orientation in window_area.

    Raises:
        KeyError: If the archetype file is missing a required field.
        ValueError: If window_area has an orientation not in ORIENTATIONS, or is empty.
    """

    def __init__(self, lat: float, lon: float, archetype_name: str) -> None:
        self.lat = lat
        self.lon = lon
        self.archetype_name = archetype_name

        # Load and unpack TABULA archetype YAML file
        # TODO refactor into helper function, appropriate location TBD
        archetype_path = f"s3://asf-home-electrification/inputs/tabula/archetype_heat_parameters/{archetype_name}.yaml"
        with fsspec.open(str(archetype_path), "r") as f:
            raw = yaml.safe_load(f)
        # heat loss through individual construction elements
        elements = raw["elements"]
        h_elements = sum(e["u"] * e["area"] * e["b_tr"] for e in elements)
        # thermal bridging is where heat bypasses insulation, modelled here as a flat surcharge
        h_bridging = raw["delta_u_tb"] * sum(e["area"] for e in elements)

        self.archetype_params = {
            "a_ref": raw["a_ref"],
            "h_tr": h_elements + h_bridging,
            "f_nu": raw["f_nu"],
            "n_air_infiltr": raw["n_air_infiltr"],
            "window_area": raw["window_area"],
            "g_gl": raw["g_gl"],
        }
        self.archetype_description = raw["description"]

        self.outside_temperature, self.solar_irradiance = self._fetch_weather()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(lat={self.lat!r}, lon={self.lon!r}, "
            f"archetype_name={self.archetype_name!r}, archetype_description={self.archetype_description!r})"
        )

    def _fetch_weather(self) -> tuple[pd.Series, pd.DataFrame]:
        """Fetch PVGIS weather, one request per window orientation.

        Outside temperature is the same for every orientation, so it is taken from the first
        response.

        Raises:
            ValueError: If an orientation in window_area is not in ORIENTATIONS, or there
                are no window orientations.

        Returns:
            tuple[pd.Series, pd.DataFrame]:
                - outside_temperature: half-hourly T2m (degC), indexed by Europe/London time.
                - solar_irradiance: half-hourly G(i), the irradiance on the tilted surface
                  (W/m2), same index, one column per orientation in window_area.
        """
        irradiance = {}
        temperature = None

        for orientation in self.archetype_params["window_area"]:
            if orientation not in ORIENTATIONS:
                raise ValueError(f"Unknown window orientation {orientation!r}; expected one of {list(ORIENTATIONS)}")
            pvgis_data = pvgis.get_hourly_pvgis_data(
                self.lat,
                self.lon,
                peakpower=1.0,  # non-meaningful stub
                loss=0,  # non-meaningful stub
                mountingplace="building",
                aspect=ORIENTATIONS[orientation]["aspect"],
                angle=ORIENTATIONS[orientation]["angle"],
            )
            half_hourly = _hourly_to_half_hourly(pvgis_data)
            irradiance[orientation] = pd.Series(half_hourly["G(i)"].to_numpy(dtype=float), index=half_hourly["time"])
            if temperature is None:  # T2m is the same for every orientation
                temperature = pd.Series(half_hourly["T2m"].to_numpy(dtype=float), index=half_hourly["time"])

        if temperature is None:
            raise ValueError(f"No window orientations for archetype {self.archetype_name!r}")

        return temperature, pd.DataFrame(irradiance)
