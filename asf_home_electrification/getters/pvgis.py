import pandas as pd
import requests


def get_solar_half_hourly_radiation(
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

    df = pd.DataFrame(data["outputs"]["hourly"])
    df["time"] = pd.to_datetime(df["time"], format="%Y%m%d:%H%M")
    df = df.rename(columns={"P": "generation_W"})
    df["generation_kWh"] = df["generation_W"] / 1000  # hourly W -> kWh

    # Create a proper half-hourly index for one year
    hh_index = pd.date_range("2005-01-01 00:00", "2023-12-31 23:30", freq="30min")

    # Each hourly value splits equally across two half-hour periods
    df_hh = df.loc[df.index.repeat(2)].reset_index(drop=True)
    df_hh["generation_kWh"] = df_hh["generation_kWh"] / 2
    df_hh["time"] = hh_index

    df_hh = df_hh.set_index("time")

    return df_hh
