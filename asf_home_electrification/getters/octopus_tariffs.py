import pandas as pd
import requests

REGIONS = list("ABCDEFGHJKLMN")

"""
DNO region codes represent electricity distribution zones in the UK.
Each code maps to a specific regional grid operator.

A - Eastern England
B - East Midlands
C - London
D - Merseyside and North Wales
E - West Midlands
F - North East England
G - North West England
H - Southern England
J - South East England
K - South Wales
L - South West England
M - Yorkshire
N - Southern Scotland
P - Northern Scotland
"""


def get_agile_rates(region: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Get Agile tariff rates from Octopus API.
    region is DNO region code e.g. "C" for London.
    start_date and end_date in format "YYYY-MM-DD". Returned period is up to, not including, end date.
    """

    url = f"https://api.octopus.energy/v1/products/AGILE-24-10-01/electricity-tariffs/E-1R-AGILE-24-10-01-{region}/standard-unit-rates/"
    params = {
        "period_from": f"{start_date}T00:00Z",
        "period_to": f"{end_date}T00:00Z",
        "page_size": 100,
    }

    all_results = []
    while url:
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        all_results.extend(data["results"])
        url = data.get("next")
        params = None

    df = pd.DataFrame(all_results)

    return df


def get_agile_export_rates(region: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Get Agile Outgoing (export) tariff rates from Octopus API.
    region is DNO region code e.g. "C" for London.
    start_date and end_date in format "YYYY-MM-DD". Returned period is up to, not including, end date.
    """

    url = f"https://api.octopus.energy/v1/products/AGILE-OUTGOING-19-05-13/electricity-tariffs/E-1R-AGILE-OUTGOING-19-05-13-{region}/standard-unit-rates/"
    params = {
        "period_from": f"{start_date}T00:00Z",
        "period_to": f"{end_date}T00:00Z",
        "page_size": 100,
    }

    all_results = []
    while url:
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        all_results.extend(data["results"])
        url = data.get("next")
        params = None

    df = pd.DataFrame(all_results)

    return df


def get_agile_standing_charge(region: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Get Agile import standing charge from Octopus API."""
    product = "AGILE-24-10-01"
    url = f"https://api.octopus.energy/v1/products/{product}/electricity-tariffs/E-1R-{product}-{region}/standing-charges/"
    params = {
        "period_from": f"{start_date}T00:00Z",
        "period_to": f"{end_date}T00:00Z",
        "page_size": 100,
    }

    all_results = []
    while url:
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        all_results.extend(data["results"])
        url = data.get("next")
        params = None

    return pd.DataFrame(all_results)


def get_flux_rates(region: str, start_date: str, end_date: str, direction: str = "import") -> pd.DataFrame:
    """Get Flux import or export tariff rates from Octopus API.
    region is DNO region code e.g. "C" for London.
    start_date and end_date in format "YYYY-MM-DD". Returned period is up to, not including, end date.
    direction is "import" or "export".
    """

    if direction == "import":
        product = "FLUX-IMPORT-23-02-14"
    elif direction == "export":
        product = "FLUX-EXPORT-23-02-14"
    else:
        raise ValueError(f"direction must be 'import' or 'export', got '{direction}'")

    url = f"https://api.octopus.energy/v1/products/{product}/electricity-tariffs/E-1R-{product}-{region}/standard-unit-rates/"
    params = {
        "period_from": f"{start_date}T00:00Z",
        "period_to": f"{end_date}T00:00Z",
        "page_size": 1500,
    }

    all_results = []
    while url:
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        all_results.extend(data["results"])
        url = data.get("next")
        params = None

    df = pd.DataFrame(all_results)

    return df


def expand_flux_to_half_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Expand Flux variable-length periods to half-hourly intervals."""

    df = df.copy()
    df["valid_from"] = pd.to_datetime(df["valid_from"])
    df["valid_to"] = pd.to_datetime(df["valid_to"])

    rows = []
    for _, row in df.iterrows():
        periods = pd.date_range(start=row["valid_from"], end=row["valid_to"], freq="30min", inclusive="left")
        for period in periods:
            rows.append(
                {
                    "valid_from": period,
                    "value_inc_vat": row["value_inc_vat"],
                    "value_exc_vat": row["value_exc_vat"],
                }
            )

    expanded = pd.DataFrame(rows)

    # Same formatting as Agile
    expanded["Settlement Date"] = expanded["valid_from"].dt.tz_convert(None).dt.floor("D")
    expanded["Settlement timestamp"] = expanded["valid_from"].dt.strftime("%H:%M")

    expanded = expanded.sort_values(by=["Settlement Date", "valid_from"], ascending=[True, True]).reset_index(drop=True)

    return expanded


def get_flux_standing_charge(region: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Get Flux import standing charge from Octopus API.
    Standing charge is on the import tariff only.
    """
    product = "FLUX-IMPORT-23-02-14"
    url = f"https://api.octopus.energy/v1/products/{product}/electricity-tariffs/E-1R-{product}-{region}/standing-charges/"
    params = {
        "period_from": f"{start_date}T00:00Z",
        "period_to": f"{end_date}T00:00Z",
        "page_size": 100,
    }

    all_results = []
    while url:
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        all_results.extend(data["results"])
        url = data.get("next")
        params = None

    return pd.DataFrame(all_results)


def get_go_rates(region: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Get Octopus Go import tariff rates from Octopus API.
    region is DNO region code e.g. "C" for London.
    start_date and end_date in format "YYYY-MM-DD". Returned period is up to, not including, end date.
    Go returns time-banded records (off-peak 00:30-05:30, peak otherwise) which must be expanded
    to half-hourly using expand_flux_to_half_hourly.
    """
    product = "GO-VAR-22-10-14"
    url = f"https://api.octopus.energy/v1/products/{product}/electricity-tariffs/E-1R-{product}-{region}/standard-unit-rates/"
    params = {
        "period_from": f"{start_date}T00:00Z",
        "period_to": f"{end_date}T00:00Z",
        "page_size": 1500,
    }

    all_results = []
    while url:
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        all_results.extend(data["results"])
        url = data.get("next")
        params = None

    df = pd.DataFrame(all_results)

    return df


def get_go_standing_charge(region: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Get Octopus Go standing charge from Octopus API."""
    product = "GO-VAR-22-10-14"
    url = f"https://api.octopus.energy/v1/products/{product}/electricity-tariffs/E-1R-{product}-{region}/standing-charges/"
    params = {
        "period_from": f"{start_date}T00:00Z",
        "period_to": f"{end_date}T00:00Z",
        "page_size": 100,
    }

    all_results = []
    while url:
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        all_results.extend(data["results"])
        url = data.get("next")
        params = None

    return pd.DataFrame(all_results)
