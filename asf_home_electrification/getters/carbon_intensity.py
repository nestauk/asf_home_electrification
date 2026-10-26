import json
import re
import time

import pandas as pd
import requests


def get_carbon_intensity_for_date(date: str, retries: int = 5, backoff: float = 2.0):
    """Get grid carbon intensity from NESO Official Carbon Intensity API.
    Returns average carbon intensity in gCO2e/kWh for specified day.
    National level carbon intensity.

    Date in YYYY-MM-DD format e.g. 2017-08-25"""

    url = f"https://api.carbonintensity.org.uk/intensity/date/{date}"

    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Fix missing values e.g. "forecast": , -> "forecast": null
            cleaned = re.sub(r":\s*,", ": null,", response.text)

            json_data = json.loads(cleaned)

            df = pd.DataFrame(json_data["data"])
            df["Settlement Period ID"] = range(1, len(df) + 1)
            df["intensity (gCO2/kWh)"] = df["intensity"].apply(
                lambda x: x["actual"] if x["actual"] is not None else x["forecast"]
            )
            return df

        except (requests.exceptions.RequestException, ValueError) as e:
            if attempt < retries - 1:
                wait = backoff * (2**attempt)
                print(f"Attempt {attempt + 1} failed for {date}: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
