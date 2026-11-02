import pandas as pd

ELEXON_COEFFICIENTS_S3_URI = (
    "s3://asf-home-electrification/inputs/demand_profiles/elexon/Default_Period_Profile_Class_Coefficient_375.csv"
)


def get_raw_elexon_profile_coefficients(path: str = ELEXON_COEFFICIENTS_S3_URI) -> pd.DataFrame:
    return pd.read_csv(path)


def get_elexon_profile_use_daily(
    df: pd.DataFrame, annual_electricity_use: float, unit: str, start_date: str = None, end_date: str = None
) -> pd.DataFrame:

    df = df.copy()

    # Add timestamp column
    df["Settlement timestamp"] = (
        pd.to_datetime("00:00") + pd.to_timedelta((df["Settlement Period ID"] - 1) * 30, unit="min")
    ).dt.strftime("%H:%M")

    # Add use column
    df[f"Electricity use ({unit})"] = df["Default Period Profile Class Coefficient"] * annual_electricity_use

    # Filter for date range if given
    if start_date and end_date:
        df["Settlement Date"] = pd.to_datetime(df["Settlement Date"], dayfirst=True)
        df = df[(df["Settlement Date"] >= start_date) & (df["Settlement Date"] <= end_date)]

    return df


def get_elexon_profile_use_average_day(
    df: pd.DataFrame, annual_electricity_use: float, unit: str, start_date: str = None, end_date: str = None
) -> pd.DataFrame:

    df = df.copy()

    # Add timestamp column
    df["Settlement timestamp"] = (
        pd.to_datetime("00:00") + pd.to_timedelta((df["Settlement Period ID"] - 1) * 30, unit="min")
    ).dt.strftime("%H:%M")

    # Add use column
    df[f"Electricity use ({unit})"] = df["Default Period Profile Class Coefficient"] * annual_electricity_use

    # Filter for date range if given
    if start_date and end_date:
        df["Settlement Date"] = pd.to_datetime(df["Settlement Date"], dayfirst=True)
        df = df[(df["Settlement Date"] >= start_date) & (df["Settlement Date"] <= end_date)]

    # Pivot
    pivoted = (
        df.groupby(["Settlement timestamp", "Profile Class ID"])[f"Electricity use ({unit})"]
        .mean()
        .unstack("Profile Class ID")
    )
    pivoted.columns.name = None
    pivoted = pivoted.reset_index()

    return pivoted
