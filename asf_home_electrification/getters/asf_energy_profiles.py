"""
Data getters used by the energy-use profiles explorer.
https://github.com/nestauk/asf_energy_use_profiles_dashboard/blob/dev/getters/data_getters.py
"""

import os

import altair as alt
import pandas as pd

outputs_path = "s3://nesta-open-data/asf_energy_use_profiles_explorer/"


def get_distribution_households() -> pd.DataFrame:
    """
    Get the CSV with distribution of households per profile (numbers and percentages).
    Contains three columns:
       - profile
       - number_of_households
       - perc_of_households

    Returns:
        pd.DataFrame: A DataFrame containing the distribution of households per profile.
    """
    profile_info = pd.read_csv(os.path.join(outputs_path, "profiles_numbers_and_percentages.csv"))
    return profile_info


def get_annual_energy_consumption_avg_per_profile() -> pd.DataFrame:
    """
    Get the CSV with average annual electricity and gas consumption per profile.
    Contains three columns:
        - profile
        - avg_annual_elec_consumption_kWh
        - avg_annual_gas_consumption_kWh

    Returns:
        pd.DataFrame: A DataFrame containing the average annual energy consumption per profile.
    """
    profile_annual_avgs = pd.read_csv(os.path.join(outputs_path, "profiles_avg_annual_usage_per_profile.csv"))
    return profile_annual_avgs


def get_hh_consumption_per_profile(energy_type: str = "electricity", season: str = "") -> pd.DataFrame:
    """
    Get the daily average and standard deviation of energy consumption per profile
    for a specific type of energy (electricity or gas) and season (winter/ summer) or all year.

    Average and standard deviation data are merged into a single DataFrame, with the following columns:
    - read_time: the time of the day (HH:MM format)
    - profile_X_avg: average consumption for profile X (for all profiles, where X is the profile number) in Wh
    - profile_X_std: standard deviation of consumption for profile X (for all profiles, where X is the profile number)

    Args:
        type (str, optional): "energy_type" for electricity and "gas" for gas. Defaults to "energy_type".
        season (str, optional): "" for all year, "summer" for summer and "winter" for winter. Defaults to "" (all year).

    Returns:
        pd.DataFrame: A DataFrame containing the daily average and standard deviation of energy consumption per profile.
    """
    if season != "":
        season = f"_{season}"

    energy_type = energy_type[:4]  # Ensure energy_type is in the format 'elec' or 'gas'

    daily_data = pd.read_csv(os.path.join(outputs_path, f"profiles_{energy_type}_consumption_day{season}.csv"))

    return daily_data


def get_contextual_information() -> pd.DataFrame:
    """
    Get the CSV with contextual information about the profiles.
    Contains columns:
        - profile
        - number_households
        - proportions and counts for a variety of contextual household and property information

    Returns:
        pd.DataFrame: A DataFrame containing the contextual information about the profiles.
    """
    context_info = pd.read_csv(os.path.join(outputs_path, "profiles_contextual_information.csv"))
    return context_info


def create_chart_daily_consumption(
    daily_data: pd.DataFrame,
    profile: int,
    energy_type: str,
    season: str = "",
    std_band: bool = True,
) -> alt.Chart:
    """
    Creates a chart showing daily average energy consumption for a specific profile and energy type.

    Args:
        daily_data (pd.DataFrame): DataFrame containing daily average energy consumption data.
        profile (int): profile number
        energy_type (str): type of energy ('electricity' or 'gas')
        season (str, optional): season ("" for all year, "summer" for summer and "winter" for winter).
                                Defaults to "" (all year/no season selected)
        colour (str, optional): colour of the line. Defaults to NESTA_COLOURS[0].
        std_band (bool, optional): True to show error band, otherwise False. Defaults to True.

    Returns:
        alt.Chart: An Altair chart object showing the daily average energy consumption for the
        specified profile, energy type and season (and optional error band).
    """
    profile_col = "profile_" + str(profile)

    title = f"Daily average {energy_type} consumption for profile {profile}"
    title = f" in {season}" if season != "" else title

    # Line chart for the mean/average consumption
    avg_line = (
        alt.Chart(daily_data, title=alt.TitleParams(text=title, anchor="middle"))
        .mark_line()
        .encode(
            x=alt.X("read_time:O", title="Time of the day"),
            y=alt.Y(f"{profile_col}_avg:Q", title=f"{energy_type} consumption (in Wh)"),
            tooltip=[
                alt.Tooltip("read_time:O", title="Half hour"),
                alt.Tooltip(
                    f"{profile_col}_avg:Q",
                    title="Average daily profile consumption",
                    format=".0f",
                ),
            ],
        )
    )

    # Calculate y_min and y_max using the mean and the standard deviation
    # y_min defines the bottom of the error band
    daily_data["y_min"] = daily_data[profile_col + "_avg"] - daily_data[profile_col + "_std"]
    # y_max defines the top of the error band
    daily_data["y_max"] = daily_data[profile_col + "_avg"] + daily_data[profile_col + "_std"]
    # Ensure y_min is not negative (as there is no negative consumption)
    daily_data["y_min"] = daily_data["y_min"].clip(lower=0)

    # If std_band is True, we plot the error band around the mean line
    if std_band:
        error_band = (
            alt.Chart(daily_data)
            .mark_area(opacity=0.3)
            .encode(
                x=alt.X("read_time:O", title="Time of the day"),
                y=alt.Y("y_min:Q", title=""),
                y2="y_max:Q",
                tooltip=[
                    alt.Tooltip("read_time:O", title="Half hour"),
                    alt.Tooltip(f"{profile_col}_avg:Q", title="Mean Value", format=".0f"),
                    alt.Tooltip("y_min:Q", title="Average - standard deviation", format=".0f"),
                    alt.Tooltip("y_max:Q", title="Average + standard deviation", format=".0f"),
                ],
            )
        )
        final_chart = error_band + avg_line
    else:
        final_chart = avg_line

    final_chart = final_chart.properties(width=600, height=400)

    return final_chart
