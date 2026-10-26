"""Work in progress: Script to run all battery scenarios."""

import os
from datetime import datetime, timedelta

import pandas as pd

from asf_faster_further_electrification.batteries import run_linear_cost_optimisation_battery
from asf_faster_further_electrification.getters import carbon_intensity, elexon_demand_profiles, octopus_tariffs

# ==============================================================================
# Demand profiles
# ==============================================================================


def generate_demand_profiles(annual_tdcv: dict) -> dict[str, pd.DataFrame]:
    """
    Fetch and scale Elexon demand profiles for low, medium, and high TDCV levels.

    Args:
        annual_tdcv (dict): Dictionary mapping demand levels to annual kWh values.

    Returns:
        dict: A dictionary of DataFrames containing the scaled demand profiles.
    """
    raw_coefficients = elexon_demand_profiles.get_raw_elexon_profile_coefficients()
    demand_profiles = {}

    for level, tdcv in annual_tdcv.items():
        # Fetch profile
        elexon_2025_df = elexon_demand_profiles.get_elexon_profile_use_daily(
            df=raw_coefficients,
            annual_electricity_use=tdcv,
            unit="kWh",
            start_date="01/01/2025",
            end_date="31/12/2025",
        )

        # Filter for Profile Class 1
        elexon_profile_class_1 = elexon_2025_df[elexon_2025_df["Profile Class ID"] == 1].reset_index(drop=True)

        # Scale because coefficients sum over Apr-Mar (not Jan-Dec)
        scale_factor = tdcv / elexon_profile_class_1["Electricity use (kWh)"].sum()
        elexon_profile_class_1["Electricity use (kWh)"] *= scale_factor

        # Store the processed dataframe
        demand_profiles[level] = elexon_2025_df

    return demand_profiles


def calculate_electrification_metrics(demand_profiles: dict) -> pd.DataFrame:
    """
    Process demand profiles to generate a single unified summary DataFrame
    containing peak coefficient shares, daily demand metrics, and battery sizing.
    """
    combined_summary_data = []

    for level, df in demand_profiles.items():
        # 1. Filter for Profile Class 1
        profile_class_1 = df[df["Profile Class ID"] == 1].copy()

        # 2. Peak Coefficient Share calculations
        peak_coeff = profile_class_1[profile_class_1["Settlement timestamp"].between("16:00", "19:30")][
            "Default Period Profile Class Coefficient"
        ].sum()

        total_coeff = profile_class_1["Default Period Profile Class Coefficient"].sum()
        peak_percentage = (peak_coeff / total_coeff) * 100

        # 3. Daily Ranges & Time-of-Day Split calculations
        daily_total = profile_class_1.groupby("Settlement Date")["Electricity use (kWh)"].sum()
        avg_daily_profile = profile_class_1.groupby("Settlement timestamp")["Electricity use (kWh)"].mean()
        total_daily_demand = avg_daily_profile.sum()

        # 4. Combine all metrics into a single row dictionary
        combined_summary_data.append(
            {
                "TDCV Level": level.capitalize(),
                "Peak Coeff Sum (16:00-19:30)": round(peak_coeff, 4),
                "Total Coeff Sum": round(total_coeff, 4),
                "Peak Share (%)": round(peak_percentage, 1),
                "Min Daily Demand (kWh)": round(daily_total.min(), 3),
                "Max Daily Demand (kWh)": round(daily_total.max(), 3),
                "Avg Daily Total (kWh)": round(total_daily_demand, 2),
            }
        )

    # Return the single combined master table
    return pd.DataFrame(combined_summary_data)


def calculate_baseline_bills(
    annual_tdcv: dict, electricity_unit_rate: float, electricity_standing_charge: float
) -> dict:
    """
    Calculate baseline annual electricity bills (£/year) for each TDCV tier.
    Expects unit rate in p/kWh and standing charge in p/day.
    Returns a dictionary lookup.
    """
    baseline_bills = {}
    for level, tdcv in annual_tdcv.items():
        bill = ((tdcv * electricity_unit_rate) + (electricity_standing_charge * 365)) / 100
        baseline_bills[level] = bill
    return baseline_bills


# ==============================================================================
# Tariff rate profiles
# ==============================================================================


# Octopus Go
def get_octopus_go_2025_region_E_import_profile() -> pd.DataFrame:
    go_price_raw = octopus_tariffs.get_go_rates(region="E", start_date="2025-01-01", end_date="2026-01-01")
    go_price = octopus_tariffs.expand_flux_to_half_hourly(go_price_raw)

    # Convert UTC valid_from to local time to get correct settlement timestamps
    go_price["valid_from_local"] = go_price["valid_from"].dt.tz_convert("Europe/London")

    # Derive Settlement Date and timestamp from local time
    go_price["Settlement Date"] = go_price["valid_from_local"].dt.date
    go_price["Settlement Date"] = pd.to_datetime(go_price["Settlement Date"])
    go_price["Settlement timestamp"] = go_price["valid_from_local"].dt.strftime("%H:%M")

    # Settlement Period ID from local time
    go_price["Settlement Period ID"] = (
        pd.to_timedelta(go_price["Settlement timestamp"] + ":00").dt.total_seconds() / 1800
    ).astype(int) + 1

    # Octopus Go cheap rate: 00:30 – 05:30 (settlement periods 2-11)
    off_peak_mask = go_price["Settlement Period ID"].between(2, 11)
    peak_mask = ~off_peak_mask

    avg_off_peak = go_price.loc[off_peak_mask, "value_inc_vat"].mean()
    avg_peak = go_price.loc[peak_mask, "value_inc_vat"].mean()

    print(f"Average off-peak price (00:30–05:30): {avg_off_peak:.2f} p/kWh")
    print(f"Average peak price    (05:30–00:30): {avg_peak:.2f} p/kWh")

    return go_price


def get_octopus_go_2025_region_E_standing_charge() -> float:
    df = octopus_tariffs.get_go_standing_charge(region="E", start_date="2025-01-01", end_date="2026-01-01")
    df["valid_from"] = pd.to_datetime(df["valid_from"])
    df["valid_to"] = pd.to_datetime(df["valid_to"])
    df["days"] = (df["valid_to"] - df["valid_from"]).dt.total_seconds() / 86400
    go_standing_charge = (df["value_inc_vat"] * df["days"]).sum() / df["days"].sum()

    return go_standing_charge


def get_octopus_agile_2025_region_E_import_profile() -> pd.DataFrame:

    if not os.path.exists("asf_faster_further_electrification/config/agile_region_E_2025.pkl"):
        print(
            "Pickle file not found: asf_faster_further_electrification/config/agile_region_E_2025.pkl. Fetching Agile data from API."
        )
        df = octopus_tariffs.get_agile_rates(region="E", start_date="2025-01-01", end_date="2026-01-01")
        df.to_pickle("asf_faster_further_electrification/config/agile_region_E_2025.pkl")

    agile_import_price = pd.read_pickle("asf_faster_further_electrification/config/agile_region_E_2025.pkl")

    # Create clean date column
    agile_import_price["valid_from"] = pd.to_datetime(agile_import_price["valid_from"])
    agile_import_price["Settlement Date"] = agile_import_price["valid_from"].dt.tz_convert(None).dt.floor("D")

    # Create clean timestamp column
    agile_import_price["Settlement timestamp"] = agile_import_price["valid_from"].dt.strftime("%H:%M")

    # Sort values
    agile_import_price = agile_import_price.sort_values(
        by=["Settlement Date", "valid_from"], ascending=[True, True]
    ).reset_index(drop=True)

    # Convert UTC to local time
    agile_import_price["valid_from_local"] = agile_import_price["valid_from"].dt.tz_convert("Europe/London")

    # Derive Settlement Date and timestamp from local time
    agile_import_price["Settlement Date"] = pd.to_datetime(agile_import_price["valid_from_local"].dt.date)
    agile_import_price["Settlement timestamp"] = agile_import_price["valid_from_local"].dt.strftime("%H:%M")

    # Settlement Period ID from local time
    agile_import_price["Settlement Period ID"] = (
        pd.to_timedelta(agile_import_price["Settlement timestamp"] + ":00").dt.total_seconds() / 1800
    ).astype(int) + 1

    # Agile peak: 16:00 – 19:00 (settlement periods 33-38)
    # Agile off-peak: 00:00 – 06:00 (settlement periods 1-12)
    # Standard: everything else

    peak_mask = agile_import_price["Settlement Period ID"].between(33, 38)
    off_peak_mask = agile_import_price["Settlement Period ID"].between(1, 12)
    standard_mask = ~peak_mask & ~off_peak_mask

    avg_peak = agile_import_price.loc[peak_mask, "value_inc_vat"].mean()
    avg_off_peak = agile_import_price.loc[off_peak_mask, "value_inc_vat"].mean()
    avg_standard = agile_import_price.loc[standard_mask, "value_inc_vat"].mean()
    avg_overall = agile_import_price["value_inc_vat"].mean()
    print(f"Average Agile peak     (16:00–19:00): {avg_peak:.2f} p/kWh")
    print(f"Average Agile off-peak (00:00–06:00): {avg_off_peak:.2f} p/kWh")
    print(f"Average Agile standard (rest of day): {avg_standard:.2f} p/kWh")
    print(f"Average Agile overall               : {avg_overall:.2f} p/kWh")

    return agile_import_price


def get_octopus_agile_2025_region_E_standing_charge() -> float:
    agile_standing_charge_df = octopus_tariffs.get_agile_standing_charge(
        region="E", start_date="2025-01-01", end_date="2026-01-01"
    )
    agile_standing_charge = agile_standing_charge_df["value_inc_vat"][0]  # p/day
    return agile_standing_charge


# ==============================================================================
# NESO Grid Carbon Intensity
# ==============================================================================


def get_grid_carbon_intensity_2025() -> pd.DataFrame:

    if not os.path.exists("asf_faster_further_electrification/config/grid_intensity_2025.pkl"):
        print(
            "Pickle file not found: asf_faster_further_electrification/config/grid_intensity_2025.pkl. Fetching grid data from API."
        )
        year = 2025
        start = datetime(year, 1, 1)
        end = datetime(year + 1, 1, 1)
        all_days = []
        current = start
        while current < end:
            date_str = current.strftime("%Y-%m-%d")
            try:
                daily_df = carbon_intensity.get_carbon_intensity_for_date(date_str)
                all_days.append(daily_df)
            except Exception as e:
                print(f"Failed {date_str}: {e}")
            current += timedelta(days=1)
        grid_intensity = pd.concat(all_days, ignore_index=True)
        grid_intensity.to_pickle("asf_faster_further_electrification/config/grid_intensity_2025.pkl")

    grid_intensity = pd.read_pickle("asf_faster_further_electrification/config/grid_intensity_2025.pkl")

    # Clean date columns
    grid_intensity["from"] = pd.to_datetime(grid_intensity["from"])
    grid_intensity["Settlement Date"] = grid_intensity["from"].dt.tz_convert(None).dt.floor("D")

    # Create clean timestamp column
    grid_intensity["Settlementtimestamp"] = grid_intensity["from"].dt.strftime("%H:%M")

    # Convert UTC valid_from to local time to get correct settlement timestamps
    grid_intensity["from_local"] = pd.to_datetime(grid_intensity["from"]).dt.tz_convert("Europe/London")
    grid_intensity["Settlement Date"] = pd.to_datetime(grid_intensity["from_local"].dt.date)
    grid_intensity["Settlement timestamp"] = grid_intensity["from_local"].dt.strftime("%H:%M")
    grid_intensity["Settlement Period ID"] = (
        pd.to_timedelta(grid_intensity["Settlement timestamp"] + ":00").dt.total_seconds() / 1800
    ).astype(int) + 1

    # Renumber sequentially per day to handle DST
    grid_intensity["Settlement Period ID"] = grid_intensity.groupby("Settlement Date").cumcount() + 1

    # Verify
    # by_date = grid_intensity.groupby("Settlement Date").size()
    # print(f"30 Mar: {by_date.get(pd.Timestamp('2025-03-30'), 'missing')}")
    # print(f"26 Oct: {by_date.get(pd.Timestamp('2025-10-26'), 'missing')}")
    # print(f"Unusual days: {(by_date != 48).sum()}")

    # Daily min and max
    daily_min = grid_intensity.groupby("Settlement Date")["intensity (gCO2/kWh)"].min()
    daily_max = grid_intensity.groupby("Settlement Date")["intensity (gCO2/kWh)"].max()
    daily_ratio = daily_max / daily_min
    print("\n=== DAILY VARIATION ===")
    print(f"Average daily min    : {daily_min.mean():.1f} gCO2/kWh")
    print(f"Average daily max    : {daily_max.mean():.1f} gCO2/kWh")
    print(f"Average daily ratio  : {daily_ratio.mean():.1f}x")
    print(f"Max daily ratio      : {daily_ratio.max():.1f}x on {daily_ratio.idxmax().date()}")

    # Annual extremes
    print("\n=== ANNUAL EXTREMES ===")
    print(f"Annual minimum       : {grid_intensity['intensity (gCO2/kWh)'].min():.1f} gCO2/kWh")
    print(f"Annual maximum       : {grid_intensity['intensity (gCO2/kWh)'].max():.1f} gCO2/kWh")
    print(
        f"Annual ratio         : {grid_intensity['intensity (gCO2/kWh)'].max() / grid_intensity['intensity (gCO2/kWh)'].min():.1f}x"
    )

    return grid_intensity


# ==============================================================================
# Processing demand, tariff and carbon intensity profiles
# ==============================================================================


def process_all_profile_data(
    demand_dict: dict[str, pd.DataFrame],
    go_price_df: pd.DataFrame,
    agile_price_df: pd.DataFrame,
    grid_intensity_df: pd.DataFrame,
) -> tuple[list, dict[str, dict], dict, dict, dict]:

    dates = demand_dict["low"]["Settlement Date"].unique().tolist()

    # FIX THIS
    # TODO
    daily_tdcv_profiles = {}
    daily_go_import_profile = {}
    for date in dates:
    daily_tdcv_profiles = {
        level: dict(tuple(demand_dict[level].groupby("Settlement Date"))) for level in ["low", "medium", "high"]
    }

    return (
        dates,
        daily_tdcv_profiles,
        daily_go_import_price_profile,
        daily_agile_import_price_profile,
        daily_grid_intensity_profile,
    )


# ==============================================================================
# Run installed battery scenarios
# ==============================================================================


def run_import_only_battery_scenario(
    capacity: float,
    inverter_power: float,
    soc_init: float,
    lifespan: int,
    daily_tdcv_profiles: dict,
    daily_import_price_profile: dict,
    tariff_standing_charge: float,
    baseline_bills_dict: dict,
) -> pd.DataFrame:
    results = {}
    for level in ["low", "medium", "high"]:
        all_days = {}
        for date in dates:
            daily_result = run_linear_cost_optimisation_battery(
                time_series=daily_tdcv_profiles[level][date]["Settlement timestamp"].tolist(),
                demand_profile=daily_tdcv_profiles[level][date]["Electricity use (kWh)"].tolist(),
                prices=daily_import_price_profile[date]["value_inc_vat"].tolist(),
                capacity=capacity,
                inverter_power=inverter_power,
                lifespan=lifespan,
                max_life_cycles=6_000,
                soc_init=soc_init,
                soc_min=0.1,
                soc_max=0.9,
                delta_t=0.5,
                charge_efficiency=0.95,
                discharge_efficiency=0.95,
            )

            daily_result = daily_result.copy()
            daily_result["Settlement Date"] = date
            soc_init = daily_result["soc_kwh"].iloc[-1] / capacity
            all_days[date] = daily_result

        # Battery optimised bill
        battery_annual = sum(all_days[date]["cost_p"].sum() for date in dates) / 100
        battery_annual += (tariff_standing_charge * 365) / 100

        # No battery on new tariff
        no_battery_annual = (
            sum(
                sum(
                    d * p
                    for d, p in zip(
                        daily_tdcv_profiles[level][date]["Electricity use (kWh)"].tolist(),
                        daily_import_price_profile[date]["value_inc_vat"].tolist(),
                    )
                )
                for date in dates
            )
            / 100
        )
        no_battery_annual += (tariff_standing_charge * 365) / 100

        results[level] = {
            "battery_annual": battery_annual,
            "no_battery_annual": no_battery_annual,
        }

    results_df = pd.DataFrame(
        [
            {
                "tdcv_level": level,
                "tariff": "Octopus Go",
                "capacity_kwh": capacity,
                "inverter_kw": inverter_power,
                "baseline_annual_gbp": baseline_bills_dict[level],
                "no_battery_annual_gbp": results[level]["no_battery_annual"],
                "battery_annual_gbp": results[level]["battery_annual"],
            }
            for level in ["low", "medium", "high"]
        ]
    )

    return results_df


# ==============================================================================
# Execution
# ==============================================================================

if __name__ == "__main__":
    # 0. Define constants / inputs
    ANNUAL_TDCV = {"low": 1_800, "medium": 2_700, "high": 4_100}
    ELECTRICITY_UNIT_RATE = 24.6744  # p/kWh, Apr-Jun26 price cap
    ELECTRICITY_STANDING_CHARGE = 57.2061  # p/day, Apr-Jun26 price cap

    excel_filename = "outputs/data/battery_scenarios.xlsx"

    # 1. Demand profiles
    print("Generating demand profiles...")
    demand_dict = generate_demand_profiles(ANNUAL_TDCV)

    print("Creating demand summary table...")
    demand_summary_table = calculate_electrification_metrics(demand_dict)
    print(f"Saving demand summary table to {excel_filename}...")
    with pd.ExcelWriter(excel_filename, engine="openpyxl") as writer:
        demand_summary_table.to_excel(writer, sheet_name="Demand Summary", index=False)

    # 2. Baseline bill
    baseline_bills_dict = calculate_baseline_bills(ANNUAL_TDCV, ELECTRICITY_UNIT_RATE, ELECTRICITY_STANDING_CHARGE)

    # 3. Tariff profiles
    print("Generating 2025 tariff price profiles...")
    go_price_df = get_octopus_go_2025_region_E_import_profile()
    go_standing_charge = get_octopus_go_2025_region_E_standing_charge()
    agile_price_df = get_octopus_agile_2025_region_E_import_profile()
    agile_standing_charge = get_octopus_agile_2025_region_E_standing_charge()

    # 4. Grid intensity profiles
    print("Generating 2025 grid intensity profile...")
    grid_intensity_df = get_grid_carbon_intensity_2025()

    # 5. Converting profile dataframes into dictionaries where keys are dates of the year
    (
        dates,
        daily_tdcv_profiles,
        daily_go_import_price_profile,
        daily_agile_import_price_profile,
        daily_grid_intensity_profile,
    ) = process_all_profile_data(
        demand_dict=demand_dict,
        go_price_df=go_price_df,
        agile_price_df=agile_price_df,
        grid_intensity_df=grid_intensity_df,
    )

    # 6. Run battery scenarios
    SOC_INIT = 0.5

    # 5 kWh / 3.68 kW inverter with Octopus Go (import only)
    installed_5kwh_go_results_df = run_import_only_battery_scenario(
        capacity=5.0,
        inverter_power=3.68,
        soc_init=SOC_INIT,
        lifespan=12,
        daily_tdcv_profiles=daily_tdcv_profiles,
        daily_import_price_profile=daily_go_import_price_profile,
        tariff_standing_charge=go_standing_charge,
        baseline_bills_dict=baseline_bills_dict,
    )

    # 5 kWh / 3.68 kW inverter with Octopus Agile (import only)
    installed_5kwh_agile_results_df = run_import_only_battery_scenario(
        capacity=5.0,
        inverter_power=3.68,
        soc_init=SOC_INIT,
        lifespan=12,
        daily_tdcv_profiles=daily_tdcv_profiles,
        daily_import_price_profile=daily_agile_import_price_profile,
        tariff_standing_charge=agile_standing_charge,
        baseline_bills_dict=baseline_bills_dict,
    )

    # 10 kWh / 5.0 kW inverter with Octopus Go (import only)
    installed_10kwh_go_results_df = run_import_only_battery_scenario(
        capacity=10.0,
        inverter_power=5.0,
        soc_init=SOC_INIT,
        lifespan=12,
        daily_tdcv_profiles=daily_tdcv_profiles,
        daily_import_price_profile=daily_go_import_price_profile,
        tariff_standing_charge=go_standing_charge,
        baseline_bills_dict=baseline_bills_dict,
    )

    # 10 kWh / 5.0 kW inverter with Octopus Agile (import only)
    installed_10kwh_agile_results_df = run_import_only_battery_scenario(
        capacity=10.0,
        inverter_power=5.0,
        soc_init=SOC_INIT,
        lifespan=12,
        daily_tdcv_profiles=daily_tdcv_profiles,
        daily_import_price_profile=daily_agile_import_price_profile,
        tariff_standing_charge=agile_standing_charge,
        baseline_bills_dict=baseline_bills_dict,
    )

    all_installed_results_df = pd.concat(
        [
            installed_5kwh_go_results_df,
            installed_5kwh_agile_results_df,
            installed_10kwh_go_results_df,
            installed_10kwh_agile_results_df,
        ],
        ignore_index=True,
    )

    with pd.ExcelWriter(excel_filename, engine="openpyxl") as writer:
        all_installed_results_df.to_excel(writer, sheet_name="Installed battery, import only", index=False)

    print("Results Excel file generated successfully.")
