import pandas as pd


def calculate_solar_household_energy_balance(
    demand_profile_df: pd.DataFrame,
    radiation_profile_df: pd.DataFrame,
    import_price_profile_df: pd.DataFrame,
    export_price_profile_df: pd.DataFrame,
) -> pd.DataFrame:
    """Simulates household adoption of solar panels and calculates import and export costs assuming
    their demand profile does not change."""

    energy_balance = demand_profile_df.copy(deep=True)

    # Add solar generation column
    energy_balance["Solar generation (kWh)"] = radiation_profile_df["generation_kWh"].values  # TODO refactor col name

    # Self consumption is the lesser of generation and demand
    energy_balance["Self consumption (kWh)"] = energy_balance[["Solar generation (kWh)", "Electricity use (kWh)"]].min(
        axis=1
    )

    # Export is any surplus generation beyond demand
    energy_balance["Export (kWh)"] = (
        energy_balance["Solar generation (kWh)"] - energy_balance["Electricity use (kWh)"]
    ).clip(lower=0)

    # Grid draw is any demand not met by solar
    energy_balance["Grid draw (kWh)"] = (
        energy_balance["Electricity use (kWh)"] - energy_balance["Solar generation (kWh)"]
    ).clip(lower=0)

    # Add import rate column
    energy_balance["Import rate (p/kWh)"] = import_price_profile_df["value_inc_vat"]  # TODO refactor col name

    # Add import costs without solar, just from tariff change
    energy_balance["Import cost without solar (p)"] = (
        energy_balance["Import rate (p/kWh)"] * energy_balance["Electricity use (kWh)"]
    )

    # Calculate import costs
    energy_balance["Import cost with solar (p)"] = (
        energy_balance["Import rate (p/kWh)"] * energy_balance["Grid draw (kWh)"]
    )

    # Add export rate column
    energy_balance["Export rate (p/kWh)"] = export_price_profile_df["value_inc_vat"]  # TODO refactor col name

    # Calculate export costs
    energy_balance["Export revenue (p)"] = energy_balance["Export rate (p/kWh)"] * energy_balance["Export (kWh)"]

    return energy_balance


def calculate_solar_household_energy_balance_no_export(
    demand_profile_df: pd.DataFrame,
    radiation_profile_df: pd.DataFrame,
    import_price_profile_df: pd.DataFrame,
) -> pd.DataFrame:
    """Simulates household adoption of solar panels and calculates import costs assuming
    their demand profile does not change. No export capability — surplus generation is curtailed."""

    energy_balance = demand_profile_df.copy(deep=True)

    # Add solar generation column
    energy_balance["Solar generation (kWh)"] = radiation_profile_df["generation_kWh"].values

    # Self consumption is the lesser of generation and demand
    energy_balance["Self consumption (kWh)"] = energy_balance[["Solar generation (kWh)", "Electricity use (kWh)"]].min(
        axis=1
    )

    # Surplus = any generation beyond demand — curtailed as no export capability
    energy_balance["Surplus (kWh)"] = (
        energy_balance["Solar generation (kWh)"] - energy_balance["Electricity use (kWh)"]
    ).clip(lower=0)

    # Grid draw is any demand not met by solar
    energy_balance["Grid draw (kWh)"] = (
        energy_balance["Electricity use (kWh)"] - energy_balance["Solar generation (kWh)"]
    ).clip(lower=0)

    # Add import rate column
    energy_balance["Import rate (p/kWh)"] = import_price_profile_df["value_inc_vat"]

    # Add import costs without solar, just from tariff change
    energy_balance["Import cost without solar (p)"] = (
        energy_balance["Import rate (p/kWh)"] * energy_balance["Electricity use (kWh)"]
    )

    # Calculate import costs
    energy_balance["Import cost with solar (p)"] = (
        energy_balance["Import rate (p/kWh)"] * energy_balance["Grid draw (kWh)"]
    )

    return energy_balance
