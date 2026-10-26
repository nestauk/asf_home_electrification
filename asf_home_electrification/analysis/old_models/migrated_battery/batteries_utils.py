import altair as alt
import pandas as pd

from asf_faster_further_electrification.batteries import Battery


def create_battery_modelling_results_chart(
    profile_label: str,
    baseline_bill: float,
    new_bill: float,
    all_days: dict,
    price_label: str,
) -> alt.Chart:

    # Combine all days into one dataframe
    combined_df = pd.concat(all_days.values(), ignore_index=True)
    combined_df["date_str"] = pd.to_datetime(combined_df["Settlement Date"]).dt.strftime("%Y-%m-%d")

    alt.data_transformers.enable("vegafusion")

    dates_available = sorted(combined_df["date_str"].unique().tolist())
    date_select = alt.binding_select(options=dates_available, name="Date: ")
    date_param = alt.param(name="selected_date", bind=date_select, value=dates_available[0])

    power_vars = ["soc_kwh", "demand_kwh", "grid_kwh"]

    df_power = combined_df.melt(
        id_vars=["date_str", "timestamp", "price_p_per_kwh"],
        value_vars=power_vars,
        var_name="variable",
        value_name="value",
    )

    base_power = alt.Chart(df_power).transform_filter(alt.datum.date_str == date_param).add_params(date_param)

    base_price = alt.Chart(combined_df).transform_filter(alt.datum.date_str == date_param)

    color_scale = alt.Scale(
        domain=["soc_kwh", "demand_kwh", "grid_kwh", "price"],
        range=["#1f77b4", "#2ca02c", "#9467bd", "#000000"],
    )

    label_expr = (
        "datum.label === 'soc_kwh' ? 'State of charge' : "
        "datum.label === 'demand_kwh' ? 'Demand' : "
        "datum.label === 'grid_kwh' ? 'Grid import' : "
        "datum.label === 'price' ? 'Price (p/kWh)' : datum.label"
    )

    power_lines = base_power.mark_line(interpolate="step-after").encode(
        x=alt.X(
            "timestamp:O",
            title="Time of day",
            axis=alt.Axis(labelAngle=-45, labelOverlap=False),
        ),
        y=alt.Y("value:Q", title="Energy (kWh)"),
        color=alt.Color(
            "variable:N",
            scale=color_scale,
            legend=alt.Legend(title="variable", labelExpr=label_expr),
        ),
        tooltip=[
            alt.Tooltip("timestamp:O", title="Time of day"),
            alt.Tooltip("variable:N", title="Series"),
            alt.Tooltip("value:Q", title="Value", format=".2f"),
        ],
    )

    price_line = (
        base_price.mark_line(interpolate="step-after")
        .encode(
            x=alt.X("timestamp:O"),
            y=alt.Y(
                "price_p_per_kwh:Q",
                title=f"Price (p/kWh) ({price_label})",
                axis=alt.Axis(orient="right"),
            ),
            color=alt.Color(
                "variable:N",
                scale=alt.Scale(domain=["price"], range=["#000000"]),
                legend=alt.Legend(title="variable", labelExpr=label_expr),
            ),
            tooltip=[
                alt.Tooltip("timestamp:O", title="Time of day"),
                alt.Tooltip("price_p_per_kwh:Q", title="Price (p/kWh)", format=".2f"),
            ],
        )
        .transform_calculate(variable="'price'")
    )

    saving_pct = (baseline_bill - new_bill) / baseline_bill * 100

    chart = (
        alt.layer(power_lines, price_line)
        .resolve_scale(y="independent", color="shared")
        .properties(
            width=800,
            height=400,
            title=[
                f"Household: {profile_label}",
                f"Baseline bill: £{baseline_bill:.2f}, New bill: £{new_bill:.2f}, Savings: {saving_pct:.0f}%",
            ],
        )
    )

    return chart


def calculate_bill_summary(
    baseline_electricity_unit_rate: float,
    baseline_electricity_standing_charge: float,
    annual_consumption: float,
    demand_profile: pd.DataFrame,
    tariff_prices: pd.DataFrame,
    tariff_standing_charge: float,
    battery_results: pd.DataFrame,
) -> pd.DataFrame:
    """Small summary table for bill calculations."""

    # Function to calculate a baseline bill for given unit rate and standing charge
    std_bill = lambda v, s, c: ((v * c) + (s * 365)) / 100

    # baseline bill
    baseline_bill = std_bill(baseline_electricity_unit_rate, baseline_electricity_standing_charge, annual_consumption)

    # Battery optimised bill
    battery_annual = battery_results["cost_p"].sum() / 100
    battery_annual += (tariff_standing_charge * 365) / 100

    # No battery on Agile
    no_battery_annual = (
        demand_profile.merge(tariff_prices, on=["Settlement Date", "Settlement timestamp"])
        .assign(cost=lambda df: df["value_inc_vat"] * (df["Electricity use (kWh)"] * 2700 / 100))["cost"]
        .sum()
    )
    no_battery_annual += (tariff_standing_charge * 365) / 100

    return pd.DataFrame(
        [
            ["Baseline SVT", baseline_bill],
            ["Agile no battery", no_battery_annual],
            ["Agile + battery", battery_annual],
            ["Saving", no_battery_annual - battery_annual],
        ],
        columns=["Condition", "Electricity Bill (£)"],
    )


def calculate_import_summary(battery_results: pd.DataFrame) -> pd.DataFrame:
    """Small summary table for import, charge, discharge statistics."""
    return pd.DataFrame(
        [
            ["Total charge", battery_results["charge_kwh"].sum()],
            ["Total discharge", battery_results["discharge_kwh"].sum()],
            ["Total import", battery_results["grid_kwh"].sum()],
        ],
        columns=["condition", "kWh"],
    )


def calculate_energy_flows_summary(
    battery_results: pd.DataFrame, battery: Battery, annual_consumption: float
) -> pd.DataFrame:
    """Small summary table for energy flow statistics."""
    res = calculate_import_summary(battery_results)
    # Daily cycling
    avg_daily_discharge = res.iloc[1, 1] / 365
    usable_capacity = battery.total_capacity * (battery.state_of_charge_max - battery.state_of_charge_min)
    avg_daily_cycles = avg_daily_discharge / usable_capacity

    return pd.DataFrame(
        [
            ["Discharge/charge ratio", res.iloc[1, 1] / res.iloc[0, 1]],
            ["Import vs annual demand", f"{res.iloc[2, 1]:.1f} vs {annual_consumption} kWh"],
            ["Charge as % of demand", res.iloc[0, 1] / annual_consumption * 100],
            ["Avg daily discharge", avg_daily_discharge],
            ["Avg daily cycles", avg_daily_cycles],
            ["Max allowed cycles", (battery.lifecycles_max / (battery.lifespan * 365))],
        ],
        columns=["condition", "statistic"],
    )
