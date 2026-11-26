"""Tests for the HeatingSystem heat balance model.

Covers half-hour thermal balance math, capacity sharing between DHW tank and
space heating, multi-step temperature chaining, fuel demand calculations, and
input validation.

Usage:
    uv run pytest -m heating
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from asf_home_electrification.core_model.heating import HeatingSystem
from asf_home_electrification.core_model.hot_water import HotWaterTank

pytestmark = pytest.mark.heating


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def dummy_dwelling():
    """Minimal dwelling object with 2023 UTC-indexed weather series."""
    idx = pd.date_range("2023-01-05 00:00", periods=6, freq="30min", tz="UTC")
    return SimpleNamespace(  # using SimpleNamespace to avoid S3 and PVGIS calls
        archetype_name="TEST.ARCHETYPE",
        archetype_params={
            "a_ref": 100.0,
            "h_tr": 150.0,
            "f_nu": 1.0,
            "n_air_infiltr": 0.5,
            "window_area": {"south": 10.0, "horizontal": 5.0},
            "g_gl": 0.6,
        },
        outside_temperature=pd.Series([5.0] * len(idx), index=idx),
        solar_irradiance=pd.DataFrame(
            {"south": [200.0] * len(idx), "horizontal": [100.0] * len(idx)},
            index=idx,
        ),
    )


@pytest.fixture
def standard_tank() -> HotWaterTank:
    return HotWaterTank(
        volume_litres=210.0,
        t_cold_feed=10.0,
        t_hot_water=55.0,
        t_room=20.0,
        ref_standing_loss_kwh_per_day=1.08,
        ref_delta_t=45.0,
    )


# ============================================================================
# Initialisation
# ============================================================================


def test_init_defaults_and_tank_assignment(standard_tank):
    hp = HeatingSystem("heat pump", 8.0, fuel="electricity", efficiency=3.0)
    assert hp.efficiency_dhw == 3.0
    assert hp.tank is None

    hp.tank = standard_tank
    assert hp.tank is standard_tank


# ============================================================================
# Half-hour heat balance: run_heat_balance_half_hour()
# ============================================================================


def test_half_hour_heat_balance_reaches_target_with_large_capacity(dummy_dwelling):
    hp = HeatingSystem("gas boiler", 24.0, fuel="gas", efficiency=0.9)
    res = hp.run_heat_balance_half_hour(
        dwelling=dummy_dwelling,
        t_start=18.0,
        t_target=20.0,
        step_start="2026-01-05 00:00",
    )

    # h_ve = 0.34 * (0.4 + 0.5) * 100 * 2.5 = 76.5 W/K; t_mean = 19.0, t_out = 5.0
    # q_tr = 1.0 * 150 * 14 * 0.5 / 1000 = 1.05 kWh
    # q_ve = 1.0 * 76.5 * 14 * 0.5 / 1000 = 0.5355 kWh
    # q_sol = (0.6*(1-0.3)*0.9*0.6*10*200 + 0.8*(1-0.3)*0.9*0.6*5*100) * 0.5 / 1000 = 0.3024 kWh
    # q_int = 3.0 * 100 * 0.5 / 1000 = 0.15 kWh
    # q_mass = (45 * 100 / 0.5) * 2.0 * 0.5 / 1000 = 9.0 kWh
    assert res["q_transmission_kwh"] == pytest.approx(1.05)
    assert res["q_ventilation_kwh"] == pytest.approx(0.5355)
    assert res["q_solar_kwh"] == pytest.approx(0.3024)
    assert res["q_internal_kwh"] == pytest.approx(0.15)
    assert res["q_mass_kwh"] == pytest.approx(9.0)
    assert res["q_required_kwh"] == pytest.approx(10.1331)
    assert res["q_delivered_kwh"] == pytest.approx(10.1331)
    assert res["q_dhw_kwh"] == 0.0
    assert res["t_end_c"] == pytest.approx(20.0)


def test_half_hour_dhw_takes_priority_over_space_heating(dummy_dwelling, standard_tank):
    # 4 kW system -> 2.0 kWh max per half hour; tank starts cold (20 degC) and takes all 2.0 kWh
    hp = HeatingSystem("heat pump", 4.0, fuel="electricity", efficiency=3.0, efficiency_dhw=2.0, tank=standard_tank)
    res = hp.run_heat_balance_half_hour(
        dwelling=dummy_dwelling,
        t_start=18.0,
        t_target=21.0,
        step_start="2026-01-05 00:00",
        t_dhw_start=20.0,
    )

    assert res["q_dhw_kwh"] == pytest.approx(2.0)
    assert res["q_delivered_kwh"] == pytest.approx(0.0)
    assert res["t_end_c"] < 18.0
    assert "t_tank_end_c" in res


# ============================================================================
# Multi-step simulation: run_heat_balance_time_period()
# ============================================================================


def test_time_period_chains_temperatures_across_steps(dummy_dwelling, standard_tank):
    idx = pd.date_range("2026-01-05 00:00", "2026-01-05 01:00", freq="30min", inclusive="left", tz="Europe/London")
    temps = pd.Series([20.0, 20.0], index=idx)
    draws = pd.Series([0.0, 30.0], index=idx)

    hp = HeatingSystem("heat pump", 6.0, fuel="electricity", efficiency=3.0, tank=standard_tank)
    df = hp.run_heat_balance_time_period(
        dwelling=dummy_dwelling,
        start="2026-01-05 00:00",
        end="2026-01-05 01:00",
        t_initial=16.0,
        t_target=temps,
        t_setpoint=temps,
        dhw_draw_litres=draws,
    )

    assert len(df) == 2
    assert df["t_start_c"].iloc[1] == pytest.approx(df["t_end_c"].iloc[0])
    assert df["t_tank_start_c"].iloc[1] == pytest.approx(df["t_tank_end_c"].iloc[0])


def test_time_period_computes_fuel_demand_with_different_efficiencies(dummy_dwelling, standard_tank):
    idx = pd.date_range("2026-01-05 00:00", "2026-01-05 00:30", freq="30min", inclusive="left", tz="Europe/London")
    temps = pd.Series([20.0], index=idx)

    hp = HeatingSystem("heat pump", 6.0, fuel="electricity", efficiency=3.0, efficiency_dhw=2.0, tank=standard_tank)
    df = hp.run_heat_balance_time_period(
        dwelling=dummy_dwelling,
        start="2026-01-05 00:00",
        end="2026-01-05 00:30",
        t_initial=16.0,
        t_target=temps,
        t_setpoint=temps,
        t_tank_initial=45.0,
    )

    row = df.iloc[0]
    assert row["q_total_kwh"] == pytest.approx(row["q_delivered_kwh"] + row["q_dhw_kwh"])
    assert row["q_unmet_kwh"] == pytest.approx(row["q_required_kwh"] - row["q_delivered_kwh"])
    assert row["electricity_demand_kwh"] == pytest.approx(row["q_delivered_kwh"] / 3.0 + row["q_dhw_kwh"] / 2.0)


# ============================================================================
# Error handling & input validation
# ============================================================================


def test_init_raises_on_invalid_tank_type():
    with pytest.raises(TypeError, match="tank must be a HotWaterTank or None"):
        HeatingSystem("heat pump", 8.0, fuel="electricity", efficiency=3.0, tank="not_a_tank")


def test_half_hour_raises_on_empty_archetype_param(dummy_dwelling):
    dummy_dwelling.archetype_params["window_area"] = {}
    hp = HeatingSystem("gas boiler", 24.0, fuel="gas", efficiency=0.9)

    with pytest.raises(ValueError, match="Empty archetype params"):
        hp.run_heat_balance_half_hour(
            dwelling=dummy_dwelling,
            t_start=18.0,
            t_target=20.0,
            step_start="2026-01-05 00:00",
        )


def test_time_period_raises_when_target_is_missing_steps(dummy_dwelling):
    idx = pd.date_range("2026-01-05 00:00", "2026-01-05 00:30", freq="30min", inclusive="left", tz="Europe/London")
    hp = HeatingSystem("gas boiler", 24.0, fuel="gas", efficiency=0.9)

    with pytest.raises(ValueError, match="t_target is missing"):
        hp.run_heat_balance_time_period(
            dwelling=dummy_dwelling,
            start="2026-01-05 00:00",
            end="2026-01-05 00:30",
            t_initial=18.0,
            t_target=pd.Series(dtype=float),  # pass empty series
            t_setpoint=pd.Series([20.0], index=idx),
        )


def test_time_period_raises_when_dhw_given_without_tank(dummy_dwelling):
    idx = pd.date_range("2026-01-05 00:00", "2026-01-05 00:30", freq="30min", inclusive="left", tz="Europe/London")
    temps = pd.Series([20.0], index=idx)
    hp = HeatingSystem("gas boiler", 24.0, fuel="gas", efficiency=0.9)

    with pytest.raises(ValueError, match="no tank"):
        hp.run_heat_balance_time_period(
            dwelling=dummy_dwelling,
            start="2026-01-05 00:00",
            end="2026-01-05 00:30",
            t_initial=18.0,
            t_target=temps,
            t_setpoint=temps,
            dhw_draw_litres=pd.Series([10.0], index=idx),
        )
