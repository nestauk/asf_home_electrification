"""Tests for the HotWaterTank thermal model and half-hour step simulation.

Covers parameter validation, derived thermal properties (heat capacity and
standing loss coefficients), energy balance across half-hour timesteps, capacity
clipping, and passive cool-down above target temperature.

Usage:
    uv run pytest -m hot_water
"""

import pytest

from asf_home_electrification.core_model.hot_water import WATER_KJ_PER_L_K, HotWaterTank

pytestmark = pytest.mark.hot_water


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def standard_tank() -> HotWaterTank:
    """210 L cylinder at 55 degC in a 20 degC room with 10 degC cold feed."""
    return HotWaterTank(
        volume_litres=210.0,
        t_cold_feed=10.0,
        t_hot_water=55.0,
        t_room=20.0,
        ref_standing_loss_kwh_per_day=1.08,
        ref_delta_t=45.0,
    )


# ============================================================================
# Initialisation & validation
# ============================================================================


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"volume_litres": 0.0}, "volume_litres must be positive"),
        ({"volume_litres": -50.0}, "volume_litres must be positive"),
        ({"ref_standing_loss_kwh_per_day": -0.1}, "ref_standing_loss_kwh_per_day cannot be negative"),
        ({"ref_delta_t": 0.0}, "ref_delta_t must be positive"),
        ({"ref_delta_t": -10.0}, "ref_delta_t must be positive"),
        ({"t_cold_feed": 55.0}, "t_cold_feed .* must be below t_hot_water"),
        ({"t_cold_feed": 60.0}, "t_cold_feed .* must be below t_hot_water"),
        ({"t_room": 55.0}, "t_room .* must be below t_hot_water"),
        ({"t_room": 60.0}, "t_room .* must be below t_hot_water"),
    ],
)
def test_init_rejects_invalid_parameters(overrides, match):
    params = {
        "volume_litres": 210.0,
        "t_cold_feed": 10.0,
        "t_hot_water": 55.0,
        "t_room": 20.0,
        "ref_standing_loss_kwh_per_day": 1.08,
        "ref_delta_t": 45.0,
        **overrides,
    }
    with pytest.raises(ValueError, match=match):
        HotWaterTank(**params)


# ============================================================================
# Thermal properties
# ============================================================================


def test_thermal_properties(standard_tank):
    # 210 * 4.184 / 3600 = 0.2440666... kWh/K
    assert standard_tank.heat_capacity_kwh_per_k == pytest.approx(210.0 * WATER_KJ_PER_L_K / 3600)
    # 1.08 * 1000 / 24 / 45 = 1.0 W/K
    assert standard_tank.heat_loss_coefficient_w_per_k == pytest.approx(1.0)
    # 1.08 * (55 - 20) / 45 = 0.84 kWh/day
    assert standard_tank.standing_loss_kwh_day == pytest.approx(0.84)


# ============================================================================
# run_half_hour_step()
# ============================================================================


def test_step_reaches_target_when_capacity_is_sufficient(standard_tank):
    # Start at 50 degC, target defaults to 55 degC, 20 L draw, 10 kW heater (max 5 kWh)
    res = standard_tank.run_half_hour_step(t_start=50.0, draw_litres=20.0, heating_system_size_kw=10.0)

    expected_q_draw = 20.0 * WATER_KJ_PER_L_K / 3600 * (50.0 - 10.0)
    # heat_loss_coeff = 1.0 W/K -> 1.0 * (50 - 20) * 0.5 / 1000 = 0.015 kWh
    expected_q_standing = 0.015
    expected_q_dhw = standard_tank.heat_capacity_kwh_per_k * 5.0 + expected_q_draw + expected_q_standing

    assert res["t_tank_end_c"] == pytest.approx(55.0)
    assert res["q_dhw_draw_kwh"] == pytest.approx(expected_q_draw)
    assert res["q_dhw_standing_kwh"] == pytest.approx(expected_q_standing)
    assert res["q_dhw_kwh"] == pytest.approx(expected_q_dhw)


def test_step_clips_heat_delivered_to_system_capacity(standard_tank):
    # Start cold (20 degC) with a small 2 kW heater -> max 1.0 kWh in 0.5 h
    res = standard_tank.run_half_hour_step(t_start=20.0, draw_litres=0.0, heating_system_size_kw=2.0)

    assert res["q_dhw_kwh"] == pytest.approx(1.0)
    assert res["q_dhw_draw_kwh"] == 0.0
    assert res["q_dhw_standing_kwh"] == 0.0  # t_start == t_room
    expected_t_end = 20.0 + 1.0 / standard_tank.heat_capacity_kwh_per_k
    assert res["t_tank_end_c"] == pytest.approx(expected_t_end)
    assert res["t_tank_end_c"] < 55.0


def test_step_does_not_actively_cool_above_target(standard_tank):
    # Tank at 60 degC with custom t_target=45 degC -> no heating, drifts down via losses
    res = standard_tank.run_half_hour_step(t_start=60.0, draw_litres=10.0, heating_system_size_kw=8.0, t_target=45.0)

    assert res["q_dhw_kwh"] == 0.0
    assert res["t_tank_end_c"] < 60.0
    expected_t_end = 60.0 - (res["q_dhw_draw_kwh"] + res["q_dhw_standing_kwh"]) / standard_tank.heat_capacity_kwh_per_k
    assert res["t_tank_end_c"] == pytest.approx(expected_t_end)


@pytest.mark.parametrize("bad_draw", [-1.0, 210.0, 250.0])
def test_step_rejects_out_of_range_draw_litres(standard_tank, bad_draw):
    with pytest.raises(ValueError, match="draw_litres must be between 0 and the tank volume"):
        standard_tank.run_half_hour_step(t_start=55.0, draw_litres=bad_draw, heating_system_size_kw=8.0)
