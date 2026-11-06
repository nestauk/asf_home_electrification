import math
import pandas
import pytest

from asf_home_electrification.tariffs.cost_curve import TransmissionCostCurve
from asf_home_electrification.tariffs.schedule import SimplePeakSchedule

# Applies 'schedule' to every test function in this module
pytestmark = pytest.mark.cost_curve

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def peak_schedule() -> SimplePeakSchedule:
    """Schedule defining peak periods from 16:00 to 19:00 daily."""
    return SimplePeakSchedule(peak_start_time="16:00", peak_end_time="19:00")

@pytest.fixture
def single_day_consumption() -> pandas.Series:
    """A full 48-period day (2023-01-01) with flat 1.0 kWh consumption per HH."""
    index = pandas.date_range(
        "2023-01-01 00:00:00",
        "2023-01-01 23:30:00",
        freq="30min",
        tz="Europe/London",
    )
    return pandas.Series(1.0, index=index, name="consumption_kwh")

# ---------------------------------------------------------------------------
# 1. Initialization and Parameter Validation
# ---------------------------------------------------------------------------

def test_valid_revenue_splits(peak_schedule):
    # £1,000 revenue: 40% locational (£400), 60% residual (£600)
    # Residual: 70% standing (£420), 30% volumetric (£180)
    curve = TransmissionCostCurve(
        revenue=1000.0, alpha=0.6, beta=0.7, schedule=peak_schedule
    )

    assert math.isclose(curve.locational_revenue, 400.0)
    assert math.isclose(curve.residual_revenue, 600.0)
    assert math.isclose(curve.residual_standing_revenue, 420.0)
    assert math.isclose(curve.residual_volumetric_revenue, 180.0)


@pytest.mark.parametrize("invalid_alpha", [-0.1, 1.05, 2.0])
def test_invalid_alpha_raises_value_error(invalid_alpha):
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        TransmissionCostCurve(revenue=1000.0, alpha=invalid_alpha, beta=0.5)


@pytest.mark.parametrize("invalid_beta", [-0.1, 1.05, 2.0])
def test_invalid_beta_raises_value_error(invalid_beta):
    with pytest.raises(ValueError, match="beta must be between 0 and 1"):
        TransmissionCostCurve(revenue=1000.0, alpha=0.5, beta=invalid_beta)

# ---------------------------------------------------------------------------
# Standing and Volumetric Residual Charges
# ---------------------------------------------------------------------------

def test_calculate_standing_charge():
    # £36,500 standing revenue over 100 meters across 365 days
    # Rate: 36,500 * 100p / 100 meters / 365 days = 100 p/meter/day
    curve = TransmissionCostCurve(revenue=36500.0, alpha=1.0, beta=1.0)
    rate = curve.calculate_standing_charge(number_of_meters=100, days=365)
    assert math.isclose(rate, 100.0)


@pytest.mark.parametrize("meters, days", [(0, 365), (-10, 365), (100, 0), (100, -30)])
def test_calculate_standing_charge_invalid_inputs(meters, days):
    curve = TransmissionCostCurve(revenue=1000.0, alpha=1.0, beta=1.0)
    with pytest.raises(ValueError):
        curve.calculate_standing_charge(number_of_meters=meters, days=days)


def test_calculate_residual_volumetric_charge():
    # £50 residual volumetric revenue over 1,000 kWh
    # Rate: 50 * 100p / 1000 kWh = 5.0 p/kWh
    curve = TransmissionCostCurve(revenue=100.0, alpha=1.0, beta=0.5)
    rate = curve.calculate_residual_volumetric_charge(total_consumption_kwh=1000.0)
    assert math.isclose(rate, 5.0)


@pytest.mark.parametrize("invalid_kwh", [0.0, -100.0])
def test_calculate_residual_volumetric_charge_invalid_kwh(invalid_kwh):
    curve = TransmissionCostCurve(revenue=100.0, alpha=1.0, beta=0.5)
    with pytest.raises(ValueError, match="Total consumption must be a positive number"):
        curve.calculate_residual_volumetric_charge(total_consumption_kwh=invalid_kwh)

# ---------------------------------------------------------------------------
# Peak Locational Rate Calculations
# ---------------------------------------------------------------------------

def test_calculate_peak_locational_rate(peak_schedule, single_day_consumption):
    # 16:00 to 19:00 has 6 half-hour periods (16:00, 16:30, 17:00, 17:30, 18:00, 18:30)
    # 1.0 kWh per period = 6.0 kWh peak consumption
    # Locational revenue = £600 (alpha=0.4 on £1,000)
    # Unit rate = (600 * 100) / 6.0 = 10,000 p/kWh
    curve = TransmissionCostCurve(
        revenue=1000.0, alpha=0.4, beta=0.5, schedule=peak_schedule
    )
    rate = curve.calculate_peak_locational_rate(single_day_consumption)
    assert math.isclose(rate, 10000.0)


def test_zero_locational_revenue_returns_zero_rate(single_day_consumption):
    # alpha=1.0 means no locational charge, no schedule required
    curve = TransmissionCostCurve(revenue=1000.0, alpha=1.0, beta=0.5, schedule=None)
    rate = curve.calculate_peak_locational_rate(single_day_consumption)
    assert rate == 0.0


def test_precomputed_weights_avoid_recomputing(peak_schedule, single_day_consumption):
    curve = TransmissionCostCurve(
        revenue=1000.0, alpha=0.5, beta=0.5, schedule=peak_schedule
    )
    weights = curve.get_weights("2023-01-01", "2023-01-01")
    rate = curve.calculate_peak_locational_rate(single_day_consumption, weights=weights)
    assert rate > 0.0


def test_missing_schedule_raises_value_error_when_needed(single_day_consumption):
    curve = TransmissionCostCurve(revenue=1000.0, alpha=0.5, beta=0.5, schedule=None)
    with pytest.raises(ValueError, match="You have not specified a WeightingSchedule"):
        curve.calculate_peak_locational_rate(single_day_consumption)


def test_empty_consumption_raises_value_error(peak_schedule):
    curve = TransmissionCostCurve(
        revenue=1000.0, alpha=0.5, beta=0.5, schedule=peak_schedule
    )
    with pytest.raises(ValueError, match="consumption_kwh cannot be empty"):
        curve.calculate_peak_locational_rate(pandas.Series(dtype=float))


def test_zero_peak_consumption_raises_value_error(peak_schedule, single_day_consumption):
    curve = TransmissionCostCurve(
        revenue=1000.0, alpha=0.5, beta=0.5, schedule=peak_schedule
    )
    # Zero out only the peak hours (16:00 to 19:00)
    consumption = single_day_consumption.copy()
    consumption.loc["2023-01-01 16:00:00":"2023-01-01 18:30:00"] = 0.0

    with pytest.raises(ValueError, match="Total consumption during weighted periods must be > 0"):
        curve.calculate_peak_locational_rate(consumption)

# ---------------------------------------------------------------------------
# Full Cost Curve Generation
# ---------------------------------------------------------------------------

def test_cost_curve_structure_and_additions(peak_schedule, single_day_consumption):
    curve_model = TransmissionCostCurve(
        revenue=1000.0, alpha=0.4, beta=0.5, schedule=peak_schedule
    )
    df = curve_model.calculate_cost_curve(single_day_consumption)

    assert list(df.columns) == ["locational", "residual", "total"]
    assert len(df) == 48
    # Mathematical check: total == locational + residual across all rows
    pandas.testing.assert_series_equal(
        df["total"], df["locational"] + df["residual"], check_names=False
    )


def test_peak_vs_off_peak_rates(peak_schedule, single_day_consumption):
    curve_model = TransmissionCostCurve(
        revenue=1000.0, alpha=0.4, beta=0.5, schedule=peak_schedule
    )
    df = curve_model.calculate_cost_curve(single_day_consumption)

    # Expected: Peak locational rate is 10,000 p/kWh; Residual is 20,000 / 48 p/kWh
    peak_period = "2023-01-01 17:00:00"
    off_peak_period = "2023-01-01 12:00:00"

    assert math.isclose(df.loc[peak_period, "locational"], 10000.0)
    assert df.loc[off_peak_period, "locational"] == 0.0

    # Residual rate is volumetric flat across all settlement periods
    expected_residual = (400.0 * 0.5 * 100) / 48.0
    assert math.isclose(df.loc[peak_period, "residual"], expected_residual)
    assert math.isclose(df.loc[off_peak_period, "residual"], expected_residual)


def test_override_residual_consumption_kwh(peak_schedule, single_day_consumption):
    # Scale residual charge by an external system-level baseline volume
    curve_model = TransmissionCostCurve(
        revenue=1000.0, alpha=0.5, beta=0.5, schedule=peak_schedule
    )
    # Residual revenue = £250. Baseline = 25,000 kWh -> 1.0 p/kWh
    df = curve_model.calculate_cost_curve(
        single_day_consumption, residual_consumption_kwh=25000.0
    )
    assert math.isclose(df["residual"].iloc[0], 1.0)


def test_pure_residual_regime_without_schedule(single_day_consumption):
    # When alpha=1.0, no schedule is supplied; locational should be 0.0 everywhere
    curve_model = TransmissionCostCurve(
        revenue=500.0, alpha=1.0, beta=0.0, schedule=None
    )
    df = curve_model.calculate_cost_curve(single_day_consumption)

    assert (df["locational"] == 0.0).all()
    # All revenue (£500) recovered volumetrically over 48 kWh
    expected_rate = (500.0 * 100) / 48.0
    assert math.isclose(df["residual"].iloc[0], expected_rate)
