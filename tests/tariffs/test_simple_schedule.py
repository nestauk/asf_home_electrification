from datetime import time
import pandas
import pytest

from asf_home_electrification.tariffs.schedule import SimplePeakSchedule

# Applies 'tariffs' to every test function in this module
pytestmark = pytest.mark.schedule

# ============================================================================
# 1. Validation & Initialization Tests
# ============================================================================

def test_valid_initialization():
    schedule = SimplePeakSchedule("08:00", "17:00")
    assert schedule.peak_start_time == time(8, 0)
    assert schedule.peak_end_time == time(17, 0)
    assert schedule.weekdays_only is False
    assert schedule.winter_only is False


@pytest.mark.parametrize("invalid_time", ["08:15", "08:45", "08:01", "25:00", "not-a-time"])
def test_invalid_minute_or_format_raises_error(invalid_time):
    with pytest.raises(ValueError):
        SimplePeakSchedule(invalid_time, "17:00")

    with pytest.raises(ValueError):
        SimplePeakSchedule("08:00", invalid_time)


def test_start_time_after_end_time_raises_error():
    with pytest.raises(ValueError, match="peak_end_time must be later than peak_start_time"):
        # Current behaviour does not allow peaks to wrap around midnight
        # This test will fail if that changes
        SimplePeakSchedule("17:00", "08:00")


def test_midnight_special_case():
    """Ensure the 23:30 to 00:00 edge case is explicitly allowed."""
    schedule = SimplePeakSchedule("23:30", "00:00")
    assert schedule.peak_start_time == time(23, 30)
    assert schedule.peak_end_time == time(0, 0)


@pytest.mark.parametrize("invalid_month", ["00", "13", "foo", "1", ""])
def test_invalid_winter_months_raise_error(invalid_month):
    with pytest.raises(ValueError):
        SimplePeakSchedule("08:00", "17:00", winter_start=invalid_month)

# ============================================================================
# 2. Logic & Weights Tests
# ============================================================================

def test_generate_weights_single_day_length():
    """A standard non-daylight savings time day must contain exactly 48 settlement periods."""
    schedule = SimplePeakSchedule("08:00", "10:00")
    weights = schedule.generate_weights("2026-11-03", "2026-11-03")  # Tuesday

    assert len(weights) == 48
    assert weights.name == "weight"
    assert isinstance(weights.index, pandas.DatetimeIndex)


def test_peak_window_values():
    schedule = SimplePeakSchedule("08:00", "09:30")
    weights = schedule.generate_weights("2026-11-03", "2026-11-03")

    # 08:00, 08:30, 09:00 should be 1 (inclusive start, exclusive end)
    # 07:30 and 09:30 should be 0
    assert weights.loc["2026-11-03 07:30:00+00:00"] == 0
    assert weights.loc["2026-11-03 08:00:00+00:00"] == 1
    assert weights.loc["2026-11-03 08:30:00+00:00"] == 1
    assert weights.loc["2026-11-03 09:00:00+00:00"] == 1
    assert weights.loc["2026-11-03 09:30:00+00:00"] == 0
    assert weights.sum() == 3


def test_weekdays_only():
    schedule = SimplePeakSchedule("08:00", "10:00", weekdays_only=True)
    # 2026-06-05 is Friday, 2026-06-06 is Saturday
    weights = schedule.generate_weights("2026-10-30", "2026-10-31")

    friday_peak = weights.loc["2026-10-30 08:00:00+00:00":"2026-10-30 09:30:00+00:00"]
    saturday_peak = weights.loc["2026-10-31 08:00:00+00:00":"2026-10-31 09:30:00+00:00"]

    assert (friday_peak == 1).all()
    assert (saturday_peak == 0).all()


def test_winter_only_wraparound():
    """Winter defined as Nov to Feb (months 11, 12, 01, 02)."""
    schedule = SimplePeakSchedule("08:00", "10:00", winter_only=True, winter_start="11", winter_end="02")

    # November (in winter) vs October (outside winter)
    nov_weights = schedule.generate_weights("2026-11-03", "2026-11-03")
    oct_weights = schedule.generate_weights("2026-10-15", "2026-10-15")

    assert nov_weights.sum() > 0
    assert oct_weights.sum() == 0

# ============================================================================
# 3. Timezone / Daylight Saving Time Tests (Europe/London)
# ============================================================================

def test_dst_spring_forward():
    """UK clocks spring forward on the last Sunday of March (46 half-hours)."""
    schedule = SimplePeakSchedule("08:00", "10:00")
    # 2026-03-29 is the UK spring transition date
    weights = schedule.generate_weights("2026-03-29", "2026-03-29")
    assert len(weights) == 46


def test_dst_fall_back():
    """UK clocks fall back on the last Sunday of October (50 half-hours)."""
    schedule = SimplePeakSchedule("08:00", "10:00")
    # 2026-10-25 is the UK autumn transition date
    weights = schedule.generate_weights("2026-10-25", "2026-10-25")
    assert len(weights) == 50

# ============================================================================
# 4. Representations
# ============================================================================

def test_repr_roundtrip():
    schedule = SimplePeakSchedule("08:00", "17:00", weekdays_only=True, winter_only=True)
    # The repr string should be valid code that instantiates an equivalent object
    recreated = eval(repr(schedule))
    assert repr(schedule) == repr(recreated)
