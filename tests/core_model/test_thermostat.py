"""Tests for Thermostat setpoint schedules and preheat target generation.

Covers schedule parsing and validation, weekday/weekend half-hourly setpoint
generation (including UK DST transitions), and preheat lead-time logic.

Usage:
    uv run pytest -m thermostat
"""

from datetime import time

import pandas as pd
import pytest

from asf_home_electrification.core_model.thermostat import Thermostat, _parse_schedule

pytestmark = pytest.mark.thermostat


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def default_thermostat() -> Thermostat:
    """Standard weekday/weekend thermostat with 60-minute preheat."""
    return Thermostat(
        weekday={"00:00": 16.0, "06:30": 21.0, "08:30": 16.0, "17:00": 21.0, "22:30": 16.0},
        weekend={"00:00": 16.0, "07:30": 21.0, "22:30": 16.0},
        preheat_minutes=60,
    )


# ============================================================================
# _parse_schedule & validation
# ============================================================================


def test_parse_schedule_sorts_chronologically():
    parsed = _parse_schedule({"06:30": 21.0, "00:00": 16.0, "22:30": 16.0})
    assert parsed == [(time(0, 0), 16.0), (time(6, 30), 21.0), (time(22, 30), 16.0)]


@pytest.mark.parametrize(
    ("bad_schedule", "match"),
    [
        ({}, "empty"),
        ({"06:00": 20.0}, "must start at '00:00'"),
        ({"00:00": 16.0, "00:00:00": 18.0}, "duplicate times"),
        ({"00:00": 16.0, "6:30": 21.0}, "Invalid isoformat string"),
    ],
)
def test_parse_schedule_rejects_invalid_inputs(bad_schedule, match):
    with pytest.raises(ValueError, match=match):
        _parse_schedule(bad_schedule)


@pytest.mark.parametrize("bad_preheat", [-30, 15, 45])
def test_thermostat_rejects_invalid_preheat_minutes(bad_preheat):
    with pytest.raises(ValueError, match="preheat_minutes"):
        Thermostat(weekday={"00:00": 18.0}, weekend={"00:00": 18.0}, preheat_minutes=bad_preheat)


def test_thermostat_revalidates_after_mutation(default_thermostat):
    default_thermostat.preheat_minutes = 15
    with pytest.raises(ValueError, match="preheat_minutes"):
        default_thermostat.setpoints("2026-01-05", "2026-01-06")


# ============================================================================
# Setpoints
# ============================================================================


def test_setpoints_distinguishes_weekday_and_weekend(default_thermostat):
    # 2026-01-09 is Friday (weekday), 2026-01-10 is Saturday (weekend)
    sp = default_thermostat.setpoints("2026-01-09", "2026-01-11")

    assert sp.name == "t_setpoint_c"
    assert str(sp.index.tz) == "Europe/London"
    assert len(sp) == 96

    # At 06:30: Friday is already 21.0, Saturday is still 16.0 (rises at 07:30)
    assert sp["2026-01-09 06:30"] == 21.0
    assert sp["2026-01-10 06:30"] == 16.0
    assert sp["2026-01-10 07:30"] == 21.0
    assert sp["2026-01-10 22:30"] == 16.0


def test_setpoints_handles_uk_dst_transitions(default_thermostat):
    # Spring forward (2026-03-29): 01:00 -> 02:00 skipped -> 46 half-hour steps
    spring = default_thermostat.setpoints("2026-03-29", "2026-03-30")
    assert len(spring) == 46

    # Fall back (2026-10-25): 01:00-02:00 repeated -> 50 half-hour steps
    autumn = default_thermostat.setpoints("2026-10-25", "2026-10-26")
    assert len(autumn) == 50


def test_setpoints_raises_on_empty_period(default_thermostat):
    with pytest.raises(ValueError, match="No time intervals"):
        default_thermostat.setpoints("2026-01-05", "2026-01-05")


# ============================================================================
# Pre-heating targets
# ============================================================================


def test_targets_brings_rises_forward_by_preheat_minutes(default_thermostat):
    sp = default_thermostat.setpoints("2026-01-05", "2026-01-06")  # Monday
    targets = default_thermostat.targets(sp)

    assert targets.name == "t_target_c"
    pd.testing.assert_index_equal(targets.index, sp.index)

    # 06:30 rise to 21.0 with 60m preheat -> 05:30 and 06:00 become 21.0; 05:00 stays 16.0
    assert targets["2026-01-05 05:00"] == 16.0
    assert targets["2026-01-05 05:30"] == 21.0
    assert targets["2026-01-05 06:00"] == 21.0
    # Drop at 08:30 is not brought forward
    assert targets["2026-01-05 08:00"] == 21.0
    assert targets["2026-01-05 08:30"] == 16.0


def test_targets_respects_min_rise():
    thermostat = Thermostat(
        weekday={"00:00": 16.0, "02:00": 16.4, "05:00": 21.0},
        weekend={"00:00": 16.0},
        preheat_minutes=60,
        min_rise=0.5,
    )
    sp = thermostat.setpoints("2026-01-05", "2026-01-06")
    targets = thermostat.targets(sp)

    # 02:00 rise is 0.4 (< min_rise 0.5) -> no preheat at 01:00 or 01:30
    assert targets["2026-01-05 01:00"] == 16.0
    assert targets["2026-01-05 01:30"] == 16.0
    assert targets["2026-01-05 02:00"] == 16.4

    # 05:00 rise is 4.6 (>= min_rise 0.5) -> preheats at 04:00 and 04:30
    assert targets["2026-01-05 03:30"] == 16.4
    assert targets["2026-01-05 04:00"] == 21.0
    assert targets["2026-01-05 04:30"] == 21.0


def test_targets_clamps_preheat_at_series_start_without_wrapping():
    thermostat = Thermostat(
        weekday={"00:00": 16.0, "01:00": 21.0, "02:00": 16.0},
        weekend={"00:00": 16.0},
        preheat_minutes=120,
    )
    sp = thermostat.setpoints("2026-01-05", "2026-01-06")
    targets = thermostat.targets(sp)

    # 01:00 rise wants 120m preheat (back to 23:00 previous day), clamped to 00:00
    assert targets["2026-01-05 00:00"] == 21.0
    assert targets["2026-01-05 00:30"] == 21.0
    # Must not wrap around via negative indexing to the end of the series
    assert targets["2026-01-05 23:00"] == 16.0
    assert targets["2026-01-05 23:30"] == 16.0
