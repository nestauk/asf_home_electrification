from datetime import time
import pandas
import pytest

from asf_home_electrification.tariffs.schedule import SimplePeakSchedule, MultiBandSchedule, BandWindow

# Applies 'tariffs' to every test function in this module
pytestmark = pytest.mark.schedule

# ============================================================================
# 1. SimplePeakSchedule: Validation & Initialisation Tests
# ============================================================================

def test_valid_initialisation():
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
    """Ensure windows ending at 00:00 (midnight) are allowed."""
    schedule = SimplePeakSchedule("23:30", "00:00")
    assert schedule.peak_start_time == time(23, 30)
    assert schedule.peak_end_time == time(0, 0)

    evening_schedule = SimplePeakSchedule("19:00", "00:00")
    assert evening_schedule.peak_start_time == time(19, 0)
    assert evening_schedule.peak_end_time == time(0, 0)


@pytest.mark.parametrize("invalid_month", ["00", "13", "foo", "1", ""])
def test_invalid_winter_months_raise_error(invalid_month):
    with pytest.raises(ValueError):
        SimplePeakSchedule("08:00", "17:00", winter_start=invalid_month)

# ============================================================================
# 2. SimplePeakSchedule: Logic & Weights Tests
# ============================================================================

def test_generate_weights_single_day_length():
    """A standard non-daylight savings time day must contain exactly 48 settlement periods."""
    schedule = SimplePeakSchedule("08:00", "10:00")
    weights = schedule.generate_weights("2026-11-03", "2026-11-03")  # Tuesday

    assert len(weights) == 48
    assert weights.name == "weight"
    assert weights.dtype == "float64"
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
    formatted_times = weights.index.strftime("%H:%M").tolist()

    assert len(weights) == 46
    assert "01:00" not in formatted_times
    assert "01:30" not in formatted_times


def test_dst_fall_back():
    """UK clocks fall back on the last Sunday of October (50 half-hours)."""
    schedule = SimplePeakSchedule("08:00", "10:00")
    # 2026-10-25 is the UK autumn transition date
    weights = schedule.generate_weights("2026-10-25", "2026-10-25")

    assert len(weights) == 50
    
    for expected in [
        "2026-10-25 01:00:00+01:00",
        "2026-10-25 01:30:00+01:00",
        "2026-10-25 01:00:00+00:00",
        "2026-10-25 01:30:00+00:00",
       ]:
        assert pandas.Timestamp(expected) in weights.index

# ============================================================================
# 4. BandWindow Tests
# ============================================================================

def test_band_window_valid_initialisation_and_normalisation():
    window = BandWindow(
        name="  Red ",
        start_time="16:00",
        end_time="19:30",
        weight=1,
        days_of_week=[4, 2, 0, 2, 1, 3],
        months=[2, 12, 1, 11, 12],
    )
    assert window.name == "Red"
    assert window.start_time == time(16, 0)
    assert window.end_time == time(19, 30)
    assert window.weight == 1.0
    assert isinstance(window.weight, float)
    assert window.days_of_week == (0, 1, 2, 3, 4)
    assert window.months == (1, 2, 11, 12)


def test_band_window_defaults():
    window = BandWindow("Amber", "07:00", "23:00", weight=0.15)
    assert window.days_of_week == (0, 1, 2, 3, 4, 5, 6)
    assert window.months is None


@pytest.mark.parametrize("invalid_name", ["", "   ", 123, None])
def test_band_window_invalid_name_raises_error(invalid_name):
    with pytest.raises(ValueError, match="name must be a non-empty string"):
        BandWindow(invalid_name, "16:00", "19:00", weight=1.0)


@pytest.mark.parametrize("invalid_weight", [-0.01, -1, True, False, "1.0"])
def test_band_window_invalid_weight_raises_error(invalid_weight):
    with pytest.raises(ValueError, match="weight must be a non-negative number"):
        BandWindow("Red", "16:00", "19:00", weight=invalid_weight)


@pytest.mark.parametrize("invalid_days", [[], [-1], [7], [True], ["0"]])
def test_band_window_invalid_days_of_week_raises_error(invalid_days):
    with pytest.raises(ValueError, match="days_of_week"):
        BandWindow("Red", "16:00", "19:00", weight=1.0, days_of_week=invalid_days)


@pytest.mark.parametrize("invalid_months", [[], [0], [13], [False], ["11"]])
def test_band_window_invalid_months_raises_error(invalid_months):
    with pytest.raises(ValueError, match="months"):
        BandWindow("Red", "16:00", "19:00", weight=1.0, months=invalid_months)


def test_band_window_time_validation_and_mutation():
    with pytest.raises(ValueError, match="end_time must be later than start_time"):
        BandWindow("Red", "19:00", "16:00", weight=1.0)

    window = BandWindow("Red", "16:00", "19:00", weight=1.0)
    with pytest.raises(ValueError, match="end_time must be later than start_time"):
        window.start_time = "20:00"

    with pytest.raises(ValueError, match="end_time must be later than start_time"):
        window.end_time = "15:00"

def test_band_window_equality():
    w1 = BandWindow("Red", "16:00", "19:00", weight=1.0, days_of_week=[4, 3, 2, 1, 0])
    w2 = BandWindow("Red", "16:00", "19:00", weight=1.0, days_of_week=(0, 1, 2, 3, 4))
    w3 = BandWindow("Amber", "16:00", "19:00", weight=0.15, days_of_week=(0, 1, 2, 3, 4))

    assert w1 == w2
    assert w1 != w3
    assert w1 != "not-a-window"

# ============================================================================
# 5. MultiBandSchedule Tests
# ============================================================================

def test_multiband_invalid_initialisation():
    valid_window = BandWindow("Red", "16:00", "19:00", weight=1.0)

    with pytest.raises(ValueError, match="default_weight"):
        MultiBandSchedule([valid_window], default_weight=-0.1)

    with pytest.raises(ValueError, match="default_weight"):
        MultiBandSchedule([valid_window], default_weight=True)

    with pytest.raises(ValueError, match="default_band_name"):
        MultiBandSchedule([valid_window], default_band_name="  ")

    with pytest.raises(TypeError, match="BandWindow"):
        MultiBandSchedule(["not-a-window"])  # type: ignore[list-item]

def test_multiband_empty_windows_returns_default_weight():
    schedule = MultiBandSchedule([], default_weight=0.05)
    weights = schedule.generate_weights("2026-11-03", "2026-11-03")

    assert len(weights) == 48
    assert weights.dtype == "float64"
    assert (weights == 0.05).all()

def test_multiband_rag_profile_weekday_and_weekend():
    """Verify typical DNO Red/Amber/Green behaviour across weekdays and weekends."""
    schedule = MultiBandSchedule(
        default_band_name="Green",
        default_weight=0.02,
        windows=[
            BandWindow("Amber", "07:00", "23:00", weight=0.15, days_of_week=(0, 1, 2, 3, 4)),
            BandWindow("Red", "16:00", "19:00", weight=1.00, days_of_week=(0, 1, 2, 3, 4)),
            BandWindow("Amber", "16:00", "20:00", weight=0.15, days_of_week=(5, 6)),
        ],
    )

    # 2026-10-30 is Friday, 2026-10-31 is Saturday
    weights = schedule.generate_weights("2026-10-30", "2026-10-31")

    # Friday (weekday)
    assert weights.loc["2026-10-30 03:00:00+00:00"] == pytest.approx(0.02)  # Green
    assert weights.loc["2026-10-30 10:00:00+00:00"] == pytest.approx(0.15)  # Morning Amber
    assert weights.loc["2026-10-30 16:00:00+00:00"] == pytest.approx(1.00)  # Red start (inclusive)
    assert weights.loc["2026-10-30 18:30:00+00:00"] == pytest.approx(1.00)  # Red last period
    assert weights.loc["2026-10-30 19:00:00+00:00"] == pytest.approx(0.15)  # Evening Amber
    assert weights.loc["2026-10-30 23:00:00+00:00"] == pytest.approx(0.02)  # Night Green

    # Saturday (weekend): no Red band, only 16:00-20:00 Amber
    assert weights.loc["2026-10-31 10:00:00+00:00"] == pytest.approx(0.02)
    assert weights.loc["2026-10-31 17:30:00+00:00"] == pytest.approx(0.15)
    assert weights.loc["2026-10-31 20:00:00+00:00"] == pytest.approx(0.02)


def test_multiband_overlap_precedence_independent_of_order():
    """Higher-weight bands must override overlapping lower-weight bands regardless of input order."""
    amber = BandWindow("Amber", "07:00", "23:00", weight=0.15)
    red = BandWindow("Red", "16:00", "19:00", weight=1.00)

    sched_amber_first = MultiBandSchedule([amber, red], default_weight=0.02)
    sched_red_first = MultiBandSchedule([red, amber], default_weight=0.02)

    w1 = sched_amber_first.generate_weights("2026-11-03", "2026-11-03")
    w2 = sched_red_first.generate_weights("2026-11-03", "2026-11-03")

    pandas.testing.assert_series_equal(w1, w2)
    assert w1.loc["2026-11-03 17:00:00+00:00"] == pytest.approx(1.00)


def test_multiband_midnight_end_time():
    """A window ending at '00:00' covers up to 23:30 and does not bleed into 00:00 the next day."""
    schedule = MultiBandSchedule(
        default_weight=0.0,
        windows=[BandWindow("Evening", "22:00", "00:00", weight=0.5)],
    )
    weights = schedule.generate_weights("2026-11-03", "2026-11-04")

    assert weights.loc["2026-11-03 21:30:00+00:00"] == 0.0
    assert weights.loc["2026-11-03 22:00:00+00:00"] == 0.5
    assert weights.loc["2026-11-03 23:30:00+00:00"] == 0.5
    assert weights.loc["2026-11-04 00:00:00+00:00"] == 0.0


def test_multiband_month_filtering():
    """Windows restricted to specific months only apply during those months."""
    schedule = MultiBandSchedule(
        default_weight=0.02,
        windows=[
            BandWindow("WinterRed", "16:00", "19:00", weight=1.0, months=(11, 12, 1, 2)),
        ],
    )

    nov_weights = schedule.generate_weights("2026-11-03", "2026-11-03")  # Nov (GMT)
    jul_weights = schedule.generate_weights("2026-07-15", "2026-07-15")  # Jul (BST)

    assert nov_weights.loc["2026-11-03 17:00:00+00:00"] == pytest.approx(1.0)
    assert jul_weights.loc["2026-07-15 17:00:00+01:00"] == pytest.approx(0.02)


def test_multiband_dst_transitions():
    """MultiBandSchedule inherits Europe/London DST handling (46 and 50 settlement periods)."""
    schedule = MultiBandSchedule([BandWindow("Red", "16:00", "19:00", weight=1.0)])

    spring_weights = schedule.generate_weights("2026-03-29", "2026-03-29")
    autumn_weights = schedule.generate_weights("2026-10-25", "2026-10-25")

    assert len(spring_weights) == 46
    assert len(autumn_weights) == 50

# ============================================================================
# 6. Representations
# ============================================================================

def test_simple_peak_repr_roundtrip():
    schedule = SimplePeakSchedule("08:00", "17:00", weekdays_only=True, winter_only=True)
    recreated = eval(repr(schedule))
    assert repr(schedule) == repr(recreated)


def test_band_window_repr_roundtrip_and_str():
    window = BandWindow("Red", "16:00", "19:00", weight=1.0, days_of_week=(0, 1, 2, 3, 4))
    recreated = eval(repr(window))
    assert recreated == window
    assert repr(window) == repr(recreated)
    assert str(window) == "Red: 16:00 to 19:00 (weekdays only, weight=1.0)"


def test_multiband_repr_roundtrip_and_str():
    schedule = MultiBandSchedule(
        default_band_name="Green",
        default_weight=0.02,
        windows=[
            BandWindow("Amber", "07:00", "23:00", weight=0.15, days_of_week=(0, 1, 2, 3, 4)),
            BandWindow("Red", "16:00", "19:00", weight=1.0, days_of_week=(0, 1, 2, 3, 4)),
        ],
    )
    recreated = eval(repr(schedule))
    assert repr(schedule) == repr(recreated)
    assert recreated.windows == schedule.windows
    assert recreated.default_weight == schedule.default_weight
    assert recreated.default_band_name == schedule.default_band_name

    rendered = str(schedule)
    assert "Default (Green): weight=0.02" in rendered
    assert "Red: 16:00-19:00 (weekdays) -> weight=1.0" in rendered
