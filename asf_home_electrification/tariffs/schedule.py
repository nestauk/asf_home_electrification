from abc import ABC, abstractmethod
from datetime import date, time, timedelta
from typing import Iterable, Sequence

import pandas

# Shared functions

def _parse_half_hour(value: str, field_name: str) -> time:
    """Parse an '%H:%M' string and enforce 30-minute settlement boundaries."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string of format %H:%M, not {value!r}.")
    try:
        parsed = time.fromisoformat(value)
    except ValueError:
        raise ValueError(
            f"{field_name} '{value}' was not recognised as a valid time of format %H:%M"
        ) from None

    if parsed.minute not in (0, 30) or parsed.second != 0 or parsed.microsecond != 0:
        raise ValueError(
            f"{field_name} must be on either the hour or half-hour, not '{value}'."
        )
    return time(parsed.hour, parsed.minute)

def _validate_window(start: time, end: time,
                     start_field: str, end_field: str) -> None:
    """Validate that the peak end time occurs after the peak start time.

    Treats an end time of 00:00 as midnight at the end of the day,
    allowing peak windows to run through the end of the day from any start time.

    Args:
        start: Inclusive start time of the peak window.
        end: Exclusive end time of the peak window.
        start_field: attribute name of start time. 
        end_field: attribute name of end time

    Raises:
        ValueError: If `end` is not `00:00` and is earlier than or equal to `start`.
    """
    # Treat 00:00 as 24:00 (end of day), which is later than any start time
    if end != time(0, 0) and end <= start:
        raise ValueError(f"{end_field} must be later than {start_field}.")

class WeightingSchedule(ABC):
    """Abstract base class for generating half-hourly settlement period weights.

    Subclasses define specific weighting methodologies (e.g., flat, seasonal, or
    profile-based) to distribute or scale quantities across 30-minute settlement
    periods over a given date range. Concrete implementations must also provide
    explicit string representations for serialization, logging, and display.
    """

    @staticmethod
    def _build_settlement_index(start_date: str, end_date: str) -> pandas.DatetimeIndex:
        """Validate ISO date strings and build a half-hourly Europe/London DatetimeIndex."""
        try:
            start = date.fromisoformat(start_date)
        except ValueError:
            raise ValueError(
                f"start_date '{start_date}' was not recognised as a valid date of format %Y-%m-%d"
            ) from None

        try:
            end = date.fromisoformat(end_date)
        except ValueError:
            raise ValueError(
                f"end_date '{end_date}' was not recognised as a valid date of format %Y-%m-%d"
            ) from None

        if end < start:
            raise ValueError("end_date must be after start_date.")

        return pandas.date_range(
            start,
            end + timedelta(days=1),
            freq="30min",
            tz="Europe/London",
            inclusive="left",
        )

    @abstractmethod
    def generate_weights(self, start_date: str, end_date: str) -> pandas.Series:
        """Returns a Series object of half-hourly settlement periods from start_date to end_date."""
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


class SimplePeakSchedule(WeightingSchedule):
    """Generates daily recurring peak window weights.

    Args:
        peak_start_time: Start of peak (inclusive) as formatted time string (%H:%M)
        peak_end_time: End of peak (exclusive) as formatted time string (%H:%M)
        weekdays_only: If True, weights are 0 on weekends. Default: False.
        winter_only: If True, weights are 0 outside of winter_start and winter_end. Default: False.
        winter_start: Month to start winter peak (inclusive) as two digit number string '01' (Jan) to '12' (Dec)
        winter_end: Month to end winter peak (inclusive) as two digit number '01' (Jan) to '12' (Dec)
    """

    def __init__(
        self,
        peak_start_time: str,
        peak_end_time: str,
        weekdays_only: bool=False,
        winter_only: bool=False,
        winter_start: str='11',
        winter_end: str='02'
    ) -> None:
        self.peak_start_time = peak_start_time
        self.peak_end_time = peak_end_time
        self.weekdays_only = weekdays_only
        self.winter_only = winter_only
        self.winter_start = winter_start
        self.winter_end = winter_end

    @property
    def peak_start_time(self) -> time:
        return self._peak_start_time

    @peak_start_time.setter
    def peak_start_time(self, value: str) -> None:
        new_start = _parse_half_hour(value, "peak_start_time")
    
        if hasattr(self, "_peak_end_time"):
            _validate_window(new_start, self._peak_end_time, "peak_start_time", "peak_end_time")

        self._peak_start_time = new_start

    @property
    def peak_end_time(self) -> time:
        return self._peak_end_time

    @peak_end_time.setter
    def peak_end_time(self, value: str) -> None:
        new_end = _parse_half_hour(value, "peak_end_time")
        _validate_window(self.peak_start_time, new_end, "peak_start_time", "peak_end_time")

        self._peak_end_time = new_end

    @property
    def winter_start(self) -> str:
        return self._winter_start

    @winter_start.setter
    def winter_start(self, value: str) -> None:
        if not (isinstance(value, str) and int(value) > 0 and int(value) <= 12 and len(value) == 2):
            raise ValueError(f"winter start must be a two digit string e.g. '01' for Jan or '12' for Dec, not {value}")
        self._winter_start = value

    @property
    def winter_end(self) -> str:
        return self._winter_end

    @winter_end.setter
    def winter_end(self, value: str) -> None:
        if not (isinstance(value, str) and int(value) > 0 and int(value) <= 12 and len(value) == 2):
            raise ValueError(f"winter end must be a two digit string e.g. '01' for Jan or '12' for Dec, not {value}")
        self._winter_end = value

    def generate_weights(self, start_date: str, end_date: str) -> pandas.Series:
        """Generate binary weights for a given peak-time schedule.

        Args:
            start_date: Date on which to start schedule (inclusive) as formatted date string (%Y-%m-%d)
            end_date: Date on which to end schedule (inclusive) as formatted date string (%Y-%m-%d)
        """
        # Create time index
        index = self._build_settlement_index(start_date, end_date)

        # Initialise weights as Series of 0s
        weights = pandas.Series(data = 0.0, index=index, dtype='float64')

        # Apply peak schedule options
        if self.peak_end_time == time(0, 0):
            in_peak = weights.index.time >= self.peak_start_time
        else:
            in_peak = (weights.index.time >= self.peak_start_time) & (weights.index.time < self.peak_end_time)
        weights[in_peak] = 1.0

        # apply weekend condition
        if self.weekdays_only:
            # If saturday or sunday set weight to 0.
            weights[weights.index.day_of_week.isin([5,6])] = 0.0

        # Apply winter condition
        if self.winter_only:
            # If outside of winter period, set weight to 0.
            if int(self.winter_start) <= int(self.winter_end):
                # E.g. winter is defined as nov-dec
                weights[~weights.index.month.isin(range(int(self.winter_start), int(self.winter_end) + 1))] = 0.0
            else: # winter wraps around e.g. Nov - Feb
                weights[~((weights.index.month >= int(self.winter_start))
                          | (weights.index.month <= int(self.winter_end)))] = 0.0

        return weights.rename("weight")

    def __repr__(self) -> str:
        params = [
            f"peak_start_time={self.peak_start_time.strftime('%H:%M')!r}",
            f"peak_end_time={self.peak_end_time.strftime('%H:%M')!r}",
            f"weekdays_only={self.weekdays_only!r}",
            f"winter_only={self.winter_only!r}"
            ]
        if self.winter_only:
            params.extend([
                f"winter_start={self.winter_start!r}",
                f"winter_end={self.winter_end!r}"
                ])

        return f"{self.__class__.__name__}({', '.join(params)})"

    def __str__(self) -> str:
        days = "weekdays only" if self.weekdays_only else "all days"
        season = (
            f", winter (months {self.winter_start}–{self.winter_end})"
            if self.winter_only
            else ""
        )
        return (
            f"Peak window: {self.peak_start_time.strftime("%H:%M")} to {self.peak_end_time.strftime("%H:%M")} "
            f"({days}{season})"
        )


class BandWindow:
    """Defines a single recurring time-of-use window and its relative weight.

    Args:
        name: Band label (e.g. 'Red', 'Amber').
        start_time: Start of window (inclusive) as formatted time string (%H:%M).
        end_time: End of window (exclusive) as formatted time string (%H:%M). Use '00:00' for midnight.
        weight: Non-negative relative rate weight for this window.
        days_of_week: Iterable of weekday integers 0 (Mon) to 6 (Sun). Default: all days.
        months: Optional iterable of active months 1 (Jan) to 12 (Dec). Default: None.
    """
    def __init__(
        self,
        name: str,
        start_time: str,
        end_time: str,
        weight: float,
        days_of_week: Iterable[int] = tuple(range(7)),
        months: Iterable[int] | None = None,
    ) -> None:
        self.name = name
        self.start_time = start_time
        self.end_time = end_time
        self.weight = weight
        self.days_of_week = days_of_week
        self.months = months

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("name must be a non-empty string.")
        self._name = value.strip()

    @property
    def start_time(self) -> time:
        return self._start_time

    @start_time.setter
    def start_time(self, value: str) -> None:
        new_start = _parse_half_hour(value, "start_time")
        if hasattr(self, "_end_time"):
            _validate_window(new_start, self._end_time, "start_time", "end_time")

        self._start_time = new_start

    @property
    def end_time(self) -> time:
        return self._end_time

    @end_time.setter
    def end_time(self, value: str) -> None:
        new_end = _parse_half_hour(value, "end_time")
        _validate_window(self.start_time, new_end, "start_time", "end_time")
        self._end_time = new_end

    @property
    def weight(self) -> float:
        return self._weight

    @weight.setter
    def weight(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"weight must be a non-negative number, not {value!r}.")
        self._weight = float(value)

    @property
    def days_of_week(self) -> tuple[int, ...]:
        return self._days_of_week

    @days_of_week.setter
    def days_of_week(self, value: Iterable[int]) -> None:
        """By convention 0=monday...6=sunday"""
        days = tuple(set(value))
        if not days or any(isinstance(d, bool) or not isinstance(d, int) or d not in range(7) for d in days):
            raise ValueError("days_of_week must be a non-empty iterable of integers between 0 and 6.")
        self._days_of_week = days

    @property
    def months(self) -> tuple[int, ...] | None:
        return self._months

    @months.setter
    def months(self, value: Iterable[int] | None) -> None:
        if value is None:
            self._months = None
            return
        month_tuple = tuple(set(value))
        if not month_tuple or any(
            isinstance(m, bool) or not isinstance(m, int) or m not in range(1, 13)
            for m in month_tuple
        ):
            raise ValueError("months must be None or a non-empty iterable of integers in 1..12.")
        self._months = month_tuple

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BandWindow):
            return NotImplemented
        return (
            self.name == other.name
            and self.start_time == other.start_time
            and self.end_time == other.end_time
            and self.weight == other.weight
            and set(self.days_of_week) == set(other.days_of_week)
            and (
                (self.months is None and other.months is None)
                or (self.months is not None and other.months is not None and set(self.months) == set(other.months))
            )
        )

    def __repr__(self) -> str:
        params = [
            f"name={self.name!r}",
            f"start_time={self.start_time.strftime('%H:%M')!r}",
            f"end_time={self.end_time.strftime('%H:%M')!r}",
            f"weight={self.weight!r}",
        ]
        if set(self.days_of_week) != set(range(7)):
            params.append(f"days_of_week={self.days_of_week!r}")
        if self.months is not None:
            params.append(f"months={self.months!r}")
        return f"{self.__class__.__name__}({', '.join(params)})"

    def __str__(self) -> str:
        days = (
            "weekdays only" if set(self.days_of_week) == set(range(5))
            else "weekends only" if set(self.days_of_week) == set(range(5,7))
            else "all days" if set(self.days_of_week) == set(range(7))
            else f"days {self.days_of_week}"
        )
        season = f", months {self.months}" if self.months is not None else ""
        return (
            f"{self.name}: {self.start_time.strftime('%H:%M')} to {self.end_time.strftime('%H:%M')} "
            f"({days}{season}, weight={self.weight})"
        )

class MultiBandSchedule(WeightingSchedule):
    """Generates half-hourly weights across multiple time-of-use bands (e.g. DUoS RAG).

    Periods not covered by any BandWindow receive `default_weight` (e.g. Green).
    If windows overlap, higher-weight windows take precedence over lower-weight ones.
    """

    def __init__(
        self,
        windows: Sequence[BandWindow],
        default_weight: float = 0.0,
        default_band_name: str = "Green",
    ) -> None:
        if not isinstance(default_weight, float):
            raise ValueError(f"default_weight must be a float, got {type(default_weight)}")
        
        if default_weight < 0:
            raise ValueError(f"default_weight must be non-negative, got {default_weight}")

        if any(not isinstance(w, BandWindow) for w in windows):
            raise TypeError("windows must contain only BandWindow instances.")

        if not isinstance(default_band_name, str) or not default_band_name.strip():
            raise ValueError("default_band_name must be a non-empty string.")

        self.windows: tuple[BandWindow, ...] = tuple(windows)
        self.default_weight: float = float(default_weight)
        self.default_band_name: str = default_band_name


    def generate_weights(self, start_date: str, end_date: str) -> pandas.Series:
        """Generate weights for a given peak-time schedule.

        Args:
            start_date: Date on which to start schedule (inclusive) as formatted date string (%Y-%m-%d)
            end_date: Date on which to end schedule (inclusive) as formatted date string (%Y-%m-%d)
        """
        # Create time index
        index = self._build_settlement_index(start_date, end_date)

        # create Series with default weights and index
        weights = pandas.Series(data=self.default_weight, index=index, dtype="float64")

        # Apply in ascending weight order so higher-tier bands (e.g. Red) cleanly
        # override broader underlying bands (e.g. Amber) if their windows overlap.
        for window in sorted(self.windows, key=lambda w: w.weight):
            if window.end_time == time(0, 0):
                mask = weights.index.time >= window.start_time
            else:
                mask = (weights.index.time >= window.start_time) & (weights.index.time < window.end_time)

            mask &= weights.index.day_of_week.isin(window.days_of_week)

            if window.months is not None:
                mask &= weights.index.month.isin(window.months)

            weights[mask] = float(window.weight)

        return weights.rename("weight")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"windows={list(self.windows)!r}, "
            f"default_weight={self.default_weight!r}, "
            f"default_band_name={self.default_band_name!r})"
        )

    def __str__(self) -> str:
        lines = [f"Default ({self.default_band_name}): weight={self.default_weight}"]
        for w in self.windows:
            days = (
                "weekdays" if set(w.days_of_week) == set(range(5))
                else "weekends" if set(w.days_of_week) == set(range(5,7))
                else "all days" if set(w.days_of_week) == set(range(7))
                else f"days={w.days_of_week}"
            )
            season = f", months={w.months}" if w.months else ""
            lines.append(
                f"  - {w.name}: {w.start_time.strftime("%H:%M")}-{w.end_time.strftime("%H:%M")} ({days}{season}) -> weight={w.weight}"
            )
        return "\n".join(lines)