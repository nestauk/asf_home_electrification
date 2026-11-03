from abc import ABC, abstractmethod
from datetime import date, time, timedelta

import pandas


class WeightingSchedule(ABC):
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
        winter_only: If True, weights are 0 outside of winter_start and winter_stop. Default: False.
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
        try:
            parsed_time = time.fromisoformat(value)
        except ValueError:
            raise ValueError(f"peak_start_time '{value}' was not recognised as a valid time of format %H:%M") from None

        if parsed_time.minute not in (0,30):
            raise ValueError(f"peak_start_time must be on either the hour or half-hour, not '{value}'.")

        self._peak_start_time = time(parsed_time.hour, parsed_time.minute)

    @property
    def peak_end_time(self) -> time:
        return self._peak_end_time

    @peak_end_time.setter
    def peak_end_time(self, value: str) -> None:
        try:
            parsed_time = time.fromisoformat(value)
        except ValueError:
            raise ValueError(f"peak_end_time '{value}' was not recognised as a valid time of format %H:%M") from None

        if parsed_time.minute not in (0,30):
            raise ValueError(f"peak_end_time must be on either the hour or half-hour, not '{value}'.")

        if parsed_time <= self.peak_start_time:
            if parsed_time == time(0,0) and self.peak_start_time == time(23,30):
                # Handle special case in which you specifically want a peak from 23:30 to 00:00
                pass
            else:
                raise ValueError("peak_end_time must be later than peak_start_time.")

        self._peak_end_time = time(parsed_time.hour, parsed_time.minute)

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
        try:
            start_date = date.fromisoformat(start_date)
        except ValueError:
            raise ValueError(
                f"start_date '{start_date}' was not recognised as a valid date of format %Y-%m-%d"
                ) from None

        try:
            end_date = date.fromisoformat(end_date)
        except ValueError:
            raise ValueError(f"end_date '{end_date}' was not recognised as a valid date of format %Y-%m-%d") from None

        if end_date < start_date:
            raise ValueError("end_date must be after start_date.")

        # Create time index
        index = pandas.date_range(
            start_date,
            end_date + timedelta(days=1),
            freq="30min",
            tz='Europe/London',
            inclusive='left')

        # Initialise weights as Series of 0s
        weights = pandas.Series(data = 0, index=index)

        # Apply peak schedule options
        if self.peak_end_time == time(0, 0):
            in_peak = weights.index.time >= self.peak_start_time
        else:
            in_peak = (weights.index.time >= self.peak_start_time) & (weights.index.time < self.peak_end_time)
        weights[in_peak] = 1

        # apply weekend condition
        if self.weekdays_only:
            # If saturday or sunday set weight to 0.
            weights[weights.index.day_of_week.isin([5,6])] = 0

        # Apply winter condition
        if self.winter_only:
            # If outside of winter period, set weight to 0.
            if int(self.winter_start) <= int(self.winter_end):
                # E.g. winter is defined as nov-dec
                weights[~weights.index.month.isin(range(int(self.winter_start), int(self.winter_end) + 1))] = 0
            else: # winter wraps around e.g. Nov - Feb
                weights[~((weights.index.month >= int(self.winter_start))
                          | (weights.index.month <= int(self.winter_end)))] = 0

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
