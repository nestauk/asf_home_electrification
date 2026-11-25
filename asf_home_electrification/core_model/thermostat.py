"""Thermostat schedules for the heating model.

A Thermostat holds daily setpoint schedules (one for weekdays, one for weekends) and a
fixed-lead-time preheat rule. It produces half-hourly Series on a Europe/London index:

    - setpoints(): the scheduled temperature in each half hour.
    - targets(): the temperature the system should heat towards, which is the setpoint
      brought forward by the preheat lead time before each rise.

The same class works for any temperature schedule, such as hot water tank targets
(use setpoints() and ignore targets(), as preheating does not apply to a tank).

Conventions:
    - Schedule times are UK local clock time. The returned index is tz-aware Europe/London,
      so it has 46 or 50 steps on the clock-change days.
    - Temperatures are in degC.

Typical use:
    thermostat = Thermostat(
        weekday={"00:00": 16.0, "06:30": 21.0, "08:30": 16.0, "17:00": 21.0, "22:30": 16.0},
        weekend={"00:00": 16.0, "07:30": 21.0, "22:30": 16.0},
        preheat_minutes=60,
    )
    t_setpoint = thermostat.setpoints("2026-01-05", "2026-01-12")
    t_target = thermostat.targets(t_setpoint)
"""

from dataclasses import dataclass
from datetime import time

import pandas as pd

TIMESTEP = 30  # minutes, length of one model step


def _parse_schedule(schedule: dict[str, float]) -> list[tuple[time, float]]:
    """Parse a schedule into [(start time, setpoint), ...] sorted by time.

    Args:
        schedule: {"HH:MM": temperature in degC}. Each time is when that temperature
            takes effect, and it holds until the next time.

    Raises:
        ValueError: If the schedule is empty, has duplicate times, has a time that is not
            a valid zero-padded "HH:MM", or does not start at "00:00".

    Returns:
        The (time, setpoint) pairs in chronological order.
    """
    if not schedule:
        raise ValueError("Given schedule is empty")
    parsed = sorted((time.fromisoformat(t), float(v)) for t, v in schedule.items())
    if len({t for t, _ in parsed}) != len(parsed):
        raise ValueError("Given schedule has duplicate times")
    if parsed[0][0] != time(0, 0):
        raise ValueError("Given schedule must start at '00:00'")
    return parsed


@dataclass
class Thermostat:
    """Thermostat settings: daily setpoint schedules plus a fixed preheat rule.

    Args:
        weekday (dict[str, float]): Schedule for Monday to Friday, as {"HH:MM": setpoint in
            degC}. Each time is when that setpoint takes effect, and it holds until the
            next one. Must include "00:00".
        weekend (dict[str, float]): Same, for Saturday and Sunday.
        preheat_minutes (int, optional): How long before each setpoint rise to start heating
            towards the new setpoint. Must be a multiple of the timestep (30). Defaults to 0.
        min_rise (float, optional): Smallest setpoint rise, in degC, that triggers preheating.
            Defaults to 0.5.

    Raises:
        ValueError: If a schedule is invalid (see _parse_schedule) or preheat_minutes is
            negative or not a multiple of the timestep.
    """

    weekday: dict[str, float]
    weekend: dict[str, float]
    preheat_minutes: int = 0
    min_rise: float = 0.5

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Check the settings are usable. Called at construction and by each method, because
        the fields can be changed after construction.

        Raises:
            ValueError: If preheat_minutes is invalid or either schedule is invalid.
        """
        if self.preheat_minutes < 0 or self.preheat_minutes % TIMESTEP != 0:
            raise ValueError(f"preheat_minutes must be a non-negative multiple of {TIMESTEP} minutes.")
        _parse_schedule(self.weekday)
        _parse_schedule(self.weekend)

    def setpoints(self, start: str, end: str) -> pd.Series:
        """Build the comfort setpoint for every half hour in a period.

        Saturday and Sunday use the weekend schedule and other days use the weekday one.
        Days are local days, and times are compared against local clock time.

        Args:
            start (str): Start of the period, as UK local time (inclusive), e.g. "2026-01-05".
            end (str): End of the period, as UK local time (exclusive).

        Raises:
            ValueError: If the settings are invalid or the period contains no half hours.

        Returns:
            pd.Series: Setpoint in degC for each half hour, named "t_setpoint_c", indexed by
                tz-aware Europe/London timestamps.
        """
        self._validate()
        index = pd.date_range(start, end, freq="30min", inclusive="left", tz="Europe/London")
        if len(index) == 0:
            raise ValueError(f"No time intervals between {start!r} and {end!r}")

        weekday_sched = _parse_schedule(self.weekday)
        weekend_sched = _parse_schedule(self.weekend)

        values = []
        for timestamp in index:
            schedule = weekend_sched if timestamp.day_of_week >= 5 else weekday_sched
            # latest change that has already started
            values.append(next(v for t, v in reversed(schedule) if t <= timestamp.time()))
        return pd.Series(values, index=index, name="t_setpoint_c")

    def targets(self, t_setpoint: pd.Series) -> pd.Series:
        """Build the heating target for every half hour, preheating ahead of setpoint rises.

        Whenever the setpoint rises by at least min_rise, the new setpoint is applied as the
        target for preheat_minutes before the rise. The target is never lower than the
        setpoint, so overlapping windows keep the higher value.

        Preheating only uses steps inside the t_setpoint series, so a rise within preheat_minutes
        of the first step gets a shortened preheat window. E.g., a 120 minute preheat before a 01:00
        rise on the first day only gets 60 minutes, from 00:00.

        Args:
            t_setpoint (pd.Series): Scheduled setpoints, e.g. from setpoints().

        Raises:
            ValueError: If the settings are invalid.

        Returns:
            pd.Series: Heating target in degC, named "t_target_c", with the same index as
                t_setpoint.
        """
        self._validate()
        n_steps = self.preheat_minutes // TIMESTEP
        setpoints = t_setpoint.tolist()
        targets = setpoints.copy()

        for i in range(1, len(setpoints)):
            if setpoints[i] - setpoints[i - 1] >= self.min_rise:
                for j in range(max(i - n_steps, 0), i):
                    targets[j] = max(targets[j], setpoints[i])  # never lower an existing target

        return pd.Series(targets, index=t_setpoint.index, name="t_target_c")
