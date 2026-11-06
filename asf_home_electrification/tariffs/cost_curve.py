from typing import Optional

import pandas

from asf_home_electrification.tariffs.schedule import WeightingSchedule


class TransmissionCostCurve:
    """Define a transmission cost curve.
    
    Attributes:
        revenue: Total domestic transmission revenue in £.
        alpha: Balancing factor that directs cost recovery proportionately to the locational demand (alpha=0) or the residual (alpha=1).
        beta: Balancing factor that directs the residual to either a volumetric (0) or standing (1) charge.
        schedule: WeightingSchedule that defines when peak periods occur.
    """
    def __init__(self, revenue: float, alpha: float, beta: float, schedule: Optional[WeightingSchedule]=None) -> None:
        """Initialises a TransmissionCostCurve.

        Args:
            revenue: Total domestic transmission revenue in £.
            alpha: Balancing factor between locational demand and residual.
            beta: Residual balancing factor between volumetric or fixed (standing) charge.
            schedule: Optional WeightingSchedule that defines when peak periods occur.
        """
        self.revenue = revenue
        self.alpha = alpha
        self.beta = beta
        self.schedule = schedule

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"alpha must be between 0 and 1, got {value}")
        self._alpha = value

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"beta must be between 0 and 1, got {value}")
        self._beta = value

    @property
    def locational_revenue(self) -> float:
        return self.revenue * (1.0 - self.alpha)

    @property
    def residual_revenue(self) -> float:
        return self.revenue * self.alpha
    
    @property
    def residual_standing_revenue(self) -> float:
        """Total residual revenue to be recovered via standing charges."""
        return self.residual_revenue * self.beta

    @property
    def residual_volumetric_revenue(self) -> float:
        """Total residual revenue to be recovered via volumetric charges."""
        return self.residual_revenue * (1.0 - self.beta)

    def calculate_standing_charge(self, number_of_meters: int, days: int = 365) -> float:
        """Calculates the standing charge in p/meter/day.
        
        Args:
            number_of_meters: Number of meters over which to spread the standing charge.
            days: Denominator for daily charges (default: 365)
        """
        if number_of_meters <= 0:
            raise ValueError("Number of meters must be a positive integer.")

        if days <= 0:
            raise ValueError("days must be a positive integer.")

        return self.residual_standing_revenue * 100 / number_of_meters / days

    def calculate_residual_volumetric_charge(self, total_consumption_kwh: float) -> float:
        """Calculates the flat volumetric charge from the residual in p/kWh.
        
        Args:
            total_consumption_kwh: total consumption over which to spread the residual (kWh).
        """
        if total_consumption_kwh <= 0:
            raise ValueError("Total consumption must be a positive number.")

        return self.residual_volumetric_revenue * 100 / total_consumption_kwh

    def get_weights(self, start_date: str, end_date: str) -> pandas.Series:
        """Generates weights from schedule."""
        if not self.schedule:
            raise ValueError(
                "You have not specified a WeightingSchedule, pass one to 'schedule' on object creation."
                )
        return self.schedule.generate_weights(start_date, end_date)

    def calculate_peak_locational_rate(self, consumption_kwh: pandas.Series, weights: Optional[pandas.Series]=None) -> float:
        """Calculates the unit rate applied to active settlement periods in p/kWh.
        
        The consumption_kwh argument should be a pandas Series with a datetime index of settlement periods.

        The consumption_kwh index min and max are used to create the weights and so must not have gaps.

        Args:
            consumption_kwh: Half-hourly consumption volumes as a pandas Series.
            weights: Optionally provide precomputed weights.
        """
        if consumption_kwh.empty:
            raise ValueError("consumption_kwh cannot be empty.")
        
        if self.locational_revenue <= 0.0:
            # Short cut calculations where there is no locational_revenue
            return 0.0

        if weights is None:
            weights = self.get_weights(
                consumption_kwh.index.min().strftime("%Y-%m-%d"),
                consumption_kwh.index.max().strftime("%Y-%m-%d"))

        if not consumption_kwh.index.equals(weights.index):
            raise ValueError("schedule index and consumption index do not match.")
        
        peak_consumption = (consumption_kwh * weights).sum()

        if peak_consumption <= 0:
            raise ValueError("Total consumption during weighted periods must be > 0.")

        return (self.locational_revenue * 100) / peak_consumption

    def calculate_cost_curve(self, consumption_kwh: pandas.Series,
                             residual_consumption_kwh: Optional[float]=None,
                             weights: Optional[pandas.Series]=None) -> pandas.DataFrame:
        """Returns the unit rate (p/kWh) for every settlement period.
        
        The consumption_kwh argument should be a pandas Series with a datetime index of settlement periods.
        
        The consumption_kwh index min and max are used to create the weights and so must not have gaps.

        If residual_consumption_kwh is None, volumetric residual uses consumption_kwh if needed.

        Args:
            consumption_kwh: Half-hourly consumption volumes as a pandas Series.
            residual_consumption_kwh: optional total consumption value used for volumetric residual.
            weights: optionally provide pre-computed weights.
        """

        if consumption_kwh.empty:
            raise ValueError("consumption_kwh cannot be empty.")

        if self.locational_revenue > 0:
            if weights is None:
                weights = self.get_weights(
                    consumption_kwh.index.min().strftime("%Y-%m-%d"),
                    consumption_kwh.index.max().strftime("%Y-%m-%d"))
            peak_rate = self.calculate_peak_locational_rate(consumption_kwh, weights)
            locational_series = weights * peak_rate
        else:
            locational_series = 0.0

        if (residual_consumption_kwh is not None) and (self.residual_volumetric_revenue > 0):
            residual_rate = self.calculate_residual_volumetric_charge(residual_consumption_kwh)
        elif self.residual_volumetric_revenue > 0:
            residual_rate = self.calculate_residual_volumetric_charge(consumption_kwh.sum())
        else:
            residual_rate = 0.0

        curve = pandas.DataFrame(
            {
                "locational": locational_series,
                "residual": residual_rate,
            },
            index=consumption_kwh.index,
        )
        curve["total"] = curve["locational"] + curve["residual"]
        return curve
