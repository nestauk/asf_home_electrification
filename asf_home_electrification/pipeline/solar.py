import numpy as np
import pandas as pd

from asf_home_electrification.getters import pvgis


class SolarPV:
    def __init__(
        self,
        peak_power_kw: float,
        lat: float,
        lon: float,
        loss: float,
        mounting_place: str,  # "free" or "building"
        aspect: float,  # 0=south 90=west -90=east
        angle: float,
    ) -> None:
        """Initialise a solar PV system with its technical parameters and orientation.

        Args:
            peak_power_kw (float): Nominal power of the PV system in kW.
            lat (float): Latitude in decimal degrees. South is negative.
            lon (float): Longitude in decimal degrees. West is negative.
            loss (float): Sum of system losses in percent.
            mounting_place (str): Type of mounting of the PV modules. Choices are "free" for free-standing and
                "building" for building integrated.
            aspect (float): Orientation angle of the fixed PV system where 0=south, 90=west and -90=east.
            angle (float): Inclination angle from horizontal plane of the fixed PV system.

        Raises:
            ValueError: If invalid mounting place is given.
        """

        if mounting_place not in ("free", "building"):
            raise ValueError("Valid mounting place choices are 'free' or 'building'.")

        self.peak_power_kw = peak_power_kw
        self.lat = lat
        self.lon = lon
        self.loss = loss
        self.mounting_place = mounting_place
        self.aspect = aspect
        self.angle = angle

    def half_hourly_pvgis_data(self, year: int) -> pd.DataFrame:
        """Half-hourly PV generation and weather data for one year of PVGIS data.

        PVGIS returns hourly values. Each hourly row is duplicated into two
        half-hour rows, with the second 30 minutes after the first. Timestamps
        are converted from UTC to Europe/London so the index matches the demand
        and tariff profiles, and settlement columns are added in the same
        format as those profiles.

        Args:
            year (int): Calendar year of PVGIS data to return, between 2005 and 2023.
                Leap years (e.g. 2008, 2012, 2016, 2020) have 17,568 rows
                instead of 17,520.

        Returns:
            DataFrame indexed by local half-hourly timestamp, with columns:

            - generation_W: PV power output in W (PVGIS field "P") at the
            hourly time step.
            - generation_kWh: PV energy generated in the half-hour, in kWh.
            - G(i): global in-plane irradiance in W/m2.
            - H_sun: sun height (elevation) in degrees.
            - T2m: air temperature in degrees C.
            - WS10m: wind speed at 10 m in m/s.
            - Int: PVGIS flag column (0 throughout in the sample checked).
            - Settlement Date: local calendar date.
            - Settlement Period ID: elapsed half-hours since local midnight
              plus one, so it reaches 46 and 50 on clock-change days.
            - Settlement timestamp: local time as "HH:MM".

            The PVGIS fields are the hourly values repeated across both
            half-hours of the hour. generation_kWh is calculated from
            generation_W as power times half an hour. Any other columns the
            getter returns are repeated in the same way.

        Raises:
            ValueError: If the year is outside the range covered by the PVGIS
                data version in use, or PVGIS returned no data for it.
        """

        if not pvgis.PVGIS_FIRST_YEAR <= year <= pvgis.PVGIS_LAST_YEAR:
            raise ValueError(f"PVGIS data covers {pvgis.PVGIS_FIRST_YEAR} to {pvgis.PVGIS_LAST_YEAR}, got {year}.")

        pvgis_output = pvgis.get_hourly_radiation_data(
            lat=self.lat,
            lon=self.lon,
            peakpower=self.peak_power_kw,
            loss=self.loss,
            mountingplace=self.mounting_place,
            aspect=self.aspect,
            angle=self.angle,
        )

        pvgis_output["time"] = pd.to_datetime(pvgis_output["time"], format="%Y%m%d:%H%M")
        pvgis_output = pvgis_output.loc[pvgis_output["time"].dt.year == year]
        if pvgis_output.empty:
            raise ValueError(f"No PVGIS data for {year}.")
        pvgis_output = pvgis_output.rename(columns={"P": "generation_W"})

        # Two half-hour rows per hourly row
        half_hourly = pvgis_output.loc[pvgis_output.index.repeat(2)].reset_index(drop=True)
        offsets = pd.to_timedelta([0, 30] * len(pvgis_output), unit="min")
        half_hourly["time"] = half_hourly["time"].dt.floor("h") + offsets

        # Power (W) over a half-hour -> energy (kWh)
        half_hourly["generation_kWh"] = half_hourly["generation_W"] / 1000 * 0.5

        # PVGIS times are UTC, convert to local time to match demand and tariffs
        half_hourly["time"] = half_hourly["time"].dt.tz_localize("UTC").dt.tz_convert("Europe/London")
        half_hourly = half_hourly.set_index("time")
        half_hourly.index.name = "timestamp"

        # Settlement columns, built the same way as for the demand and price profiles
        local_midnight = half_hourly.index.normalize()
        half_hourly["Settlement Date"] = local_midnight.tz_localize(None)
        half_hourly["Settlement Period ID"] = ((half_hourly.index - local_midnight) / pd.Timedelta(minutes=30)).astype(
            int
        ) + 1
        half_hourly["Settlement timestamp"] = half_hourly.index.strftime("%H:%M")

        return half_hourly

    def generation_profile(self, index: pd.DatetimeIndex) -> pd.Series:
        """PV generation on a given time index, using the latest year of PVGIS weather.

        The latest PVGIS year is used as a representative year for any year being
        modelled. The index must be a full year of half-hourly timestamps with the
        same number of days as the PVGIS year, and must match it row for row in UTC
        month, day, hour and minute. Leap years and partial years are not supported.

        Args:
            index (pd.DatetimeIndex): Timezone-aware half-hourly timestamps to
                produce generation for, e.g. the index of a demand profile.

        Raises:
            ValueError: If the index has a different number of timestamps from
                the PVGIS year, has repeated month, day, hour and minute values
                in UTC, or differs from the PVGIS year in UTC month, day, hour
                and minute for any row.

        Returns:
            pd.Series: PV energy generated in each half-hour in kWh, on `index`.
        """

        # Use latest year as representative of any year
        pvgis_data = self.half_hourly_pvgis_data(year=pvgis.PVGIS_LAST_YEAR)

        # Check we can use the given index
        index_keys = index.tz_convert("UTC").strftime("%m-%d %H:%M")
        pvgis_keys = pvgis_data.index.tz_convert("UTC").strftime("%m-%d %H:%M")

        if len(index_keys) != len(pvgis_keys):
            raise ValueError(
                f"index has {len(index_keys)} timestamps but PVGIS {pvgis.PVGIS_LAST_YEAR} has "
                f"{len(pvgis_keys)}. A full year of half-hourly timestamps with the same number "
                "of days is required."
            )
        if not (index_keys.is_unique and pvgis_keys.is_unique):
            raise ValueError("index or PVGIS data has repeated month, day, hour and minute values in UTC.")
        if not (index_keys == pvgis_keys).all():
            raise ValueError(
                f"index differs from PVGIS {pvgis.PVGIS_LAST_YEAR} in UTC month, day, hour and minute "
                f"in {int((index_keys != pvgis_keys).sum())} rows."
            )

        return pd.Series(pvgis_data["generation_kWh"].to_numpy(), index=index, name="generation_kwh")

    def run_household_energy_balance(
        self,
        demand_profile: pd.Series,
        import_price_profile: pd.Series,
        export_price_profile: pd.Series,
    ) -> pd.DataFrame:
        """Household energy balance with solar PV and no battery.

        Solar generation is used in the home first, any surplus is exported, and
        any demand not covered by generation is imported. Surplus is curtailed
        (not exported) in intervals where the export price is negative.

        Generation uses the latest year of PVGIS weather as a representative
        year, matched to the profiles by UTC month, day, hour and minute. See
        `generation_profile` for the requirements on the index.

        Args:
            demand_profile (pd.Series): Household demand per half-hour in kWh.
            import_price_profile (pd.Series): Electricity import price per
                half-hour in p/kWh.
            export_price_profile (pd.Series): Electricity export price per
                half-hour in p/kWh.

        Raises:
            ValueError: If the inputs do not share the same index, the index is
                not a DatetimeIndex, or the index cannot be matched to the PVGIS
                year (see `generation_profile`).

        Returns:
            pd.DataFrame: One row per half-hour, on the same index as the
            inputs, with columns: demand_kwh, solar_generation_kwh,
            self_consumption_kwh (the lesser of demand and generation),
            grid_export_kwh, grid_import_kwh, import_price_p_per_kwh,
            export_price_p_per_kwh, import_cost_p, export_revenue_p and cost_p
            (net cost: import cost minus export revenue). Cost columns are in
            pence. Curtailed generation appears in solar_generation_kwh but in
            neither self_consumption_kwh nor grid_export_kwh.
        """

        if not (
            demand_profile.index.equals(import_price_profile.index)
            and demand_profile.index.equals(export_price_profile.index)
        ):
            raise ValueError("Inputs must share the same index.")
        if not isinstance(demand_profile.index, pd.DatetimeIndex):
            raise ValueError("Inputs must be indexed by a DatetimeIndex.")

        # Construction solar radiation profile series
        solar_generation_profile = self.generation_profile(index=demand_profile.index)

        # Construct household energy balance dataframe
        energy_balance = pd.DataFrame(index=demand_profile.index)
        energy_balance["demand_kwh"] = demand_profile.values
        energy_balance["solar_generation_kwh"] = solar_generation_profile.values

        # Self consumption is the lesser of generation and demand
        energy_balance["self_consumption_kwh"] = energy_balance[["demand_kwh", "solar_generation_kwh"]].min(axis=1)

        # Export is any surplus generation beyond what can be self consumed
        energy_balance["grid_export_kwh"] = np.where(
            export_price_profile.to_numpy() < 0,
            0.0,  # curtail if export prices are negative
            energy_balance["solar_generation_kwh"] - energy_balance["self_consumption_kwh"],
        )

        # Grid import is any demand not met by self consumption
        energy_balance["grid_import_kwh"] = energy_balance["demand_kwh"] - energy_balance["self_consumption_kwh"]

        # Import and export prices and costs
        energy_balance["import_price_p_per_kwh"] = import_price_profile.values
        energy_balance["export_price_p_per_kwh"] = export_price_profile.values

        energy_balance["import_cost_p"] = energy_balance["import_price_p_per_kwh"] * energy_balance["grid_import_kwh"]
        energy_balance["export_revenue_p"] = (
            energy_balance["export_price_p_per_kwh"] * energy_balance["grid_export_kwh"]
        )

        energy_balance["cost_p"] = energy_balance["import_cost_p"] - energy_balance["export_revenue_p"]

        return energy_balance

    def run_household_energy_balance_no_export(
        self,
        demand_profile: pd.Series,
        import_price_profile: pd.Series,
    ) -> pd.DataFrame:
        """Household energy balance with solar PV, no battery and no export.

        Solar generation is used in the home first, and any surplus is curtailed
        because export is not allowed. Any demand not covered by generation is
        imported.

        Generation uses the latest year of PVGIS weather as a representative
        year, matched to the profiles by UTC month, day, hour and minute. PVGIS
        data is converted to half-hourly intervals, so the demand and price
        profiles must both be half-hourly and share the same timezone-aware
        index. See `generation_profile` for the requirements on the index.

        Args:
            demand_profile (pd.Series): Household demand per half-hour in kWh.
            import_price_profile (pd.Series): Electricity import price per
                half-hour in p/kWh.

        Raises:
            ValueError: If the inputs do not share the same index, the index is
                not a DatetimeIndex, or the index cannot be matched to the PVGIS
                year (see `generation_profile`).

        Returns:
            pd.DataFrame: One row per half-hour, on the same index as the
            inputs, with columns: demand_kwh, solar_generation_kwh,
            self_consumption_kwh (the lesser of demand and generation),
            curtailed_kwh (surplus generation that is lost), grid_import_kwh,
            import_price_p_per_kwh, import_cost_p and cost_p (pence).
        """
        if not (demand_profile.index.equals(import_price_profile.index)):
            raise ValueError("Inputs must share the same index.")
        if not isinstance(demand_profile.index, pd.DatetimeIndex):
            raise ValueError("Inputs must be indexed by a DatetimeIndex.")

        # Construction solar radiation profile series
        solar_generation_profile = self.generation_profile(index=demand_profile.index)

        # Construct household energy balance dataframe
        energy_balance = pd.DataFrame(index=demand_profile.index)
        energy_balance["demand_kwh"] = demand_profile.values
        energy_balance["solar_generation_kwh"] = solar_generation_profile.values

        # Self consumption is the lesser of generation and demand
        energy_balance["self_consumption_kwh"] = energy_balance[["demand_kwh", "solar_generation_kwh"]].min(axis=1)

        # Curtailment is any surplus generation beyond what can be self consumed
        energy_balance["curtailed_kwh"] = (
            energy_balance["solar_generation_kwh"] - energy_balance["self_consumption_kwh"]
        )

        # Grid import is any demand not met by self consumption
        energy_balance["grid_import_kwh"] = energy_balance["demand_kwh"] - energy_balance["self_consumption_kwh"]

        # Import prices and costs
        energy_balance["import_price_p_per_kwh"] = import_price_profile.values
        energy_balance["import_cost_p"] = energy_balance["import_price_p_per_kwh"] * energy_balance["grid_import_kwh"]

        return energy_balance
