"""Hot water tank model for a heating system.

A fully mixed cylinder: one volume at one temperature. Each half hour it loses heat to the
draw (cold feed water replaces what is drawn) and to the room (standing loss), and the
heating system adds heat towards a target temperature, up to its capacity.

Conventions:
    - Energies are kWh over the half hour, temperatures are degC.
    - The tank's temperature is not stored on the object. The caller passes the start
      temperature to each step and carries the end temperature to the next one.

Simplifications:
    - Fully mixed, so no stratification. A real tank keeps its top hot for longer than this
      model suggests.
    - The tank is never actively cooled. Above its target it just drifts down.
    - Litres drawn are litres of tank water, so the heat removed depends on tank temperature.

Typical use:
    tank = HotWaterTank(
        volume_litres=210, t_cold_feed=10.0, t_hot_water=55.0, t_room=20.0,
        ref_standing_loss_kwh_per_day=1.01, ref_delta_t=45.0,
    )
    result = tank.run_half_hour_step(t_start=55.0, draw_litres=20.0, heating_system_size_kw=8.0)
"""

WATER_KJ_PER_L_K = 4.184  # specific heat capacity of water, taking 1 L = 1 kg


class HotWaterTank:
    """A fully mixed hot water cylinder.

    Args:
        volume_litres (float): Tank volume, in litres.
        t_cold_feed (float): Cold feed water temperature, in degC.
        t_hot_water (float): Temperature the water is stored at, in degC. Also the default
            target when heating the tank.
        t_room (float): Temperature of the space the tank sits in, in degC.
        ref_standing_loss_kwh_per_day (float): Standing heat loss as quoted on the technical
            specification, in kWh/24h.
        ref_delta_t (float): Tank-to-room temperature difference the quoted standing loss was
            measured at, in K. For example 45 for a 65 degC tank in a 20 degC room.

    Raises:
        ValueError: If the volume or reference temperature difference is not positive, the
            quoted standing loss is negative, or t_cold_feed or t_room is not below
            t_hot_water.
    """

    def __init__(
        self,
        volume_litres: float,
        t_cold_feed: float,
        t_hot_water: float,
        t_room: float,
        ref_standing_loss_kwh_per_day: float,
        ref_delta_t: float,
    ) -> None:

        if volume_litres <= 0:
            raise ValueError(f"volume_litres must be positive, got {volume_litres}")
        if ref_standing_loss_kwh_per_day < 0:
            raise ValueError("ref_standing_loss_kwh_per_day cannot be negative")
        if ref_delta_t <= 0:
            raise ValueError(f"ref_delta_t must be positive, got {ref_delta_t}")
        if not t_cold_feed < t_hot_water:
            raise ValueError(f"t_cold_feed ({t_cold_feed}) must be below t_hot_water ({t_hot_water})")
        if not t_room < t_hot_water:
            raise ValueError(f"t_room ({t_room}) must be below t_hot_water ({t_hot_water})")

        self.volume_litres = volume_litres
        self.t_cold_feed = t_cold_feed
        self.t_hot_water = t_hot_water
        self.t_room = t_room
        self.ref_standing_loss_kwh_per_day = ref_standing_loss_kwh_per_day
        self.ref_delta_t = ref_delta_t

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(volume_litres={self.volume_litres}, t_cold_feed={self.t_cold_feed}, "
            f"t_hot_water={self.t_hot_water}, t_room={self.t_room}, "
            f"ref_standing_loss_kwh_per_day={self.ref_standing_loss_kwh_per_day}, "
            f"ref_delta_t={self.ref_delta_t})"
        )

    @property
    def heat_capacity_kwh_per_k(self) -> float:
        """Energy to raise the whole tank by 1 K, in kWh/K."""
        return self.volume_litres * WATER_KJ_PER_L_K / 3600

    @property
    def heat_loss_coefficient_w_per_k(self) -> float:
        """Heat lost to the room per kelvin of tank-to-room difference, in W/K.

        Derived from the quoted standing loss and the temperature difference it was
        measured at. The step method uses this with the current tank temperature.
        """
        return self.ref_standing_loss_kwh_per_day * 1000 / 24 / self.ref_delta_t

    @property
    def standing_loss_kwh_day(self) -> float:
        """Standing loss with the tank held at t_hot_water in a room at t_room, in kWh/day.

        The quoted loss scaled by actual / reference temperature difference. This is a summary
        figure for checking and comparing tanks, not what the step uses, because the tank
        temperature changes through the day.
        """
        actual_delta_t = self.t_hot_water - self.t_room
        return self.ref_standing_loss_kwh_per_day * actual_delta_t / self.ref_delta_t

    def run_half_hour_step(
        self,
        *,
        t_start: float,
        draw_litres: float,
        heating_system_size_kw: float,
        t_target: float | None = None,
    ) -> dict:
        """Advance the tank by one half-hour timestep.

        Removes the heat of the draw and the standing loss, then heats the tank towards
        t_target, limited by the heating system's capacity. If the capacity is enough, the
        tank ends the step at t_target. The tank is never actively cooled, so a tank above
        its target just drifts down.

        Args:
            t_start (float): Tank temperature at the start of the step, in degC.
            draw_litres (float): Hot water drawn during the step, in litres. Cold feed water
                replaces it immediately. Must be at least 0 and below the tank volume.
            heating_system_size_kw (float): Maximum thermal output of the heating system, in
                kW. Limits the heat that can go into the tank in this step.
            t_target (float | None, optional): Temperature to heat the tank towards, in degC.
                Defaults to self.t_hot_water.

        Raises:
            ValueError: If draw_litres is negative or not below the tank volume.

        Returns:
            dict: Results for the half-hour step, with these keys:
                - q_dhw_kwh: heat delivered to the tank by the heating system (kWh).
                - q_dhw_draw_kwh: heat removed by the draw (kWh).
                - q_dhw_standing_kwh: heat lost from the tank to the room (kWh).
                - t_tank_end_c: tank temperature at the end of the step (degC).
        """

        if not 0 <= draw_litres < self.volume_litres:
            raise ValueError(f"draw_litres must be between 0 and the tank volume, got {draw_litres}")

        t_target = self.t_hot_water if t_target is None else t_target

        dt = 0.5  # hours
        max_heat_kwh = heating_system_size_kw * dt  # kWh: kW for half an hour

        # heat needed to replace drawn water
        q_draw = draw_litres * WATER_KJ_PER_L_K / 3600 * (t_start - self.t_cold_feed)  # kWh

        # heat needed to replace standing losses
        q_standing = self.heat_loss_coefficient_w_per_k * (t_start - self.t_room) * dt / 1000  # kWh

        # heat needed to reach hot water target
        q_required = max(self.heat_capacity_kwh_per_k * (t_target - t_start) + q_draw + q_standing, 0.0)  # kWh
        q_delivered = min(q_required, max_heat_kwh)  # kWh

        t_end = t_start + (q_delivered - q_draw - q_standing) / self.heat_capacity_kwh_per_k  # degC

        return {
            "q_dhw_kwh": q_delivered,
            "q_dhw_draw_kwh": q_draw,
            "q_dhw_standing_kwh": q_standing,
            "t_tank_end_c": t_end,
        }
