"""
Sustainability Configuration — Business Rules & Emission Factors.

All parameters are configurable via environment variables.
Import this module anywhere:
    from config.sustainability_config import SustainabilityConfig
"""

import os
from dataclasses import dataclass, field


@dataclass
class SustainabilityConfig:
    """
    Centralized configuration for the sustainability module.
    All values can be overridden via environment variables prefixed with SUST_.
    """

    # ── Inventory Rules ───────────────────────────────────────
    REORDER_THRESHOLD_PCT: float = 0.25
    """Reorder when inventory falls below this % of max capacity."""

    MAX_DELAY_DAYS: int = 3
    """Maximum days an order can be delayed for consolidation."""

    MAX_INVENTORY_CAPACITY: int = 2000
    """Maximum units a store can hold per product."""

    INITIAL_INVENTORY_PCT: float = 0.70
    """Starting inventory as a fraction of max capacity."""

    # ── Logistics Costs ───────────────────────────────────────
    SHIPPING_COST_PER_TRIP: float = 450.0
    """Cost in USD per delivery trip."""

    FUEL_COST_PER_KM: float = 0.85
    """Diesel cost in USD per kilometre."""

    DISTANCE_KM: float = 150.0
    """Average one-way warehouse-to-store distance in km."""

    # ── Emission Factors ──────────────────────────────────────
    CO2_EMISSION_FACTOR: float = 2.68
    """Kilograms of CO₂ emitted per litre of diesel burned."""

    FUEL_CONSUMPTION_RATE: float = 0.35
    """Litres of diesel consumed per km (heavy delivery truck)."""

    # ── Truck Capacity ────────────────────────────────────────
    TRUCK_CAPACITY_UNITS: int = 5000
    """Maximum product units a single truck can carry."""

    def __post_init__(self):
        """Override defaults with environment variables if set."""
        env_map = {
            "SUST_REORDER_THRESHOLD": ("REORDER_THRESHOLD_PCT", float),
            "SUST_MAX_DELAY_DAYS": ("MAX_DELAY_DAYS", int),
            "SUST_MAX_INVENTORY": ("MAX_INVENTORY_CAPACITY", int),
            "SUST_INITIAL_INVENTORY": ("INITIAL_INVENTORY_PCT", float),
            "SUST_SHIPPING_COST": ("SHIPPING_COST_PER_TRIP", float),
            "SUST_FUEL_COST_KM": ("FUEL_COST_PER_KM", float),
            "SUST_DISTANCE_KM": ("DISTANCE_KM", float),
            "SUST_CO2_FACTOR": ("CO2_EMISSION_FACTOR", float),
            "SUST_FUEL_RATE": ("FUEL_CONSUMPTION_RATE", float),
            "SUST_TRUCK_CAPACITY": ("TRUCK_CAPACITY_UNITS", int),
        }
        for env_key, (attr, cast) in env_map.items():
            val = os.getenv(env_key)
            if val is not None:
                try:
                    setattr(self, attr, cast(val))
                except (ValueError, TypeError):
                    pass  # Keep default if env var is malformed

    @property
    def reorder_threshold_units(self) -> int:
        """Absolute reorder threshold in units."""
        return int(self.REORDER_THRESHOLD_PCT * self.MAX_INVENTORY_CAPACITY)

    @property
    def initial_inventory_units(self) -> int:
        """Initial inventory level in units."""
        return int(self.INITIAL_INVENTORY_PCT * self.MAX_INVENTORY_CAPACITY)

    @property
    def round_trip_km(self) -> float:
        """Round-trip distance in km."""
        return self.DISTANCE_KM * 2.0

    @property
    def fuel_per_trip(self) -> float:
        """Litres of fuel consumed per round trip."""
        return self.round_trip_km * self.FUEL_CONSUMPTION_RATE

    @property
    def co2_per_trip(self) -> float:
        """Kilograms of CO₂ emitted per round trip."""
        return self.fuel_per_trip * self.CO2_EMISSION_FACTOR

    @property
    def total_cost_per_trip(self) -> float:
        """Total cost per trip (shipping + fuel)."""
        fuel_cost = self.fuel_per_trip * self.FUEL_COST_PER_KM
        return self.SHIPPING_COST_PER_TRIP + fuel_cost


# ── Singleton instance ────────────────────────────────────────
DEFAULT_CONFIG = SustainabilityConfig()
