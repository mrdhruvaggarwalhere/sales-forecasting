"""
Carbon Emission Calculator — Environmental Impact Estimation.

Computes fuel consumption, CO₂ emissions, and logistics costs for
both naive (individual) and optimized (consolidated) shipping scenarios.

Usage:
    from src.sustainability.carbon_calculator import compare_scenarios
    report = compare_scenarios(naive_trips=15, optimized_trips=8, config=config)
"""

import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EmissionMetrics:
    """Emission and cost metrics for a single logistics scenario."""
    num_trips: int
    total_distance_km: float
    fuel_litres: float
    co2_kg: float
    shipping_cost: float
    fuel_cost: float
    total_cost: float


@dataclass
class CarbonReport:
    """Comparison of naive vs optimized logistics."""
    naive: EmissionMetrics
    optimized: EmissionMetrics
    trips_saved: int
    fuel_saved_litres: float
    co2_saved_kg: float
    cost_saved: float
    reduction_pct: float  # % reduction in CO₂


def calculate_emissions(num_trips: int, config) -> EmissionMetrics:
    """
    Calculates total emissions and costs for a given number of trips.

    Args:
        num_trips: Number of delivery round trips.
        config: SustainabilityConfig instance.

    Returns:
        EmissionMetrics with all computed values.
    """
    if num_trips <= 0:
        return EmissionMetrics(
            num_trips=0, total_distance_km=0, fuel_litres=0,
            co2_kg=0, shipping_cost=0, fuel_cost=0, total_cost=0,
        )

    total_distance = config.round_trip_km * num_trips
    fuel = total_distance * config.FUEL_CONSUMPTION_RATE
    co2 = fuel * config.CO2_EMISSION_FACTOR
    shipping_cost = config.SHIPPING_COST_PER_TRIP * num_trips
    fuel_cost = fuel * config.FUEL_COST_PER_KM
    total_cost = shipping_cost + fuel_cost

    return EmissionMetrics(
        num_trips=num_trips,
        total_distance_km=round(total_distance, 1),
        fuel_litres=round(fuel, 2),
        co2_kg=round(co2, 2),
        shipping_cost=round(shipping_cost, 2),
        fuel_cost=round(fuel_cost, 2),
        total_cost=round(total_cost, 2),
    )


def compare_scenarios(naive_trips: int, optimized_trips: int, config) -> CarbonReport:
    """
    Compares the environmental impact of naive vs optimized logistics.

    Args:
        naive_trips: Total trips without consolidation (1 per reorder event).
        optimized_trips: Total trips after consolidation.
        config: SustainabilityConfig instance.

    Returns:
        CarbonReport with both scenarios and computed savings.
    """
    logger.info("Calculating carbon emissions for both scenarios...")

    naive = calculate_emissions(naive_trips, config)
    optimized = calculate_emissions(optimized_trips, config)

    trips_saved = naive.num_trips - optimized.num_trips
    fuel_saved = naive.fuel_litres - optimized.fuel_litres
    co2_saved = naive.co2_kg - optimized.co2_kg
    cost_saved = naive.total_cost - optimized.total_cost

    reduction_pct = (co2_saved / naive.co2_kg * 100) if naive.co2_kg > 0 else 0.0

    report = CarbonReport(
        naive=naive,
        optimized=optimized,
        trips_saved=trips_saved,
        fuel_saved_litres=round(fuel_saved, 2),
        co2_saved_kg=round(co2_saved, 2),
        cost_saved=round(cost_saved, 2),
        reduction_pct=round(reduction_pct, 1),
    )

    logger.info(f"  Naive:     {naive.num_trips} trips, {naive.co2_kg:.1f} kg CO₂, ${naive.total_cost:,.0f}")
    logger.info(f"  Optimized: {optimized.num_trips} trips, {optimized.co2_kg:.1f} kg CO₂, ${optimized.total_cost:,.0f}")
    logger.info(f"  Savings:   {trips_saved} trips, {co2_saved:.1f} kg CO₂, ${cost_saved:,.0f} ({reduction_pct:.1f}%)")

    return report
