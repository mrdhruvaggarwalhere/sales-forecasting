"""
Shipment Optimizer — AI Consolidation Engine.

Simulates per-product inventory over a forecast horizon, detects reorder points,
and consolidates nearby orders to reduce total shipment trips.

Usage:
    from src.sustainability.shipment_optimizer import run_optimization
    result = run_optimization(product_forecasts, config)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class ReorderEvent:
    """A single product reorder event detected during inventory simulation."""
    product: str
    day: int
    date: str
    quantity_needed: int
    current_inventory: int
    urgency: str  # "critical", "warning", "planned"


@dataclass
class ConsolidatedShipment:
    """A group of reorder events combined into one trip."""
    day: int
    date: str
    products: List[str]
    quantities: Dict[str, int]
    total_units: int
    products_consolidated: int
    savings_vs_separate: int  # number of trips saved


@dataclass
class InventorySnapshot:
    """Daily inventory state for a single product."""
    day: int
    date: str
    product: str
    inventory_level: int
    demand: int
    reorder_triggered: bool
    refill_amount: int


@dataclass
class OptimizationResult:
    """Complete output of the optimization engine."""
    naive_trips: int
    optimized_trips: int
    trips_saved: int
    consolidated_shipments: List[ConsolidatedShipment]
    reorder_events: List[ReorderEvent]
    inventory_timeline: Dict[str, List[InventorySnapshot]]
    products_analyzed: int


# ── Inventory Simulation ──────────────────────────────────────────────────────

def simulate_inventory(
    product: str,
    daily_demand: List[float],
    dates: List[str],
    config,
) -> Tuple[List[ReorderEvent], List[InventorySnapshot]]:
    """
    Simulates daily inventory for a single product over the forecast horizon.

    Args:
        product: Product identifier (e.g., "FOODS_3_090").
        daily_demand: List of forecasted daily demand values.
        dates: List of date strings corresponding to each day.
        config: SustainabilityConfig instance.

    Returns:
        Tuple of (reorder_events, inventory_snapshots).
    """
    events: List[ReorderEvent] = []
    snapshots: List[InventorySnapshot] = []

    inventory = config.initial_inventory_units
    threshold = config.reorder_threshold_units
    capacity = config.MAX_INVENTORY_CAPACITY

    for day_idx, (demand_raw, date_str) in enumerate(zip(daily_demand, dates)):
        demand = max(0, int(round(demand_raw)))
        reorder_triggered = False
        refill = 0

        # Consume inventory
        inventory = max(0, inventory - demand)

        # Check if reorder is needed
        if inventory <= threshold:
            refill = capacity - inventory
            urgency = "critical" if inventory <= threshold * 0.5 else "warning" if inventory <= threshold else "planned"

            events.append(ReorderEvent(
                product=product,
                day=day_idx,
                date=date_str,
                quantity_needed=refill,
                current_inventory=inventory,
                urgency=urgency,
            ))

            # Refill inventory
            inventory = capacity
            reorder_triggered = True

        snapshots.append(InventorySnapshot(
            day=day_idx,
            date=date_str,
            product=product,
            inventory_level=inventory,
            demand=demand,
            reorder_triggered=reorder_triggered,
            refill_amount=refill,
        ))

    return events, snapshots


# ── Reorder Detection ─────────────────────────────────────────────────────────

def detect_reorder_events(
    product_forecasts: Dict,
    config,
) -> Tuple[List[ReorderEvent], Dict[str, List[InventorySnapshot]]]:
    """
    Runs inventory simulation across all products in the forecast.

    Args:
        product_forecasts: Dict with "dates", "data" keys from product_forecasts.json.
        config: SustainabilityConfig instance.

    Returns:
        Tuple of (all_reorder_events, inventory_timelines_by_product).
    """
    dates = product_forecasts.get("dates", [])
    data = product_forecasts.get("data", {})

    all_events: List[ReorderEvent] = []
    timelines: Dict[str, List[InventorySnapshot]] = {}

    for product_name, product_data in data.items():
        # Use predictions as the forecasted demand
        daily_demand = product_data.get("predictions", [])

        if not daily_demand or not dates:
            logger.warning(f"Skipping {product_name}: no forecast data.")
            continue

        events, snapshots = simulate_inventory(product_name, daily_demand, dates, config)
        all_events.extend(events)
        timelines[product_name] = snapshots

        logger.info(f"  {product_name}: {len(events)} reorder events detected.")

    # Sort all events chronologically
    all_events.sort(key=lambda e: (e.day, e.product))
    logger.info(f"Total reorder events across all products: {len(all_events)}")

    return all_events, timelines


# ── Shipment Consolidation ────────────────────────────────────────────────────

def consolidate_shipments(
    events: List[ReorderEvent],
    config,
) -> List[ConsolidatedShipment]:
    """
    Groups nearby reorder events into consolidated shipments.

    Algorithm:
        1. Sort events by day.
        2. For each unassigned event, create a new shipment group.
        3. Look ahead MAX_DELAY_DAYS for other events that can join this group.
        4. Mark all grouped events as assigned.

    Args:
        events: All reorder events sorted by day.
        config: SustainabilityConfig instance.

    Returns:
        List of ConsolidatedShipment objects.
    """
    if not events:
        return []

    max_delay = config.MAX_DELAY_DAYS
    truck_cap = config.TRUCK_CAPACITY_UNITS
    assigned = set()
    shipments: List[ConsolidatedShipment] = []

    for i, anchor in enumerate(events):
        if i in assigned:
            continue

        # Start a new consolidated group anchored at this event
        group_products = [anchor.product]
        group_quantities = {anchor.product: anchor.quantity_needed}
        group_total = anchor.quantity_needed
        assigned.add(i)

        # Look ahead for events within the delay window
        for j in range(i + 1, len(events)):
            if j in assigned:
                continue

            candidate = events[j]

            # Must be within delay window
            if candidate.day - anchor.day > max_delay:
                break  # Events are sorted, so no more candidates

            # Don't add if same product already in group
            if candidate.product in group_quantities:
                continue

            # Check truck capacity
            if group_total + candidate.quantity_needed > truck_cap:
                continue

            # Add to consolidated group
            group_products.append(candidate.product)
            group_quantities[candidate.product] = candidate.quantity_needed
            group_total += candidate.quantity_needed
            assigned.add(j)

        shipments.append(ConsolidatedShipment(
            day=anchor.day,
            date=anchor.date,
            products=group_products,
            quantities=group_quantities,
            total_units=group_total,
            products_consolidated=len(group_products),
            savings_vs_separate=len(group_products) - 1,
        ))

    return shipments


# ── Naive Baseline ────────────────────────────────────────────────────────────

def calculate_naive_trips(events: List[ReorderEvent]) -> int:
    """
    Baseline: every reorder event = 1 separate shipment trip.
    """
    return len(events)


# ── Pipeline Orchestrator ─────────────────────────────────────────────────────

def run_optimization(product_forecasts: Dict, config) -> OptimizationResult:
    """
    Full optimization pipeline:
    1. Detect reorder events via inventory simulation.
    2. Consolidate shipments.
    3. Compare against naive baseline.

    Args:
        product_forecasts: Dict from product_forecasts.json.
        config: SustainabilityConfig instance.

    Returns:
        OptimizationResult with all metrics and timelines.
    """
    logger.info("=" * 50)
    logger.info("Running Shipment Optimization Engine...")
    logger.info("=" * 50)

    # 1. Detect reorder events
    events, timelines = detect_reorder_events(product_forecasts, config)

    # 2. Calculate naive baseline
    naive = calculate_naive_trips(events)

    # 3. Consolidate shipments
    consolidated = consolidate_shipments(events, config)
    optimized = len(consolidated)

    # 4. Build result
    result = OptimizationResult(
        naive_trips=naive,
        optimized_trips=optimized,
        trips_saved=naive - optimized,
        consolidated_shipments=consolidated,
        reorder_events=events,
        inventory_timeline=timelines,
        products_analyzed=len(timelines),
    )

    logger.info(f"Naive trips: {naive}, Optimized: {optimized}, Saved: {result.trips_saved}")
    logger.info("Shipment optimization complete.")

    return result
