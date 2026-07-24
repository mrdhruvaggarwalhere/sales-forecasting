"""
Recommendation Engine — AI-Powered Sustainability Suggestions.

Generates human-readable, actionable recommendations based on
shipment consolidation results, reorder events, and carbon savings.

Usage:
    from src.sustainability.recommendation_engine import generate_recommendations
    recs = generate_recommendations(consolidated, events, carbon_report, config)
"""

import logging
from dataclasses import dataclass
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """A single sustainability recommendation."""
    rec_type: str          # "combine", "delay", "advance", "reduce", "avoid_overstock"
    priority: str          # "high", "medium", "low"
    title: str
    message: str
    products: List[str]
    estimated_savings: str
    icon: str


def generate_recommendations(
    consolidated_shipments: list,
    reorder_events: list,
    carbon_report,
    config,
) -> List[Recommendation]:
    """
    Generates contextual sustainability recommendations.

    Args:
        consolidated_shipments: List of ConsolidatedShipment objects.
        reorder_events: List of ReorderEvent objects.
        carbon_report: CarbonReport from the carbon calculator.
        config: SustainabilityConfig instance.

    Returns:
        List of Recommendation objects.
    """
    logger.info("Generating AI sustainability recommendations...")
    recs: List[Recommendation] = []

    # ── 1. Combine Order Recommendations ──────────────────────
    for shipment in consolidated_shipments:
        if shipment.products_consolidated >= 2:
            products_str = " and ".join(shipment.products)
            trips_saved = shipment.savings_vs_separate
            co2_saved = round(config.co2_per_trip * trips_saved, 1)
            cost_saved = round(config.total_cost_per_trip * trips_saved, 0)

            recs.append(Recommendation(
                rec_type="combine",
                priority="high",
                title=f"Combine {shipment.products_consolidated} Orders on Day {shipment.day + 1}",
                message=(
                    f"{products_str} are all expected to reach reorder levels within a "
                    f"{config.MAX_DELAY_DAYS}-day window around {shipment.date}. "
                    f"Consolidating into a single shipment will save {trips_saved} trip(s), "
                    f"reduce CO₂ emissions by {co2_saved} kg, and cut logistics costs by ${cost_saved:,.0f}."
                ),
                products=shipment.products,
                estimated_savings=f"${cost_saved:,.0f} saved, {co2_saved} kg CO₂ avoided",
                icon="🔗",
            ))

    # ── 2. Delay Order Recommendations ────────────────────────
    # Find events that could have been delayed (non-critical urgency)
    delayable = [e for e in reorder_events if e.urgency == "planned"]
    if delayable:
        products = list(set(e.product for e in delayable))
        recs.append(Recommendation(
            rec_type="delay",
            priority="medium",
            title=f"Consider Delaying {len(delayable)} Non-Urgent Orders",
            message=(
                f"Products {', '.join(products[:3])}{'...' if len(products) > 3 else ''} "
                f"have reorder events that are not yet critical. Delaying these by "
                f"1-{config.MAX_DELAY_DAYS} days may allow them to be bundled with "
                f"other orders, further reducing shipments."
            ),
            products=products,
            estimated_savings=f"Up to {len(delayable)} additional trips could be saved",
            icon="⏳",
        ))

    # ── 3. Advance Order Recommendations ──────────────────────
    # Find critical events that could benefit from earlier ordering
    critical = [e for e in reorder_events if e.urgency == "critical"]
    if critical:
        products = list(set(e.product for e in critical))
        recs.append(Recommendation(
            rec_type="advance",
            priority="high",
            title=f"Advance Orders for {len(products)} Critical Products",
            message=(
                f"Products {', '.join(products[:3])} hit critically low inventory levels "
                f"(below {int(config.REORDER_THRESHOLD_PCT * 50)}% of capacity). "
                f"Consider advancing future orders by 1-2 days to maintain safety stock "
                f"and prevent stockouts while keeping shipments consolidated."
            ),
            products=products,
            estimated_savings="Reduced stockout risk + better consolidation windows",
            icon="⚡",
        ))

    # ── 4. Reduce Quantity / Avoid Overstocking ───────────────
    # If any product has very frequent reorders, suggest capacity adjustment
    from collections import Counter
    product_reorder_count = Counter(e.product for e in reorder_events)
    frequent_reorders = {p: c for p, c in product_reorder_count.items() if c >= 3}

    if frequent_reorders:
        for product, count in frequent_reorders.items():
            recs.append(Recommendation(
                rec_type="reduce",
                priority="low",
                title=f"Optimize Inventory Capacity for {product}",
                message=(
                    f"{product} triggered {count} reorder events in the 30-day forecast "
                    f"period. This high frequency suggests either the inventory capacity "
                    f"({config.MAX_INVENTORY_CAPACITY} units) is too low for demand, or "
                    f"demand is unusually spiky. Consider increasing storage capacity or "
                    f"implementing just-in-time ordering for this product."
                ),
                products=[product],
                estimated_savings="Smoother replenishment cycle, fewer emergency orders",
                icon="📦",
            ))

    # ── 5. Overall Environmental Impact Recommendation ────────
    if carbon_report and carbon_report.co2_saved_kg > 0:
        recs.append(Recommendation(
            rec_type="avoid_overstock",
            priority="medium",
            title="Environmental Impact Summary",
            message=(
                f"By consolidating shipments, you can reduce carbon emissions by "
                f"{carbon_report.co2_saved_kg:.1f} kg CO₂ ({carbon_report.reduction_pct:.1f}% reduction), "
                f"save {carbon_report.fuel_saved_litres:.1f} litres of fuel, and cut logistics "
                f"costs by ${carbon_report.cost_saved:,.0f}. This is equivalent to "
                f"{_trees_equivalent(carbon_report.co2_saved_kg)} trees absorbing CO₂ for one day."
            ),
            products=[],
            estimated_savings=f"{carbon_report.co2_saved_kg:.1f} kg CO₂ reduced",
            icon="🌱",
        ))

    logger.info(f"Generated {len(recs)} recommendations.")
    return recs


def _trees_equivalent(co2_kg: float) -> str:
    """Converts CO₂ savings to equivalent number of trees (1 tree ≈ 21 kg CO₂/year ≈ 0.058 kg/day)."""
    trees_per_day = co2_kg / 0.058
    if trees_per_day >= 1000:
        return f"{trees_per_day / 1000:,.1f}K"
    return f"{trees_per_day:,.0f}"
