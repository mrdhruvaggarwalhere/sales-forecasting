"""
Sustainability Metrics — Orchestrator Module.

Ties together the shipment optimizer, carbon calculator, and recommendation engine.
Loads forecast data, runs the full analysis, computes dashboard KPIs,
and returns a structured dict for the Streamlit dashboard.

Usage:
    from src.sustainability.sustainability_metrics import run_sustainability_analysis
    results = run_sustainability_analysis()
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any
from dataclasses import asdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
SUSTAINABILITY_DIR = PROJECT_ROOT / "data" / "sustainability"


def run_sustainability_analysis(
    product_forecasts: Dict = None,
    config=None,
) -> Dict[str, Any]:
    """
    Runs the complete sustainability analysis pipeline.

    Args:
        product_forecasts: Pre-loaded product_forecasts.json dict (optional).
        config: SustainabilityConfig instance (optional, uses defaults).

    Returns:
        Dict containing all sustainability metrics, KPIs, recommendations,
        and timeline data for dashboard consumption.
    """
    from config.sustainability_config import SustainabilityConfig
    from src.sustainability.shipment_optimizer import run_optimization
    from src.sustainability.carbon_calculator import compare_scenarios
    from src.sustainability.recommendation_engine import generate_recommendations

    logger.info("=" * 60)
    logger.info("  SUSTAINABILITY ANALYSIS PIPELINE")
    logger.info("=" * 60)

    # ── 1. Load config ────────────────────────────────────────
    if config is None:
        config = SustainabilityConfig()

    # ── 2. Load forecast data ─────────────────────────────────
    if product_forecasts is None:
        forecast_path = DATASET_DIR / "product_forecasts.json"
        if not forecast_path.is_file():
            logger.error(f"Product forecasts not found at {forecast_path}")
            return _empty_results()

        with open(forecast_path) as f:
            product_forecasts = json.load(f)

    if not product_forecasts or "data" not in product_forecasts:
        logger.error("Invalid product_forecasts format.")
        return _empty_results()

    # ── 3. Run shipment optimization ──────────────────────────
    optimization = run_optimization(product_forecasts, config)

    # ── 4. Calculate carbon emissions ─────────────────────────
    carbon_report = compare_scenarios(
        optimization.naive_trips,
        optimization.optimized_trips,
        config,
    )

    # ── 5. Generate AI recommendations ────────────────────────
    recommendations = generate_recommendations(
        optimization.consolidated_shipments,
        optimization.reorder_events,
        carbon_report,
        config,
    )

    # ── 6. Compute dashboard KPIs ─────────────────────────────
    sustainability_score = _compute_sustainability_score(optimization, carbon_report)
    optimized_order_pct = (
        (optimization.trips_saved / optimization.naive_trips * 100)
        if optimization.naive_trips > 0 else 0.0
    )

    # ── 7. Build results dict ─────────────────────────────────
    results = {
        "kpis": {
            "co2_saved_kg": carbon_report.co2_saved_kg,
            "trips_reduced": optimization.trips_saved,
            "fuel_saved_litres": carbon_report.fuel_saved_litres,
            "cost_saved": carbon_report.cost_saved,
            "optimized_order_pct": round(optimized_order_pct, 1),
            "sustainability_score": sustainability_score,
            "naive_trips": optimization.naive_trips,
            "optimized_trips": optimization.optimized_trips,
            "products_analyzed": optimization.products_analyzed,
        },
        "carbon_report": {
            "naive": {
                "trips": carbon_report.naive.num_trips,
                "distance_km": carbon_report.naive.total_distance_km,
                "fuel_litres": carbon_report.naive.fuel_litres,
                "co2_kg": carbon_report.naive.co2_kg,
                "total_cost": carbon_report.naive.total_cost,
            },
            "optimized": {
                "trips": carbon_report.optimized.num_trips,
                "distance_km": carbon_report.optimized.total_distance_km,
                "fuel_litres": carbon_report.optimized.fuel_litres,
                "co2_kg": carbon_report.optimized.co2_kg,
                "total_cost": carbon_report.optimized.total_cost,
            },
            "reduction_pct": carbon_report.reduction_pct,
        },
        "consolidated_shipments": [
            {
                "day": s.day,
                "date": s.date,
                "products": s.products,
                "quantities": s.quantities,
                "total_units": s.total_units,
                "products_consolidated": s.products_consolidated,
                "trips_saved": s.savings_vs_separate,
            }
            for s in optimization.consolidated_shipments
        ],
        "reorder_events": [
            {
                "product": e.product,
                "day": e.day,
                "date": e.date,
                "quantity_needed": e.quantity_needed,
                "current_inventory": e.current_inventory,
                "urgency": e.urgency,
            }
            for e in optimization.reorder_events
        ],
        "inventory_timeline": {
            product: [
                {
                    "day": s.day,
                    "date": s.date,
                    "inventory_level": s.inventory_level,
                    "demand": s.demand,
                    "reorder_triggered": s.reorder_triggered,
                    "refill_amount": s.refill_amount,
                }
                for s in snapshots
            ]
            for product, snapshots in optimization.inventory_timeline.items()
        },
        "recommendations": [
            {
                "type": r.rec_type,
                "priority": r.priority,
                "title": r.title,
                "message": r.message,
                "products": r.products,
                "estimated_savings": r.estimated_savings,
                "icon": r.icon,
            }
            for r in recommendations
        ],
        "config": {
            "reorder_threshold_pct": config.REORDER_THRESHOLD_PCT,
            "max_delay_days": config.MAX_DELAY_DAYS,
            "max_inventory_capacity": config.MAX_INVENTORY_CAPACITY,
            "shipping_cost_per_trip": config.SHIPPING_COST_PER_TRIP,
            "distance_km": config.DISTANCE_KM,
            "co2_emission_factor": config.CO2_EMISSION_FACTOR,
            "fuel_consumption_rate": config.FUEL_CONSUMPTION_RATE,
        },
    }

    # ── 8. Save results to disk ───────────────────────────────
    _save_results(results)

    logger.info("=" * 60)
    logger.info("  SUSTAINABILITY ANALYSIS COMPLETE")
    logger.info(f"  Score: {sustainability_score}/100 | CO₂ Saved: {carbon_report.co2_saved_kg:.1f} kg")
    logger.info("=" * 60)

    return results


def _compute_sustainability_score(optimization, carbon_report) -> int:
    """
    Computes a 0-100 sustainability score based on multiple factors.

    Scoring breakdown:
        - Trip reduction %: 40 points max
        - CO₂ reduction %: 30 points max
        - Consolidation ratio: 20 points max
        - Cost efficiency: 10 points max
    """
    score = 0

    # Trip reduction (40 pts)
    if optimization.naive_trips > 0:
        trip_pct = optimization.trips_saved / optimization.naive_trips
        score += min(40, int(trip_pct * 100))

    # CO₂ reduction (30 pts)
    score += min(30, int(carbon_report.reduction_pct * 0.3))

    # Consolidation ratio (20 pts) — avg products per shipment
    if optimization.consolidated_shipments:
        avg_consolidated = sum(
            s.products_consolidated for s in optimization.consolidated_shipments
        ) / len(optimization.consolidated_shipments)
        score += min(20, int((avg_consolidated - 1) * 10))

    # Cost efficiency (10 pts)
    if carbon_report.naive.total_cost > 0:
        cost_pct = carbon_report.cost_saved / carbon_report.naive.total_cost
        score += min(10, int(cost_pct * 25))

    return min(100, max(0, score))


def _save_results(results: Dict) -> None:
    """Saves sustainability results to data/sustainability/."""
    os.makedirs(SUSTAINABILITY_DIR, exist_ok=True)
    output_path = SUSTAINABILITY_DIR / "sustainability_results.json"

    try:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.warning(f"Failed to save results: {e}")


def _empty_results() -> Dict:
    """Returns empty result structure when no data is available."""
    return {
        "kpis": {
            "co2_saved_kg": 0, "trips_reduced": 0, "fuel_saved_litres": 0,
            "cost_saved": 0, "optimized_order_pct": 0, "sustainability_score": 0,
            "naive_trips": 0, "optimized_trips": 0, "products_analyzed": 0,
        },
        "carbon_report": {"naive": {}, "optimized": {}, "reduction_pct": 0},
        "consolidated_shipments": [],
        "reorder_events": [],
        "inventory_timeline": {},
        "recommendations": [],
        "config": {},
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    results = run_sustainability_analysis()
    kpis = results["kpis"]
    print(f"\n🌱 Sustainability Score: {kpis['sustainability_score']}/100")
    print(f"🚚 Trips Reduced: {kpis['trips_reduced']}")
    print(f"⛽ Fuel Saved: {kpis['fuel_saved_litres']:.1f} L")
    print(f"🌍 CO₂ Saved: {kpis['co2_saved_kg']:.1f} kg")
    print(f"💰 Cost Saved: ${kpis['cost_saved']:,.0f}")
