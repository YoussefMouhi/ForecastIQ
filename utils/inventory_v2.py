"""Inventory planning and stock trajectory simulation for ForecastIQ."""

import numpy as np
import pandas as pd


SERVICE_FACTORS = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}


def compute_inventory_metrics(
    forecast,
    forecast_std=None,
    lead_time_days=7,
    service_level=0.95,
    current_stock=None,
):
    demand = np.asarray(forecast, dtype=float)

    if demand.ndim != 1 or not demand.size:
        raise ValueError("Forecast must be a nonempty daily series.")

    if not np.isfinite(demand).all() or (demand < 0).any():
        raise ValueError("Forecast values must be finite and nonnegative.")

    if lead_time_days < 1:
        raise ValueError("Lead time must be positive.")

    if forecast_std is None:
        uncertainty = demand * 0.20
    else:
        uncertainty = np.asarray(forecast_std, dtype=float)
        if uncertainty.shape != demand.shape:
            raise ValueError("Uncertainty shape must match demand.")

    average = float(demand.mean())
    total = float(demand.sum())
    daily_std = float(uncertainty.mean())

    safety = (
        SERVICE_FACTORS.get(service_level, 1.645)
        * daily_std
        * np.sqrt(lead_time_days)
    )

    threshold = average * lead_time_days + safety

    days_remaining = None
    risk_indicator = None
    reorder = None

    if current_stock is not None:
        if current_stock < 0:
            raise ValueError("Current stock cannot be negative.")

        days_remaining = current_stock / max(average, 1e-8)

        risk_indicator = 100 * np.clip(
            1 - current_stock / (threshold + 1e-8),
            0, 1
        )

        reorder = bool(current_stock <= threshold)

    return {
        "avg_daily_demand": round(average, 2),
        "total_forecast": round(total, 2),
        "safety_stock": round(float(safety), 2),
        "reorder_point": round(float(threshold), 2),
        "days_of_stock": (
            round(float(days_remaining), 1)
            if days_remaining is not None else None
        ),
        "stockout_prob": (
            round(float(risk_indicator), 1)
            if risk_indicator is not None else None
        ),
        "needs_reorder": reorder,
        "service_level": service_level,
        "lead_time_days": lead_time_days,
    }


def simulate_inventory_trajectory(
    forecast,
    starting_stock,
    reorder_point,
    order_qty,
    lead_time_days=7,
):
    demand = np.asarray(forecast, dtype=float)

    if not np.isfinite(demand).all() or (demand < 0).any():
        raise ValueError("Demand must be finite and nonnegative.")

    if min(starting_stock, reorder_point, order_qty) < 0:
        raise ValueError("Inventory quantities cannot be negative.")

    if lead_time_days < 1:
        raise ValueError("Lead time must be positive.")

    available = float(starting_stock)
    scheduled_arrival = None
    records = []

    for index, requested in enumerate(demand):
        day = index + 1

        if scheduled_arrival is not None:
            if index >= scheduled_arrival:
                available += order_qty
                scheduled_arrival = None

        sold = min(available, requested)
        unavailable = max(requested - available, 0)
        available -= sold

        reorder = (
            available <= reorder_point
            and scheduled_arrival is None
        )

        if reorder:
            scheduled_arrival = index + lead_time_days

        records.append({
            "day": day,
            "forecast_demand": round(float(requested), 2),
            "actual_sales": round(float(sold), 2),
            "lost_sales": round(float(unavailable), 2),
            "stock_level": round(float(available), 2),
            "in_transit": (
                round(order_qty, 2)
                if scheduled_arrival is not None else 0
            ),
            "stockout": bool(available <= 0),
            "reorder": bool(reorder),
        })

    return pd.DataFrame(records)
