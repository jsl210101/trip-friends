"""Travel-time and fare estimates.

These are haversine approximations with fixed speeds. They are not live
OneMap routes or LTA fares, and every figure is labelled an estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

EARTH_KM = 6371.0

MODE_SPECS: dict[str, dict[str, float]] = {
    "walk": {"kmh": 4.5, "wait_min": 0.0},
    "transit": {"kmh": 22.0, "wait_min": 8.0},
    "ride_hail": {"kmh": 30.0, "wait_min": 5.0},
}

TRAVEL_SOURCE = "Haversine distance with fixed mode speeds (not a live OneMap route)"


@dataclass(frozen=True)
class LegEstimate:
    mode: str
    minutes: int
    cost_sgd: float
    distance_km: float
    is_estimate: bool = True
    source: str = TRAVEL_SOURCE


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_KM * math.asin(math.sqrt(a))


def _mode_cost(mode: str, distance_km: float, party_size: int) -> float:
    party = max(1, party_size)
    if mode == "walk":
        return 0.0
    if mode == "transit":
        per_person = min(2.60, 0.92 + 0.08 * distance_km)
        return round(per_person * party, 2)
    if mode == "ride_hail":
        vehicles = max(1, math.ceil(party / 4))
        return round(vehicles * (4.0 + 1.05 * distance_km), 2)
    raise KeyError(mode)


def estimate_leg(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    modes: list[str],
    party_size: int,
    remaining_budget: float | None = None,
) -> LegEstimate | None:
    """Fastest selected mode whose estimated fare still fits the remaining budget.

    Returns None when every selected mode is over the remaining budget.
    A zero or negative remaining budget still allows a free walk.
    """
    selected = [mode for mode in modes if mode in MODE_SPECS]
    if not selected:
        selected = ["walk"]
    distance = haversine_km(lat1, lon1, lat2, lon2)
    options: list[LegEstimate] = []
    for mode in selected:
        spec = MODE_SPECS[mode]
        if distance < 0.05:
            minutes = 1
        else:
            minutes = int(round(spec["wait_min"] + distance / spec["kmh"] * 60.0))
            minutes = max(1, minutes)
        cost = _mode_cost(mode, distance, party_size)
        options.append(LegEstimate(mode=mode, minutes=minutes, cost_sgd=cost, distance_km=round(distance, 3)))
    options.sort(key=lambda leg: (leg.minutes, leg.cost_sgd, leg.mode))
    if remaining_budget is None:
        return options[0]
    for leg in options:
        if leg.cost_sgd <= remaining_budget + 1e-6:
            return leg
    return None
