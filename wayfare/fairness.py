"""Weighted Nash social welfare with a satisfaction floor.

Maximise prod_k U_k ** w_k, equivalently sum_k w_k * log(U_k),
subject to U_k >= beta for every member.

w_k is the agreed influence. It is never rewritten to manufacture fairness.
The floor vetoes a plan that leaves someone below beta. If every compared
plan fails the floor, search still prefers the plan that lifts more members
over the floor and then the least-satisfied member.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def weighted_nsw_product(utilities: list[float], weights: list[float]) -> float:
    if len(utilities) != len(weights):
        raise ValueError("Utilities and weights must be the same length.")
    if not utilities:
        return 0.0
    product = 1.0
    for utility, weight in zip(utilities, weights):
        if utility <= 0:
            return 0.0
        product *= utility ** weight
    return product


def weighted_nsw_log(utilities: list[float], weights: list[float]) -> float:
    if len(utilities) != len(weights):
        raise ValueError("Utilities and weights must be the same length.")
    if not utilities:
        return float("-inf")
    total = 0.0
    for utility, weight in zip(utilities, weights):
        if utility <= 0:
            return float("-inf")
        total += weight * math.log(utility)
    return total


def floor_failures(utilities: list[float], beta: float) -> list[int]:
    return [index for index, utility in enumerate(utilities) if utility < beta]


@dataclass
class PlanUtility:
    utilities: list[float]
    weights: list[float]
    beta: float
    day_length_penalty: float = 0.0
    collected: float = 0.0

    @property
    def meets_floor(self) -> bool:
        return bool(self.utilities) and all(utility >= self.beta for utility in self.utilities)

    @property
    def min_u(self) -> float:
        return min(self.utilities) if self.utilities else 0.0

    @property
    def max_u(self) -> float:
        return max(self.utilities) if self.utilities else 0.0

    @property
    def sum_u(self) -> float:
        return float(sum(self.utilities))

    @property
    def gap(self) -> float:
        return self.max_u - self.min_u

    @property
    def n_above_floor(self) -> int:
        return sum(1 for utility in self.utilities if utility >= self.beta)

    @property
    def nsw_log(self) -> float:
        return weighted_nsw_log(self.utilities, self.weights)

    @property
    def nsw_product(self) -> float:
        return weighted_nsw_product(self.utilities, self.weights)

    @property
    def objective(self) -> float:
        """Collected orienteering prize minus a penalty for uneven days.

        The prize is sum_k w_k * sum_v S_k(v). Maximising the product of
        averaged utilities on its own collapses to a one-stop plan, because
        U_k is a mean. The floor still vetoes a high-prize plan that leaves
        someone below beta, and weighted NSW breaks ties.
        """
        return self.collected - self.day_length_penalty


def plan_is_better(candidate: PlanUtility | None, incumbent: PlanUtility | None) -> bool:
    """True when the candidate should replace the incumbent.

    A plan that clears the fairness floor outranks one that does not.
    Among plans on the same side of the floor, the higher collected prize
    (after the day-length penalty) wins. Near-ties break toward weighted
    NSW and then the least-satisfied member. Weights are not modified.
    """
    if candidate is None:
        return False
    if incumbent is None:
        return True
    if candidate.meets_floor != incumbent.meets_floor:
        return candidate.meets_floor
    if not candidate.meets_floor and not incumbent.meets_floor:
        if candidate.n_above_floor != incumbent.n_above_floor:
            return candidate.n_above_floor > incumbent.n_above_floor
        if abs(candidate.min_u - incumbent.min_u) > 1e-6:
            return candidate.min_u > incumbent.min_u
    if candidate.objective > incumbent.objective + 1e-4:
        return True
    if incumbent.objective > candidate.objective + 1e-4:
        return False
    if math.isfinite(candidate.nsw_log) and math.isfinite(incumbent.nsw_log):
        if abs(candidate.nsw_log - incumbent.nsw_log) > 1e-7:
            return candidate.nsw_log > incumbent.nsw_log
    return candidate.min_u > incumbent.min_u + 1e-6


def select_floor(
    plans: list[list[float]],
    betas: list[float] | None = None,
) -> dict[str, float] | None:
    """Grid-search a floor on already simulated utility vectors.

    Chooses the highest beta at which at least one plan is feasible, and
    among such betas the one whose best NSW plan keeps the highest minimum
    satisfaction. This does not rewrite member weights. It is an offline
    helper for the default floor, not a model trained on user data.
    """
    betas = betas or [0.3, 0.4, 0.5, 0.6, 0.7]
    best: dict[str, float] | None = None
    for beta in betas:
        feasible = [plan for plan in plans if plan and all(utility >= beta for utility in plan)]
        if not feasible:
            continue
        chosen = max(feasible, key=lambda plan: (min(plan), sum(plan)))
        record = {"beta": beta, "min_u": min(chosen), "sum_u": float(sum(chosen))}
        if best is None or record["beta"] > best["beta"] or (
            record["beta"] == best["beta"] and record["min_u"] > best["min_u"]
        ):
            best = record
    return best
