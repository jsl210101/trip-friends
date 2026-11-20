"""Weighted linear activity scores.

S_k(v) = sum_f alpha_f * x_f^(k)(v), with each x in [0, 1] and oriented
so that higher is better. Criterion weights are normalised to sum to 1.
If a feature is missing (unknown price), it is omitted and the remaining
weights are renormalised. The missing value is never guessed.

Preference match is the interest-weighted average of the venue's 0–1
attribute tags over interests that the member rates above zero.
"""

from __future__ import annotations

from triper.models import FEATURES, FEATURE_LABELS, INTERESTS, Member, Venue

# Reference points for unit-interval features. Documented approximations.
TRAVEL_REF_MIN = 90.0
COST_REF_SGD = 60.0


def normalise_alphas(alphas: dict[str, float]) -> dict[str, float]:
    """Normalise criterion weights onto the simplex. Empty input becomes equal weights."""
    clean = {feature: max(0.0, float(alphas.get(feature, 0.0))) for feature in FEATURES}
    total = sum(clean.values())
    if total <= 0:
        share = 1.0 / len(FEATURES)
        return {feature: share for feature in FEATURES}
    return {feature: clean[feature] / total for feature in FEATURES}


def preference_match(member: Member, venue: Venue) -> float:
    weighted = [
        (float(member.preferences.get(key, 0.0)), float(venue.tags.get(key, 0.0)))
        for key in INTERESTS
        if float(member.preferences.get(key, 0.0)) > 0
    ]
    if not weighted:
        return 0.0
    denominator = sum(weight for weight, _ in weighted)
    return sum(weight * tag for weight, tag in weighted) / denominator


def feature_vector(
    member: Member,
    venue: Venue,
    travel_min: float,
) -> dict[str, float | None]:
    """Build x_f^(k)(v). Cost is None when the catalogue price is unknown.

    Transport cost is enforced in the budget check, not inside this feature,
    so a missing activity price is omitted rather than filled with the fare.
    """
    travel = 1.0 - min(max(travel_min, 0.0), TRAVEL_REF_MIN) / TRAVEL_REF_MIN
    if venue.price_per_person_sgd is None:
        cost: float | None = None
    else:
        cost = 1.0 - min(max(venue.price_per_person_sgd, 0.0), COST_REF_SGD) / COST_REF_SGD
    tolerance = min(1.0, max(0.0, member.crowd_tolerance))
    crowd = 1.0 - min(1.0, max(0.0, venue.crowd_estimate)) * (1.0 - tolerance)
    return {
        "preference_match": preference_match(member, venue),
        "travel_time": min(1.0, max(0.0, travel)),
        "cost": None if cost is None else min(1.0, max(0.0, cost)),
        "crowd": min(1.0, max(0.0, crowd)),
    }


def score_features(
    features: dict[str, float | None],
    alphas: dict[str, float],
) -> tuple[float, dict[str, float], list[str]]:
    """Return S, per-feature contributions, and omitted feature names.

    Contributions already include the normalised weight, so they sum to S.
    """
    present = {feature: float(features[feature]) for feature in FEATURES if features.get(feature) is not None}
    omitted = [feature for feature in FEATURES if feature not in present]
    base = normalise_alphas(alphas)
    if not present:
        return 0.0, {}, omitted
    weight_sum = sum(base[feature] for feature in present)
    contributions = {feature: (base[feature] / weight_sum) * present[feature] for feature in present}
    return sum(contributions.values()), contributions, omitted


def score_member_venue(
    member: Member,
    venue: Venue,
    alphas: dict[str, float],
    travel_min: float,
) -> tuple[float, dict[str, float], list[str]]:
    return score_features(feature_vector(member, venue, travel_min), alphas)


def top_factor(contributions: dict[str, float]) -> str | None:
    if not contributions:
        return None
    return max(contributions, key=lambda feature: (contributions[feature], feature))


def factor_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " "))


def normalise_satisfaction(scores: list[float]) -> float:
    """U from the mean of S_k(v), scaled to [0, 1].

    Each S_k(v) is already on [0, 1] because features are normalised and
    criterion weights sum to 1. Normalise therefore clips the mean.
    An empty itinerary has no satisfaction (0), which the fairness floor
    rejects whenever beta > 0.
    """
    if not scores:
        return 0.0
    mean = sum(scores) / len(scores)
    return min(1.0, max(0.0, mean))
