"""Contrastive explanations taken from the same scores that built the plan.

The largest contribution answers why a stop was chosen. The factor with the
largest score gap answers why a named alternative lost. The smallest single
criterion-weight change that reverses the pair is found by search, then the
weights are renormalised exactly as scoring does. A hard-constraint rejection
has no weight-based reversal.

Templates are the source of truth. A model may only rephrase them, and only
when the rephrased top factor still matches.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from wayfare.models import FEATURES, Member
from wayfare.scoring import factor_label, score_features, top_factor


@dataclass
class Explanation:
    venue_id: str
    venue_name: str
    top_factor: str
    contributions: dict[str, float]
    alternative_id: str | None
    alternative_name: str | None
    gap_factor: str | None
    why: str
    why_not: str
    reversal: str
    wording: str = "template"


def _group(
    feature_rows: list[dict[str, float | None]],
    weights: list[float],
    alphas: dict[str, float],
) -> tuple[float, dict[str, float], list[str]]:
    total = 0.0
    weight_sum = 0.0
    contrib_acc: dict[str, float] = {}
    contrib_weight: dict[str, float] = {}
    omitted: set[str] = set()
    for features, weight in zip(feature_rows, weights):
        score, contributions, missing = score_features(features, alphas)
        omitted.update(missing)
        total += weight * score
        weight_sum += weight
        for feature, value in contributions.items():
            contrib_acc[feature] = contrib_acc.get(feature, 0.0) + weight * value
            contrib_weight[feature] = contrib_weight.get(feature, 0.0) + weight
    if weight_sum <= 0:
        return 0.0, {}, sorted(omitted)
    contributions = {
        feature: contrib_acc[feature] / contrib_weight[feature] for feature in contrib_acc
    }
    return total / weight_sum, contributions, sorted(omitted)


def _gap_factor(chosen: dict[str, float], alternative: dict[str, float]) -> str | None:
    factors = set(chosen) | set(alternative)
    if not factors:
        return None
    return max(factors, key=lambda feature: (chosen.get(feature, 0.0) - alternative.get(feature, 0.0), feature))


def smallest_weight_reversal(
    chosen_rows: list[dict[str, float | None]],
    alternative_rows: list[dict[str, float | None]],
    weights: list[float],
    alphas: dict[str, float],
) -> tuple[str, float, float] | None:
    """Smallest absolute slider move on one criterion that reverses the group score.

    Other sliders stay at their current raw values. Scoring renormalises the
    full set, which is the same rule the optimiser uses. Returns
    (feature, old_raw, new_raw) or None when no move inside [0, 1] works.
    """
    base_chosen, _, _ = _group(chosen_rows, weights, alphas)
    base_alternative, _, _ = _group(alternative_rows, weights, alphas)
    if base_alternative > base_chosen + 1e-9:
        return None
    best: tuple[float, str, float, float] | None = None
    for feature in FEATURES:
        current = float(alphas.get(feature, 0.0))
        for direction in (1.0, -1.0):
            span = (1.0 - current) if direction > 0 else current
            if span <= 1e-6:
                continue
            low = 0.0
            high = span
            found: float | None = None
            for _ in range(18):
                mid = (low + high) / 2.0
                trial = dict(alphas)
                trial[feature] = current + direction * mid
                score_chosen, _, _ = _group(chosen_rows, weights, trial)
                score_alternative, _, _ = _group(alternative_rows, weights, trial)
                if score_alternative > score_chosen + 1e-9:
                    found = mid
                    high = mid
                else:
                    low = mid
            if found is None:
                continue
            if best is None or found < best[0] - 1e-9:
                best = (found, feature, current, current + direction * found)
    if best is None:
        return None
    _, feature, old, new = best
    return feature, old, new


def _why(venue_name: str, contributions: dict[str, float], omitted: list[str], anchor_role: str | None = None) -> str:
    leading = top_factor(contributions)
    if anchor_role == "anchor_timed":
        prefix = f"{venue_name} is a FIXED must-visit and stays at its requested time. "
    elif anchor_role == "anchor_must":
        prefix = f"{venue_name} is a must-visit anchor, so the route keeps it. "
    else:
        prefix = ""
    if leading is None:
        return prefix + f"{venue_name} was scheduled, but every scoring feature was missing so no contribution could be cited."
    ordered = sorted(contributions, key=lambda feature: (-contributions[feature], feature))
    parts = [f"{factor_label(feature)} ({contributions[feature]:.2f})" for feature in ordered]
    text = (
        prefix
        + f"{venue_name} was chosen because the largest contribution to its normalised group score "
        f"is {factor_label(leading)}. Contributions: {', '.join(parts)}."
    )
    if omitted:
        names = ", ".join(factor_label(feature) for feature in omitted)
        text += f" Omitted rather than guessed: {names}."
    return text


def explain_pair(
    venue_id: str,
    venue_name: str,
    chosen_rows: list[dict[str, float | None]],
    weights: list[float],
    alphas: dict[str, float],
    alternative_id: str | None = None,
    alternative_name: str | None = None,
    alternative_rows: list[dict[str, float | None]] | None = None,
    blocked_reason: str | None = None,
    anchor_role: str | None = None,
) -> Explanation:
    """Faithful explanation for one chosen venue against one named alternative."""
    _, contributions, omitted = _group(chosen_rows, weights, alphas)
    leading = top_factor(contributions) or "preference_match"
    why = _why(venue_name, contributions, omitted, anchor_role)
    gap_name: str | None = None
    why_not = "No alternative was left to compare."
    reversal = "There is no alternative venue to reverse against."
    if blocked_reason and alternative_name:
        why_not = (
            f"{alternative_name} cannot take this slot: {blocked_reason}. "
            "That is a hard constraint applied before scoring, so criterion weights and "
            "agreed weightage cannot put it back. The explanation does not attach the "
            "constraint to a named traveller."
        )
        reversal = (
            f"No single change to a criterion weight reverses {venue_name} and {alternative_name}, "
            "because the alternative never entered the scored set."
        )
    elif alternative_name and alternative_rows is not None:
        alt_score, alt_contrib, _ = _group(alternative_rows, weights, alphas)
        chosen_score, _, _ = _group(chosen_rows, weights, alphas)
        gap_name = _gap_factor(contributions, alt_contrib)
        gap_value = 0.0
        if gap_name is not None:
            gap_value = contributions.get(gap_name, 0.0) - alt_contrib.get(gap_name, 0.0)
        if gap_name is None:
            why_not = f"{alternative_name} has no comparable score factors."
        elif alt_score > chosen_score + 1e-9:
            why_not = (
                f"{alternative_name} has a higher group score ({alt_score:.2f} against {chosen_score:.2f}). "
                f"The factor with the largest score difference is {factor_label(gap_name)} "
                f"(gap {gap_value:.2f}). It was not scheduled because no feasible insertion "
                "improved the penalised orienteering prize while keeping the fairness floor, "
                "time windows, opening hours, budget, and must-visit anchors."
            )
            reversal = (
                f"The score ranking is already in favour of {alternative_name}. "
                "The single change that would put it on the itinerary is a feasible time or budget slot, "
                "not a criterion weight."
            )
        else:
            why_not = (
                f"{alternative_name} scores lower. The factor with the largest score difference is "
                f"{factor_label(gap_name)} (gap {gap_value:.2f})."
            )
            flip = smallest_weight_reversal(chosen_rows, alternative_rows, weights, alphas)
            if flip is None:
                reversal = (
                    f"No single criterion-weight change inside the 0–1 slider range makes "
                    f"{alternative_name} outrank {venue_name} after the weights are renormalised."
                )
            else:
                feature, old, new = flip
                direction = "Raise" if new > old else "Lower"
                reversal = (
                    f"{direction} the {factor_label(feature)} weight from {old:.2f} to {new:.2f}, "
                    f"leaving the other sliders where they are. After renormalisation that is the "
                    f"smallest single slider change that makes {alternative_name} outrank {venue_name} "
                    f"on the weighted group score."
                )
    return Explanation(
        venue_id=venue_id,
        venue_name=venue_name,
        top_factor=leading,
        contributions={key: round(value, 4) for key, value in contributions.items()},
        alternative_id=alternative_id,
        alternative_name=alternative_name,
        gap_factor=gap_name,
        why=why,
        why_not=why_not,
        reversal=reversal,
    )


def wording_is_faithful(explanation: Explanation, why: str, top: str) -> bool:
    """A rephrase is kept only when it still names the same top factor."""
    if top != explanation.top_factor:
        return False
    lowered = why.lower()
    return factor_label(explanation.top_factor) in lowered or explanation.top_factor in lowered


def apply_model_wording(explanations: list[Explanation], stops: list[dict[str, str]]) -> tuple[list[Explanation], list[str]]:
    """Replace template sentences only where the model kept the true top factor."""
    by_id = {item["venue_id"]: item for item in stops}
    updated: list[Explanation] = []
    notes: list[str] = []
    for explanation in explanations:
        item = by_id.get(explanation.venue_id)
        if item is None:
            updated.append(explanation)
            continue
        if not wording_is_faithful(explanation, item.get("why", ""), item.get("top_factor", "")):
            notes.append(explanation.venue_id)
            updated.append(explanation)
            continue
        updated.append(
            replace(
                explanation,
                why=item["why"].strip(),
                why_not=item.get("why_not", explanation.why_not).strip(),
                reversal=item.get("reversal", explanation.reversal).strip(),
                wording="model",
            )
        )
    return updated, notes


def member_weights(members: list[Member]) -> list[float]:
    return [float(member.weight) for member in members]
