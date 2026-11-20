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

from triper.i18n import feature_name, reason_phrase, t
from triper.models import FEATURES, Member
from triper.scoring import factor_label, score_features, top_factor


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
    # Structured fields let the same scores be worded in Chinese or English.
    anchor_role: str | None = None
    omitted: tuple[str, ...] = ()
    text_case: str = "no_alternative"
    gap_value: float | None = None
    chosen_group_score: float | None = None
    alternative_group_score: float | None = None
    reversal_feature: str | None = None
    reversal_old: float | None = None
    reversal_new: float | None = None
    blocked_reason: str | None = None
    wording_lang: str = "en"


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


def render_explanation(explanation: Explanation, lang: str = "en") -> tuple[str, str, str]:
    """Word a stored explanation. Model text is kept only in the language it was written."""
    if explanation.wording == "model" and explanation.wording_lang == lang:
        return explanation.why, explanation.why_not, explanation.reversal
    if explanation.anchor_role == "anchor_timed":
        prefix = t("explain.prefix.timed", lang, name=explanation.venue_name)
    elif explanation.anchor_role == "anchor_must":
        prefix = t("explain.prefix.must", lang, name=explanation.venue_name)
    else:
        prefix = ""
    if explanation.top_factor and explanation.contributions:
        ordered = sorted(explanation.contributions, key=lambda feature: (-explanation.contributions[feature], feature))
        parts = ", ".join(
            f"{feature_name(feature, lang)} ({explanation.contributions[feature]:.2f})" for feature in ordered
        )
        why = t(
            "explain.why",
            lang,
            prefix=prefix,
            name=explanation.venue_name,
            factor=feature_name(explanation.top_factor, lang),
            parts=parts,
        )
        if explanation.omitted:
            why += t(
                "explain.omitted",
                lang,
                names=", ".join(feature_name(feature, lang) for feature in explanation.omitted),
            )
    else:
        why = t("explain.why_missing", lang, prefix=prefix, name=explanation.venue_name)
    name = explanation.venue_name
    alt = explanation.alternative_name or ""
    case = explanation.text_case
    if case == "blocked":
        why_not = t(
            "explain.why_not.blocked",
            lang,
            alt=alt,
            reason=reason_phrase(explanation.blocked_reason or "", lang),
        )
        reversal = t("explain.reversal.blocked", lang, name=name, alt=alt)
    elif case == "no_factors":
        why_not = t("explain.why_not.no_factors", lang, alt=alt)
        reversal = t("explain.reversal.none", lang)
    elif case == "higher":
        why_not = t(
            "explain.why_not.higher",
            lang,
            alt=alt,
            alt_score=explanation.alternative_group_score or 0.0,
            chosen_score=explanation.chosen_group_score or 0.0,
            factor=feature_name(explanation.gap_factor or "", lang),
            gap=explanation.gap_value or 0.0,
        )
        reversal = t("explain.reversal.higher", lang, alt=alt)
    elif case == "lower":
        why_not = t(
            "explain.why_not.lower",
            lang,
            alt=alt,
            factor=feature_name(explanation.gap_factor or "", lang),
            gap=explanation.gap_value or 0.0,
        )
        if explanation.reversal_feature is None or explanation.reversal_new is None or explanation.reversal_old is None:
            reversal = t("explain.reversal.no_flip", lang, alt=alt, name=name)
        else:
            direction = t(
                "explain.raise" if explanation.reversal_new > explanation.reversal_old else "explain.lower",
                lang,
            )
            reversal = t(
                "explain.reversal.flip",
                lang,
                direction=direction,
                factor=feature_name(explanation.reversal_feature, lang),
                old=explanation.reversal_old,
                new=explanation.reversal_new,
                alt=alt,
                name=name,
            )
    else:
        why_not = t("explain.why_not.none", lang)
        reversal = t("explain.reversal.none", lang)
    return why, why_not, reversal


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
    gap_name: str | None = None
    gap_value: float | None = None
    chosen_score: float | None = None
    alt_score: float | None = None
    reversal_feature: str | None = None
    reversal_old: float | None = None
    reversal_new: float | None = None
    text_case = "no_alternative"
    if blocked_reason and alternative_name:
        text_case = "blocked"
    elif alternative_name and alternative_rows is not None:
        alt_score, alt_contrib, _ = _group(alternative_rows, weights, alphas)
        chosen_score, _, _ = _group(chosen_rows, weights, alphas)
        gap_name = _gap_factor(contributions, alt_contrib)
        if gap_name is not None:
            gap_value = contributions.get(gap_name, 0.0) - alt_contrib.get(gap_name, 0.0)
        if gap_name is None:
            text_case = "no_factors"
        elif alt_score > chosen_score + 1e-9:
            text_case = "higher"
        else:
            text_case = "lower"
            flip = smallest_weight_reversal(chosen_rows, alternative_rows, weights, alphas)
            if flip is not None:
                reversal_feature, reversal_old, reversal_new = flip
    explanation = Explanation(
        venue_id=venue_id,
        venue_name=venue_name,
        top_factor=leading,
        contributions={key: round(value, 4) for key, value in contributions.items()},
        alternative_id=alternative_id,
        alternative_name=alternative_name,
        gap_factor=gap_name,
        why="",
        why_not="",
        reversal="",
        anchor_role=anchor_role,
        omitted=tuple(omitted),
        text_case=text_case,
        gap_value=None if gap_value is None else round(gap_value, 4),
        chosen_group_score=None if chosen_score is None else round(chosen_score, 4),
        alternative_group_score=None if alt_score is None else round(alt_score, 4),
        reversal_feature=reversal_feature,
        reversal_old=None if reversal_old is None else round(reversal_old, 4),
        reversal_new=None if reversal_new is None else round(reversal_new, 4),
        blocked_reason=blocked_reason,
    )
    why, why_not, reversal = render_explanation(explanation, "en")
    explanation.why = why
    explanation.why_not = why_not
    explanation.reversal = reversal
    return explanation


def wording_is_faithful(explanation: Explanation, why: str, top: str, lang: str = "en") -> bool:
    """A rephrase is kept only when it still names the same top factor, in the UI language."""
    if top != explanation.top_factor:
        return False
    labels = {
        explanation.top_factor,
        factor_label(explanation.top_factor),
        feature_name(explanation.top_factor, "en"),
        feature_name(explanation.top_factor, "zh"),
        feature_name(explanation.top_factor, lang),
    }
    lowered = why.lower()
    if not any(label and label.lower() in lowered for label in labels):
        return False
    if lang == "zh":
        return any("\u4e00" <= char <= "\u9fff" for char in why)
    return any(character.isascii() and character.isalpha() for character in why)


def apply_model_wording(
    explanations: list[Explanation],
    stops: list[dict[str, str]],
    lang: str = "en",
) -> tuple[list[Explanation], list[str]]:
    """Replace template sentences only where the model kept the true top factor."""
    by_id = {item["venue_id"]: item for item in stops}
    updated: list[Explanation] = []
    notes: list[str] = []
    for explanation in explanations:
        item = by_id.get(explanation.venue_id)
        if item is None:
            updated.append(explanation)
            continue
        if not wording_is_faithful(explanation, item.get("why", ""), item.get("top_factor", ""), lang):
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
                wording_lang=lang,
            )
        )
    return updated, notes


def member_weights(members: list[Member]) -> list[float]:
    return [float(member.weight) for member in members]
