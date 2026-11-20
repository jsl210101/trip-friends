"""Plan a trip locally, then optionally ask a model to rephrase.

Order of work: de-identify (payload preview only), hard-constraint filter,
weighted scores, weighted NSW, orienteering heuristic, explanations from
those scores. The model is not on this path unless the caller asks it to
rephrase, and even then it cannot change the venue order.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

from triper.constraints import FilterResult, drop_reason_text, filter_venues
from triper.deidentify import PayloadRejected, assign_pseudonyms, build_llm_payload
from triper.explain import Explanation, _group, apply_model_wording, explain_pair, member_weights
from triper.i18n import t
from triper.geo import estimate_leg
from triper.llm import LLMClient, ModelCall, output_rejection, parse_wording_response, word_instructions
from triper.models import FEATURES, Member, TripInput, Venue
from triper.money import planning_budget_sgd
from triper.preferences import moving_average_update
from triper.routing import HEURISTIC_NOTE, RouteResult, optimise, pacing_variance_hours, travel_detour_ratio
from triper.scoring import feature_vector, normalise_alphas, preference_match
from triper.venues import load_venues

HARD_REASONS = ("diet", "accessibility", "hours", "budget", "weather")


@dataclass
class PlanResult:
    weights: list[float]
    beta: float
    normalised_alphas: dict[str, float]
    route: RouteResult
    explanations: list[Explanation]
    filtered: FilterResult
    metrics: dict[str, float | int | None]
    floor_message: str
    payload_preview: dict | None
    payload_error: str | None
    meets_floor: bool
    pseudonyms: dict[str, str]
    constraint_notes: list[str] | None = None
    heuristic_note: str = HEURISTIC_NOTE


def illustrative_forecast(day: date) -> dict[str, str]:
    """A labelled stand-in, not a live NEA forecast."""
    if day.weekday() >= 5:
        condition, risk = "Partly cloudy", "moderate"
    elif day.weekday() in (1, 4):
        condition, risk = "Showers possible", "higher"
    else:
        condition, risk = "Warm", "low"
    return {
        "date": day.isoformat(),
        "condition": condition,
        "rain_risk": risk,
        "source": "Illustrative sample, not a live NEA forecast",
        "as_of": "2026-09-01",
    }


def default_alphas() -> dict[str, float]:
    """Starting sliders. Scoring renormalises them. Not a claim of a tuned optimum."""
    return {
        "preference_match": 0.40,
        "travel_time": 0.25,
        "cost": 0.25,
        "crowd": 0.10,
    }


def _mean_preference(members: list[Member], venue: Venue) -> float:
    if not members:
        return 0.0
    return sum(preference_match(member, venue) for member in members) / len(members)


def _feature_rows(members: list[Member], venue: Venue, travel_min: float) -> list[dict[str, float | None]]:
    return [feature_vector(member, venue, travel_min) for member in members]


def constraint_notes(filtered: FilterResult, catalogue: dict[str, Venue], members: list[Member]) -> list[str]:
    """One line per hard-filtered venue the group would otherwise have liked.

    Shown once for the plan. Explanations do not attach the constraint to a person.
    """
    notes: list[str] = []
    for dropped in filtered.dropped:
        reason = next((item for item in dropped.reasons if item in HARD_REASONS), None)
        venue = catalogue.get(dropped.venue_id)
        if reason is None or venue is None:
            continue
        if _mean_preference(members, venue) < 0.30:
            continue
        notes.append(
            f"{venue.name} was removed before scoring: {drop_reason_text(reason)}. "
            "Criterion weights cannot override that, and the note is not tied to one traveller."
        )
    return notes


def _explanations(
    trip: TripInput,
    route: RouteResult,
    filtered: FilterResult,
    catalogue: dict[str, Venue],
) -> list[Explanation]:
    members = trip.members
    weights = member_weights(members)
    party = max(1, len(members))
    kept = {item.venue.id: item.venue for item in filtered.kept}
    used = {stop.venue_id for day in route.days for stop in day.stops}
    explanations: list[Explanation] = []
    for day in route.days:
        origin_lat, origin_lon = trip.abode_lat, trip.abode_lon
        for stop in day.stops:
            venue = kept.get(stop.venue_id) or catalogue.get(stop.venue_id)
            if venue is None:
                continue
            chosen_rows = _feature_rows(members, venue, stop.travel_min)
            alternative_id = None
            alternative_name = None
            alternative_rows = None
            best_score = None
            for other in kept.values():
                if other.id in used:
                    continue
                leg = estimate_leg(
                    origin_lat,
                    origin_lon,
                    other.lat,
                    other.lon,
                    trip.transport_modes,
                    party,
                    remaining_budget=None,
                )
                travel = float(leg.minutes) if leg else 90.0
                rows = _feature_rows(members, other, travel)
                score, _, _ = _group(rows, weights, trip.alphas)
                if best_score is None or score > best_score:
                    best_score = score
                    alternative_id = other.id
                    alternative_name = other.name
                    alternative_rows = rows
            explanation = explain_pair(
                venue.id,
                venue.name,
                chosen_rows,
                weights,
                trip.alphas,
                alternative_id,
                alternative_name,
                alternative_rows,
                anchor_role=stop.role,
            )
            explanations.append(explanation)
            origin_lat, origin_lon = venue.lat, venue.lon
    return explanations


def _floor_message(members: list[Member], utilities: list[float], beta: float, meets: bool) -> str:
    if meets:
        return (
            f"Every member is at or above the fairness floor of {beta:.2f}. "
            "Agreed weightage was not changed."
        )
    lines = [f"No feasible plan kept every member at or above the fairness floor of {beta:.2f}."]
    for member, utility in zip(members, utilities):
        if utility < beta:
            strong = [name for name, value in member.preferences.items() if value >= 0.4]
            interests = ", ".join(strong) if strong else "no interest at or above 0.4"
            lines.append(f"{member.name} is at {utility:.2f}. Their stronger stated interests are {interests}.")
    lines.append(
        "Agreed weightage was not rewritten. Lower the floor, move the criterion weights, "
        "or change a must-visit, then optimise again."
    )
    return " ".join(lines)


def _metrics(trip: TripInput, route: RouteResult, filtered: FilterResult, catalogue: dict[str, Venue]) -> dict:
    scheduled = [stop.venue_id for day in route.days for stop in day.stops]
    dropped_ids = {item.venue_id for item in filtered.dropped}
    violations = [venue_id for venue_id in scheduled if venue_id in dropped_ids]
    included = [must for must in trip.must_visits if must.venue_id in scheduled]
    blocked = [must for must in trip.must_visits if must.venue_id in dropped_ids]
    lowest = min(route.utilities) if route.utilities else 0.0
    highest = max(route.utilities) if route.utilities else 0.0
    log_value = route.plan.nsw_log
    return {
        "hard_constraint_violations": len(violations),
        "locked_in_requested": len(trip.must_visits),
        "locked_in_included": len(included),
        "locked_in_blocked": len(blocked),
        "detour_ratio": travel_detour_ratio(trip, route.days, catalogue),
        "pacing_variance_hours": round(pacing_variance_hours(route.days), 4),
        "lowest_satisfaction": round(lowest, 4),
        "satisfaction_gap": round(highest - lowest, 4),
        "nsw_product": route.plan.nsw_product,
        "nsw_log": None if log_value != log_value or log_value == float("-inf") else log_value,
        "known_cost_sgd": route.total_known_cost,
        "budget_sgd": planning_budget_sgd(trip),
        "budget_entered": trip.budget_sgd,
        "budget_currency": (trip.budget_currency or "SGD").upper(),
    }


def plan_trip(trip: TripInput, venues: list[Venue] | None = None) -> PlanResult:
    catalogue_list = list(venues) if venues is not None else load_venues()
    catalogue = {venue.id: venue for venue in catalogue_list}
    filtered = filter_venues(catalogue_list, trip)
    kept = [item.venue for item in filtered.kept]
    dropped_details = {item.venue_id: item.detail for item in filtered.dropped}
    route = optimise(trip, kept, catalogue, dropped_details)
    explanations = _explanations(trip, route, filtered, catalogue)
    meets = route.plan.meets_floor
    try:
        payload: dict | None = build_llm_payload(trip, purpose="extract_places")
        payload_error = None
    except PayloadRejected as exc:
        payload = None
        payload_error = str(exc)
    return PlanResult(
        weights=[float(member.weight) for member in trip.members],
        beta=trip.beta,
        normalised_alphas=normalise_alphas(trip.alphas),
        route=route,
        explanations=explanations,
        filtered=filtered,
        metrics=_metrics(trip, route, filtered, catalogue),
        floor_message=_floor_message(trip.members, route.utilities, trip.beta, meets),
        payload_preview=payload,
        payload_error=payload_error,
        meets_floor=meets,
        pseudonyms=assign_pseudonyms(trip.members),
        constraint_notes=constraint_notes(filtered, catalogue, trip.members),
    )


def score_facts(explanations: list[Explanation]) -> dict:
    """De-identified facts for wording. No member names or sensitive constraints."""
    return {
        "stops": [
            {
                "venue_id": item.venue_id,
                "venue_name": item.venue_name,
                "top_factor": item.top_factor,
                "contributions": item.contributions,
                "alternative_name": item.alternative_name or "",
                "gap_factor": item.gap_factor or "",
                "reversal": item.reversal,
            }
            for item in explanations
        ]
    }


def rephrase_explanations(
    trip: TripInput,
    explanations: list[Explanation],
    llm: LLMClient | None = None,
    ai_consent: bool = False,
    lang: str = "en",
) -> tuple[list[Explanation], ModelCall]:
    """Ask the model to word existing explanations. Reject anything unfaithful."""
    client = llm or LLMClient()
    try:
        payload = build_llm_payload(
            trip,
            purpose="word_explanations",
            score_facts=score_facts(explanations),
            links=[],
        )
    except PayloadRejected as exc:
        return explanations, ModelCall(payload={}, sent=False, accepted=False, rejection=t("privacy.refused", lang))
    if not ai_consent:
        return explanations, ModelCall(
            payload=payload,
            sent=False,
            accepted=False,
            rejection=t("model.consent_off", lang),
        )
    if not client.available:
        return explanations, ModelCall(
            payload=payload,
            sent=False,
            accepted=False,
            rejection=t("model.no_key", lang),
        )
    raw = client.complete_json(payload, word_instructions(lang))
    if raw is None:
        return explanations, ModelCall(
            payload=payload,
            sent=True,
            accepted=False,
            rejection=t("model.failed", lang),
        )
    secrets: list[str] = [trip.abode_address]
    for member in trip.members:
        secrets.extend([member.name, member.email, member.booking_ref])
    rejection = output_rejection(raw, secrets)
    if rejection:
        return explanations, ModelCall(payload=payload, sent=True, accepted=False, rejection=t(rejection, lang))
    parsed = parse_wording_response(raw)
    if parsed is None:
        return explanations, ModelCall(
            payload=payload,
            sent=True,
            accepted=False,
            rejection=t("model.schema", lang),
        )
    updated, rejected_ids = apply_model_wording(explanations, parsed, lang=lang)
    note = None
    if rejected_ids:
        note = t("model.unfaithful", lang, ids=", ".join(rejected_ids))
    return updated, ModelCall(payload=payload, sent=True, accepted=True, rejection=note, body={"stops": parsed})


def apply_member_feedback(
    member: Member,
    venue: Venue,
    signal: float,
    personalisation_consent: bool,
    lang: str = "en",
) -> tuple[Member, str]:
    """Update one member when they consented. Other members are simply not passed in."""
    if not personalisation_consent:
        return member, t("feedback.off", lang)
    preferences, count = moving_average_update(member, venue, signal)
    if count == member.update_count:
        return member, t("feedback.neutral", lang)
    updated = replace(member, preferences=preferences, update_count=count)
    return updated, t("feedback.updated", lang)
