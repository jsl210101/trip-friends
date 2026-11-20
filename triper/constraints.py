"""Deterministic hard-constraint filter.

Diet, accessibility, opening hours, and budget are applied before scoring.
Criterion weights and member influence cannot put a rejected venue back.
FIXED and FLEXIBLE must-visits that fail are reported; they are not forced in.

Missing opening hours or prices are flagged and kept. They are not guessed
and they are not treated as violations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from triper.models import Member, TripInput, Venue, weekday_key
from triper.money import planning_budget_sgd


@dataclass
class GroupConstraints:
    diets: set[str]
    accessibility: set[str]


@dataclass
class DroppedVenue:
    venue_id: str
    name: str
    reasons: list[str]
    detail: str


@dataclass
class KeptVenue:
    venue: Venue
    flags: list[str] = field(default_factory=list)


@dataclass
class FilterResult:
    kept: list[KeptVenue]
    dropped: list[DroppedVenue]
    constraints: GroupConstraints


def group_constraints(members: list[Member]) -> GroupConstraints:
    """Union of needs. The result is not attached to any member's name."""
    diets: set[str] = set()
    access: set[str] = set()
    for member in members:
        diets.update(member.diets)
        access.update(member.accessibility)
    return GroupConstraints(diets=diets, accessibility=access)


def _hours_overlap(venue: Venue, trip: TripInput) -> bool:
    """True when some trip day can host at least a short visit inside the window."""
    if not venue.hours_known or venue.hours is None:
        return True
    day_start_h, day_start_m = trip.day_start.split(":")
    day_end_h, day_end_m = trip.day_end.split(":")
    window_start = int(day_start_h) * 60 + int(day_start_m)
    window_end = int(day_end_h) * 60 + int(day_end_m)
    needed = min(30, venue.duration_min)
    for day in trip.dates():
        span = venue.hours.get(weekday_key(day))
        if span is None:
            continue
        start = max(span[0], window_start)
        end = min(span[1], window_end)
        if end - start >= needed:
            return True
    return False


def _catalogue_stale(last_updated: str, trip_start: str, max_age_days: int = 365) -> bool:
    try:
        age = date.fromisoformat(trip_start) - date.fromisoformat(last_updated)
    except ValueError:
        return True
    return age.days > max_age_days


def _diet_violation(venue: Venue, diets: set[str]) -> bool:
    if not diets:
        return False
    if not venue.serves_food and venue.category != "food":
        return False
    return any(need not in venue.diets_supported for need in diets)


def _access_violation(venue: Venue, access: set[str]) -> bool:
    if "wheelchair" in access and not venue.wheelchair_accessible:
        return True
    if "step_free" in access and not venue.step_free:
        return True
    return False


def filter_venues(venues: list[Venue], trip: TripInput) -> FilterResult:
    constraints = group_constraints(trip.members)
    party = max(1, len(trip.members))
    kept: list[KeptVenue] = []
    dropped: list[DroppedVenue] = []
    excluded = set(trip.excluded_venue_ids)
    for venue in venues:
        reasons: list[str] = []
        if venue.id in excluded:
            reasons.append("rejected")
        if _diet_violation(venue, constraints.diets):
            reasons.append("diet")
        if _access_violation(venue, constraints.accessibility):
            reasons.append("accessibility")
        if venue.hours_known and not _hours_overlap(venue, trip):
            reasons.append("hours")
        if (
            venue.price_per_person_sgd is not None
            and venue.price_per_person_sgd * party > planning_budget_sgd(trip) + 1e-6
        ):
            reasons.append("budget")
        if trip.weather_scenario == "heavy_rain" and venue.weather_sensitive:
            reasons.append("weather")
        if reasons:
            dropped.append(
                DroppedVenue(
                    venue_id=venue.id,
                    name=venue.name,
                    reasons=reasons,
                    detail=_detail(reasons),
                )
            )
            continue
        flags: list[str] = ["crowd_estimate"]
        if not venue.hours_known:
            flags.append("uncertain_hours")
        if not venue.price_known:
            flags.append("uncertain_price")
        if _catalogue_stale(venue.last_updated, trip.start_date):
            flags.append("stale_catalogue")
        kept.append(KeptVenue(venue=venue, flags=flags))
    return FilterResult(kept=kept, dropped=dropped, constraints=constraints)


def _detail(reasons: list[str]) -> str:
    phrases = {
        "diet": "does not meet the group's dietary requirement",
        "accessibility": "does not meet the group's accessibility requirement",
        "hours": "is closed throughout the trip's days and hours",
        "budget": "has a known price above the group budget",
        "weather": "is weather-sensitive and the heavy-rain scenario is on",
        "rejected": "was rejected for this plan",
    }
    return "; ".join(phrases[reason] for reason in reasons)


def drop_reason_text(reason: str) -> str:
    return {
        "diet": "it does not meet the group's dietary requirement",
        "accessibility": "it does not meet the group's accessibility requirement",
        "hours": "it is closed during the trip's days and hours",
        "budget": "its known price exceeds the group budget",
        "weather": "it is unsuitable in the selected heavy-rain scenario",
        "rejected": "it was rejected for this plan",
    }.get(reason, reason)
