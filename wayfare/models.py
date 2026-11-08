"""Shared records for a Wayfare trip.

Personal fields (name, email, booking reference, precise abode address,
diet, accessibility) stay inside the session. Only an allowlisted,
de-identified payload may be offered to an external model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

INTERESTS: tuple[str, ...] = (
    "nature",
    "culture",
    "food",
    "shopping",
    "nightlife",
    "adventure",
    "relaxation",
)

# Higher is better after normalisation. These are the only scoring features.
FEATURES: tuple[str, ...] = (
    "preference_match",
    "travel_time",
    "cost",
    "crowd",
)

PACES: tuple[str, ...] = ("slow", "moderate", "fast")
DIETS: tuple[str, ...] = ("halal", "vegetarian", "vegan", "no_pork", "alcohol_free")
ACCESS_NEEDS: tuple[str, ...] = ("wheelchair", "step_free")
TRANSPORT_MODES: tuple[str, ...] = ("walk", "transit", "ride_hail")
WEEKDAYS: tuple[str, ...] = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")

FEATURE_LABELS: dict[str, str] = {
    "preference_match": "preference match",
    "travel_time": "travel time",
    "cost": "cost",
    "crowd": "crowd",
}


def parse_hm(value: str) -> int:
    """Convert HH:MM to minutes after midnight."""
    hour, minute = value.strip().split(":")
    return int(hour) * 60 + int(minute)


def format_hm(minutes: int) -> str:
    minutes = int(minutes)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def weekday_key(day: date) -> str:
    return WEEKDAYS[day.weekday()]


def each_date(start_iso: str, end_iso: str) -> list[date]:
    start = date.fromisoformat(start_iso)
    end = date.fromisoformat(end_iso)
    if end < start:
        raise ValueError("Trip end date is before the start date.")
    days: list[date] = []
    cursor = start
    while cursor <= end:
        days.append(cursor)
        cursor += timedelta(days=1)
    return days


def blank_preferences() -> dict[str, float]:
    return {key: 0.0 for key in INTERESTS}


@dataclass
class Venue:
    """A catalogue place. Prices, hours, and crowd levels may be unknown.

    Unknown opening hours and prices are stored as None and must not be
    imputed. Crowd figures are estimates, never live counts.
    """

    id: str
    name: str
    aliases: list[str]
    area: str
    lat: float
    lon: float
    category: str
    tags: dict[str, float]
    duration_min: int
    price_per_person_sgd: float | None
    hours: dict[str, tuple[int, int] | None] | None
    diets_supported: set[str]
    serves_food: bool
    wheelchair_accessible: bool
    step_free: bool
    crowd_estimate: float
    indoor: bool
    weather_sensitive: bool
    source: str
    last_updated: str
    notes: str = ""

    @property
    def hours_known(self) -> bool:
        return self.hours is not None

    @property
    def price_known(self) -> bool:
        return self.price_per_person_sgd is not None


@dataclass
class Member:
    id: str
    name: str
    email: str = ""
    booking_ref: str = ""
    preferences: dict[str, float] = field(default_factory=blank_preferences)
    pace: str = "moderate"
    crowd_tolerance: float = 0.5
    diets: list[str] = field(default_factory=list)
    accessibility: list[str] = field(default_factory=list)
    weight: float = 1.0
    update_count: int = 0


@dataclass
class MustVisit:
    venue_id: str
    kind: str  # FIXED or FLEXIBLE
    on_date: str | None = None
    at_time: str | None = None


@dataclass
class LinkImport:
    url: str
    caption: str = ""


@dataclass
class TripInput:
    start_date: str
    end_date: str
    day_start: str
    day_end: str
    budget_sgd: float
    transport_modes: list[str]
    abode_label: str
    abode_address: str
    abode_lat: float
    abode_lon: float
    members: list[Member]
    must_visits: list[MustVisit] = field(default_factory=list)
    budget_currency: str = "SGD"
    # SGD per 1 unit of budget_currency, locked from the suggestion on screen.
    # None uses the offline table. The pipeline does not fetch a rate itself.
    fx_sgd_per_unit: float | None = None
    links: list[LinkImport] = field(default_factory=list)
    alphas: dict[str, float] = field(default_factory=dict)
    beta: float = 0.5
    weather_scenario: str = "forecast"
    delay_minutes: int = 0
    delay_date: str | None = None
    excluded_venue_ids: list[str] = field(default_factory=list)

    def dates(self) -> list[date]:
        return each_date(self.start_date, self.end_date)
