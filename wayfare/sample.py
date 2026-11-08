"""A fictional Singapore group used by the demo and the end-to-end test.

Names, emails, and the booking references are invented. The precise abode
address is session data and must not appear in a model payload.
"""

from __future__ import annotations

from wayfare.models import INTERESTS, LinkImport, Member, MustVisit, TripInput, blank_preferences
from wayfare.pipeline import default_alphas


def _prefs(**values: float) -> dict[str, float]:
    preferences = blank_preferences()
    for key, value in values.items():
        if key not in INTERESTS:
            raise KeyError(key)
        preferences[key] = value
    return preferences


def sample_members() -> list[Member]:
    return [
        Member(
            id="m1",
            name="Alex Chen",
            email="alex.chen@example.com",
            booking_ref="WF-1001",
            preferences=_prefs(shopping=0.9, food=0.7, nightlife=0.45),
            pace="moderate",
            crowd_tolerance=0.7,
            weight=1.0,
        ),
        Member(
            id="m2",
            name="Mei Tan",
            email="mei.tan@example.com",
            booking_ref="WF-1002",
            preferences=_prefs(shopping=0.85, food=0.8),
            pace="moderate",
            crowd_tolerance=0.6,
            weight=1.0,
        ),
        Member(
            id="m3",
            name="Jordan Lee",
            email="jordan.lee@example.com",
            booking_ref="WF-1003",
            preferences=_prefs(shopping=0.8, relaxation=0.55),
            pace="moderate",
            crowd_tolerance=0.5,
            weight=1.0,
        ),
        Member(
            id="m4",
            name="Priya Raman",
            email="priya.raman@example.com",
            booking_ref="WF-1004",
            preferences=_prefs(nature=0.95, relaxation=0.7),
            pace="moderate",
            crowd_tolerance=0.2,
            diets=["halal"],
            accessibility=["wheelchair"],
            weight=1.0,
        ),
    ]


def sample_trip() -> TripInput:
    """Two days around Marina Bay, with one flexible must-visit and one imported link."""
    return TripInput(
        start_date="2026-11-14",
        end_date="2026-11-15",
        day_start="09:30",
        day_end="21:00",
        budget_sgd=420.0,
        transport_modes=["transit", "walk"],
        abode_label="Marina Bay area",
        abode_address="10 Bayfront Avenue, Singapore 018956",
        abode_lat=1.2830,
        abode_lon=103.8600,
        members=sample_members(),
        must_visits=[
            MustVisit(venue_id="botanic-gardens", kind="FLEXIBLE"),
            # Confirmed high-confidence match from the imported link below.
            MustVisit(venue_id="gardens-by-the-bay", kind="FLEXIBLE"),
        ],
        links=[
            LinkImport(
                url="https://www.tiktok.com/@priya.eats/video/7291847562819384756?share_token=secret-token",
                caption=(
                    "Priya Raman night walk at gardens by the bay. "
                    "Email priya.raman@example.com ref WF-1004. "
                    "Stay at 10 Bayfront Avenue, Singapore 018956."
                ),
            )
        ],
        alphas=default_alphas(),
        beta=0.5,
    )
