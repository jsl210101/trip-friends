from wayfare.constraints import filter_venues
from wayfare.models import INTERESTS, Member, MustVisit, TripInput, Venue, WEEKDAYS, blank_preferences
from wayfare.pipeline import plan_trip


def _member(**kwargs) -> Member:
    preferences = blank_preferences()
    preferences.update(kwargs.pop("preferences", {}))
    return Member(id=kwargs.pop("id", "m1"), name=kwargs.pop("name", "Traveller"), preferences=preferences, **kwargs)


def _venue(**overrides) -> Venue:
    hours = {day: (8 * 60, 22 * 60) for day in WEEKDAYS}
    base = dict(
        id="cafe",
        name="Cafe",
        aliases=[],
        area="Test",
        lat=1.30,
        lon=103.85,
        category="food",
        tags={key: 0.0 for key in INTERESTS},
        duration_min=60,
        price_per_person_sgd=10,
        hours=hours,
        diets_supported=set(),
        serves_food=True,
        wheelchair_accessible=True,
        step_free=True,
        crowd_estimate=0.4,
        indoor=True,
        weather_sensitive=False,
        source="test",
        last_updated="2026-09-01",
    )
    base.update(overrides)
    return Venue(**base)


def _trip(members, venues_unused=None, **overrides) -> TripInput:
    base = dict(
        start_date="2026-11-16",
        end_date="2026-11-16",
        day_start="09:00",
        day_end="18:00",
        budget_sgd=100,
        transport_modes=["walk"],
        abode_label="Test area",
        abode_address="99 Example Crescent",
        abode_lat=1.30,
        abode_lon=103.85,
        members=members,
        alphas={"preference_match": 1, "travel_time": 0, "cost": 0, "crowd": 0},
        beta=0.5,
    )
    base.update(overrides)
    return TripInput(**base)


def test_hard_constraints_drop_and_unknowns_are_flagged_not_guessed():
    members = [
        _member(
            preferences={"food": 1},
            diets=["halal"],
            accessibility=["wheelchair"],
        )
    ]
    closed = {day: None for day in WEEKDAYS}
    venues = [
        _venue(id="halal", name="Halal Kitchen", diets_supported={"halal", "no_pork"}, tags={"food": 1}),
        _venue(id="pork", name="Pork Stall", tags={"food": 1}),
        _venue(id="stairs", name="Upstairs Gallery", category="culture", serves_food=False, wheelchair_accessible=False, step_free=False, tags={"culture": 1}),
        _venue(id="shut", name="Monday Museum", hours=closed, serves_food=False, category="culture"),
        _venue(id="pricey", name="Pricey Tasting", price_per_person_sgd=250, diets_supported={"halal"}),
        _venue(id="unknown-hours", name="Pop Up", hours=None, serves_food=False, category="culture", diets_supported=set()),
        _venue(id="unknown-price", name="Free Maybe", price_per_person_sgd=None, serves_food=False, category="culture"),
    ]
    result = filter_venues(venues, _trip(members))
    dropped = {item.venue_id: item.reasons for item in result.dropped}
    assert "diet" in dropped["pork"]
    assert "accessibility" in dropped["stairs"]
    assert "hours" in dropped["shut"]
    assert "budget" in dropped["pricey"]
    kept = {item.venue.id: item.flags for item in result.kept}
    assert "halal" in kept
    assert "uncertain_hours" in kept["unknown-hours"]
    assert "uncertain_price" in kept["unknown-price"]
    assert "crowd_estimate" in kept["halal"]
    # Unknown hours were not filled in.
    unknown = next(item.venue for item in result.kept if item.venue.id == "unknown-hours")
    assert unknown.hours is None


def test_weights_cannot_restore_a_filtered_venue():
    member = _member(preferences={"food": 1}, diets=["halal"])
    pork = _venue(id="pork", name="Pork Stall", tags={"food": 1})
    halal = _venue(id="halal", name="Halal Kitchen", diets_supported={"halal"}, tags={"food": 0.4})
    trip = _trip(
        [member],
        must_visits=[MustVisit("pork", "FLEXIBLE")],
        alphas={"preference_match": 1, "travel_time": 0, "cost": 0, "crowd": 0},
        budget_sgd=50,
    )
    result = plan_trip(trip, venues=[pork, halal])
    scheduled = {stop.venue_id for day in result.route.days for stop in day.stops}
    assert "pork" not in scheduled
    assert result.metrics["hard_constraint_violations"] == 0
    assert any("Pork Stall" in line for line in result.route.conflicts)
    assert result.weights == [1.0]
