from triper.models import INTERESTS, Member, MustVisit, TripInput, Venue, WEEKDAYS, blank_preferences
from triper.pipeline import plan_trip


def _prefs(**values: float) -> dict[str, float]:
    preferences = blank_preferences()
    preferences.update(values)
    return preferences


def _member(id_: str, name: str, **preferences: float) -> Member:
    return Member(id=id_, name=name, preferences=_prefs(**preferences), weight=1.0, crowd_tolerance=0.5)


def _venue(id_: str, name: str, interest: str, lat: float = 1.300, lon: float = 103.850) -> Venue:
    tags = {key: 0.0 for key in INTERESTS}
    tags[interest] = 1.0
    return Venue(
        id=id_,
        name=name,
        aliases=[],
        area="Test",
        lat=lat,
        lon=lon,
        category=interest,
        tags=tags,
        duration_min=45,
        price_per_person_sgd=0,
        hours={day: (7 * 60, 23 * 60) for day in WEEKDAYS},
        diets_supported=set(),
        serves_food=False,
        wheelchair_accessible=True,
        step_free=True,
        crowd_estimate=0.3,
        indoor=True,
        weather_sensitive=False,
        source="test",
        last_updated="2026-09-01",
    )


def _trip(members, musts=None, **overrides) -> TripInput:
    base = dict(
        start_date="2026-11-14",
        end_date="2026-11-14",
        day_start="09:00",
        day_end="18:00",
        budget_sgd=200,
        transport_modes=["walk"],
        abode_label="Test area",
        abode_address="1 Test Road",
        abode_lat=1.300,
        abode_lon=103.850,
        members=members,
        must_visits=musts or [],
        alphas={"preference_match": 1, "travel_time": 0, "cost": 0, "crowd": 0},
        beta=0.5,
    )
    base.update(overrides)
    return TripInput(**base)


def test_nsw_search_mixes_a_minority_interest_and_keeps_weights():
    members = [
        _member("a", "Shopper A", shopping=1),
        _member("b", "Shopper B", shopping=1),
        _member("c", "Shopper C", shopping=1),
        _member("d", "Walker", nature=1),
    ]
    venues = [
        _venue("mall-1", "Mall One", "shopping"),
        _venue("mall-2", "Mall Two", "shopping", 1.301, 103.851),
        _venue("park-1", "Park One", "nature", 1.302, 103.850),
        _venue("park-2", "Park Two", "nature", 1.301, 103.849),
    ]
    result = plan_trip(_trip(members), venues=venues)
    scheduled = [stop.venue_id for day in result.route.days for stop in day.stops]
    assert any(item.startswith("park") for item in scheduled)
    assert any(item.startswith("mall") for item in scheduled)
    assert result.meets_floor
    assert min(result.route.utilities) >= 0.5
    assert result.weights == [1, 1, 1, 1]
    assert result.metrics["hard_constraint_violations"] == 0


def test_flexible_must_and_fixed_time_anchor():
    members = [_member("a", "Walker", nature=1)]
    park = _venue("park", "Park", "nature")
    other = _venue("mall", "Mall", "shopping", 1.301, 103.851)
    timed = _venue("lookout", "Lookout", "nature", 1.3002, 103.8502)
    trip = _trip(
        members,
        musts=[
            MustVisit("park", "FLEXIBLE"),
            MustVisit("lookout", "FIXED", on_date="2026-11-14", at_time="15:00"),
        ],
        day_end="18:00",
    )
    result = plan_trip(trip, venues=[park, other, timed])
    stops = [stop for day in result.route.days for stop in day.stops]
    by_id = {stop.venue_id: stop for stop in stops}
    assert "park" in by_id
    assert by_id["lookout"].arrive_min == 15 * 60
    assert by_id["lookout"].role == "anchor_timed"
    assert result.route.days[0].stops[-1].depart_min + result.route.days[0].return_travel_min <= 18 * 60


def test_heavy_rain_blocks_a_weather_sensitive_must_visit():
    members = [_member("a", "Walker", nature=1)]
    outdoor = _venue("park", "Park", "nature")
    outdoor.weather_sensitive = True
    indoor = _venue("gallery", "Gallery", "culture", 1.301, 103.851)
    trip = _trip(
        members,
        musts=[MustVisit("park", "FLEXIBLE")],
        weather_scenario="heavy_rain",
    )
    result = plan_trip(trip, venues=[outdoor, indoor])
    scheduled = {stop.venue_id for day in result.route.days for stop in day.stops}
    assert "park" not in scheduled
    assert result.metrics["locked_in_blocked"] == 1
    assert any("Park" in line for line in result.route.conflicts)
    assert result.weights == [member.weight for member in members]
