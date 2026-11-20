import json

import pytest

from triper.constraints import filter_venues
from triper.models import INTERESTS, Member, TripInput, Venue, WEEKDAYS, blank_preferences
from triper.money import (
    CURRENCIES,
    exchange_suggestion,
    fallback_quote,
    planning_budget_sgd,
    sgd_per_unit_fallback,
)
from triper.pipeline import plan_trip


def _member() -> Member:
    preferences = blank_preferences()
    preferences["food"] = 1.0
    return Member(id="m1", name="Traveller", preferences=preferences)


def _venue(price: float) -> Venue:
    return Venue(
        id="meal",
        name="Meal",
        aliases=[],
        area="Test",
        lat=1.30,
        lon=103.85,
        category="food",
        tags={key: 0.0 for key in INTERESTS} | {"food": 1.0},
        duration_min=60,
        price_per_person_sgd=price,
        hours={day: (8 * 60, 22 * 60) for day in WEEKDAYS},
        diets_supported={"halal"},
        serves_food=True,
        wheelchair_accessible=True,
        step_free=True,
        crowd_estimate=0.4,
        indoor=True,
        weather_sensitive=False,
        source="test",
        last_updated="2026-09-01",
    )


def _trip(amount: float, currency: str, **overrides) -> TripInput:
    base = dict(
        start_date="2026-11-16",
        end_date="2026-11-16",
        day_start="09:00",
        day_end="18:00",
        budget_sgd=amount,
        budget_currency=currency,
        transport_modes=["walk"],
        abode_label="Test area",
        abode_address="99 Example Crescent",
        abode_lat=1.30,
        abode_lon=103.85,
        members=[_member()],
        alphas={"preference_match": 1, "travel_time": 0, "cost": 0, "crowd": 0},
        beta=0.4,
    )
    base.update(overrides)
    return TripInput(**base)


def test_supported_currencies_include_the_singapore_and_china_set():
    for code in ("SGD", "CNY", "USD", "EUR", "MYR", "HKD", "JPY", "THB", "GBP"):
        assert code in CURRENCIES


def test_same_number_in_another_currency_is_not_the_same_budget():
    venue = _venue(50)
    sgd = filter_venues([venue], _trip(100, "SGD"))
    jpy = filter_venues([venue], _trip(100, "JPY"))
    assert planning_budget_sgd(_trip(100, "SGD")) == 100
    assert planning_budget_sgd(_trip(100, "JPY")) == pytest.approx(100 * sgd_per_unit_fallback("JPY"))
    assert planning_budget_sgd(_trip(100, "JPY")) < 50
    assert [item.venue.id for item in sgd.kept] == ["meal"]
    assert [item.venue_id for item in jpy.dropped] == ["meal"]
    assert "budget" in jpy.dropped[0].reasons

    sgd_plan = plan_trip(_trip(100, "SGD"), venues=[venue])
    jpy_plan = plan_trip(_trip(100, "JPY"), venues=[venue])
    assert "meal" in {stop.venue_id for day in sgd_plan.route.days for stop in day.stops}
    assert "meal" not in {stop.venue_id for day in jpy_plan.route.days for stop in day.stops}
    assert jpy_plan.metrics["budget_currency"] == "JPY"
    assert jpy_plan.metrics["budget_entered"] == 100
    assert jpy_plan.metrics["budget_sgd"] < 2


def test_locked_rate_is_the_one_used_before_scoring():
    loose = _trip(100, "USD", fx_sgd_per_unit=2.0)
    tight = _trip(100, "USD")
    assert planning_budget_sgd(loose) == 200
    assert planning_budget_sgd(tight) == pytest.approx(100 * sgd_per_unit_fallback("USD"))
    venue = _venue(150)
    assert filter_venues([venue], loose).kept
    assert filter_venues([venue], tight).dropped


def test_exchange_suggestion_uses_cny_when_the_budget_is_already_sgd():
    hint = exchange_suggestion(420, "SGD", allow_network=False)
    assert hint.converted_currency == "CNY"
    assert hint.planning_sgd == 420
    assert hint.converted_amount == pytest.approx(420 * fallback_quote("SGD", "CNY").rate)
    assert hint.source == "fallback"
    assert hint.as_of == "2026-11-06"


def test_exchange_suggestion_converts_foreign_budget_into_sgd():
    hint = exchange_suggestion(5000, "CNY", allow_network=False)
    assert hint.converted_currency == "SGD"
    assert hint.planning_sgd == pytest.approx(5000 * sgd_per_unit_fallback("CNY"))
    assert hint.planning_sgd != 5000
    assert hint.rate_base == "CNY"
    assert hint.rate_quote == "SGD"


def test_live_rate_falls_back_offline_when_the_network_fails(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise TimeoutError("offline")

    monkeypatch.setattr("triper.money.urllib.request.urlopen", refuse)
    hint = exchange_suggestion(80, "EUR", allow_network=True)
    assert hint.source == "fallback"
    assert hint.planning_sgd == pytest.approx(80 * sgd_per_unit_fallback("EUR"))


def test_live_rate_parses_a_public_quote_without_sending_the_amount(monkeypatch):
    seen = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(
                {"amount": 1.0, "base": "USD", "date": "2026-11-06", "rates": {"SGD": 1.5}}
            ).encode()

    def fake_open(request, timeout=0):
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("triper.money.urllib.request.urlopen", fake_open)
    hint = exchange_suggestion(10, "USD", allow_network=True)
    assert hint.source == "live"
    assert hint.as_of == "2026-11-06"
    assert hint.planning_sgd == pytest.approx(15)
    assert "amount=10" not in seen["url"]
    assert "from=USD" in seen["url"]
    assert "to=SGD" in seen["url"]
