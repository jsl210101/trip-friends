import json

from triper.pipeline import plan_trip
from triper.sample import sample_trip
from triper.scoring import factor_label


def test_sample_group_gets_a_fair_multi_stop_plan_without_leaking_identifiers():
    trip = sample_trip()
    result = plan_trip(trip)
    scheduled = [stop.venue_id for day in result.route.days for stop in day.stops]

    assert result.meets_floor
    assert min(result.route.utilities) >= trip.beta
    assert result.weights == [member.weight for member in trip.members]
    assert "botanic-gardens" in scheduled
    assert "gardens-by-the-bay" in scheduled
    assert "song-fa-bak-kut-teh" not in scheduled
    assert "haji-lane" not in scheduled
    assert len(scheduled) >= 6
    assert result.metrics["hard_constraint_violations"] == 0
    assert result.metrics["known_cost_sgd"] <= trip.budget_sgd
    assert any("accessibility" in note and "Haji Lane" in note for note in result.constraint_notes)
    assert all("Priya" not in note for note in result.constraint_notes)

    assert result.payload_preview is not None
    blob = json.dumps(result.payload_preview).lower()
    for secret in ("priya", "bayfront", "wf-1004", "halal", "wheelchair"):
        assert secret not in blob

    for explanation in result.explanations:
        assert explanation.top_factor == max(explanation.contributions, key=explanation.contributions.get)
        assert factor_label(explanation.top_factor) in explanation.why
        assert "Priya" not in explanation.why_not
