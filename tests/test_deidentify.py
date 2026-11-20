import json

from triper.deidentify import ALLOWLIST, build_llm_payload, url_path_for_model
from triper.sample import sample_trip


def test_payload_strips_identifiers_and_keeps_the_allowlist():
    trip = sample_trip()
    payload = build_llm_payload(trip, purpose="extract_places")
    blob = json.dumps(payload).lower()

    assert set(payload) <= ALLOWLIST
    for secret in (
        "alex",
        "chen",
        "mei",
        "tan",
        "jordan",
        "lee",
        "priya",
        "raman",
        "alex.chen@example.com",
        "priya.raman@example.com",
        "wf-1001",
        "wf-1004",
        "bayfront",
        "018956",
        "share_token",
        "secret-token",
        "@priya",
        "priya.eats",
    ):
        assert secret not in blob, secret
    assert "halal" not in blob
    assert "wheelchair" not in blob
    assert "accessibility" not in blob
    assert "diet" not in blob
    assert payload["start_node_label"] == "Marina Bay area"
    assert "gardens by the bay" in blob
    assert "https://" not in blob
    assert payload["weightage"] == {
        "Person A": 1.0,
        "Person B": 1.0,
        "Person C": 1.0,
        "Person D": 1.0,
    }


def test_agreed_weightage_is_not_rewritten():
    trip = sample_trip()
    trip.members[0].weight = 2
    trip.members[3].weight = 0.5
    payload = build_llm_payload(trip, purpose="extract_places")
    assert payload["weightage"]["Person A"] == 2
    assert payload["weightage"]["Person D"] == 0.5


def test_url_path_drops_handles_queries_and_opaque_ids():
    path = url_path_for_model(
        "https://www.tiktok.com/@priya.eats/video/7291847562819384756?share_token=secret-token"
    )
    assert path == ""
    maps = url_path_for_model("https://www.google.com/maps/place/Maxwell+Food+Centre/")
    assert "Maxwell Food Centre" in maps
    assert "share_token" not in maps
