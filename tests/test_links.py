from wayfare.links import accept_extracted_names, resolve_links
from wayfare.llm import output_rejection, parse_place_response
from wayfare.sample import sample_trip
from wayfare.venues import load_venues


def test_invented_places_are_dropped_and_catalogue_places_match():
    venues = load_venues()
    matches, unmatched = accept_extracted_names(
        ["Gardens by the Bay", "Atlantis Underwater Casino"],
        venues,
        source="model",
    )
    assert [item.venue_id for item in matches] == ["gardens-by-the-bay"]
    assert unmatched == ["Atlantis Underwater Casino"]

    trip = sample_trip()
    resolution = resolve_links(trip, venues, ai_consent=False)
    ids = {item.venue_id for item in resolution.matches}
    assert "gardens-by-the-bay" in ids
    assert resolution.model_call.sent is False
    blob = str(resolution.payload).lower()
    assert "atlantis" not in blob
    assert "priya" not in blob
    assert all(item.venue_id != "atlantis-underwater-casino" for item in resolution.matches)


def test_low_confidence_partial_name_needs_confirmation():
    venues = load_venues()
    from wayfare.links import match_places

    matches = match_places("we might do gardens", venues)
    assert matches
    assert all(item.needs_confirmation for item in matches if item.confidence < 0.82)


def test_maps_path_matches_maxwell_and_model_schema_rejects_extras():
    venues = load_venues()
    trip = sample_trip()
    trip.links[0].url = "https://www.google.com/maps/place/Maxwell+Food+Centre/"
    trip.links[0].caption = ""
    resolution = resolve_links(trip, venues, ai_consent=False)
    assert any(item.venue_id == "maxwell-food-centre" for item in resolution.matches)

    assert parse_place_response(
        {"places": [{"name": "X", "area": "Y", "category": "food", "confidence": 0.4, "extra": "no"}]}
    ) is None
    assert output_rejection({"places": [], "note": "see https://evil.example/phish"}, []) is not None
