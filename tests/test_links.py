from triper.links import accept_extracted_names, match_places, resolve_links
from triper.llm import output_rejection, parse_place_response
from triper.sample import sample_trip
from triper.venues import load_venues


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
    from triper.links import match_places

    matches = match_places("we might do gardens", venues)
    assert matches
    assert all(item.needs_confirmation for item in matches if item.confidence < 0.82)


def test_xiaohongshu_and_douyin_share_cards_match_without_a_network_fetch(monkeypatch):
    def refuse_network(*_args, **_kwargs):
        raise AssertionError("share text must not be fetched")

    monkeypatch.setattr("urllib.request.urlopen", refuse_network)
    venues = load_venues()
    xhs = (
        "27 滨海湾花园夜景也太好看了 http://xhslink.com/a/GbBay01?xsec_token=secret-token "
        "复制这段话，打开【小红书】查看"
    )
    from triper.deidentify import share_links, split_share_card
    from triper.links import is_supported_import_host

    pairs = share_links(xhs)
    assert pairs and "xhslink.com" in pairs[0][0]
    assert is_supported_import_host(pairs[0][0])
    assert is_supported_import_host("https://www.xiaohongshu.com/explore/abc")
    assert is_supported_import_host("https://xhs.cn/abcd")
    path, caption = split_share_card(pairs[0][0], pairs[0][1])
    assert "xsec_token" not in path
    assert "xsec_token" not in caption
    assert "http" not in caption
    assert any(item.venue_id == "gardens-by-the-bay" for item in match_places(f"{path} {caption}", venues))

    douyin = "3.21 复制打开抖音，看看新加坡植物园 https://v.douyin.com/iRNB8abc/"
    assert is_supported_import_host(share_links(douyin)[0][0])
    path, caption = split_share_card(share_links(douyin)[0][0], share_links(douyin)[0][1])
    assert "v.douyin.com" not in caption
    assert any(item.venue_id == "botanic-gardens" for item in match_places(f"{path} {caption}", venues))

    trip = sample_trip()
    resolution = resolve_links(trip, venues, ai_consent=False)
    blob = str(resolution.payload).lower()
    assert "xhslink.com" not in blob
    assert "share_token" not in blob
    assert "http" not in blob
    assert "gardens by the bay" in blob or "滨海湾花园" in blob
    assert any(item.venue_id == "gardens-by-the-bay" for item in resolution.matches)


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
