from wayfare.models import Member, blank_preferences
from wayfare.pipeline import apply_member_feedback
from wayfare.preferences import rating_to_signal
from wayfare.venues import load_venues


def test_accept_updates_only_the_member_who_gave_feedback():
    venue = next(item for item in load_venues() if item.id == "botanic-gardens")
    actor = Member(
        id="m4",
        name="Priya Raman",
        preferences=blank_preferences() | {"nature": 0.2, "shopping": 0.8},
    )
    other = Member(
        id="m1",
        name="Alex Chen",
        preferences=blank_preferences() | {"shopping": 0.9},
    )
    other_before = dict(other.preferences)
    updated, message = apply_member_feedback(actor, venue, signal=1, personalisation_consent=True)
    assert updated.preferences["nature"] > actor.preferences["nature"]
    assert updated.id == actor.id
    assert other.preferences == other_before
    assert "only this traveller" in message

    refused, note = apply_member_feedback(actor, venue, signal=1, personalisation_consent=False)
    assert refused.preferences == actor.preferences
    assert refused.update_count == 0
    assert "not applied" in note


def test_reject_damps_the_venue_tags_and_neutral_rating_is_a_no_op():
    venue = next(item for item in load_venues() if item.id == "ion-orchard")
    member = Member(
        id="m1",
        name="Alex Chen",
        preferences=blank_preferences() | {"shopping": 0.9, "nature": 0.4},
    )
    rejected, _ = apply_member_feedback(member, venue, signal=-1, personalisation_consent=True)
    assert rejected.preferences["shopping"] < member.preferences["shopping"]
    assert abs(rejected.preferences["nature"] - member.preferences["nature"]) < 1e-9
    neutral, note = apply_member_feedback(rejected, venue, signal=rating_to_signal(3), personalisation_consent=True)
    assert neutral.preferences == rejected.preferences
    assert "neutral" in note.lower()
