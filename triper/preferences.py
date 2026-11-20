"""Per-member preference updates.

Accepts, rejects, and ratings adjust only that member's preference vector
by a simple moving average of one observation. This is not reinforcement
learning: there is no reward model, policy, or shared update.

signal > 0 pulls the vector toward the venue's tags.
signal < 0 damps the tags of the rejected venue and leaves unrelated
interests alone. signal 0 makes no change.
"""

from __future__ import annotations

from triper.models import INTERESTS, Member, Venue

# Prior observations so the first click does not replace the stated profile.
PRIOR_STRENGTH = 3


def rating_to_signal(rating: int) -> float:
    """Map a 1–5 rating onto [-1, 1]. A rating of 3 is neutral."""
    bounded = min(5, max(1, int(rating)))
    return (bounded - 3) / 2.0


def observation(member: Member, venue: Venue, signal: float) -> dict[str, float]:
    signal = max(-1.0, min(1.0, signal))
    observed: dict[str, float] = {}
    for key in INTERESTS:
        current = float(member.preferences.get(key, 0.0))
        tag = float(venue.tags.get(key, 0.0))
        if signal > 0:
            observed[key] = (1.0 - signal) * current + signal * tag
        elif signal < 0:
            strength = -signal
            observed[key] = min(1.0, max(0.0, current * (1.0 - strength * tag)))
        else:
            observed[key] = current
    return observed


def moving_average_update(
    member: Member,
    venue: Venue,
    signal: float,
) -> tuple[dict[str, float], int]:
    """Return the updated preference vector and the new observation count.

    Does not mutate the member. Callers apply the result only when
    personalisation consent is on, and only to the member who gave feedback.
    """
    signal = max(-1.0, min(1.0, signal))
    if signal == 0.0:
        return {key: float(member.preferences.get(key, 0.0)) for key in INTERESTS}, member.update_count
    observed = observation(member, venue, signal)
    count = PRIOR_STRENGTH + member.update_count
    updated = {
        key: (count * float(member.preferences.get(key, 0.0)) + observed[key]) / (count + 1.0)
        for key in INTERESTS
    }
    return updated, member.update_count + 1
