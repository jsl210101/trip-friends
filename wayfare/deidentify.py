"""De-identification gateway.

The model never receives the raw trip record. Names, emails, booking
references, and the precise abode address are removed or replaced with
Person A, Person B, and so on. Diet and accessibility stay out of the
payload entirely; they are group hard constraints applied locally.

Allowlisted fields: trip dates, budget, interests, pace, agreed weightage,
imported-link path and caption, an area or start-node label, and (for
wording only) score facts that were already computed locally.
"""

from __future__ import annotations

import json
import re
from urllib.parse import unquote, urlparse

from wayfare.models import INTERESTS, LinkImport, Member, TripInput

ALLOWLIST: frozenset[str] = frozenset(
    {
        "purpose",
        "trip_dates",
        "budget_sgd",
        "interests",
        "pace",
        "weightage",
        "links",
        "start_node_label",
        "score_facts",
    }
)

_EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
_HANDLE_RE = re.compile(r"@[A-Za-z0-9._]+")
_PHONE_RE = re.compile(r"(?:\+\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{3,4}")
_BOOKING_RE = re.compile(r"\b[A-Z]{2,6}-\d{3,}\b")
_PLATFORM_SEGMENTS = {
    "www",
    "video",
    "videos",
    "p",
    "reel",
    "reels",
    "tv",
    "maps",
    "place",
    "search",
    "explore",
    "stories",
    "share",
    "dir",
    "data",
    "watch",
}


class PayloadRejected(ValueError):
    """Raised when a payload still contains a disallowed identifier."""


def pseudonym(index: int) -> str:
    """0 -> Person A, 1 -> Person B, 26 -> Person AA."""
    letters = ""
    number = index
    while True:
        letters = chr(ord("A") + number % 26) + letters
        number = number // 26 - 1
        if number < 0:
            break
    return f"Person {letters}"


def assign_pseudonyms(members: list[Member]) -> dict[str, str]:
    """Stable pseudonyms in the order the group entered members."""
    return {member.id: pseudonym(index) for index, member in enumerate(members)}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def address_fragments(address: str, area_label: str) -> list[str]:
    """Pieces of the precise abode that must not leave the session."""
    fragments: list[str] = []
    cleaned = address.strip()
    if cleaned:
        fragments.append(cleaned)
    label_tokens = set(_tokens(area_label))
    generic = {"singapore", "road", "street", "drive", "avenue", "lane", "walk", "close"}
    for code in re.findall(r"\b\d{6}\b", address):
        fragments.append(code)
    for unit in re.findall(r"#\s*\d+\s*-\s*\d+", address):
        fragments.append(re.sub(r"\s+", "", unit))
        fragments.append(unit)
    for word in re.findall(r"[A-Za-z]{4,}", address):
        if word.lower() not in label_tokens and word.lower() not in generic:
            fragments.append(word)
    # Longer fragments first so a full address is removed before its parts.
    fragments.sort(key=len, reverse=True)
    return fragments


def redact_free_text(text: str, members: list[Member], address: str, area_label: str) -> str:
    """Strip identifiers from a caption or URL fragment."""
    redacted = text or ""
    redacted = _EMAIL_RE.sub("[redacted-email]", redacted)
    redacted = _HANDLE_RE.sub("[redacted-handle]", redacted)
    for member in members:
        if member.email:
            redacted = re.sub(re.escape(member.email), "[redacted-email]", redacted, flags=re.IGNORECASE)
        if member.booking_ref:
            redacted = re.sub(re.escape(member.booking_ref), "[redacted-booking]", redacted, flags=re.IGNORECASE)
        if member.name.strip():
            redacted = re.sub(re.escape(member.name.strip()), "[redacted-name]", redacted, flags=re.IGNORECASE)
            for part in member.name.split():
                if len(part) >= 3:
                    redacted = re.sub(rf"\b{re.escape(part)}\b", "[redacted-name]", redacted, flags=re.IGNORECASE)
    redacted = _BOOKING_RE.sub("[redacted-booking]", redacted)
    for fragment in address_fragments(address, area_label):
        if fragment:
            redacted = re.sub(re.escape(fragment), "[redacted-address]", redacted, flags=re.IGNORECASE)
    redacted = _PHONE_RE.sub("[redacted-phone]", redacted)
    redacted = re.sub(r"(?:\[redacted-name\]\s*){2,}", "[redacted-name] ", redacted)
    return re.sub(r"\s+", " ", redacted).strip()


def url_path_for_model(url: str) -> str:
    """Keep a place-like path and drop hosts, queries, handles, and opaque ids.

    The page is never opened. Query strings (tracking tokens, emails) are
    discarded. @handles are social-account identifiers and are removed.
    """
    parsed = urlparse((url or "").strip())
    raw = unquote(parsed.path or "").replace("+", " ")
    kept: list[str] = []
    for segment in raw.split("/"):
        segment = segment.strip()
        if not segment or segment.startswith("@"):
            continue
        lowered = segment.lower()
        if lowered in _PLATFORM_SEGMENTS:
            continue
        if re.fullmatch(r"\d+", segment):
            continue
        # Opaque social ids have no spaces. Hyphenated or spaced place names stay.
        if " " not in segment and "-" not in segment and re.fullmatch(r"[A-Za-z0-9_]{6,}", segment):
            continue
        kept.append(segment)
    return "/".join(kept)


def sanitise_link(link: LinkImport, members: list[Member], address: str, area_label: str) -> dict[str, str]:
    path = redact_free_text(url_path_for_model(link.url), members, address, area_label)
    caption = redact_free_text(link.caption, members, address, area_label)
    return {"url_path": path, "caption": caption}


def _assert_allowlist(payload: dict) -> None:
    extra = set(payload) - ALLOWLIST
    if extra:
        raise PayloadRejected(f"Payload has fields outside the allowlist: {sorted(extra)}")


def _walk_strings(value: object) -> list[str]:
    found: list[str] = []
    if isinstance(value, str):
        found.append(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            found.append(str(key))
            found.extend(_walk_strings(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(_walk_strings(item))
    return found


def assert_payload_safe(payload: dict, trip: TripInput) -> None:
    """Refuse to hand off a payload that still contains identifiers or sensitive labels."""
    _assert_allowlist(payload)
    blob = json.dumps(payload, ensure_ascii=False)
    lowered = blob.lower()
    leaks: list[str] = []
    for member in trip.members:
        for secret in (member.name, member.email, member.booking_ref):
            if secret and secret.lower() in lowered:
                leaks.append(secret)
        for part in member.name.split():
            if len(part) >= 3 and re.search(rf"\b{re.escape(part.lower())}\b", lowered):
                leaks.append(part)
    for fragment in address_fragments(trip.abode_address, trip.abode_label):
        if len(fragment) >= 3 and fragment.lower() in lowered:
            leaks.append(fragment)
    if _EMAIL_RE.search(blob):
        leaks.append("email-pattern")
    if _HANDLE_RE.search(blob):
        leaks.append("handle")
    # Diet and access needs are group constraints, not model inputs.
    for key in ("diet", "diets", "accessibility", "wheelchair", "booking_ref", "email", "abode_address"):
        if key in payload:
            leaks.append(key)
    if leaks:
        raise PayloadRejected("Refusing model payload; identifiers still present: " + ", ".join(sorted(set(leaks))))


def build_llm_payload(
    trip: TripInput,
    purpose: str,
    score_facts: dict | None = None,
    links: list[LinkImport] | None = None,
) -> dict:
    """Build the only JSON that may be sent to the model.

    `weightage` copies the agreed w_k. It is not rebalanced for fairness.
    """
    pseudonyms = assign_pseudonyms(trip.members)
    interests: dict[str, dict[str, float]] = {}
    pace: dict[str, str] = {}
    weightage: dict[str, float] = {}
    for member in trip.members:
        label = pseudonyms[member.id]
        interests[label] = {key: round(float(member.preferences.get(key, 0.0)), 3) for key in INTERESTS}
        pace[label] = member.pace
        weightage[label] = float(member.weight)
    payload: dict = {
        "purpose": purpose,
        "trip_dates": [trip.start_date, trip.end_date],
        "budget_sgd": trip.budget_sgd,
        "interests": interests,
        "pace": pace,
        "weightage": weightage,
        "links": [
            sanitise_link(link, trip.members, trip.abode_address, trip.abode_label)
            for link in (trip.links if links is None else links)
        ],
        "start_node_label": trip.abode_label,
    }
    if score_facts is not None:
        payload["score_facts"] = score_facts
    assert_payload_safe(payload, trip)
    return payload
