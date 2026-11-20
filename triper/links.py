"""Imported-link parsing.

The social or map page is never opened. A local matcher reads the URL path
and the optional caption, then keeps a hit only when it matches the venue
catalogue. Low-confidence hits are returned for the user to confirm.
Names that do not match, including places invented by a model, are dropped.

A model may propose {name, area, category} from the de-identified path and
caption. Its proposals go through the same catalogue match.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import unquote

from triper.deidentify import build_llm_payload, sanitise_link
from triper.i18n import t
from triper.llm import (
    EXTRACT_INSTRUCTIONS,
    LLMClient,
    ModelCall,
    output_rejection,
    parse_place_response,
)
from triper.models import LinkImport, TripInput, Venue

CONFIRMED_THRESHOLD = 0.82
POSSIBLE_THRESHOLD = 0.55

# Hosts the importer recognises. Matching still uses only the path and caption.
IMPORT_HOSTS: tuple[str, ...] = (
    "xiaohongshu.com",
    "xhslink.com",
    "xhs.cn",
    "douyin.com",
    "tiktok.com",
    "instagram.com",
    "google.com",
    "goo.gl",
    "maps.app.goo.gl",
)

_STOPWORDS = {
    "the",
    "a",
    "an",
    "at",
    "in",
    "on",
    "of",
    "and",
    "or",
    "to",
    "for",
    "by",
    "with",
    "singapore",
    "sg",
    "night",
    "walk",
    "picnic",
    "my",
    "our",
    "this",
    "that",
    "from",
    "into",
    "around",
}


@dataclass
class PlaceMatch:
    venue_id: str
    venue_name: str
    confidence: float
    needs_confirmation: bool
    source: str


@dataclass
class LinkResolution:
    matches: list[PlaceMatch]
    unmatched: list[str]
    payload: dict
    model_call: ModelCall
    notes: list[str] = field(default_factory=list)
    note_keys: list[str] = field(default_factory=list)
    model_dropped: list[str] = field(default_factory=list)


def import_host(url: str) -> str:
    from urllib.parse import urlparse

    host = (urlparse(url).netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def is_supported_import_host(url: str) -> bool:
    """True for Xiaohongshu, Douyin, and the other advertised share hosts."""
    host = import_host(url)
    return any(host == item or host.endswith("." + item) for item in IMPORT_HOSTS)


def normalise_text(value: str) -> str:
    text = unquote(value or "").lower().replace("+", " ")
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _significant(name: str) -> list[str]:
    return [token for token in normalise_text(name).split() if token not in _STOPWORDS and len(token) > 1]


def name_score(text: str, name: str) -> float:
    text_norm = normalise_text(text)
    name_norm = normalise_text(name)
    if not text_norm or not name_norm:
        return 0.0
    if any("\u4e00" <= char <= "\u9fff" for char in name_norm):
        # Chinese names are not space-separated, so a caption can hold the
        # name inside a longer sentence. Two characters avoids a one-word hit.
        if len(name_norm) >= 2 and name_norm in text_norm:
            return 1.0
        return 0.0
    padded_text = f" {text_norm} "
    padded_name = f" {name_norm} "
    if padded_name in padded_text:
        if len(name_norm) >= 4 or " " in name_norm:
            return 1.0
        return 0.7
    tokens = set(text_norm.split())
    significant = _significant(name)
    if not significant:
        return 0.0
    hits = sum(1 for token in significant if token in tokens)
    if hits == 0:
        return 0.0
    recall = hits / len(significant)
    if recall < 0.5:
        return recall * 0.5
    if recall < 1.0:
        return 0.45 + 0.35 * recall
    return 0.9


def match_places(text: str, venues: list[Venue]) -> list[PlaceMatch]:
    found: list[PlaceMatch] = []
    for venue in venues:
        score = max(name_score(text, venue.name), *(name_score(text, alias) for alias in venue.aliases))
        if score < POSSIBLE_THRESHOLD:
            continue
        found.append(
            PlaceMatch(
                venue_id=venue.id,
                venue_name=venue.name,
                confidence=round(score, 3),
                needs_confirmation=score < CONFIRMED_THRESHOLD,
                source="heuristic",
            )
        )
    found.sort(key=lambda item: (-item.confidence, item.venue_name))
    # A clear catalogue hit should not also surface weaker overlaps on the same words.
    if any(item.confidence >= CONFIRMED_THRESHOLD for item in found):
        found = [item for item in found if item.confidence >= CONFIRMED_THRESHOLD]
    if len(found) >= 2 and found[0].confidence - found[1].confidence < 0.08:
        found[0].needs_confirmation = True
    return found


def accept_extracted_names(names: list[str], venues: list[Venue], source: str) -> tuple[list[PlaceMatch], list[str]]:
    """Keep extracted names only when the catalogue matcher agrees. Drop the rest."""
    kept: list[PlaceMatch] = []
    dropped: list[str] = []
    seen: set[str] = set()
    for name in names:
        cleaned = name.strip()
        if not cleaned:
            continue
        matches = match_places(cleaned, venues)
        if not matches:
            dropped.append(cleaned)
            continue
        best = matches[0]
        if best.venue_id in seen:
            continue
        seen.add(best.venue_id)
        kept.append(
            PlaceMatch(
                venue_id=best.venue_id,
                venue_name=best.venue_name,
                confidence=best.confidence,
                needs_confirmation=best.needs_confirmation or best.confidence < CONFIRMED_THRESHOLD,
                source=source,
            )
        )
    return kept, dropped


def _forbidden(trip: TripInput) -> list[str]:
    secrets: list[str] = [trip.abode_address]
    for member in trip.members:
        secrets.extend([member.name, member.email, member.booking_ref])
        secrets.extend(part for part in member.name.split() if len(part) >= 3)
    return [secret for secret in secrets if secret]


def resolve_links(
    trip: TripInput,
    venues: list[Venue],
    llm: LLMClient | None = None,
    ai_consent: bool = False,
) -> LinkResolution:
    """Match every imported link. The model is called only with consent and a key."""
    payload = build_llm_payload(trip, purpose="extract_places")
    notes: list[str] = []
    note_keys: list[str] = []
    model_places: list[dict] = []
    call = ModelCall(payload=payload, sent=False, accepted=False, rejection=None, body=None)
    client = llm or LLMClient()
    if ai_consent and client.available:
        raw = client.complete_json(payload, EXTRACT_INSTRUCTIONS)
        call.sent = True
        if raw is None:
            call.rejection = "The model call failed. Local matching was used instead."
            notes.append(call.rejection)
            note_keys.append("link.failed")
        else:
            echoed = [sanitise_link(link, trip.members, trip.abode_address, trip.abode_label)["caption"] for link in trip.links]
            rejection = output_rejection(raw, _forbidden(trip), echoed)
            if rejection:
                call.rejection = t(rejection, "en")
                notes.append(call.rejection)
                note_keys.append(rejection)
            else:
                parsed = parse_place_response(raw)
                if parsed is None:
                    call.rejection = "The model response did not match the fixed place schema and was rejected."
                    notes.append(call.rejection)
                    note_keys.append("link.bad_schema")
                else:
                    call.accepted = True
                    call.body = {"places": parsed}
                    model_places = parsed
    elif ai_consent and not client.available:
        notes.append("AI consent is on, but no LLM_API_KEY or OPENAI_API_KEY is set. Local matching was used.")
        note_keys.append("link.no_key")
    else:
        notes.append("No model call. Places were matched locally from the URL path and caption.")
        note_keys.append("link.local")

    matches: list[PlaceMatch] = []
    unmatched: list[str] = []
    model_dropped: list[str] = []
    seen: set[str] = set()
    for link in trip.links:
        cleaned = sanitise_link(link, trip.members, trip.abode_address, trip.abode_label)
        text = f"{cleaned['url_path']} {cleaned['caption']}".strip()
        local = match_places(text, venues)
        if not local:
            label = text or link.url
            unmatched.append(label)
        for item in local:
            if item.venue_id in seen:
                continue
            seen.add(item.venue_id)
            matches.append(item)
    if model_places:
        accepted, dropped = accept_extracted_names([item["name"] for item in model_places], venues, source="model")
        for name in dropped:
            model_dropped.append(name)
            unmatched.append(f"Model suggested “{name}”, which is not in the venue catalogue and was dropped.")
        for item in accepted:
            if item.venue_id in seen:
                continue
            # A model proposal never outranks the need to confirm a weak catalogue hit.
            item.needs_confirmation = item.needs_confirmation or item.confidence < CONFIRMED_THRESHOLD
            seen.add(item.venue_id)
            matches.append(item)
    matches.sort(key=lambda item: (-item.confidence, item.venue_name))
    return LinkResolution(
        matches=matches,
        unmatched=unmatched,
        payload=payload,
        model_call=call,
        notes=notes,
        note_keys=note_keys,
        model_dropped=model_dropped,
    )


def resolution_from_caption(text: str, venues: list[Venue]) -> tuple[list[PlaceMatch], list[str]]:
    """Helper for tests and the fallback matcher. Does not call a model."""
    matches = match_places(text, venues)
    if matches:
        return matches, []
    return [], [text]
