"""Optional pretrained LLM client.

Used only for unstructured text: place extraction from a link path and
caption, and wording of explanations whose factors were already computed.
Temperature is capped at 0.2. The response must be one fixed JSON object.
Extra links, files, raw identifiers, or a verbatim echo of the caption are
rejected and the local fallback is kept.

If LLM_API_KEY or OPENAI_API_KEY is unset, every call returns None and the
app stays on the local matcher and the score templates.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

EXTRACT_KEYS = ("name", "area", "category", "confidence")
WORD_KEYS = ("venue_id", "why", "why_not", "reversal", "top_factor")

EXTRACT_INSTRUCTIONS = (
    "You extract candidate places from a de-identified URL path and caption. "
    "Return only a JSON object of the form "
    '{"places":[{"name":"string","area":"string","category":"string","confidence":0.0}]}. '
    "confidence is between 0 and 1. Do not rank places, do not plan an itinerary, "
    "do not add links or files, and do not repeat names, emails, booking references, "
    "or street addresses. If nothing is a place, return {\"places\":[]}."
)

WORD_INSTRUCTIONS = (
    "You rephrase explanations that were already computed. Return only a JSON object "
    '{"stops":[{"venue_id":"string","why":"string","why_not":"string","reversal":"string","top_factor":"string"}]}. '
    "Copy venue_id and top_factor exactly. The why sentence must name that same top factor. "
    "Do not change which venue was chosen, do not invent scores, do not add links, "
    "and do not mention a named person's diet, mobility, email, or address. "
    "Use group wording such as \"the group's accessibility requirement\"."
)


def word_instructions(lang: str = "en") -> str:
    """Ask the model to word explanations in the current interface language."""
    if lang == "zh":
        return (
            WORD_INSTRUCTIONS
            + " Write why, why_not, and reversal in Simplified Chinese. "
            + "The why sentence must include the Chinese name of the top factor "
            + "(偏好匹配, 行程时间, 费用, or 拥挤程度). Leave top_factor as the English code."
        )
    return WORD_INSTRUCTIONS + " Write why, why_not, and reversal in English."

_LINK_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)


@dataclass
class ModelCall:
    payload: dict
    sent: bool
    accepted: bool
    rejection: str | None = None
    body: dict | None = None


class LLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> None:
        if api_key is None:
            api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        self.api_key = api_key
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL") or "gpt-4o-mini"
        requested = temperature if temperature is not None else float(os.environ.get("LLM_TEMPERATURE", "0.2"))
        # Low temperature is part of the containment design, not a tunable default.
        self.temperature = min(0.2, max(0.0, float(requested)))

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def complete_json(self, payload: dict, instructions: str) -> dict | None:
        if not self.available:
            return None
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = json.loads(response.read().decode("utf-8"))
            content = raw["choices"][0]["message"]["content"]
            parsed = json.loads(content)
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError, TypeError, ValueError):
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed


def output_rejection(obj: object, forbidden: list[str], echoed: list[str] | None = None) -> str | None:
    """Return a reason to discard model output, or None when it may be parsed."""
    if not isinstance(obj, dict):
        return "model.output_object"
    blob = json.dumps(obj, ensure_ascii=False)
    if _LINK_RE.search(blob):
        return "model.output_link"
    if _EMAIL_RE.search(blob):
        return "model.output_email"
    lowered = blob.lower()
    for secret in forbidden:
        if secret and len(secret) >= 3 and secret.lower() in lowered:
            return "model.output_identifier"
    for phrase in echoed or []:
        compact = (phrase or "").strip()
        if len(compact) >= 24 and compact.lower() in lowered:
            return "model.output_echo"
    return None


def parse_place_response(obj: dict) -> list[dict] | None:
    if set(obj) != {"places"} or not isinstance(obj["places"], list):
        return None
    places: list[dict] = []
    for item in obj["places"]:
        if not isinstance(item, dict) or set(item) != set(EXTRACT_KEYS):
            return None
        if not all(isinstance(item[key], str) for key in ("name", "area", "category")):
            return None
        confidence = item["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            return None
        if not 0.0 <= float(confidence) <= 1.0:
            return None
        places.append(
            {
                "name": item["name"].strip(),
                "area": item["area"].strip(),
                "category": item["category"].strip(),
                "confidence": float(confidence),
            }
        )
    return places


def parse_wording_response(obj: dict) -> list[dict] | None:
    if set(obj) != {"stops"} or not isinstance(obj["stops"], list):
        return None
    stops: list[dict] = []
    for item in obj["stops"]:
        if not isinstance(item, dict) or set(item) != set(WORD_KEYS):
            return None
        if not all(isinstance(item[key], str) and item[key].strip() for key in WORD_KEYS):
            return None
        stops.append({key: item[key].strip() for key in WORD_KEYS})
    return stops
