"""Currency conversion for a Singapore trip.

Catalogue prices and fares are Singapore dollars. The number someone types
is converted into SGD before the budget ceiling, cost checks, or fairness
math. A live ECB reference rate is optional. The offline table is always
available and is what the planner uses unless a rate has been locked from
the suggestion the user just saw.

No personal data is sent. A live request contains only a currency pair.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

CURRENCIES: tuple[str, ...] = ("SGD", "CNY", "USD", "EUR", "MYR", "HKD", "JPY", "THB", "GBP")
DESTINATION = "SGD"
REFERENCE = "CNY"

# Units of foreign currency per 1 SGD. ECB reference via Frankfurter, 2026-11-06.
# Stored so the app still converts with no network. Not a bank or card rate.
FALLBACK_PER_SGD: dict[str, float] = {
    "SGD": 1.0,
    "CNY": 5.2055,
    "USD": 0.78028,
    "EUR": 0.69886,
    "MYR": 3.1672,
    "HKD": 6.1154,
    "JPY": 123.54,
    "THB": 26.179,
    "GBP": 0.58295,
}
FALLBACK_AS_OF = "2026-11-06"
FALLBACK_PROVIDER = "Stored ECB snapshot (Frankfurter), 2026-11-06"
LIVE_PROVIDER = "Frankfurter, ECB reference"
LIVE_URL = "https://api.frankfurter.app/latest?from={base}&to={quote}"


@dataclass(frozen=True)
class RateQuote:
    base: str
    quote: str
    rate: float
    source: str  # live or fallback
    as_of: str
    provider: str

    @property
    def sgd_per_unit(self) -> float:
        """SGD received for 1 unit of `base`, when `quote` is SGD."""
        if self.quote != DESTINATION:
            raise ValueError("sgd_per_unit is only defined for a quote in SGD.")
        return self.rate


@dataclass(frozen=True)
class ExchangeSuggestion:
    entered_amount: float
    entered_currency: str
    converted_amount: float
    converted_currency: str
    planning_sgd: float
    sgd_per_unit: float
    rate_base: str
    rate_quote: str
    rate_value: float
    source: str
    as_of: str
    provider: str


class UnknownCurrency(ValueError):
    pass


def _check(currency: str) -> str:
    code = (currency or "").upper()
    if code not in FALLBACK_PER_SGD:
        raise UnknownCurrency(code)
    return code


def fallback_quote(base: str, quote: str) -> RateQuote:
    """Offline cross rate. `rate` is units of `quote` per 1 `base`."""
    base = _check(base)
    quote = _check(quote)
    if base == quote:
        rate = 1.0
    else:
        # FALLBACK is units of that currency per 1 SGD.
        # quote per 1 base = (quote per SGD) / (base per SGD).
        rate = FALLBACK_PER_SGD[quote] / FALLBACK_PER_SGD[base]
    return RateQuote(base, quote, rate, "fallback", FALLBACK_AS_OF, FALLBACK_PROVIDER)


def sgd_per_unit_fallback(currency: str) -> float:
    currency = _check(currency)
    return 1.0 / FALLBACK_PER_SGD[currency]


def _fetch_live(base: str, quote: str, timeout: float = 2.5) -> RateQuote | None:
    if base == quote:
        return RateQuote(base, quote, 1.0, "live", FALLBACK_AS_OF, LIVE_PROVIDER)
    url = LIVE_URL.format(base=base, quote=quote)
    request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        rate = float(payload["rates"][quote])
        as_of = str(payload.get("date") or "")
    except Exception:  # noqa: BLE001 - offline, Pyodide (no sockets), or bad payload all fall back
        return None
    if rate <= 0:
        return None
    return RateQuote(base, quote, rate, "live", as_of, LIVE_PROVIDER)


def quote_pair(
    base: str,
    quote: str,
    *,
    allow_network: bool = False,
    cache: dict | None = None,
    locked_sgd_per_unit: float | None = None,
) -> RateQuote:
    """Rate as units of `quote` per 1 `base`.

    `locked_sgd_per_unit` applies only when converting that currency into SGD.
    The planner passes it so scoring uses the same figure the suggestion showed.
    """
    base = _check(base)
    quote = _check(quote)
    if locked_sgd_per_unit is not None and locked_sgd_per_unit > 0 and quote == DESTINATION:
        rate = locked_sgd_per_unit if base != DESTINATION else 1.0
        if base == DESTINATION:
            rate = 1.0
        return RateQuote(base, quote, rate, "locked", FALLBACK_AS_OF, "Rate shown with the budget")
    key = (base, quote, "live" if allow_network else "fallback")
    if cache is not None and key in cache:
        return cache[key]
    found: RateQuote | None = None
    if allow_network:
        found = _fetch_live(base, quote)
    if found is None:
        found = fallback_quote(base, quote)
    if cache is not None:
        cache[key] = found
    return found


def planning_budget_sgd(trip) -> float:
    """SGD ceiling used by filters, fares, scores, and fairness.

    `trip.budget_sgd` is the number that was typed. It is not assumed to be
    Singapore dollars unless `budget_currency` is SGD.
    """
    currency = _check(getattr(trip, "budget_currency", None) or DESTINATION)
    amount = float(trip.budget_sgd)
    locked = getattr(trip, "fx_sgd_per_unit", None)
    if currency == DESTINATION:
        return amount
    if locked is not None and float(locked) > 0:
        return amount * float(locked)
    return amount * sgd_per_unit_fallback(currency)


def exchange_suggestion(
    amount: float,
    currency: str,
    *,
    allow_network: bool = False,
    cache: dict | None = None,
) -> ExchangeSuggestion:
    """Planning hint. SGD input is also shown in CNY so the line is never blank."""
    currency = _check(currency)
    amount = float(amount)
    into_sgd = quote_pair(currency, DESTINATION, allow_network=allow_network, cache=cache)
    planning = amount * into_sgd.rate
    if currency == DESTINATION:
        reference = quote_pair(DESTINATION, REFERENCE, allow_network=allow_network, cache=cache)
        return ExchangeSuggestion(
            entered_amount=amount,
            entered_currency=currency,
            converted_amount=amount * reference.rate,
            converted_currency=REFERENCE,
            planning_sgd=amount,
            sgd_per_unit=1.0,
            rate_base=reference.base,
            rate_quote=reference.quote,
            rate_value=reference.rate,
            source=reference.source,
            as_of=reference.as_of,
            provider=reference.provider,
        )
    return ExchangeSuggestion(
        entered_amount=amount,
        entered_currency=currency,
        converted_amount=planning,
        converted_currency=DESTINATION,
        planning_sgd=planning,
        sgd_per_unit=into_sgd.rate,
        rate_base=into_sgd.base,
        rate_quote=into_sgd.quote,
        rate_value=into_sgd.rate,
        source=into_sgd.source,
        as_of=into_sgd.as_of,
        provider=into_sgd.provider,
    )


def format_money(amount: float, currency: str) -> str:
    currency = _check(currency)
    if currency == "JPY":
        return f"{amount:,.0f} {currency}"
    return f"{amount:,.2f} {currency}"


def format_rate(value: float) -> str:
    if value >= 100:
        return f"{value:,.2f}"
    if value >= 1:
        return f"{value:,.4f}"
    return f"{value:.6f}"
