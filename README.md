# Wayfare

Wayfare plans a short group trip. Deterministic rules enforce hard constraints, a weighted linear score ranks venues, weighted Nash social welfare sets a fairness floor, and an orienteering heuristic builds the route. A pretrained language model is optional and is limited to unstructured text: imported-link parsing, and wording of explanations whose factors were already computed. It does not rank venues or choose the itinerary.

The design follows the report section “Use of AI Technologies”, not only the shorter class demo. Where the live demo drafts the day with a model and checks fairness afterwards, this app filters, scores, and routes locally first.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
streamlit run app.py
```

In the app, **Load sample Singapore trip**, then **Optimise itinerary**. The sample is four fictional travellers, a two-day window around Marina Bay, a flexible must-visit (Singapore Botanic Gardens), and a TikTok-style link whose caption names Gardens by the Bay. Names, emails, the booking references, and the street address in that sample are invented.

Without a model key the app still plans, matches links against the offline catalogue, and writes explanations from the scores.

Optional model (wording and link extraction only):

```bash
export LLM_API_KEY=...          # or OPENAI_API_KEY
export LLM_BASE_URL=https://api.openai.com/v1   # optional
export LLM_MODEL=gpt-4o-mini                     # optional
```

Temperature is capped at 0.2. Do not put keys in the repo.

`python -m wayfare.demo` prints the sample plan in the terminal.

## What runs locally

1. **Plan input.** Dates, daily window, budget, transport, an abode area plus a precise address, per-person interests (nature, culture, food, shopping, nightlife, adventure, relaxation), pace, crowd tolerance, agreed influence `w_k`, FIXED or FLEXIBLE must-visits, and pasted links.
2. **De-identification gateway.** Names and emails become Person A, Person B, and so on. The only payload that may leave the session is trip dates, budget, interests, pace, agreed weightage, a start-area label, link path plus caption, and, for wording, score facts. Diet and accessibility stay behind as group constraints. The page shows that payload. The social or map page is never opened.
3. **Hard filter.** A venue that breaks the group’s diet, accessibility, known opening hours, or known budget is removed before scoring. Missing hours or prices are flagged and are not guessed. Crowd figures are labelled estimates. Weather in the catalogue is an illustrative label, not a live NEA feed. A heavy-rain re-plan drops weather-sensitive venues.
4. **Score.** For member `k` and venue `v`,

   `S_k(v) = sum_f alpha_f * x_f^(k)(v)`

   Features, all oriented so that higher is better and scaled to [0, 1]: preference match, travel time, cost, crowd. Preference match is the interest-weighted average of the venue’s tags over interests that member rates above zero. Alphas are the sliders; they are normalised to sum to 1. An unknown price omits the cost feature and renormalises the rest.
5. **Satisfaction and fairness.**

   `U_k = normalise(mean of S_k(v) over the itinerary)`, clipped to [0, 1] because each score is already on that interval.

   The floor constraint is `U_k >= beta` (the sample uses 0.5). The reported welfare is `prod_k U_k^{w_k}`, equivalently `sum_k w_k log U_k`. `w_k` is the agreed influence and is not rewritten to manufacture fairness. If every candidate fails the floor, the screen names the conflict and leaves the weights alone.
6. **Route.** Team orienteering with time windows and anchors (Vansteenwegen, Souffriau, Van Oudheusden, 2011), solved by greedy insertion and bounded local search. FIXED must-visits that carry a date and time stay locked. Other must-visits must appear somewhere feasible. The insertion prize is the total member-weighted score collected in the time window, minus a penalty for uneven days, a penalty for back-to-back stops of the same category, and a caution when hours or price are unknown. Plans that miss the floor lose to plans that meet it. Weighted NSW breaks ties. This is not an exact optimum. Maximising the product of averaged utilities on its own collapses to a one-stop plan, which is why NSW is the floor and the tie-break rather than the insertion prize.
7. **Explanations.** Each stop cites the factor with the largest contribution, the factor with the largest gap against a named alternative, and the smallest single slider move that would reverse that pair after renormalisation. Anchors say that they are must-visits. A model may rephrase these sentences only when it keeps the same top factor.
8. **Confirm, reject, swap, re-plan.** The itinerary is a proposal until you confirm it. Rejecting or swapping optimises again. After confirmation you can apply a heavy-rain scenario or a morning delay. Ratings update only that traveller’s preference vector, by a simple moving average, and only if personalisation consent is on.

## Privacy controls that are actually implemented

- Notice, and separate consent for AI generation and for personalisation.
- Allowlist and identifier stripping before any model call. Tests check that names, emails, booking references, the street address, handles, and query tokens do not appear in the payload, and that diet and accessibility are not fields on it.
- Structured output only. Extra keys, links, raw identifiers, or a long verbatim echo of the caption are rejected and the local fallback is kept.
- Trip data is session-scoped. **Clear all my data** drops it. There is no account database, so there is no identity table to join back to trips.
- The model is not fine-tuned on trip inputs, and the inputs are not sent to a marketing sink by this app.
- Group explanations say “the group’s accessibility requirement” (or dietary requirement), not “because this named person has limited mobility”.

This build does not add federated learning, local differential privacy, or hardware enclaves. It does not claim to be PDPA-compliant.

## Approximations

- Travel time and fares are haversine estimates with fixed speeds, not OneMap or LTA DataMall.
- The venue list is a small offline Singapore catalogue with a source note and a last-reviewed date. Hours, prices, diet, and access flags are curated for the demo. Stale rows and unknown hours or prices are flagged.
- Crowd levels are static estimates, not live counts. Weather is an illustrative label plus a user-triggered heavy-rain switch, not a live NEA forecast.
- The route search is capped (local-search iterations, optional stops per day, replacement pool) so a re-plan stays interactive.
- Link matching without a model is a catalogue matcher on the URL path and caption. Low-confidence hits wait for you to confirm them. Names that do not match the catalogue, including places a model invents, are dropped.
- “Normalise” on satisfaction is a clip to [0, 1]. The smallest reversal is a search over one slider at a time, then the same renormalisation the scorer uses.
- A grid helper, `select_floor`, can pick a beta on simulated utility vectors. The app default is the report’s example floor of 0.5, not a value fitted on this group.

## Tests

```bash
pytest
```

Covered: identifier stripping and the allowlist; hard-constraint filtering, including a weight that cannot restore a removed venue; NSW preferring a balanced utility vector over a majority-only vector whose sum is higher; the fairness-floor veto; explanations citing the real top factor and a slider move that does reverse the pair; link extraction dropping an invented place; must-visit anchors; a minority interest kept in the route; personalisation that does not touch the other traveller; and the sample group plan.
