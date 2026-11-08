"""Wayfare session UI.

The page is a thin shell over the local pipeline. Trip inputs live in
st.session_state and are discarded by "Clear all my data". Nothing is written
to a database.
"""

from __future__ import annotations

import os
from dataclasses import replace
from datetime import date, datetime

import streamlit as st

from wayfare.deidentify import PayloadRejected, build_llm_payload
from wayfare.links import resolve_links
from wayfare.llm import LLMClient
from wayfare.models import (
    FEATURES,
    INTERESTS,
    LinkImport,
    Member,
    MustVisit,
    TripInput,
    blank_preferences,
    format_hm,
)
from wayfare.pipeline import (
    apply_member_feedback,
    default_alphas,
    illustrative_forecast,
    plan_trip,
    rephrase_explanations,
)
from wayfare.preferences import rating_to_signal
from wayfare.sample import sample_trip
from wayfare.scoring import factor_label
from wayfare.venues import ABODES, load_venues

st.set_page_config(page_title="Wayfare", page_icon="WF", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: #f6f3ec; }
    h1, h2, h3 { font-family: Georgia, "Iowan Old Style", Palatino, serif; letter-spacing: -0.02em; }
    div[data-testid="stSidebar"] { background: #e7efe9; }
    .wayfare-banner {
        background: #123c3a;
        color: #f6f3ec;
        padding: 0.9rem 1.1rem;
        border-radius: 12px;
        margin-bottom: 0.8rem;
    }
    .wayfare-banner p { margin: 0.2rem 0 0; color: #d5e4df; }
    </style>
    """,
    unsafe_allow_html=True,
)

VENUES = load_venues()
VENUE_BY_ID = {venue.id: venue for venue in VENUES}
NOTICE = """
Before any external model call, Wayfare de-identifies the trip. Names, email
addresses, booking references, and the precise abode address stay in this
session. The model, if you consent and a key is configured, receives only an
allowlisted payload: trip dates, budget, interests, pace, agreed weightage, a
start-area label, and the path and caption of imported links. Dietary and
accessibility needs are applied locally as group hard constraints and are not
sent as one named person's sensitive label.

Trip inputs are used only to plan the current itinerary. They are not used to
train or fine-tune a model, and they are not used for marketing. Nothing is
written to a database. Closing the session or choosing **Clear all my data**
discards it.

These are the controls this build actually implements. They are not a claim of
PDPA compliance, and they are not a guarantee of privacy. Pseudonymous interest
vectors are still behavioural data; the control is minimisation.
"""


def _init() -> None:
    ss = st.session_state
    if ss.get("ready"):
        return
    ss.ready = True
    ss.members = []
    ss.musts = []
    ss.links = []
    ss.excluded = []
    ss.plan = None
    ss.link_result = None
    ss.confirmed = False
    ss.notice = False
    ss.ai_consent = False
    ss.personal_consent = False
    ss.weather = "forecast"
    ss.delay_minutes = 0
    ss.alphas = default_alphas()
    ss.beta = 0.5
    ss.start_date = date(2026, 11, 14)
    ss.end_date = date(2026, 11, 15)
    ss.day_start = "09:30"
    ss.day_end = "21:00"
    ss.budget = 420.0
    ss.modes = ["transit", "walk"]
    ss.abode_label = "Marina Bay area"
    ss.abode_address = ""
    ss.feedback = []
    ss.model_note = ""
    ss._seq = 20


def _abode(label: str) -> dict:
    for item in ABODES:
        if item["label"] == label:
            return item
    return ABODES[0]


def _sync_member_controls(member: Member) -> None:
    for interest in INTERESTS:
        st.session_state.setdefault(f"pref_{member.id}_{interest}", float(member.preferences.get(interest, 0.0)))
    st.session_state.setdefault(f"weight_{member.id}", float(member.weight))
    st.session_state.setdefault(f"pace_{member.id}", member.pace)
    st.session_state.setdefault(f"crowd_{member.id}", float(member.crowd_tolerance))
    st.session_state.setdefault(f"diet_{member.id}", list(member.diets))
    st.session_state.setdefault(f"access_{member.id}", list(member.accessibility))


def _load_sample() -> None:
    trip = sample_trip()
    ss = st.session_state
    ss.members = trip.members
    ss.musts = list(trip.must_visits)
    ss.links = list(trip.links)
    ss.excluded = []
    ss.plan = None
    ss.link_result = None
    ss.confirmed = False
    ss.weather = "forecast"
    ss.delay_minutes = 0
    ss.alphas = dict(trip.alphas)
    ss.beta = trip.beta
    ss.start_date = date.fromisoformat(trip.start_date)
    ss.end_date = date.fromisoformat(trip.end_date)
    ss["start_date_input"] = ss.start_date
    ss["end_date_input"] = ss.end_date
    ss.day_start = trip.day_start
    ss.day_end = trip.day_end
    ss.budget = trip.budget_sgd
    ss.modes = list(trip.transport_modes)
    ss["modes_input"] = list(trip.transport_modes)
    ss.abode_label = trip.abode_label
    ss["abode_label_input"] = trip.abode_label
    ss.abode_address = trip.abode_address
    ss["weather_input"] = "forecast"
    ss["delay_input"] = 0
    ss.feedback = []
    ss.model_note = "Sample link is loaded. Match it, then add Gardens by the Bay if you want the import to count as well as the must-visit already on the trip."
    for feature, value in trip.alphas.items():
        ss[f"alpha_{feature}"] = value
    ss["beta_slider"] = trip.beta
    ss["budget_input"] = trip.budget_sgd
    ss["day_start_input"] = trip.day_start
    ss["day_end_input"] = trip.day_end
    ss["abode_address_input"] = trip.abode_address
    for member in trip.members:
        for interest in INTERESTS:
            ss[f"pref_{member.id}_{interest}"] = float(member.preferences.get(interest, 0.0))
        ss[f"weight_{member.id}"] = float(member.weight)
        ss[f"pace_{member.id}"] = member.pace
        ss[f"crowd_{member.id}"] = float(member.crowd_tolerance)
        ss[f"diet_{member.id}"] = list(member.diets)
        ss[f"access_{member.id}"] = list(member.accessibility)


def _clear() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def _members_from_state() -> list[Member]:
    updated: list[Member] = []
    for member in st.session_state.members:
        _sync_member_controls(member)
        preferences = {
            interest: float(st.session_state[f"pref_{member.id}_{interest}"]) for interest in INTERESTS
        }
        updated.append(
            replace(
                member,
                preferences=preferences,
                weight=float(st.session_state[f"weight_{member.id}"]),
                pace=st.session_state[f"pace_{member.id}"],
                crowd_tolerance=float(st.session_state[f"crowd_{member.id}"]),
                diets=list(st.session_state[f"diet_{member.id}"]),
                accessibility=list(st.session_state[f"access_{member.id}"]),
            )
        )
    return updated


def _alphas_from_state() -> dict[str, float]:
    alphas = dict(st.session_state.alphas)
    for feature in FEATURES:
        key = f"alpha_{feature}"
        if key in st.session_state:
            alphas[feature] = float(st.session_state[key])
    return alphas


def _current_trip() -> TripInput:
    label = st.session_state.get("abode_label_input", st.session_state.abode_label)
    preset = _abode(label)
    address = st.session_state.get("abode_address_input", st.session_state.abode_address)
    start = st.session_state.get("start_date_input", st.session_state.start_date)
    end = st.session_state.get("end_date_input", st.session_state.end_date)
    if isinstance(start, datetime):
        start = start.date()
    if isinstance(end, datetime):
        end = end.date()
    delay = int(st.session_state.get("delay_input", st.session_state.delay_minutes))
    return TripInput(
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        day_start=st.session_state.get("day_start_input", st.session_state.day_start).strip(),
        day_end=st.session_state.get("day_end_input", st.session_state.day_end).strip(),
        budget_sgd=float(st.session_state.get("budget_input", st.session_state.budget)),
        transport_modes=list(st.session_state.get("modes_input", st.session_state.modes)),
        abode_label=label,
        abode_address=address.strip(),
        abode_lat=float(preset["lat"]),
        abode_lon=float(preset["lon"]),
        members=_members_from_state(),
        must_visits=list(st.session_state.musts),
        links=list(st.session_state.links),
        alphas=_alphas_from_state(),
        beta=float(st.session_state.get("beta_slider", st.session_state.beta)),
        weather_scenario=st.session_state.get("weather_input", st.session_state.weather),
        delay_minutes=delay,
        delay_date=start.isoformat() if delay else None,
        excluded_venue_ids=list(st.session_state.excluded),
    )


def _validate(trip: TripInput) -> list[str]:
    errors = []
    if not trip.members:
        errors.append("Add at least one traveller.")
    if trip.end_date < trip.start_date:
        errors.append("The end date is before the start date.")
    try:
        if trip.day_end <= trip.day_start:
            errors.append("The day needs a start time before the end time. Use 24-hour HH:MM.")
    except ValueError:
        errors.append("Daily times must look like 09:30.")
    if trip.budget_sgd <= 0:
        errors.append("Budget has to be above zero.")
    if not trip.transport_modes:
        errors.append("Pick at least one transport mode.")
    if len(trip.dates()) > 4:
        errors.append("This demo catalogue is meant for one to four days.")
    return errors


def _ai_allowed() -> bool:
    return bool(st.session_state.notice and st.session_state.ai_consent)


def _render_banner() -> None:
    st.markdown(
        """
        <div class="wayfare-banner">
          <strong>Wayfare</strong>
          <p>A group itinerary planner. Hard constraints, scores, weighted Nash social welfare, and the route are computed in this session. A model, if used at all, only reads de-identified text.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_members() -> None:
    st.subheader("Travellers")
    st.caption(
        "Preference sliders are each person's own profile. Agreed influence is w_k. "
        "The fairness floor does not rewrite it. Diet and access become group hard constraints."
    )
    if not st.session_state.members:
        st.info("No travellers yet. Load the sample group, or add someone below.")
    for index, member in enumerate(list(st.session_state.members)):
        _sync_member_controls(member)
        with st.expander(f"{member.name or 'Traveller'}  ·  {member.id}", expanded=index == 0):
            columns = st.columns([2, 1, 1])
            columns[0].write(f"Email (session only): {member.email or '—'}")
            columns[1].write(f"Booking ref (session only): {member.booking_ref or '—'}")
            if columns[2].button("Remove", key=f"remove_{member.id}"):
                st.session_state.members = [item for item in st.session_state.members if item.id != member.id]
                st.rerun()
            pref_cols = st.columns(4)
            for offset, interest in enumerate(INTERESTS):
                pref_cols[offset % 4].slider(
                    interest.capitalize(),
                    0.0,
                    1.0,
                    key=f"pref_{member.id}_{interest}",
                )
            meta = st.columns(4)
            meta[0].selectbox("Pace", ["slow", "moderate", "fast"], key=f"pace_{member.id}")
            meta[1].slider("Crowd tolerance", 0.0, 1.0, key=f"crowd_{member.id}", help="0 prefers quiet. 1 is at ease in a crowd.")
            meta[2].number_input("Agreed influence w", min_value=0.1, max_value=5.0, step=0.1, key=f"weight_{member.id}")
            meta[3].caption("Higher w means more influence. It is not a fairness patch.")
            st.multiselect(
                "Dietary requirements (group hard constraint)",
                ["halal", "vegetarian", "vegan", "no_pork", "alcohol_free"],
                key=f"diet_{member.id}",
            )
            st.multiselect(
                "Accessibility requirements (group hard constraint)",
                ["wheelchair", "step_free"],
                key=f"access_{member.id}",
            )
    with st.expander("Add a traveller"):
        with st.form("add_member"):
            name = st.text_input("Name")
            email = st.text_input("Email")
            booking = st.text_input("Booking reference")
            pace = st.selectbox("Pace", ["moderate", "slow", "fast"])
            submitted = st.form_submit_button("Add traveller")
        if submitted:
            if not name.strip():
                st.error("A name is needed so the group can tell profiles apart. It stays in the session.")
            else:
                st.session_state._seq += 1
                member = Member(
                    id=f"m{st.session_state._seq}",
                    name=name.strip(),
                    email=email.strip(),
                    booking_ref=booking.strip(),
                    preferences=blank_preferences(),
                    pace=pace,
                )
                st.session_state.members.append(member)
                _sync_member_controls(member)
                st.rerun()


def _render_places() -> None:
    st.subheader("Must-visits and imported links")
    left, right = st.columns(2)
    with left:
        venue_id = st.selectbox("Place", [venue.id for venue in VENUES], format_func=lambda item: VENUE_BY_ID[item].name)
        kind = st.selectbox("Kind", ["FLEXIBLE", "FIXED"])
        on_date = st.text_input("FIXED date (YYYY-MM-DD, optional)", value="")
        at_time = st.text_input("FIXED time (HH:MM, optional)", value="")
        if st.button("Add must-visit"):
            if kind == "FIXED" and at_time and not on_date:
                st.error("A clock time needs a date, so the anchor locks to the right day.")
            else:
                st.session_state.musts.append(
                    MustVisit(
                        venue_id=venue_id,
                        kind=kind,
                        on_date=on_date.strip() or None,
                        at_time=at_time.strip() or None,
                    )
                )
                st.rerun()
        if st.session_state.musts:
            for index, must in enumerate(list(st.session_state.musts)):
                name = VENUE_BY_ID[must.venue_id].name if must.venue_id in VENUE_BY_ID else must.venue_id
                label = f"{name} · {must.kind}"
                if must.on_date:
                    label += f" · {must.on_date}"
                if must.at_time:
                    label += f" {must.at_time}"
                cols = st.columns([4, 1])
                cols[0].write(label)
                if cols[1].button("Drop", key=f"drop_must_{index}"):
                    st.session_state.musts.pop(index)
                    st.rerun()
        else:
            st.caption("No must-visits yet. FLEXIBLE ones must appear somewhere. FIXED ones with a time stay locked.")
    with right:
        url = st.text_input("Link (TikTok, Instagram, Google Maps, or the web)")
        caption = st.text_area("Optional caption", height=80)
        if st.button("Save link"):
            if url.strip():
                st.session_state.links.append(LinkImport(url.strip(), caption.strip()))
                st.rerun()
        for index, link in enumerate(st.session_state.links):
            st.write(f"{index + 1}. {link.url}")
            if link.caption:
                st.caption(link.caption)
        if st.button("Match imported links"):
            trip = _current_trip()
            consent = _ai_allowed()
            if st.session_state.ai_consent and not st.session_state.notice:
                st.warning("Acknowledge the notice on the De-identify tab before a model call. Matching stayed local.")
            result = resolve_links(trip, VENUES, llm=LLMClient(), ai_consent=consent)
            st.session_state.link_result = result
        result = st.session_state.link_result
        if result is not None:
            for note in result.notes:
                st.caption(note)
            if not result.matches and not result.unmatched:
                st.info("No place text to match yet.")
            chosen = []
            for match in result.matches:
                confidence = f"{match.confidence:.2f}"
                flag = "confirm this lower-confidence match" if match.needs_confirmation else "high-confidence catalogue match"
                match_key = f"match_{match.venue_id}_{match.source}"
                st.session_state.setdefault(match_key, not match.needs_confirmation)
                if st.checkbox(f"{match.venue_name} ({confidence}, {match.source}, {flag})", key=match_key):
                    chosen.append(match.venue_id)
            for text in result.unmatched:
                st.warning(f"Dropped, no catalogue match: {text}")
            if st.button("Add selected matches as flexible must-visits") and chosen:
                existing = {must.venue_id for must in st.session_state.musts}
                for venue_id in chosen:
                    if venue_id not in existing:
                        st.session_state.musts.append(MustVisit(venue_id, "FLEXIBLE"))
                st.rerun()


def _render_inputs() -> None:
    st.header("1. Plan input")
    st.caption("Nothing is stored until you confirm, and even then it stays in this session.")
    top = st.columns(4)
    top[0].date_input("Start date", key="start_date_input")
    top[1].date_input("End date", key="end_date_input")
    top[2].text_input("Day starts", key="day_start_input")
    top[3].text_input("Day ends", key="day_end_input")
    second = st.columns(3)
    second[0].number_input("Group budget (SGD)", min_value=0.0, step=10.0, key="budget_input")
    second[1].multiselect("Transport modes", ["walk", "transit", "ride_hail"], key="modes_input")
    second[2].selectbox("Abode area (sent, at most)", [item["label"] for item in ABODES], key="abode_label_input")
    st.text_input(
        "Precise abode address (local start and end node only, never sent)",
        key="abode_address_input",
        help="Used as each day's start and end coordinate. The model receives the area label only.",
    )
    _render_members()
    _render_places()


def _render_privacy() -> None:
    st.header("2. De-identify")
    st.markdown(NOTICE)
    st.checkbox("I have read this notice", key="notice")
    st.checkbox("Consent to AI generation (link parsing and explanation wording only)", key="ai_consent")
    st.checkbox("Consent to personalisation (this traveller's own moving-average update)", key="personal_consent")
    key_on = bool(os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    st.caption(
        "Model key detected. Calls use low temperature and a fixed JSON schema."
        if key_on
        else "No LLM_API_KEY or OPENAI_API_KEY. Link matching and explanations stay on the local fallback."
    )
    st.subheader("Payload that would be sent")
    if not _members_from_state():
        st.info("Add a traveller to preview the allowlisted payload.")
        return
    try:
        payload = build_llm_payload(_current_trip(), purpose="extract_places")
    except PayloadRejected as exc:
        st.error(str(exc))
        return
    except ValueError as exc:
        st.error(str(exc))
        return
    st.json(payload)
    st.caption("This preview is not a model call. Diet, accessibility, names, emails, booking references, and the street address are absent.")


def _render_filter(plan) -> None:
    kept_col, dropped_col = st.columns(2)
    with kept_col:
        st.markdown("**Still eligible**")
        for item in plan.filtered.kept:
            flags = ", ".join(item.flags)
            price = "price unknown" if item.venue.price_per_person_sgd is None else f"SGD {item.venue.price_per_person_sgd:.0f} / person"
            st.write(f"{item.venue.name} · {item.venue.area} · {price} · crowd {item.venue.crowd_estimate:.2f} (estimate)")
            st.caption(f"{flags}. Source: {item.venue.source}. Last reviewed {item.venue.last_updated}.")
    with dropped_col:
        st.markdown("**Removed before scoring**")
        if not plan.filtered.dropped:
            st.caption("No venue failed a hard constraint.")
        for item in plan.filtered.dropped:
            st.write(f"{item.name}")
            st.caption(item.detail)


def _optimise() -> None:
    trip = _current_trip()
    errors = _validate(trip)
    if errors:
        for error in errors:
            st.error(error)
        return
    try:
        st.session_state.plan = plan_trip(trip, VENUES)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.session_state.confirmed = False


def _render_optimise() -> None:
    st.header("3. Filter and optimise")
    st.caption(
        "Hard constraints run first. S_k(v) is a weighted sum of normalised features. "
        "U_k is the normalised mean of those scores. The route collects member-weighted "
        "score inside the day, and it is vetoed when any U_k is below beta."
    )
    cols = st.columns(4)
    labels = {
        "preference_match": "Preference match",
        "travel_time": "Travel time",
        "cost": "Cost",
        "crowd": "Crowd",
    }
    for index, feature in enumerate(FEATURES):
        st.session_state.setdefault(f"alpha_{feature}", float(st.session_state.alphas.get(feature, 0.25)))
        cols[index].slider(labels[feature], 0.0, 1.0, key=f"alpha_{feature}")
    st.slider("Fairness floor beta", 0.0, 0.9, key="beta_slider", help="A plan that leaves anyone below this floor is vetoed. w_k is not rewritten.")
    if st.button("Optimise itinerary", type="primary"):
        _optimise()
    plan = st.session_state.plan
    if plan is None:
        st.info("Optimise to see who was filtered out and which plan the heuristic kept.")
        return
    alphas = plan.normalised_alphas
    st.write(
        "Normalised criterion weights: "
        + ", ".join(f"{factor_label(feature)} {alphas[feature]:.2f}" for feature in FEATURES)
    )
    _render_filter(plan)


def _render_plan() -> None:
    st.header("4. Explain and confirm")
    plan = st.session_state.plan
    if plan is None:
        st.info("Optimise a plan first. You can swap, reject, or move the weights, then optimise again.")
        return
    trip = _current_trip()
    metrics = plan.metrics
    cards = st.columns(4)
    cards[0].metric("Floor", "Met" if plan.meets_floor else "Vetoed", f"beta {plan.beta:.2f}")
    cards[1].metric("Lowest satisfaction", f"{metrics['lowest_satisfaction']:.2f}", f"gap {metrics['satisfaction_gap']:.2f}")
    product = metrics["nsw_product"]
    cards[2].metric("Weighted NSW", f"{product:.3f}" if isinstance(product, float) else "—")
    cards[3].metric("Known cost", f"SGD {metrics['known_cost_sgd']:.0f}", f"budget {metrics['budget_sgd']:.0f}")
    st.write(plan.floor_message)
    extra = st.columns(3)
    detour = metrics["detour_ratio"]
    extra[0].write(f"Detour ratio (estimate): {detour:.2f}" if isinstance(detour, float) else "Detour ratio: —")
    extra[1].write(f"Pacing variance (hours²): {metrics['pacing_variance_hours']:.2f}")
    extra[2].write(f"Hard-constraint violations in the plan: {metrics['hard_constraint_violations']}")
    st.caption(
        "Detour ratio compares planned travel with a travel-only nearest-neighbour order of the same stops. "
        "It is not proof of an optimal tour. Crowd, fares, and travel times are estimates."
    )
    if plan.constraint_notes:
        st.subheader("Hard constraints the weights cannot override")
        for note in plan.constraint_notes:
            st.warning(note)
    if plan.route.conflicts:
        st.subheader("Must-visits that could not be placed")
        for line in plan.route.conflicts:
            st.error(line)
    st.subheader("Satisfaction")
    for member_id, utility in zip(plan.route.scores, plan.route.utilities):
        member = next((item for item in st.session_state.members if item.id == member_id), None)
        name = member.name if member else member_id
        label = plan.pseudonyms.get(member_id, "pseudonym")
        weight = member.weight if member else ""
        st.write(f"{name} ({label} in any model payload): **{utility:.2f}** · agreed influence {weight}")
    for day in plan.route.days:
        forecast = illustrative_forecast(date.fromisoformat(day.day))
        st.subheader(day.day)
        st.caption(
            f"Window {format_hm(day.start_min)}–{format_hm(day.end_min)}. "
            f"Illustrative weather: {forecast['condition']}, rain risk {forecast['rain_risk']} "
            f"({forecast['source']}, as of {forecast['as_of']})."
        )
        if not day.stops:
            st.write("No stops fit this day.")
        for stop in day.stops:
            venue = VENUE_BY_ID.get(stop.venue_id)
            title = venue.name if venue else stop.venue_id
            uncertain = []
            if stop.uncertain_hours:
                uncertain.append("opening hours unknown, not guessed")
            if stop.uncertain_price:
                uncertain.append("price unknown, not guessed")
            suffix = f" — {', '.join(uncertain)}" if uncertain else ""
            st.markdown(
                f"**{format_hm(stop.arrive_min)}–{format_hm(stop.depart_min)} {title}** "
                f"({stop.role.replace('_', ' ')}{suffix})"
            )
            if venue:
                st.caption(
                    f"Travel {stop.travel_min} min by {stop.travel_mode}, about SGD {stop.travel_cost:.2f} "
                    f"(estimate, not a live OneMap route). Crowd {venue.crowd_estimate:.2f} (estimate). "
                    f"Last reviewed {venue.last_updated}."
                )
        st.caption(
            f"Return to the abode: {day.return_travel_min} min by {day.return_mode}, "
            f"about SGD {day.return_cost:.2f} (estimate). Known spend this day SGD {day.known_cost:.2f}."
        )
    st.subheader("Why this, and why not that")
    for explanation in plan.explanations:
        with st.expander(f"{explanation.venue_name} · top factor: {factor_label(explanation.top_factor)} ({explanation.wording})"):
            st.write(explanation.why)
            st.write(explanation.why_not)
            st.write(explanation.reversal)
    with st.expander("Rephrase explanations with the model"):
        st.caption("The model may only reword the factors already computed. A sentence that cites the wrong factor is discarded.")
        if st.button("Rephrase"):
            if not _ai_allowed():
                st.session_state.model_note = (
                    "No model call. Tick the notice and AI consent, and set an API key, or keep the local wording."
                )
            else:
                updated, call = rephrase_explanations(trip, plan.explanations, llm=LLMClient(), ai_consent=True)
                plan.explanations = updated
                st.session_state.plan = plan
                st.session_state.model_note = call.rejection or "Model wording accepted where the top factor still matched."
                st.session_state.last_sent_payload = call.payload
        if st.session_state.model_note:
            st.info(st.session_state.model_note)
        if st.session_state.get("last_sent_payload"):
            st.caption("Exact payload from the last wording call")
            st.json(st.session_state.last_sent_payload)
    st.subheader("Change the plan")
    scheduled = [stop.venue_id for day in plan.route.days for stop in day.stops]
    if not scheduled:
        st.warning("The heuristic did not place any stops. Relax the floor, the hours, or a must-visit, then optimise again.")
        return
    change = st.columns(2)
    with change[0]:
        reject_id = st.selectbox("Reject a stop", scheduled, format_func=lambda item: VENUE_BY_ID[item].name, key="reject_id")
        if st.button("Reject and optimise again"):
            if reject_id not in st.session_state.excluded:
                st.session_state.excluded.append(reject_id)
            st.session_state.musts = [must for must in st.session_state.musts if must.venue_id != reject_id]
            _optimise()
            st.rerun()
    with change[1]:
        swap_from = st.selectbox("Swap this stop", scheduled, format_func=lambda item: VENUE_BY_ID[item].name, key="swap_from")
        options = [venue.id for venue in VENUES if venue.id not in scheduled and venue.id not in st.session_state.excluded]
        swap_to = st.selectbox(
            "With",
            options or [""],
            format_func=lambda item: VENUE_BY_ID[item].name if item in VENUE_BY_ID else "No replacement left",
            key="swap_to",
        )
        if st.button("Swap and optimise again") and swap_to:
            if swap_from not in st.session_state.excluded:
                st.session_state.excluded.append(swap_from)
            st.session_state.musts = [must for must in st.session_state.musts if must.venue_id != swap_from]
            st.session_state.musts.append(MustVisit(swap_to, "FLEXIBLE"))
            _optimise()
            st.rerun()
    st.caption("Move the criterion sliders or the floor on the previous tab and optimise again if you want a different balance.")
    if st.button("Confirm this plan for the session"):
        st.session_state.confirmed = True
    if not st.session_state.confirmed:
        st.info("The plan is a proposal. Confirm it when you want the re-plan and rating controls.")
        return
    st.success("Confirmed in this session only. It is still not written to a database.")
    st.subheader("Re-plan")
    replan = st.columns(3)
    replan[0].selectbox("Weather scenario", ["forecast", "heavy_rain"], key="weather_input")
    replan[1].number_input("Delay on the first morning (minutes)", min_value=0, max_value=240, step=15, key="delay_input")
    if replan[2].button("Re-optimise"):
        _optimise()
        st.rerun()
    st.caption("Heavy rain drops weather-sensitive venues and optimises again. A delay shortens the first day's window.")
    st.subheader("Rate a stop")
    st.caption("A rating updates only the traveller you pick, and only if personalisation consent is on. It is a simple moving average, not reinforcement learning.")
    rate_cols = st.columns(3)
    rate_venue = rate_cols[0].selectbox("Stop", scheduled, format_func=lambda item: VENUE_BY_ID[item].name, key="rate_venue")
    rate_member = rate_cols[1].selectbox("As", [member.id for member in trip.members], format_func=lambda item: next(m.name for m in trip.members if m.id == item), key="rate_member")
    rating = rate_cols[2].select_slider("Rating", options=[1, 2, 3, 4, 5], value=4)
    if st.button("Save rating"):
        member = next(item for item in st.session_state.members if item.id == rate_member)
        venue = VENUE_BY_ID[rate_venue]
        updated, message = apply_member_feedback(
            member,
            venue,
            rating_to_signal(int(rating)),
            personalisation_consent=bool(st.session_state.personal_consent),
        )
        st.session_state.members = [updated if item.id == updated.id else item for item in st.session_state.members]
        if st.session_state.personal_consent:
            for interest, value in updated.preferences.items():
                st.session_state[f"pref_{updated.id}_{interest}"] = value
        st.session_state.feedback.append(message)
    for message in st.session_state.feedback[-4:]:
        st.caption(message)
    with st.expander("What the heuristic does, and what it does not claim"):
        st.write(plan.heuristic_note)
        st.write(
            "S_k(v) = sum of alpha_f times x_f^(k)(v), after the alphas are normalised and any missing "
            "feature (an unknown price) is omitted rather than guessed. U_k is that member's mean score, "
            "clipped to [0, 1]. The search maximises collected score subject to U_k >= beta. "
            "Weighted NSW is prod U_k^w_k. w_k stays at the agreed influence."
        )


def main() -> None:
    _init()
    st.session_state.setdefault("start_date_input", st.session_state.start_date)
    st.session_state.setdefault("end_date_input", st.session_state.end_date)
    st.session_state.setdefault("day_start_input", st.session_state.day_start)
    st.session_state.setdefault("day_end_input", st.session_state.day_end)
    st.session_state.setdefault("budget_input", st.session_state.budget)
    st.session_state.setdefault("modes_input", st.session_state.modes)
    st.session_state.setdefault("abode_label_input", st.session_state.abode_label)
    st.session_state.setdefault("abode_address_input", st.session_state.abode_address)
    st.session_state.setdefault("beta_slider", st.session_state.beta)
    st.session_state.setdefault("weather_input", st.session_state.weather)
    st.session_state.setdefault("delay_input", st.session_state.delay_minutes)
    for feature in FEATURES:
        st.session_state.setdefault(f"alpha_{feature}", float(st.session_state.alphas.get(feature, 0.25)))

    with st.sidebar:
        st.header("Session")
        if st.button("Load sample Singapore trip", type="primary"):
            _load_sample()
            st.rerun()
        if st.button("Clear all my data"):
            _clear()
            st.rerun()
        st.caption("Clearing discards travellers, links, the plan, and ratings. There is no account table.")
        st.write(f"AI consent: {'on' if st.session_state.ai_consent else 'off'}")
        st.write(f"Personalisation: {'on' if st.session_state.personal_consent else 'off'}")
        st.write(f"Plan: {'confirmed' if st.session_state.confirmed else 'not confirmed'}")
    _render_banner()
    input_tab, privacy_tab, optimise_tab, plan_tab = st.tabs(
        ["1. Plan input", "2. De-identify", "3. Filter and optimise", "4. Explain and confirm"]
    )
    with input_tab:
        _render_inputs()
    with privacy_tab:
        _render_privacy()
    with optimise_tab:
        _render_optimise()
    with plan_tab:
        _render_plan()


main()
