"""Wayfare session UI.

The page is a thin shell over the local pipeline. Trip inputs live in
st.session_state and are discarded by "Clear all my data". Nothing is written
to a database. Labels follow the session language; scores do not.
"""

from __future__ import annotations

import os
from dataclasses import replace
from datetime import date, datetime

import streamlit as st

from wayfare.deidentify import PayloadRejected, build_llm_payload, share_links
from wayfare.explain import render_explanation
from wayfare.i18n import DEFAULT_LANG, floor_text, t, venue_caption
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
from wayfare.scoring import preference_match
from wayfare.venues import ABODES, load_venues

st.set_page_config(page_title="Wayfare", page_icon="WF", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: #f6f3ec; }
    html, body, [class*="css"] {
        font-family: "Segoe UI", "PingFang SC", "Noto Sans SC", "Microsoft YaHei", sans-serif;
    }
    h1, h2, h3 {
        font-family: Georgia, "Songti SC", "STSong", "Noto Serif SC", serif;
        letter-spacing: -0.02em;
    }
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
_HARD = ("diet", "accessibility", "hours", "budget", "weather")
_WX = {
    "Partly cloudy": "wx.partly_cloudy",
    "Showers possible": "wx.showers",
    "Warm": "wx.warm",
    "moderate": "risk.moderate",
    "higher": "risk.higher",
    "low": "risk.low",
}


def lang() -> str:
    return st.session_state.get("lang", DEFAULT_LANG)


def tr(key: str, **kwargs: object) -> str:
    return t(key, lang(), **kwargs)


def _named(venue_id: str) -> str:
    venue = VENUE_BY_ID.get(venue_id)
    if venue is None:
        return venue_id
    return venue_caption(venue.name, venue.aliases, lang())


def _init() -> None:
    ss = st.session_state
    if ss.get("ready"):
        return
    kept_lang = ss.get("lang", DEFAULT_LANG)
    ss.ready = True
    ss.lang = kept_lang if kept_lang in ("zh", "en") else DEFAULT_LANG
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
    ss.model_note = "note.sample"
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
    kept = st.session_state.get("lang", DEFAULT_LANG)
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.lang = kept


def _members_from_state() -> list[Member]:
    updated: list[Member] = []
    for member in st.session_state.members:
        _sync_member_controls(member)
        preferences = {interest: float(st.session_state[f"pref_{member.id}_{interest}"]) for interest in INTERESTS}
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
        errors.append(tr("error.members"))
    if trip.end_date < trip.start_date:
        errors.append(tr("error.dates"))
    try:
        if trip.day_end <= trip.day_start:
            errors.append(tr("error.times"))
    except ValueError:
        errors.append(tr("error.time_format"))
    if trip.budget_sgd <= 0:
        errors.append(tr("error.budget"))
    if not trip.transport_modes:
        errors.append(tr("error.modes"))
    if len(trip.dates()) > 4:
        errors.append(tr("error.span"))
    return errors


def _ai_allowed() -> bool:
    return bool(st.session_state.notice and st.session_state.ai_consent)


def _show_note(note: str) -> None:
    if not note:
        return
    st.info(tr(note) if note.startswith("note.") or note.startswith("model.") or note.startswith("link.") else note)


def _constraint_lines(plan, members: list[Member]) -> list[str]:
    lines = []
    for dropped in plan.filtered.dropped:
        reason = next((item for item in dropped.reasons if item in _HARD), None)
        venue = VENUE_BY_ID.get(dropped.venue_id)
        if reason is None or venue is None or not members:
            continue
        mean = sum(preference_match(member, venue) for member in members) / len(members)
        if mean < 0.30:
            continue
        lines.append(
            tr(
                "block.line",
                name=venue_caption(venue.name, venue.aliases, lang()),
                reason=tr(f"reason.{reason}"),
            )
        )
    return lines


def _conflict_lines(plan, trip: TripInput) -> list[str]:
    scheduled = {stop.venue_id for day in plan.route.days for stop in day.stops}
    lines = []
    seen: set[str] = set()
    for must in trip.must_visits:
        if must.venue_id in scheduled or must.venue_id in seen:
            continue
        seen.add(must.venue_id)
        kind = tr(f"kind.{must.kind.upper()}")
        name = _named(must.venue_id)
        dropped = next((item for item in plan.filtered.dropped if item.venue_id == must.venue_id), None)
        if dropped and dropped.reasons:
            reason = tr(f"reason.{dropped.reasons[0]}")
            lines.append(tr("conflict.blocked", name=name, kind=kind, reason=reason))
            continue
        bits = " ".join(part for part in (must.on_date, must.at_time) if part)
        when = tr("conflict.when", when=bits) if bits else ""
        lines.append(tr("conflict.slot", name=name, kind=kind, when=when))
    if plan.route.conflicts and not any(day.stops for day in plan.route.days):
        lines.append(tr("conflict.infeasible"))
    return lines


def _render_banner() -> None:
    st.markdown(
        f"""
        <div class="wayfare-banner">
          <strong>Wayfare</strong>
          <p>{tr("banner.body")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_members() -> None:
    st.subheader(tr("members.title"))
    st.caption(tr("members.help"))
    if not st.session_state.members:
        st.info(tr("members.empty"))
    for index, member in enumerate(list(st.session_state.members)):
        _sync_member_controls(member)
        with st.expander(f"{member.name or tr('members.name')}  ·  {member.id}", expanded=index == 0):
            columns = st.columns([2, 1, 1])
            columns[0].write(tr("members.email", value=member.email or tr("members.none")))
            columns[1].write(tr("members.booking", value=member.booking_ref or tr("members.none")))
            if columns[2].button(tr("members.remove"), key=f"remove_{member.id}"):
                st.session_state.members = [item for item in st.session_state.members if item.id != member.id]
                st.rerun()
            pref_cols = st.columns(4)
            for offset, interest in enumerate(INTERESTS):
                pref_cols[offset % 4].slider(tr(f"interest.{interest}"), 0.0, 1.0, key=f"pref_{member.id}_{interest}")
            meta = st.columns(4)
            meta[0].selectbox(
                tr("members.pace"),
                ["slow", "moderate", "fast"],
                format_func=lambda value: tr(f"pace.{value}"),
                key=f"pace_{member.id}",
            )
            meta[1].slider(tr("members.crowd"), 0.0, 1.0, key=f"crowd_{member.id}", help=tr("members.crowd_help"))
            meta[2].number_input(tr("members.weight"), min_value=0.1, max_value=5.0, step=0.1, key=f"weight_{member.id}")
            meta[3].caption(tr("members.weight_help"))
            st.multiselect(
                tr("members.diet"),
                ["halal", "vegetarian", "vegan", "no_pork", "alcohol_free"],
                format_func=lambda value: tr(f"diet.{value}"),
                key=f"diet_{member.id}",
            )
            st.multiselect(
                tr("members.access"),
                ["wheelchair", "step_free"],
                format_func=lambda value: tr(f"access.{value}"),
                key=f"access_{member.id}",
            )
    with st.expander(tr("members.add")):
        with st.form("add_member"):
            name = st.text_input(tr("members.name"))
            email = st.text_input(tr("members.email_field"))
            booking = st.text_input(tr("members.booking_field"))
            pace = st.selectbox(
                tr("members.pace"),
                ["moderate", "slow", "fast"],
                format_func=lambda value: tr(f"pace.{value}"),
            )
            submitted = st.form_submit_button(tr("members.add_button"))
        if submitted:
            if not name.strip():
                st.error(tr("members.name_required"))
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
    st.subheader(tr("places.title"))
    left, right = st.columns(2)
    with left:
        venue_id = st.selectbox(
            tr("places.place"),
            [venue.id for venue in VENUES],
            format_func=_named,
        )
        kind = st.selectbox(
            tr("places.kind"),
            ["FLEXIBLE", "FIXED"],
            format_func=lambda value: tr(f"kind.{value}"),
        )
        on_date = st.text_input(tr("places.date"), value="")
        at_time = st.text_input(tr("places.time"), value="")
        if st.button(tr("places.add_must")):
            if kind == "FIXED" and at_time and not on_date:
                st.error(tr("places.need_date"))
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
                label = f"{_named(must.venue_id)} · {tr(f'kind.{must.kind.upper()}')}"
                if must.on_date:
                    label += f" · {must.on_date}"
                if must.at_time:
                    label += f" {must.at_time}"
                cols = st.columns([4, 1])
                cols[0].write(label)
                if cols[1].button(tr("places.drop"), key=f"drop_must_{index}"):
                    st.session_state.musts.pop(index)
                    st.rerun()
        else:
            st.caption(tr("places.none"))
    with right:
        pasted = st.text_area(tr("places.paste"), height=110, help=tr("places.paste_help"))
        extra = st.text_area(tr("places.caption"), height=68)
        if st.button(tr("places.save")):
            blob = "\n".join(part.strip() for part in (pasted, extra) if part and part.strip())
            pairs = share_links(blob)
            if pairs:
                for url, caption in pairs:
                    st.session_state.links.append(LinkImport(url, caption))
                st.rerun()
            elif blob:
                st.session_state.links.append(LinkImport(blob, ""))
                st.rerun()
        for index, link in enumerate(st.session_state.links):
            st.write(f"{index + 1}. {link.url or link.caption}")
            if link.caption and link.url:
                st.caption(link.caption)
        if st.button(tr("places.match")):
            trip = _current_trip()
            consent = _ai_allowed()
            if st.session_state.ai_consent and not st.session_state.notice:
                st.warning(tr("places.notice_first"))
            st.session_state.link_result = resolve_links(trip, VENUES, llm=LLMClient(), ai_consent=consent)
        result = st.session_state.link_result
        if result is not None:
            for key in result.note_keys:
                st.caption(tr(key))
            if not result.matches and not result.unmatched and not result.model_dropped:
                st.info(tr("places.empty_match"))
            chosen = []
            for match in result.matches:
                confidence = f"{match.confidence:.2f}"
                flag = tr("places.confirm") if match.needs_confirmation else tr("places.high")
                match_key = f"match_{match.venue_id}_{match.source}"
                st.session_state.setdefault(match_key, not match.needs_confirmation)
                label = f"{_named(match.venue_id)} ({confidence}, {match.source}, {flag})"
                if st.checkbox(label, key=match_key):
                    chosen.append(match.venue_id)
            for text in result.unmatched:
                if text.startswith("Model suggested"):
                    continue
                st.warning(tr("places.dropped", text=text))
            for name in result.model_dropped:
                st.warning(tr("places.model_dropped", name=name))
            if st.button(tr("places.add_matches")) and chosen:
                existing = {must.venue_id for must in st.session_state.musts}
                for venue_id in chosen:
                    if venue_id not in existing:
                        st.session_state.musts.append(MustVisit(venue_id, "FLEXIBLE"))
                st.rerun()


def _render_inputs() -> None:
    st.header(tr("tab.input"))
    st.caption(tr("input.caption"))
    top = st.columns(4)
    top[0].date_input(tr("input.start"), key="start_date_input")
    top[1].date_input(tr("input.end"), key="end_date_input")
    top[2].text_input(tr("input.day_start"), key="day_start_input")
    top[3].text_input(tr("input.day_end"), key="day_end_input")
    second = st.columns(3)
    second[0].number_input(tr("input.budget"), min_value=0.0, step=10.0, key="budget_input")
    second[1].multiselect(
        tr("input.modes"),
        ["walk", "transit", "ride_hail"],
        format_func=lambda value: tr(f"mode.{value}"),
        key="modes_input",
    )
    second[2].selectbox(
        tr("input.abode"),
        [item["label"] for item in ABODES],
        format_func=lambda value: tr(f"area.{value}"),
        key="abode_label_input",
    )
    st.text_input(tr("input.address"), key="abode_address_input", help=tr("input.address_help"))
    _render_members()
    _render_places()


def _render_privacy() -> None:
    st.header(tr("privacy.title"))
    st.markdown(tr("privacy.notice"))
    st.checkbox(tr("privacy.read"), key="notice")
    st.checkbox(tr("privacy.ai"), key="ai_consent")
    st.checkbox(tr("privacy.personal"), key="personal_consent")
    key_on = bool(os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    st.caption(tr("privacy.key_on") if key_on else tr("privacy.key_off"))
    st.subheader(tr("privacy.payload"))
    if not _members_from_state():
        st.info(tr("privacy.need_member"))
        return
    try:
        payload = build_llm_payload(_current_trip(), purpose="extract_places")
    except PayloadRejected:
        st.error(tr("privacy.refused"))
        return
    except ValueError:
        st.error(tr("error.generic"))
        return
    st.json(payload)
    st.caption(tr("privacy.preview_note"))


def _render_filter(plan) -> None:
    kept_col, dropped_col = st.columns(2)
    with kept_col:
        st.markdown(f"**{tr('opt.kept')}**")
        for item in plan.filtered.kept:
            flags = ", ".join(tr(f"flag.{flag}") for flag in item.flags)
            if item.venue.price_per_person_sgd is None:
                price = tr("opt.price_unknown")
            else:
                price = tr("opt.price", amount=item.venue.price_per_person_sgd)
            crowd = tr("opt.crowd", value=item.venue.crowd_estimate)
            st.write(f"{_named(item.venue.id)} · {item.venue.area} · {price} · {crowd}")
            st.caption(tr("opt.source", flags=flags, source=tr("catalogue.source"), updated=item.venue.last_updated))
    with dropped_col:
        st.markdown(f"**{tr('opt.dropped')}**")
        if not plan.filtered.dropped:
            st.caption(tr("opt.none_dropped"))
        for item in plan.filtered.dropped:
            st.write(_named(item.venue_id))
            st.caption("; ".join(tr(f"reason.{reason}") for reason in item.reasons))


def _optimise() -> None:
    trip = _current_trip()
    errors = _validate(trip)
    if errors:
        for error in errors:
            st.error(error)
        return
    try:
        st.session_state.plan = plan_trip(trip, VENUES)
    except ValueError:
        st.error(tr("error.generic"))
        return
    st.session_state.confirmed = False


def _render_optimise() -> None:
    st.header(tr("opt.title"))
    st.caption(tr("opt.help"))
    cols = st.columns(4)
    for index, feature in enumerate(FEATURES):
        st.session_state.setdefault(f"alpha_{feature}", float(st.session_state.alphas.get(feature, 0.25)))
        cols[index].slider(tr(f"feature.{feature}"), 0.0, 1.0, key=f"alpha_{feature}")
    st.slider(tr("opt.beta"), 0.0, 0.9, key="beta_slider", help=tr("opt.beta_help"))
    if st.button(tr("opt.button"), type="primary"):
        _optimise()
    plan = st.session_state.plan
    if plan is None:
        st.info(tr("opt.empty"))
        return
    weights = ", ".join(f"{tr(f'feature.{feature}')} {plan.normalised_alphas[feature]:.2f}" for feature in FEATURES)
    st.write(tr("opt.weights", weights=weights))
    _render_filter(plan)


def _render_plan() -> None:
    st.header(tr("plan.title"))
    plan = st.session_state.plan
    if plan is None:
        st.info(tr("plan.empty"))
        return
    trip = _current_trip()
    metrics = plan.metrics
    cards = st.columns(4)
    cards[0].metric(
        tr("plan.floor"),
        tr("plan.floor_met") if plan.meets_floor else tr("plan.floor_veto"),
        tr("plan.floor_delta", beta=plan.beta),
    )
    cards[1].metric(
        tr("plan.lowest"),
        f"{metrics['lowest_satisfaction']:.2f}",
        tr("plan.gap", gap=metrics["satisfaction_gap"]),
    )
    product = metrics["nsw_product"]
    cards[2].metric(tr("plan.nsw"), f"{product:.3f}" if isinstance(product, float) else "—")
    cards[3].metric(tr("plan.cost"), f"SGD {metrics['known_cost_sgd']:.0f}", tr("plan.budget_delta", budget=metrics["budget_sgd"]))
    st.write(floor_text(trip.members, plan.route.utilities, plan.beta, plan.meets_floor, lang()))
    extra = st.columns(3)
    detour = metrics["detour_ratio"]
    extra[0].write(tr("plan.detour", value=detour) if isinstance(detour, float) else tr("plan.detour_none"))
    extra[1].write(tr("plan.pacing", value=metrics["pacing_variance_hours"]))
    extra[2].write(tr("plan.violations", count=metrics["hard_constraint_violations"]))
    st.caption(tr("plan.detour_help"))
    blocks = _constraint_lines(plan, trip.members)
    if blocks:
        st.subheader(tr("plan.blocks"))
        for note in blocks:
            st.warning(note)
    conflicts = _conflict_lines(plan, trip)
    if conflicts:
        st.subheader(tr("plan.unplaced"))
        for line in conflicts:
            st.error(line)
    st.subheader(tr("plan.satisfaction"))
    for member_id, utility in zip(plan.route.scores, plan.route.utilities):
        member = next((item for item in st.session_state.members if item.id == member_id), None)
        name = member.name if member else member_id
        label = plan.pseudonyms.get(member_id, "pseudonym")
        weight = member.weight if member else ""
        st.write(tr("plan.member_line", name=name, pseudo=label, utility=utility, weight=weight))
    for day in plan.route.days:
        forecast = illustrative_forecast(date.fromisoformat(day.day))
        st.subheader(day.day)
        condition = tr(_WX.get(forecast["condition"], "wx.warm"))
        risk = tr(_WX.get(forecast["rain_risk"], "risk.low"))
        st.caption(
            tr("plan.window", start=format_hm(day.start_min), end=format_hm(day.end_min))
            + " "
            + tr("plan.weather", condition=condition, risk=risk, source=tr("wx.source"), as_of=forecast["as_of"])
        )
        if not day.stops:
            st.write(tr("plan.no_stops"))
        for stop in day.stops:
            venue = VENUE_BY_ID.get(stop.venue_id)
            title = _named(stop.venue_id)
            uncertain = []
            if stop.uncertain_hours:
                uncertain.append(tr("flag.hours_unknown"))
            if stop.uncertain_price:
                uncertain.append(tr("flag.price_unknown"))
            suffix = f" — {', '.join(uncertain)}" if uncertain else ""
            st.markdown(
                f"**{format_hm(stop.arrive_min)}–{format_hm(stop.depart_min)} {title}** "
                f"({tr(f'role.{stop.role}')}{suffix})"
            )
            if venue:
                st.caption(
                    tr(
                        "plan.travel",
                        minutes=stop.travel_min,
                        mode=tr(f"mode.{stop.travel_mode}"),
                        cost=stop.travel_cost,
                        crowd=venue.crowd_estimate,
                        updated=venue.last_updated,
                    )
                )
        st.caption(
            tr(
                "plan.back",
                minutes=day.return_travel_min,
                mode=tr(f"mode.{day.return_mode}"),
                cost=day.return_cost,
                known=day.known_cost,
            )
        )
    st.subheader(tr("plan.why_title"))
    for explanation in plan.explanations:
        why, why_not, reversal = render_explanation(explanation, lang())
        wording = tr("wording.model") if explanation.wording == "model" and explanation.wording_lang == lang() else tr("wording.template")
        with st.expander(tr("plan.expander", name=_named(explanation.venue_id), factor=tr(f"feature.{explanation.top_factor}"), wording=wording)):
            st.write(why)
            st.write(why_not)
            st.write(reversal)
    with st.expander(tr("plan.rephrase")):
        st.caption(tr("plan.rephrase_help"))
        if st.button(tr("plan.rephrase_button")):
            if not _ai_allowed():
                st.session_state.model_note = "plan.no_model"
            else:
                updated, call = rephrase_explanations(
                    trip,
                    plan.explanations,
                    llm=LLMClient(),
                    ai_consent=True,
                    lang=lang(),
                )
                plan.explanations = updated
                st.session_state.plan = plan
                st.session_state.model_note = call.rejection or "plan.wording_ok"
                st.session_state.last_sent_payload = call.payload
        if st.session_state.model_note:
            _show_note(st.session_state.model_note)
        if st.session_state.get("last_sent_payload"):
            st.caption(tr("plan.sent"))
            st.json(st.session_state.last_sent_payload)
    st.subheader(tr("plan.change"))
    scheduled = [stop.venue_id for day in plan.route.days for stop in day.stops]
    if not scheduled:
        st.warning(tr("plan.no_schedule"))
        return
    change = st.columns(2)
    with change[0]:
        reject_id = st.selectbox(tr("plan.reject"), scheduled, format_func=_named, key="reject_id")
        if st.button(tr("plan.reject_button")):
            if reject_id not in st.session_state.excluded:
                st.session_state.excluded.append(reject_id)
            st.session_state.musts = [must for must in st.session_state.musts if must.venue_id != reject_id]
            _optimise()
            st.rerun()
    with change[1]:
        swap_from = st.selectbox(tr("plan.swap"), scheduled, format_func=_named, key="swap_from")
        options = [venue.id for venue in VENUES if venue.id not in scheduled and venue.id not in st.session_state.excluded]
        swap_to = st.selectbox(
            tr("plan.with"),
            options or [""],
            format_func=lambda item: _named(item) if item in VENUE_BY_ID else tr("plan.no_replacement"),
            key="swap_to",
        )
        if st.button(tr("plan.swap_button")) and swap_to:
            if swap_from not in st.session_state.excluded:
                st.session_state.excluded.append(swap_from)
            st.session_state.musts = [must for must in st.session_state.musts if must.venue_id != swap_from]
            st.session_state.musts.append(MustVisit(swap_to, "FLEXIBLE"))
            _optimise()
            st.rerun()
    st.caption(tr("plan.sliders"))
    if st.button(tr("plan.confirm")):
        st.session_state.confirmed = True
    if not st.session_state.confirmed:
        st.info(tr("plan.proposal"))
        return
    st.success(tr("plan.confirmed"))
    st.subheader(tr("plan.replan"))
    replan = st.columns(3)
    replan[0].selectbox(
        tr("plan.weather_scenario"),
        ["forecast", "heavy_rain"],
        format_func=lambda value: tr(f"weather.{value}"),
        key="weather_input",
    )
    replan[1].number_input(tr("plan.delay"), min_value=0, max_value=240, step=15, key="delay_input")
    if replan[2].button(tr("plan.reopt")):
        _optimise()
        st.rerun()
    st.caption(tr("plan.replan_help"))
    st.subheader(tr("plan.rate"))
    st.caption(tr("plan.rate_help"))
    rate_cols = st.columns(3)
    rate_venue = rate_cols[0].selectbox(tr("plan.stop"), scheduled, format_func=_named, key="rate_venue")
    rate_member = rate_cols[1].selectbox(
        tr("plan.as"),
        [member.id for member in trip.members],
        format_func=lambda item: next(member.name for member in trip.members if member.id == item),
        key="rate_member",
    )
    rating = rate_cols[2].select_slider(tr("plan.rating"), options=[1, 2, 3, 4, 5], value=4)
    if st.button(tr("plan.save_rating")):
        member = next(item for item in st.session_state.members if item.id == rate_member)
        venue = VENUE_BY_ID[rate_venue]
        updated, message = apply_member_feedback(
            member,
            venue,
            rating_to_signal(int(rating)),
            personalisation_consent=bool(st.session_state.personal_consent),
            lang=lang(),
        )
        st.session_state.members = [updated if item.id == updated.id else item for item in st.session_state.members]
        if st.session_state.personal_consent:
            for interest, value in updated.preferences.items():
                st.session_state[f"pref_{updated.id}_{interest}"] = value
        st.session_state.feedback.append(message)
    for message in st.session_state.feedback[-4:]:
        st.caption(message)
    with st.expander(tr("plan.heuristic")):
        st.write(tr("heuristic.note"))
        st.write(tr("plan.formula"))


def main() -> None:
    _init()
    st.session_state.setdefault("lang", DEFAULT_LANG)
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
        st.radio(
            tr("lang.label"),
            ["zh", "en"],
            format_func=lambda value: tr(f"lang.{value}"),
            horizontal=True,
            key="lang",
        )
        st.header(tr("nav.session"))
        if st.button(tr("nav.sample"), type="primary"):
            _load_sample()
            st.rerun()
        if st.button(tr("nav.clear")):
            _clear()
            st.rerun()
        st.caption(tr("nav.clear_help"))
        state = tr("status.on") if st.session_state.ai_consent else tr("status.off")
        st.write(tr("status.ai", state=state))
        state = tr("status.on") if st.session_state.personal_consent else tr("status.off")
        st.write(tr("status.personal", state=state))
        state = tr("status.confirmed") if st.session_state.confirmed else tr("status.draft")
        st.write(tr("status.plan", state=state))
    _render_banner()
    input_tab, privacy_tab, optimise_tab, plan_tab = st.tabs(
        [tr("tab.input"), tr("tab.privacy"), tr("tab.optimise"), tr("tab.explain")]
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
