"""Orienteering heuristic with time windows and must-visit anchors.

Select and order activities inside each day's time, then maximise weighted
Nash social welfare subject to the fairness floor. FIXED must-visits with a
clock time stay locked. Other must-visits must appear somewhere in the trip.
Search is greedy insertion followed by bounded local search, with a penalty
on unequal day lengths. This is not an exact optimum.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

from triper.fairness import PlanUtility, plan_is_better
from triper.geo import estimate_leg
from triper.models import Member, MustVisit, TripInput, Venue, format_hm, parse_hm, weekday_key
from triper.money import planning_budget_sgd
from triper.scoring import normalise_satisfaction, score_member_venue

# Local-search breadth is capped so a re-plan stays interactive.
MAX_LOCAL_ITERS = 8
MAX_OPTIONAL_PER_DAY = 3
REPLACEMENT_POOL = 6
# Hours-squared. Large enough to prefer a spare day when two insertions
# have a similar prize, and small enough that a useful stop is still added.
DAY_LENGTH_LAMBDA = 0.6

HEURISTIC_NOTE = (
    "Routing is a heuristic for the team orienteering problem with time windows: "
    "greedy insertion, then bounded local search (relocate, swap, 2-opt, removal, "
    "and replacements from the unused venues with the highest mean preference match). "
    "The prize is the total member-weighted score collected in the time window, "
    "subject to the fairness floor on averaged satisfaction U_k. Maximising the "
    "product of those averages on its own would collapse to a very short plan, "
    "so weighted NSW is the floor and the tie-break, not the insertion prize. "
    "A penalty on unequal day lengths discourages stacking one day, and a back-to-back "
    "same-category penalty keeps the heuristic from scheduling two meals in a row when "
    "another category still fits. Venues with unknown hours or price stay eligible "
    "and are flagged, but the heuristic subtracts a caution from their prize so a "
    "missing value is not treated as a perfect score. "
    "This is not an exact optimum. FIXED must-visits that carry a time stay locked; "
    "FLEXIBLE must-visits must appear somewhere in the trip and may move."
)


@dataclass
class Node:
    venue_id: str
    role: str  # anchor_timed, anchor_must, optional
    fixed_start: int | None = None
    fixed_date: str | None = None


@dataclass
class ScheduledStop:
    venue_id: str
    role: str
    arrive_min: int
    depart_min: int
    dwell_min: int
    travel_min: int
    travel_cost: float
    travel_mode: str
    uncertain_hours: bool
    uncertain_price: bool


@dataclass
class ScheduledDay:
    day: str
    stops: list[ScheduledStop]
    return_travel_min: int
    return_cost: float
    return_mode: str
    start_min: int
    end_min: int
    activity_min: int
    known_cost: float


@dataclass
class Evaluation:
    plan: PlanUtility
    days: list[ScheduledDay]
    scores: dict[str, list[float]]
    total_known_cost: float


@dataclass
class RouteResult:
    days: list[ScheduledDay]
    utilities: list[float]
    weights: list[float]
    plan: PlanUtility
    scores: dict[str, list[float]]
    conflicts: list[str]
    total_known_cost: float
    note: str = HEURISTIC_NOTE


def pace_multiplier(members: list[Member]) -> float:
    """The slowest stated pace sets dwell time, so the plan stays feasible for them."""
    table = {"slow": 1.25, "moderate": 1.0, "fast": 0.8}
    if not members:
        return 1.0
    return max(table.get(member.pace, 1.0) for member in members)


def dwell_minutes(venue: Venue, members: list[Member]) -> int:
    return max(20, int(round(venue.duration_min * pace_multiplier(members))))


def _clone(days: dict[str, list[Node]]) -> dict[str, list[Node]]:
    return {day: [replace(node) for node in nodes] for day, nodes in days.items()}


def _ids(days: dict[str, list[Node]]) -> set[str]:
    return {node.venue_id for nodes in days.values() for node in nodes}


def _delay_minutes(trip: TripInput, day_iso: str, first_iso: str) -> int:
    if not trip.delay_minutes:
        return 0
    if trip.delay_date is None:
        return int(trip.delay_minutes) if day_iso == first_iso else 0
    return int(trip.delay_minutes) if trip.delay_date == day_iso else 0


def _bounds(trip: TripInput, day_iso: str, first_iso: str) -> tuple[int, int] | None:
    start = parse_hm(trip.day_start) + _delay_minutes(trip, day_iso, first_iso)
    end = parse_hm(trip.day_end)
    if start >= end:
        return None
    return start, end


def _place(
    venue: Venue,
    day_iso: str,
    arrive_guess: int,
    dwell: int,
    day_end: int,
    fixed_start: int | None,
) -> tuple[int, int, bool] | None:
    if fixed_start is not None and arrive_guess > fixed_start:
        return None
    uncertain = not venue.hours_known or venue.hours is None
    if uncertain:
        arrive = fixed_start if fixed_start is not None else arrive_guess
        depart = arrive + dwell
        if depart > day_end:
            return None
        return arrive, depart, True
    span = venue.hours.get(weekday_key(date.fromisoformat(day_iso))) if venue.hours else None
    if span is None:
        return None
    open_min, close_min = span
    if fixed_start is not None:
        depart = fixed_start + dwell
        if fixed_start < open_min or depart > close_min or depart > day_end:
            return None
        return fixed_start, depart, False
    arrive = max(arrive_guess, open_min)
    depart = arrive + dwell
    if depart > close_min or depart > day_end:
        return None
    return arrive, depart, False


def simulate(
    days: dict[str, list[Node]],
    trip: TripInput,
    venues: dict[str, Venue],
    members: list[Member],
) -> Evaluation | None:
    party = max(1, len(members))
    ordered_days = list(days)
    if not ordered_days:
        return None
    seen: set[str] = set()
    for nodes in days.values():
        for node in nodes:
            if node.venue_id in seen:
                return None
            seen.add(node.venue_id)
    first = ordered_days[0]
    total_cost = 0.0
    scheduled: list[ScheduledDay] = []
    scores: dict[str, list[float]] = {member.id: [] for member in members}
    collected = 0.0
    budget_sgd = planning_budget_sgd(trip)
    modes = trip.transport_modes or ["walk"]
    for day_iso in ordered_days:
        bounds = _bounds(trip, day_iso, first)
        if bounds is None:
            return None
        start_min, end_min = bounds
        clock = start_min
        lat, lon = trip.abode_lat, trip.abode_lon
        day_stops: list[ScheduledStop] = []
        day_cost = 0.0
        activity = 0
        for node in days[day_iso]:
            if node.fixed_date and node.fixed_date != day_iso:
                return None
            venue = venues.get(node.venue_id)
            if venue is None:
                return None
            remaining = budget_sgd - total_cost - day_cost
            leg = estimate_leg(lat, lon, venue.lat, venue.lon, modes, party, remaining_budget=remaining)
            if leg is None:
                return None
            dwell = dwell_minutes(venue, members)
            placed = _place(venue, day_iso, clock + leg.minutes, dwell, end_min, node.fixed_start)
            if placed is None:
                return None
            arrive, depart, uncertain_hours = placed
            activity_cost = 0.0 if venue.price_per_person_sgd is None else venue.price_per_person_sgd * party
            if leg.cost_sgd + activity_cost > remaining + 1e-6:
                return None
            for member in members:
                score, _, _ = score_member_venue(member, venue, trip.alphas, leg.minutes)
                scores[member.id].append(score)
                collected += float(member.weight) * score
            weight_sum = sum(float(member.weight) for member in members)
            if day_stops and venues[day_stops[-1].venue_id].category == venue.category:
                # Heuristic variety term so two stops of one category are not
                # glued together when a different category still fits.
                collected -= 0.45 * weight_sum
            if not venue.hours_known or venue.price_per_person_sgd is None:
                # Unknown hours or price stay eligible, but they do not get a
                # free ride in the prize. The score itself is not imputed.
                collected -= 0.35 * weight_sum
            day_cost += leg.cost_sgd + activity_cost
            activity += dwell
            day_stops.append(
                ScheduledStop(
                    venue_id=venue.id,
                    role=node.role,
                    arrive_min=arrive,
                    depart_min=depart,
                    dwell_min=dwell,
                    travel_min=leg.minutes,
                    travel_cost=leg.cost_sgd,
                    travel_mode=leg.mode,
                    uncertain_hours=uncertain_hours,
                    uncertain_price=venue.price_per_person_sgd is None,
                )
            )
            clock = depart
            lat, lon = venue.lat, venue.lon
        remaining = budget_sgd - total_cost - day_cost
        back = estimate_leg(lat, lon, trip.abode_lat, trip.abode_lon, modes, party, remaining_budget=remaining)
        if back is None or clock + back.minutes > end_min:
            return None
        if back.cost_sgd - remaining > 1e-6:
            return None
        day_cost += back.cost_sgd
        total_cost += day_cost
        scheduled.append(
            ScheduledDay(
                day=day_iso,
                stops=day_stops,
                return_travel_min=back.minutes,
                return_cost=back.cost_sgd,
                return_mode=back.mode,
                start_min=start_min,
                end_min=end_min,
                activity_min=activity,
                known_cost=round(day_cost, 2),
            )
        )
    utilities = [normalise_satisfaction(scores[member.id]) for member in members]
    weights = [float(member.weight) for member in members]
    penalty = _day_penalty(scheduled)
    plan = PlanUtility(
        utilities=utilities,
        weights=weights,
        beta=trip.beta,
        day_length_penalty=penalty,
        collected=collected,
    )
    return Evaluation(plan=plan, days=scheduled, scores=scores, total_known_cost=round(total_cost, 2))


def _day_penalty(days: list[ScheduledDay]) -> float:
    if len(days) < 2:
        return 0.0
    hours = [day.activity_min / 60.0 for day in days]
    mean = sum(hours) / len(hours)
    variance = sum((hour - mean) ** 2 for hour in hours) / len(hours)
    return DAY_LENGTH_LAMBDA * variance


def _better_eval(candidate: Evaluation | None, incumbent: Evaluation | None) -> bool:
    if candidate is None:
        return False
    if incumbent is None:
        return True
    return plan_is_better(candidate.plan, incumbent.plan)


def _insert(days: dict[str, list[Node]], day: str, index: int, node: Node) -> dict[str, list[Node]]:
    trial = _clone(days)
    trial[day].insert(index, node)
    return trial


def _best_insertion(
    days: dict[str, list[Node]],
    node: Node,
    trip: TripInput,
    venues: dict[str, Venue],
    members: list[Member],
    allowed_days: list[str] | None,
    optional_cap: bool,
) -> tuple[dict[str, list[Node]], Evaluation] | None:
    best_days: dict[str, list[Node]] | None = None
    best_eval: Evaluation | None = None
    day_list = allowed_days or list(days)
    for day in day_list:
        if day not in days:
            continue
        if optional_cap and node.role == "optional":
            optional = sum(1 for item in days[day] if item.role == "optional")
            if optional >= MAX_OPTIONAL_PER_DAY:
                continue
        for index in range(len(days[day]) + 1):
            trial = _insert(days, day, index, node)
            evaluation = simulate(trial, trip, venues, members)
            if _better_eval(evaluation, best_eval):
                best_days, best_eval = trial, evaluation
    if best_days is None or best_eval is None:
        return None
    return best_days, best_eval


def _optional_count(days: dict[str, list[Node]], day: str) -> int:
    return sum(1 for node in days[day] if node.role == "optional")


def _local_search(
    days: dict[str, list[Node]],
    trip: TripInput,
    venues: dict[str, Venue],
    members: list[Member],
    pool: list[Venue],
) -> dict[str, list[Node]]:
    current = days
    for _ in range(MAX_LOCAL_ITERS):
        incumbent = simulate(current, trip, venues, members)
        best_days = None
        best_eval = incumbent
        day_names = list(current)
        for day in day_names:
            for index, node in enumerate(current[day]):
                if node.role != "optional":
                    continue
                trial = _clone(current)
                trial[day].pop(index)
                evaluation = simulate(trial, trip, venues, members)
                if _better_eval(evaluation, best_eval):
                    best_days, best_eval = trial, evaluation
        for src_day in day_names:
            for index, node in enumerate(current[src_day]):
                if node.role == "anchor_timed":
                    continue
                for dst_day in day_names:
                    if node.role == "optional" and src_day != dst_day:
                        if _optional_count(current, dst_day) >= MAX_OPTIONAL_PER_DAY:
                            continue
                    limit = len(current[dst_day]) + 1
                    for dst_index in range(limit):
                        if src_day == dst_day and dst_index in (index, index + 1):
                            continue
                        trial = _clone(current)
                        moved = trial[src_day].pop(index)
                        insert_at = dst_index
                        if src_day == dst_day and insert_at > index:
                            insert_at -= 1
                        insert_at = max(0, min(insert_at, len(trial[dst_day])))
                        trial[dst_day].insert(insert_at, moved)
                        evaluation = simulate(trial, trip, venues, members)
                        if _better_eval(evaluation, best_eval):
                            best_days, best_eval = trial, evaluation
        pairs: list[tuple[str, int]] = []
        for day in day_names:
            for index, node in enumerate(current[day]):
                if node.role != "anchor_timed":
                    pairs.append((day, index))
        for left in range(len(pairs)):
            for right in range(left + 1, len(pairs)):
                day_a, index_a = pairs[left]
                day_b, index_b = pairs[right]
                trial = _clone(current)
                trial[day_a][index_a], trial[day_b][index_b] = trial[day_b][index_b], trial[day_a][index_a]
                evaluation = simulate(trial, trip, venues, members)
                if _better_eval(evaluation, best_eval):
                    best_days, best_eval = trial, evaluation
        for day in day_names:
            sequence = current[day]
            for start in range(len(sequence)):
                for end in range(start + 2, len(sequence) + 1):
                    if any(node.role == "anchor_timed" for node in sequence[start:end]):
                        continue
                    trial = _clone(current)
                    trial[day][start:end] = list(reversed(trial[day][start:end]))
                    evaluation = simulate(trial, trip, venues, members)
                    if _better_eval(evaluation, best_eval):
                        best_days, best_eval = trial, evaluation
        used = _ids(current)
        for venue in pool:
            if venue.id in used:
                continue
            for day in day_names:
                for index, node in enumerate(current[day]):
                    if node.role != "optional":
                        continue
                    trial = _clone(current)
                    trial[day][index] = Node(venue.id, "optional")
                    evaluation = simulate(trial, trip, venues, members)
                    if _better_eval(evaluation, best_eval):
                        best_days, best_eval = trial, evaluation
        if best_days is None:
            break
        current = best_days
    return current


def _mean_preference(venue: Venue, members: list[Member]) -> float:
    from triper.scoring import preference_match

    if not members:
        return 0.0
    return sum(preference_match(member, venue) for member in members) / len(members)


def optimise(
    trip: TripInput,
    kept: list[Venue],
    catalogue: dict[str, Venue],
    dropped_details: dict[str, str],
) -> RouteResult:
    """Build one itinerary. `dropped_details` maps a blocked must-visit id to a group-level reason."""
    venues = {venue.id: venue for venue in kept}
    members = trip.members
    days: dict[str, list[Node]] = {day.isoformat(): [] for day in trip.dates()}
    conflicts: list[str] = []
    if not days:
        raise ValueError("The trip has no dates.")
    if not members:
        raise ValueError("Add at least one traveller before optimising.")

    placed: set[str] = set()
    timed: list[MustVisit] = []
    dated: list[MustVisit] = []
    flexible: list[MustVisit] = []
    for must in trip.must_visits:
        if must.venue_id in placed:
            continue
        placed.add(must.venue_id)
        kind = must.kind.upper()
        if kind == "FIXED" and must.at_time:
            timed.append(must)
        elif kind == "FIXED" and must.on_date:
            dated.append(must)
        else:
            flexible.append(must)

    def place_must(must: MustVisit, node: Node, allowed: list[str] | None) -> None:
        if must.venue_id not in venues:
            name = catalogue[must.venue_id].name if must.venue_id in catalogue else must.venue_id
            reason = dropped_details.get(must.venue_id, "it is not available in the filtered catalogue")
            conflicts.append(
                f"{name} is a {must.kind.upper()} must-visit but was not scheduled: {reason}. "
                "Criterion weights cannot override that."
            )
            return
        found = _best_insertion(days, node, trip, venues, members, allowed, optional_cap=False)
        if found is None:
            name = venues[must.venue_id].name
            when = ""
            if must.on_date or must.at_time:
                when = f" (requested {must.on_date or 'any day'} {must.at_time or ''})".rstrip()
            conflicts.append(
                f"{name} is a {must.kind.upper()} must-visit{when} but no feasible slot fits "
                f"the time window, opening hours, travel, and budget. "
                f"The search did not drop a hard constraint to force it in."
            )
            return
        updated, _ = found
        days.clear()
        days.update(updated)

    for must in sorted(timed, key=lambda item: (item.on_date or "", item.at_time or "")):
        allowed = [must.on_date] if must.on_date else None
        node = Node(must.venue_id, "anchor_timed", parse_hm(must.at_time or "00:00"), must.on_date)
        place_must(must, node, allowed)
    for must in dated:
        node = Node(must.venue_id, "anchor_must", None, must.on_date)
        place_must(must, node, [must.on_date] if must.on_date else None)
    for must in flexible:
        node = Node(must.venue_id, "anchor_must")
        place_must(must, node, None)

    candidates = [venue for venue in kept if venue.id not in _ids(days)]
    while candidates:
        incumbent = simulate(days, trip, venues, members)
        best: tuple[dict[str, list[Node]], Evaluation, str] | None = None
        for venue in candidates:
            found = _best_insertion(
                days,
                Node(venue.id, "optional"),
                trip,
                venues,
                members,
                None,
                optional_cap=True,
            )
            if found is None:
                continue
            trial_days, evaluation = found
            if not _better_eval(evaluation, incumbent):
                continue
            if best is None or _better_eval(evaluation, best[1]):
                best = (trial_days, evaluation, venue.id)
        if best is None:
            break
        days.clear()
        days.update(best[0])
        chosen = best[2]
        candidates = [venue for venue in candidates if venue.id != chosen]

    ranked_pool = sorted(kept, key=lambda venue: _mean_preference(venue, members), reverse=True)[:REPLACEMENT_POOL]
    # Replacements must be allowed to consider venues outside the top preference list
    # only through the pool; anchors are already placed.
    searched = _local_search(days, trip, venues, members, ranked_pool)
    evaluation = simulate(searched, trip, venues, members)
    if evaluation is None:
        conflicts.append("The heuristic could not produce a feasible timetable for these constraints.")
        weights = [float(member.weight) for member in members]
        empty = PlanUtility(utilities=[0.0 for _ in members], weights=weights, beta=trip.beta)
        return RouteResult(
            days=[],
            utilities=empty.utilities,
            weights=weights,
            plan=empty,
            scores={member.id: [] for member in members},
            conflicts=conflicts,
            total_known_cost=0.0,
        )
    return RouteResult(
        days=evaluation.days,
        utilities=evaluation.plan.utilities,
        weights=list(evaluation.plan.weights),
        plan=evaluation.plan,
        scores=evaluation.scores,
        conflicts=conflicts,
        total_known_cost=evaluation.total_known_cost,
    )


def travel_detour_ratio(trip: TripInput, days: list[ScheduledDay], venues: dict[str, Venue]) -> float | None:
    """Planned travel time divided by a travel-only nearest-neighbour order of the same stops.

    This is an estimate of extra travel, not a proof of optimality. Values below 1
    can appear when the timetable was forced onto a slower mode by the budget.
    """
    party = max(1, len(trip.members))
    ratios: list[float] = []
    for day in days:
        if not day.stops:
            continue
        planned = sum(stop.travel_min for stop in day.stops) + day.return_travel_min
        pending = [venues[stop.venue_id] for stop in day.stops if stop.venue_id in venues]
        lat, lon = trip.abode_lat, trip.abode_lon
        nearest = 0
        while pending:
            ranked: list[tuple[int, Venue]] = []
            for venue in pending:
                leg = estimate_leg(
                    lat, lon, venue.lat, venue.lon, trip.transport_modes, party, remaining_budget=None
                )
                ranked.append((leg.minutes if leg else 10**6, venue))
            minutes, chosen = min(ranked, key=lambda item: (item[0], item[1].id))
            nearest += minutes
            lat, lon = chosen.lat, chosen.lon
            pending = [venue for venue in pending if venue.id != chosen.id]
        back = estimate_leg(lat, lon, trip.abode_lat, trip.abode_lon, trip.transport_modes, party, None)
        nearest += back.minutes if back else 0
        if nearest > 0:
            ratios.append(planned / nearest)
    if not ratios:
        return None
    return sum(ratios) / len(ratios)


def pacing_variance_hours(days: list[ScheduledDay]) -> float:
    if not days:
        return 0.0
    hours = [day.activity_min / 60.0 for day in days]
    mean = sum(hours) / len(hours)
    return sum((hour - mean) ** 2 for hour in hours) / len(hours)


def describe_stop(stop: ScheduledStop) -> str:
    return (
        f"{format_hm(stop.arrive_min)}–{format_hm(stop.depart_min)} "
        f"after {stop.travel_min} min {stop.travel_mode} (estimate)"
    )
