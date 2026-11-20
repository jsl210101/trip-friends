"""Print a sample plan. Useful without Streamlit: python -m triper.demo."""

from __future__ import annotations

from triper.models import format_hm
from triper.money import exchange_suggestion, format_money, format_rate
from triper.pipeline import plan_trip
from triper.sample import sample_trip
from triper.venues import load_venues


def main() -> None:
    trip = sample_trip()
    result = plan_trip(trip)
    names = {venue.id: venue.name for venue in load_venues()}
    print("Triper sample plan")
    if trip.links:
        print(f"Sample import: {trip.links[0].url} (Xiaohongshu short link, not opened)")
    print(f"Floor met: {result.meets_floor}")
    print(result.floor_message)
    for member, utility in zip(trip.members, result.route.utilities):
        print(f"  {member.name}: U={utility:.3f}  weight={member.weight}")
    print(f"NSW product: {result.metrics['nsw_product']}")
    hint = exchange_suggestion(trip.budget_sgd, trip.budget_currency, allow_network=False)
    print(
        f"Budget: {format_money(hint.entered_amount, hint.entered_currency)} "
        f"≈ {format_money(hint.converted_amount, hint.converted_currency)} "
        f"({hint.source}, 1 {hint.rate_base} = {format_rate(hint.rate_value)} {hint.rate_quote}, {hint.as_of})"
    )
    print(f"Known cost: {result.metrics['known_cost_sgd']} / {result.metrics['budget_sgd']} SGD")
    print(f"Hard-constraint violations: {result.metrics['hard_constraint_violations']}")
    for day in result.route.days:
        print(f"\n{day.day}  {format_hm(day.start_min)}–{format_hm(day.end_min)}")
        for stop in day.stops:
            flags = []
            if stop.uncertain_hours:
                flags.append("hours uncertain")
            if stop.uncertain_price:
                flags.append("price uncertain")
            suffix = f" [{', '.join(flags)}]" if flags else ""
            print(
                f"  {format_hm(stop.arrive_min)} {names.get(stop.venue_id, stop.venue_id)} "
                f"({stop.role}, {stop.travel_min} min {stop.travel_mode}){suffix}"
            )
        print(f"  return {day.return_travel_min} min {day.return_mode}")
    if result.constraint_notes:
        print("\nHard constraints:")
        for line in result.constraint_notes:
            print(f"  - {line}")
    if result.route.conflicts:
        print("\nConflicts:")
        for line in result.route.conflicts:
            print(f"  - {line}")
    print("\nExplanations:")
    for item in result.explanations:
        print(f"- {item.venue_name} [{item.top_factor}]")
        print(f"  {item.why}")
        print(f"  {item.why_not}")
        print(f"  {item.reversal}")


if __name__ == "__main__":
    main()
