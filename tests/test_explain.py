from wayfare.explain import apply_model_wording, explain_pair, smallest_weight_reversal
from wayfare.scoring import factor_label, score_features, top_factor


def test_explanation_cites_the_actual_top_factor_and_a_real_reversal():
    alphas = {"preference_match": 0.25, "travel_time": 0.25, "cost": 0.25, "crowd": 0.25}
    chosen = {
        "preference_match": 0.95,
        "travel_time": 0.70,
        "cost": 0.70,
        "crowd": 0.70,
    }
    alternative = {
        "preference_match": 0.20,
        "travel_time": 0.95,
        "cost": 0.95,
        "crowd": 0.75,
    }
    chosen_score, contributions, _ = score_features(chosen, alphas)
    alternative_score, _, _ = score_features(alternative, alphas)
    assert chosen_score > alternative_score
    leading = top_factor(contributions)
    explanation = explain_pair(
        "gardens",
        "Gardens by the Bay",
        [chosen],
        [1.0],
        alphas,
        "ion",
        "ION Orchard",
        [alternative],
    )
    assert explanation.top_factor == leading == "preference_match"
    assert explanation.top_factor == max(explanation.contributions, key=explanation.contributions.get)
    assert factor_label(explanation.top_factor) in explanation.why
    assert explanation.gap_factor == "preference_match"
    assert "preference match" in explanation.why_not
    flip = smallest_weight_reversal([chosen], [alternative], [1.0], alphas)
    assert flip is not None
    feature, _old, new = flip
    trial = dict(alphas)
    trial[feature] = new
    revised_chosen, _, _ = score_features(chosen, trial)
    revised_alternative, _, _ = score_features(alternative, trial)
    assert revised_alternative > revised_chosen
    assert factor_label(feature) in explanation.reversal


def test_hard_constraint_explanation_does_not_name_a_person():
    features = {"preference_match": 0.8, "travel_time": 0.5, "cost": 0.5, "crowd": 0.5}
    explanation = explain_pair(
        "zam",
        "Zam Zam Restaurant",
        [features],
        [1.0],
        {"preference_match": 1, "travel_time": 0, "cost": 0, "crowd": 0},
        alternative_id="song-fa",
        alternative_name="Song Fa Bak Kut Teh",
        blocked_reason="it does not meet the group's dietary requirement",
    )
    assert "group's dietary requirement" in explanation.why_not
    assert "never entered the scored set" in explanation.reversal
    assert "Priya" not in explanation.why_not
    assert "limited mobility" not in explanation.why_not
    assert "preference match" in explanation.why


def test_unfaithful_model_wording_is_discarded():
    features = {"preference_match": 0.9, "travel_time": 0.2, "cost": 0.2, "crowd": 0.2}
    explanation = explain_pair(
        "park",
        "Park",
        [features],
        [1],
        {"preference_match": 1, "travel_time": 0, "cost": 0, "crowd": 0},
    )
    original = explanation.why
    updated, notes = apply_model_wording(
        [explanation],
        [
            {
                "venue_id": "park",
                "why": "Park was chosen because of cost.",
                "why_not": "No alternative.",
                "reversal": "None.",
                "top_factor": "cost",
            }
        ],
    )
    assert notes == ["park"]
    assert updated[0].wording == "template"
    assert updated[0].why == original
    faithful, notes = apply_model_wording(
        [explanation],
        [
            {
                "venue_id": "park",
                "why": "Park was chosen because the largest contribution is preference match.",
                "why_not": "No alternative was left to compare.",
                "reversal": "There is no alternative venue to reverse against.",
                "top_factor": "preference_match",
            }
        ],
    )
    assert notes == []
    assert faithful[0].wording == "model"
    assert "preference match" in faithful[0].why
