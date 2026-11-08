from wayfare.fairness import plan_is_better, weighted_nsw_product, PlanUtility, select_floor


def test_nsw_prefers_a_balanced_plan_over_a_majority_plan():
    majority = [0.95, 0.95, 0.95, 0.20]
    balanced = [0.70, 0.70, 0.70, 0.70]
    weights = [1, 1, 1, 1]
    assert sum(majority) > sum(balanced)
    assert weighted_nsw_product(balanced, weights) > weighted_nsw_product(majority, weights)


def test_fairness_floor_vetoes_the_majority_plan():
    majority = PlanUtility([0.95, 0.95, 0.95, 0.20], [1, 1, 1, 1], beta=0.5)
    balanced = PlanUtility([0.70, 0.70, 0.70, 0.70], [1, 1, 1, 1], beta=0.5)
    assert not majority.meets_floor
    assert balanced.meets_floor
    assert plan_is_better(balanced, majority)
    # The veto does not rewrite the agreed weights.
    assert majority.weights == [1, 1, 1, 1]


def test_higher_agreed_weight_tilts_nsw_without_being_rewritten():
    nature = [0.90, 0.45]
    shopping = [0.45, 0.90]
    nature_influence = [3, 1]
    shopping_influence = [1, 3]
    assert weighted_nsw_product(nature, nature_influence) > weighted_nsw_product(shopping, nature_influence)
    assert weighted_nsw_product(shopping, shopping_influence) > weighted_nsw_product(nature, shopping_influence)
    assert nature_influence == [3, 1]


def test_floor_grid_keeps_a_feasible_beta():
    plans = [
        [0.95, 0.95, 0.95, 0.20],
        [0.62, 0.60, 0.58, 0.55],
    ]
    chosen = select_floor(plans, betas=[0.3, 0.5, 0.7])
    assert chosen is not None
    assert chosen["beta"] == 0.5
    assert chosen["min_u"] >= 0.5
