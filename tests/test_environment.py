from kitchen_ops_env.models import KitchenAction
from kitchen_ops_env.server.kitchen_environment import KitchenOpsEnvironment


def test_easy_path_scores_high() -> None:
    env = KitchenOpsEnvironment()
    obs = env.reset(task_id="breakfast_omelette")
    assert obs.scenario_id == "breakfast_omelette"

    env.step(KitchenAction(action_type="PREP_COMPONENT", order_id="ord_easy_1", component_id="omelette_base"))
    env.step(KitchenAction(action_type="PREP_COMPONENT", order_id="ord_easy_1", component_id="omelette_veg_mix"))
    env.step(KitchenAction(action_type="COOK_DISH", order_id="ord_easy_1"))
    env.step(KitchenAction(action_type="ASSEMBLE_DISH", order_id="ord_easy_1"))
    final = env.step(KitchenAction(action_type="SERVE_ORDER", order_id="ord_easy_1"))

    assert final.done is True
    assert env.grade_episode() >= 0.95


def test_substitution_changes_order_state() -> None:
    env = KitchenOpsEnvironment()
    env.reset(task_id="dinner_rush_stockout")

    obs = env.step(
        KitchenAction(
            action_type="SUBSTITUTE_INGREDIENT",
            order_id="ord_hard_1",
            ingredient_id="paneer",
            source_id="tofu",
        )
    )

    substitutions = next(
        order["substitutions"]
        for order in obs.service_board
        if order["order_id"] == "ord_hard_1"
    )
    assert substitutions["paneer"]["ingredient_id"] == "tofu"
    assert 0.0 <= env.grade_episode() <= 1.0

