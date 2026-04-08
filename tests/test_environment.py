from inference import choose_action
from kitchen_ops_env.models import KitchenAction
from kitchen_ops_env.scenario_generator import TASK_IDS
from kitchen_ops_env.server.kitchen_environment import KitchenOpsEnvironment


def play_first_available(env: KitchenOpsEnvironment) -> float:
    obs = env.reset(task_id="dinner_rush_stockout")

    while not obs.done:
        payload = {
            key: value
            for key, value in obs.available_actions[0].items()
            if key in {"action_type", "order_id", "component_id", "ingredient_id", "quantity", "source_id", "notes"}
        }
        obs = env.step(KitchenAction(**payload))

    return env.grade_episode()


def test_easy_path_scores_high() -> None:
    env = KitchenOpsEnvironment()
    obs = env.reset(task_id="breakfast_omelette")
    assert obs.scenario_id == "breakfast_omelette"
    assert "Open tickets:" in obs.briefing
    assert "Last move:" not in obs.briefing
    assert len(obs.available_actions) >= 1

    env.step(
        KitchenAction(action_type="PREP_COMPONENT", order_id="ord_easy_1", component_id="omelette_base")
    )
    env.step(
        KitchenAction(action_type="PREP_COMPONENT", order_id="ord_easy_1", component_id="omelette_veg_mix")
    )
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

    substitutions = env.state.orders["ord_hard_1"]["substitutions"]
    assert substitutions["Paneer"] == "Tofu"
    assert "Current swaps: Paneer -> Tofu." in obs.briefing
    assert 0.0 <= env.grade_episode() <= 1.0


def test_idle_play_scores_low() -> None:
    env = KitchenOpsEnvironment()
    obs = env.reset(task_id="dinner_rush_stockout")

    while not obs.done:
        obs = env.step(KitchenAction(action_type="CHECK_PROGRESS"))

    assert env.grade_episode() <= 0.05


def test_first_available_policy_is_not_enough() -> None:
    env = KitchenOpsEnvironment()
    score = play_first_available(env)

    assert score < 0.9


def test_generated_task_count() -> None:
    assert len(TASK_IDS) >= 6


def test_state_is_sanitized() -> None:
    env = KitchenOpsEnvironment()
    env.reset(task_id="double_shortage_service")
    state = env.state

    tomato = state.inventory["tomato"]
    assert "source_note" not in tomato
    assert "restock_pack" not in tomato
    assert "restock_cost" not in tomato
    assert set(tomato).issubset({"name", "quantity", "unit", "running_low", "unit_cost"})


def test_deterministic_baseline_clears_generated_tasks() -> None:
    for task_id in TASK_IDS:
        env = KitchenOpsEnvironment()
        obs = env.reset(task_id=task_id)
        history: list[str] = []
        step = 0

        while not obs.done:
            step += 1
            action = choose_action(step, obs, history)
            obs = env.step(action)
            history.append(
                f"step={step} action={action.action_type} reward={obs.reward:.2f} task={task_id}"
            )

        assert env.grade_episode() >= 0.7
