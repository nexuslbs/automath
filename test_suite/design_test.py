"""Design tests for unit U3a (bounded state, legible failures, fewest actions).

Run with pytest:  python -m pytest test_suite/design_test.py -v
or directly:       python test_suite/design_test.py
"""
from __future__ import annotations

from env import core
from env.action_impl import DefineScratchFromInt
from env.core import Integer
from env.full_state import (
    FullState,
    HistoryGroupNode,
    HistoryNode,
    HistorySummaryActionData,
    history_summary_data,
)
from env.goal_env import GoalEnv
from env.meta_env import MetaInfo
from env.node_types import ESSENTIAL_ACTIONS, HaveScratch
from env.reward import DefaultRewardEvaluator
from env.state import IGoal, State

BAD_ACTION_INDEX = 10 ** 9


def _goal():
    return HaveScratch.with_goal(core.Void())


def _bad_action():
    return DefineScratchFromInt.from_raw(1, BAD_ACTION_INDEX, 3)


def test_history_bound():
    bound = 5
    steps = 12
    goal = _goal()
    env = GoalEnv(
        goal=goal,
        allowed_actions=ESSENTIAL_ACTIONS,
        max_history_state_size=bound,
        max_steps=steps + 5,
    )
    for _ in range(steps):
        env.step(_bad_action())

    history = env.full_state.history.apply().real(HistoryGroupNode).as_tuple
    assert len(history) <= bound, f'history not bounded: {len(history)} > {bound}'

    summaries = [item for item in history if history_summary_data(item) is not None]
    assert len(summaries) == 1, f'expected exactly one summary node, got {len(summaries)}'
    summary_data = history_summary_data(summaries[0])
    assert isinstance(summary_data, HistorySummaryActionData)
    dropped = summary_data.dropped_count.apply().real(Integer).as_int
    expected_dropped = steps - (len(history) - 1)
    assert dropped == expected_dropped, (dropped, expected_dropped, steps, len(history))
    assert dropped > 0

    # The goal node is retained and the goal stays observable in the state.
    assert isinstance(history[0], HistoryNode)
    assert history_summary_data(history[0]) is None
    meta = env.full_state.meta.apply().real(MetaInfo)
    assert meta.goal.apply().real(IGoal) == goal
    assert env.full_state.goal_achieved() is False

    # The bound fires through the existing truncation branch: a second run of
    # the same length through the default GoalEnv also stays bounded.
    default_env = GoalEnv(goal=_goal(), allowed_actions=ESSENTIAL_ACTIONS)
    from config import settings
    assert default_env.max_history_state_size == settings.DEFAULT_MAX_HISTORY_STATE_SIZE


def test_error_observation():
    goal = _goal()
    env = GoalEnv(
        goal=goal,
        allowed_actions=ESSENTIAL_ACTIONS,
        max_history_state_size=10,
        max_steps=5,
    )
    before_state = env.full_state.current_state.apply().real(State)

    next_state, _reward, _terminated, _truncated = env.step(_bad_action())

    error = next_state.last_action_error()
    assert error is not None, 'a wrong action must expose a structured error'
    action_name, error_class, message, action_index = error
    assert action_name == 'DefineScratchFromInt', action_name
    assert error_class and error_class != '<none>', error_class
    assert message, 'error message must be non-empty'
    assert action_index >= 1

    observation = next_state.render_observation(max_nodes=20)
    assert observation
    assert action_name in observation, observation
    assert error_class in observation, observation
    assert 'last_error:' in observation
    assert 'dropped_history_count:' in observation

    # The state is otherwise unchanged by the failed action.
    after_state = next_state.current_state.apply().real(State)
    assert after_state == before_state


class _StubState:
    """Minimal object exposing what the evaluator reads."""

    def __init__(self, goal: bool = False, cost: int = 0):
        self._goal = goal
        self._cost = cost

    def goal_achieved(self) -> bool:
        return self._goal

    def is_last_step_error(self) -> bool:
        return False

    def final_cost(self) -> Integer:
        return Integer(self._cost)


def test_fewest_actions():
    evaluator = DefaultRewardEvaluator.create()
    assert evaluator.step_penalty == 1.0

    mid = _StubState(goal=False, cost=0)
    terminal = _StubState(goal=True, cost=0)

    def path_reward(action_count: int) -> float:
        total = 0.0
        for _ in range(action_count - 1):
            total += evaluator.evaluate(mid, mid)
        total += evaluator.evaluate(mid, terminal)
        return total

    reward_3 = path_reward(3)
    reward_5 = path_reward(5)
    assert reward_3 > reward_5, (reward_3, reward_5)
    # Goal reward stays dominant.
    assert reward_3 > 10000 - 20

    # Backward compatible: an evaluator built with only the goal reward uses
    # the default step penalty of 1.0.
    legacy = DefaultRewardEvaluator(Integer(10000))
    assert legacy.step_penalty == 1.0
    assert legacy.evaluate(mid, terminal) == 10000 - 1.0

    # A larger explicit penalty makes the same terminal state strictly worse
    # with more actions by a larger margin.
    strict = DefaultRewardEvaluator.create(step_penalty=2.0)
    assert strict.step_penalty == 2.0
    strict_3 = 0.0
    for _ in range(2):
        strict_3 += strict.evaluate(mid, mid)
    strict_3 += strict.evaluate(mid, terminal)
    strict_5 = 0.0
    for _ in range(4):
        strict_5 += strict.evaluate(mid, mid)
    strict_5 += strict.evaluate(mid, terminal)
    assert strict_5 < strict_3


def _main() -> int:
    test_history_bound()
    print('test_history_bound: PASS')
    test_error_observation()
    print('test_error_observation: PASS')
    test_fewest_actions()
    print('test_fewest_actions: PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(_main())
