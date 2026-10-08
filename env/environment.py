import time
from env import core, node_data, symbol
from env import action as action_module
from env import full_state as full_state_module
from env import reward as reward_module

class Environment:
    def __init__(
        self,
        initial_state: full_state_module.FullState,
        reward_evaluator: reward_module.IRewardEvaluator | None = None,
        max_steps: int | None = None,
    ):
        self._initial_state = initial_state
        self._full_state = initial_state
        self._reward_evaluator = reward_evaluator or reward_module.DefaultRewardEvaluator.create()
        self._max_steps = max_steps
        self._current_step = 0
        self._step_times: list[float] = []

    @property
    def full_state(self) -> full_state_module.FullState:
        return self._full_state

    @property
    def reward_evaluator(self) -> reward_module.IRewardEvaluator | None:
        return self._reward_evaluator

    @property
    def max_steps(self) -> int | None:
        return self._max_steps

    def action_space_size(self):
        return self._full_state.action_space_size()

    def reset(self) -> full_state_module.FullState:
        self._full_state = self._initial_state
        self._current_step = 0
        self._step_times = []
        core.INode.clear_cache()
        return self._full_state

    def step(
        self,
        action: action_module.IAction[full_state_module.FullState],
    ) -> tuple[full_state_module.FullState, float, bool, bool]:
        reward_evaluator = self._reward_evaluator
        current_state = self._full_state
        step_start = time.perf_counter()
        next_state = action.run_action(current_state)
        self._step_times.append(time.perf_counter() - step_start)
        reward = reward_evaluator.evaluate(
            current_state,
            next_state)
        self._current_step += 1
        terminated = next_state.goal_achieved()
        truncated = (
            (self._current_step >= self._max_steps and not terminated)
            if self._max_steps is not None
            else False
        )
        self._full_state = next_state
        return next_state, reward, terminated, truncated

    @property
    def step_times(self) -> tuple[float, ...]:
        return tuple(self._step_times)

    def total_time(self) -> float:
        return sum(self._step_times)

    @property
    def as_symbol(self) -> symbol.Symbol:
        return self.symbol(self.full_state)

    @property
    def current_state_symbol(self) -> symbol.Symbol:
        current_state = self.full_state.current_state.apply()
        return self.symbol(current_state)

    def symbol(self, node: core.BaseNode) -> symbol.Symbol:
        node_types = self.full_state.node_types()
        return symbol.Symbol(node, node_types)

    def to_data(self) -> node_data.NodeData:
        node_types = self.full_state.node_types()
        result = node_data.NodeData(
            node=self.full_state,
            node_types=node_types)
        return result

    def node_data(self, node: core.INode) -> node_data.NodeData:
        node_types = self.full_state.node_types()
        result = node_data.NodeData(
            node=node,
            node_types=node_types)
        return result
