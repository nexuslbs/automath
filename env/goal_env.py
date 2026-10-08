import typing
import functools
from config import settings
from env import meta_env
from env import full_state
from env import reward
from env import action
from env.environment import Environment
from env.env_utils import load_all_subclasses_sorted

T = typing.TypeVar("T")

@functools.cache
def _get_meta(
    goal: meta_env.IGoal,
    allowed_actions: tuple[type[action.IAction], ...] | None,
    max_history_state_size: int | None = None,
    max_steps: int | None = None,
):
    all_types = tuple([t.as_type() for t in load_all_subclasses_sorted()])
    allowed_actions_typed = tuple([
        t.as_type()
        for t in allowed_actions
    ]) if allowed_actions is not None else None
    meta = meta_env.MetaInfo.with_defaults(
        goal=goal,
        all_types=all_types,
        allowed_actions=allowed_actions_typed,
        max_history_state_size=max_history_state_size,
        max_steps=max_steps,
    )
    return meta

class GoalEnv(Environment):

    def __init__(
        self,
        goal: meta_env.IGoal,
        fn_initial_state: typing.Callable[
            [meta_env.MetaInfo],
            full_state.FullState,
        ] | None = None,
        reward_evaluator: reward.IRewardEvaluator | None = None,
        allowed_actions: tuple[type[action.IAction], ...] | None = None,
        max_history_state_size: int | None = None,
        max_steps: int | None = None,
    ):
        if max_history_state_size is None:
            max_history_state_size = settings.DEFAULT_MAX_HISTORY_STATE_SIZE

        meta = _get_meta(
            goal=goal,
            allowed_actions=allowed_actions,
            max_history_state_size=max_history_state_size,
            max_steps=max_steps,
        )

        initial_state = (
            fn_initial_state(meta)
            if fn_initial_state
            else full_state.FullState.with_node(meta))

        super().__init__(
            initial_state=initial_state,
            reward_evaluator=reward_evaluator,
            max_steps=max_steps,
        )

        self._max_history_state_size = max_history_state_size

    @classmethod
    def default_node_types(cls):
        return tuple(load_all_subclasses_sorted())

    @property
    def max_history_state_size(self) -> int | None:
        return self._max_history_state_size
