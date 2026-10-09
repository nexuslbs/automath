"""Bounded macro-action layer over the typed primitive action catalogue.

Design unit U3 (fix 2).  The primitive catalogue is large: ``ESSENTIAL_ACTIONS``
(``env/node_types.py``) exposes 22 typed actions and the full basic-action set is
bigger, and every action carries three integer arguments whose meaning depends on
the type (``env/action.py``).  That is the action space a learner would have to
search (see ``docs/REPORT.md`` section 3(ii)).

This module adds a small, deterministic catalogue of MACRO-ACTIONS.  A
macro-action is a named sequence of primitive actions with FIXED arguments.  The
agent selects ONE macro-action as a single decision; the environment expands it
and replays the primitives through the ordinary ``Environment.step`` path, so:

* the reward evaluator (``env/reward.py``) is applied to every primitive
  transition and summed over the macro-action;
* the bounded history (``config/settings.py``
  ``DEFAULT_MAX_HISTORY_STATE_SIZE``) still applies, because nothing bypasses
  ``BaseAction.run_action_details``;
* the agent-facing selection space is exactly the macro-action catalogue.

The catalogue is deliberately small (a handful of macros per goal family), not
22-34 typed actions x 3 integer arguments.

Two ways to obtain a catalogue:

* ``catalogue_for_state(state)`` builds the built-in deterministic default for
  the state's goal family (see ``_DEFAULT_CATALOGUES``);
* ``catalogue_from_json(path)`` loads the SAME shape from a JSON file.  This is
  the injection seam for option C (``docs/REPORT.md`` section 5(C)): a later
  unit may have an LLM propose the macro-action candidate list offline, while
  the RL agent still selects online.  This unit only ships the seam plus the
  built-in default; it builds no LLM pass.

Argument tokens
---------------
A macro step holds three FIXED primitive arguments.  Most are plain ints.
Where a primitive needs the *position* of a type inside a meta index group
(``from_int:<TypeName>``), the step stores the token below and the expansion
resolves it against the concrete environment.  The resolution is deterministic
for a given environment and keeps the catalogue independent of import order.
"""
from __future__ import annotations

import json
import typing
from dataclasses import dataclass, field

from env import core, meta_env
from env.action import RawAction
from env.full_state import FullState
from env.state import IGoal

__all__ = [
    'MacroActionError',
    'MacroStep',
    'MacroAction',
    'MacroActionEnv',
    'catalogue_for_state',
    'catalogue_from_json',
    'default_catalogue',
    'goal_family',
    'FROM_INT_TOKEN_PREFIX',
]

FROM_INT_TOKEN_PREFIX = 'from_int:'

# Goal family names.  A family groups cases whose macro recipe is the same shape.
FAMILY_RESULT = 'result'
FAMILY_SCRATCH = 'scratch'
FAMILY_GENERIC = 'generic'


class MacroActionError(ValueError):
    """A macro-action cannot be parsed, resolved or expanded in this env."""


@dataclass(frozen=True)
class MacroStep:
    """One primitive action with its fixed arguments.

    ``action`` is the primitive action type name as it appears in
    ``env/node_types.py`` / ``env/action_impl.py``.  ``args`` holds three fixed
    ints, or the token string ``"from_int:<TypeName>"`` for an argument that is
    the position of ``<TypeName>`` inside the environment's from-int index group.
    """

    action: str
    args: tuple[typing.Union[int, str], ...] = (0, 0, 0)

    def __post_init__(self) -> None:
        if len(self.args) != 3:
            raise MacroActionError(
                f'step {self.action!r} needs exactly 3 args, got {len(self.args)}')
        for arg in self.args:
            if isinstance(arg, bool) or not isinstance(arg, (int, str)):
                raise MacroActionError(
                    f'step {self.action!r} arg must be int or str, got {arg!r}')
            if isinstance(arg, str) and not arg.startswith(FROM_INT_TOKEN_PREFIX):
                raise MacroActionError(
                    f'step {self.action!r} unknown arg token {arg!r}; '
                    f'only {FROM_INT_TOKEN_PREFIX}<TypeName> is supported')

    def resolved_args(self, env: 'MacroActionEnv') -> tuple[int, int, int]:
        return tuple(
            _from_int_index(env, str(arg)[len(FROM_INT_TOKEN_PREFIX):])
            if isinstance(arg, str) else int(arg)
            for arg in self.args
        )  # type: ignore[return-value]

    def describe(self, env: 'MacroActionEnv') -> str:
        resolved = self.resolved_args(env)
        return f'{self.action}({resolved[0]}, {resolved[1]}, {resolved[2]})'


@dataclass(frozen=True)
class MacroAction:
    """A named, deterministic sequence of primitive actions."""

    name: str
    description: str
    family: str
    steps: tuple[MacroStep, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.name:
            raise MacroActionError('macro-action needs a name')
        if not self.steps:
            raise MacroActionError(f'macro-action {self.name!r} has no steps')

    def expand(self, env: 'MacroActionEnv') -> tuple[RawAction, ...]:
        """Expand to the concrete ``RawAction`` sequence for ``env``."""
        return tuple(
            RawAction.with_raw_args(
                _basic_action_index(env, step.action),
                *step.resolved_args(env),
            )
            for step in self.steps
        )

    def describe(self, env: 'MacroActionEnv') -> str:
        return ' ; '.join(step.describe(env) for step in self.steps)


# --------------------------------------------------------------- resolution --
def _meta(state: FullState) -> meta_env.MetaInfo:
    return state.meta.apply().real(meta_env.MetaInfo)


def _basic_action_index(env: 'MacroActionEnv', action_name: str) -> int:
    """1-based position of ``action_name`` in the allowed basic-action group."""
    allowed = _meta(env.full_state).allowed_basic_actions.apply().real(
        meta_env.GeneralTypeGroup).as_tuple
    for index, type_node in enumerate(allowed, start=1):
        if type_node.type.__name__ == action_name:
            return index
    raise MacroActionError(
        f'primitive action {action_name!r} is not in the allowed basic-action '
        f'catalogue of this environment')


def _from_int_index(env: 'MacroActionEnv', type_name: str) -> int:
    """1-based position of ``type_name`` in the environment from-int group."""
    group = env.full_state.meta.apply().nested_arg((
        meta_env.MetaInfo.idx_from_int_group,
        meta_env.SubtypeOuterGroup.idx_subtypes,
    )).apply().cast(meta_env.GeneralTypeGroup)
    for index, type_node in enumerate(group.as_tuple, start=1):
        if type_node.type.__name__ == type_name:
            return index
    raise MacroActionError(
        f'type {type_name!r} is not in the from-int index group of this '
        f'environment')


def goal_family(state: FullState) -> str:
    """Map the state goal to a macro-action family name."""
    goal = _meta(state).goal.apply().real(IGoal)
    name = type(goal).__name__
    if name == 'HaveResultScratch':
        return FAMILY_RESULT
    if name == 'HaveScratch':
        return FAMILY_SCRATCH
    return FAMILY_GENERIC


# ------------------------------------------------------- built-in catalogue --
_SCRATCH_IDX = 1

# result family: arithmetic / boolean ``HaveResultScratch`` cases.  The recipe is
# the one the repo's own arithmetic test uses (test_suite/arithmetic_test.py
# run_single_eq): create scratch 1, define it from IntBoolean, then verify it.
_RESULT_MACROS: tuple[MacroAction, ...] = (
    MacroAction(
        name='result_true',
        family=FAMILY_RESULT,
        description='Create scratch 1, write IntBoolean(true), verify it',
        steps=(
            MacroStep('CreateScratch', (0, 0, 0)),
            MacroStep('DefineScratchFromInt',
                      (_SCRATCH_IDX, 'from_int:IntBoolean', 1)),
            MacroStep('VerifyGoal',
                      (0, 'from_int:StateScratchIndex', _SCRATCH_IDX)),
        ),
    ),
    MacroAction(
        name='result_false',
        family=FAMILY_RESULT,
        description='Create scratch 1, write IntBoolean(false), verify it',
        steps=(
            MacroStep('CreateScratch', (0, 0, 0)),
            MacroStep('DefineScratchFromInt',
                      (_SCRATCH_IDX, 'from_int:IntBoolean', 0)),
            MacroStep('VerifyGoal',
                      (0, 'from_int:StateScratchIndex', _SCRATCH_IDX)),
        ),
    ),
    MacroAction(
        name='result_write_true',
        family=FAMILY_RESULT,
        description='Create scratch 1 and write IntBoolean(true), no verify',
        steps=(
            MacroStep('CreateScratch', (0, 0, 0)),
            MacroStep('DefineScratchFromInt',
                      (_SCRATCH_IDX, 'from_int:IntBoolean', 1)),
        ),
    ),
    MacroAction(
        name='result_check',
        family=FAMILY_RESULT,
        description='Verify scratch 1 against the goal, no write',
        steps=(
            MacroStep('VerifyGoal',
                      (0, 'from_int:StateScratchIndex', _SCRATCH_IDX)),
        ),
    ),
)

# scratch family: indices / control-flow ``HaveScratch`` cases (goal is Void, so
# the useful macros shape or clear the shared scratch).  Generic states fall back
# to this family.
_SCRATCH_MACROS: tuple[MacroAction, ...] = (
    MacroAction(
        name='scratch_new',
        family=FAMILY_SCRATCH,
        description='Create scratch 1',
        steps=(MacroStep('CreateScratch', (0, 0, 0)),),
    ),
    MacroAction(
        name='scratch_clear',
        family=FAMILY_SCRATCH,
        description='Clear scratch 1',
        steps=(MacroStep('ClearScratch', (_SCRATCH_IDX, 0, 0)),),
    ),
    MacroAction(
        name='scratch_check',
        family=FAMILY_SCRATCH,
        description='Verify scratch 1 against the goal',
        steps=(
            MacroStep('VerifyGoal',
                      (0, 'from_int:StateScratchIndex', _SCRATCH_IDX)),
        ),
    ),
)

_DEFAULT_CATALOGUES: dict[str, tuple[MacroAction, ...]] = {
    FAMILY_RESULT: _RESULT_MACROS,
    FAMILY_SCRATCH: _SCRATCH_MACROS,
    FAMILY_GENERIC: _SCRATCH_MACROS,
}


def default_catalogue() -> tuple[MacroAction, ...]:
    """The full built-in catalogue (all families), deterministic order."""
    return _RESULT_MACROS + _SCRATCH_MACROS


def catalogue_for_state(state: FullState) -> tuple[MacroAction, ...]:
    """The built-in catalogue for ``state``'s goal family."""
    return _DEFAULT_CATALOGUES[goal_family(state)]


# ----------------------------------------------------------- JSON injection --
def catalogue_from_json(path: str) -> tuple[MacroAction, ...]:
    """Load a macro-action catalogue from a JSON file (option C seam).

    Accepted shapes: a bare list of macro objects, or an object with a
    ``"macros"`` list.  Every macro object is::

        {"name": str, "description": str, "family": str,
         "steps": [{"action": str, "args": [int|str, int|str, int|str]}, ...]}

    A malformed file raises ``MacroActionError`` with the offending field.
    """
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
    except OSError as e:
        raise MacroActionError(f'cannot read macro catalogue {path!r}: {e}') from e
    except json.JSONDecodeError as e:
        raise MacroActionError(f'macro catalogue {path!r} is not JSON: {e}') from e

    raw_macros = data.get('macros') if isinstance(data, dict) else data
    if not isinstance(raw_macros, list) or not raw_macros:
        raise MacroActionError(
            f'macro catalogue {path!r} must hold a non-empty list of macros '
            f'(or an object with "macros")')

    macros: list[MacroAction] = []
    seen: set[str] = set()
    for i, item in enumerate(raw_macros):
        if not isinstance(item, dict):
            raise MacroActionError(f'macro #{i} must be an object, got {item!r}')
        name = item.get('name')
        if not isinstance(name, str) or not name:
            raise MacroActionError(f'macro #{i} has no non-empty "name"')
        if name in seen:
            raise MacroActionError(f'duplicate macro name {name!r}')
        seen.add(name)
        raw_steps = item.get('steps')
        if not isinstance(raw_steps, list) or not raw_steps:
            raise MacroActionError(f'macro {name!r} has no non-empty "steps"')
        steps: list[MacroStep] = []
        for j, raw_step in enumerate(raw_steps):
            if not isinstance(raw_step, dict):
                raise MacroActionError(
                    f'macro {name!r} step #{j} must be an object')
            action_name = raw_step.get('action')
            if not isinstance(action_name, str) or not action_name:
                raise MacroActionError(
                    f'macro {name!r} step #{j} has no "action"')
            args = raw_step.get('args', [0, 0, 0])
            if not isinstance(args, list) or len(args) != 3:
                raise MacroActionError(
                    f'macro {name!r} step #{j} "args" must be a list of 3')
            steps.append(MacroStep(action_name, tuple(args)))  # type: ignore[arg-type]
        family = item.get('family', FAMILY_GENERIC)
        if not isinstance(family, str) or not family:
            raise MacroActionError(f'macro {name!r} "family" must be a string')
        macros.append(MacroAction(
            name=name,
            description=str(item.get('description', '')),
            family=family,
            steps=tuple(steps),
        ))
    return tuple(macros)


# ------------------------------------------------------------ environment ----
class MacroActionEnv:
    """Wrap a ``GoalEnv`` so the agent selects macro-actions only.

    It is deliberately a thin adapter, not a second environment: ``step``
    resolves the selected macro-action to its ``RawAction`` sequence and calls
    the wrapped ``GoalEnv.step`` once per primitive, so reward and history stay
    on the single existing path.  The returned reward is the sum over the
    macro-action; ``terminated``/``truncated`` stop the expansion early.
    """

    def __init__(
        self,
        env,
        catalogue: typing.Sequence[MacroAction] | None = None,
    ):
        self._env = env
        self._catalogue: tuple[MacroAction, ...] = (
            tuple(catalogue) if catalogue is not None
            else catalogue_for_state(env.full_state)
        )
        if not self._catalogue:
            raise MacroActionError('macro-action catalogue is empty')
        self._last_macro: MacroAction | None = None
        self._last_trace: list[dict] = []

    # ---- wrapped env surface ----
    @property
    def inner_env(self):
        return self._env

    @property
    def full_state(self) -> FullState:
        return self._env.full_state

    @property
    def max_steps(self):
        return self._env.max_steps

    @property
    def reward_evaluator(self):
        return self._env.reward_evaluator

    @property
    def step_times(self):
        return self._env.step_times

    @property
    def max_history_state_size(self):
        return getattr(self._env, 'max_history_state_size', None)

    def reset(self) -> FullState:
        self._last_macro = None
        self._last_trace = []
        return self._env.reset()

    # ---- macro surface ----
    @property
    def macro_actions(self) -> tuple[MacroAction, ...]:
        return self._catalogue

    @property
    def last_macro(self) -> MacroAction | None:
        return self._last_macro

    @property
    def last_primitive_trace(self) -> list[dict]:
        """Per-primitive records of the last macro step (raw evidence)."""
        return list(self._last_trace)

    def action_space_size(self) -> int:
        """Number of macro-actions the agent chooses between."""
        return len(self._catalogue)

    def resolve(self, macro) -> MacroAction:
        """Accept a MacroAction, a RawAction (1-based index) or an int index."""
        if isinstance(macro, MacroAction):
            return macro
        if isinstance(macro, RawAction):
            index = macro.action_index.apply().real(core.IInt).as_int
        elif isinstance(macro, int) and not isinstance(macro, bool):
            index = macro
        else:
            raise MacroActionError(f'cannot select macro-action from {macro!r}')
        if not 1 <= index <= len(self._catalogue):
            raise MacroActionError(
                f'macro-action index {index} outside 1..{len(self._catalogue)}')
        return self._catalogue[index - 1]

    def expand(self, macro) -> tuple[RawAction, ...]:
        return self.resolve(macro).expand(self)

    def step(self, macro) -> tuple[FullState, float, bool, bool]:
        macro_action = self.resolve(macro)
        raw_actions = macro_action.expand(self)
        total_reward = 0.0
        terminated = False
        truncated = False
        trace: list[dict] = []
        for position, step in enumerate(macro_action.steps, start=1):
            if position > len(raw_actions):
                break
            next_state, reward, terminated, truncated = self._env.step(
                raw_actions[position - 1])
            total_reward += reward
            trace.append({
                'primitive': position,
                'action': step.action,
                'reward': round(reward, 6),
                'ok': next_state.last_action_error() is None,
                'cost': _safe_cost(next_state),
                'goal': bool(next_state.goal_achieved()),
            })
            if terminated or truncated:
                break
        self._last_macro = macro_action
        self._last_trace = trace
        return self._env.full_state, total_reward, terminated, truncated


def _safe_cost(state: FullState) -> int:
    try:
        return state.final_cost().as_int
    except Exception:  # pragma: no cover - defensive, mirrors run_case
        return 0
