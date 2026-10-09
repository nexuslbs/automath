"""dynamic_env: a GENERIC environment whose node types and axioms are DATA.

The engine (:mod:`dynamic_env.engine`) is a fixed interpreter over a JSON spec
(:mod:`dynamic_env.spec`). Environments ship as spec FILES under
``data/dynamic_env/``; nothing about a concrete environment is hardcoded in the
engine.

Quick use::

    from dynamic_env import DynamicEnv, load_spec
    env = DynamicEnv(load_spec("data/dynamic_env/spec_multi_step.json"))
    state = env.reset()
    for action in env.legal_actions(state):
        ...
"""

from .engine import (
    Action,
    DynamicEnv,
    EvalError,
    IllegalAction,
    State,
    StepResult,
    bfs_minimal_word,
    count_goal_words,
    goal_reached,
    legal_actions,
    minimal_length,
    step,
    try_step,
)
from .spec import (
    SPEC_DIR,
    Spec,
    SpecError,
    list_spec_files,
    load_spec,
    load_spec_by_id,
    spec_from_dict,
)

__all__ = [
    "Action",
    "DynamicEnv",
    "EvalError",
    "IllegalAction",
    "State",
    "StepResult",
    "SPEC_DIR",
    "Spec",
    "SpecError",
    "bfs_minimal_word",
    "count_goal_words",
    "goal_reached",
    "legal_actions",
    "list_spec_files",
    "load_spec",
    "load_spec_by_id",
    "minimal_length",
    "spec_from_dict",
    "step",
    "try_step",
]
