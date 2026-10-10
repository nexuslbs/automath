"""Unit tests for the real-engine pop action set (task 4349, FIX 1, unit 2).

Run::

    /opt/automath/venv/bin/python -m evolution_trainer.solvability_tests

Every assertion drives the REAL ``dynamic_env`` engine. The central case is an
"unseen40-type perturbed form": a start state carrying an off-canonical work
node that makes the goal guard false. It is NOT solvable under the ``current``
action set (no action removes a node) and IS solvable under ``pop`` (discard the
offending work node, then replay the canonical word). No training, stdlib only.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from dynamic_env.engine import (
    Action,
    DynamicEnv,
    bfs_minimal_word,
    work_stack,
)
from dynamic_env.spec import Spec, spec_from_dict

from .size_selection import canonical_subtree_prune
from .solvability import SOLVABLE, classify

PERTURBED_SPEC: dict = {
    "spec_id": "perturbed_form_unit2",
    "description": "unseen40-type perturbed form: an off-canonical work node blocks the guard",
    "node_types": {
        "cell": {"semantics": "literal", "role": "value", "value_field": "x",
                 "fields": [{"name": "x", "type": "int"}]},
        "mark": {"semantics": "literal", "role": "tag", "value_field": "m",
                 "fields": [{"name": "m", "type": "str"}]},
        "box": {"semantics": "combination", "role": "plain",
                "fields": [{"name": "tag", "type": "node"},
                           {"name": "items", "type": "nodes"}]},
        "goalbit": {"semantics": "literal", "role": "objective",
                    "value_field": "v", "fields": [{"name": "v", "type": "int"}]},
    },
    "combine_axioms": [
        {"id": "ax_succ", "tag": "succ", "arity": 1, "op": "succ", "active": True}
    ],
    "build_axioms": [
        {"id": "build_succ", "tag_node": "m_succ", "arity": 1,
         "node_type": "box", "operand_type": None, "active": True}
    ],
    "dynamic_axioms": [],
    "nodes": {
        "c2": {"type": "cell", "x": 2},
        "m_succ": {"type": "mark", "m": "succ"},
        "g": {"type": "goalbit", "v": 0},
    },
    "objectives": {"g": 1},
    "guards": {"g": {"op": "and", "args": [
        {"op": "exists_value", "cmp": "eq", "value": 3, "combination_only": True},
        {"op": "count_nodes", "cmp": "eq", "value": 4},
    ]}},
    "dynamic_node_type": None,
    "max_combine_arity": 0,
    "step_cost": 0.01,
    "goal_reward": 1.0,
    "max_steps": 5,
}


class CheckFailure(Exception):
    def __init__(self, name: str, detail: str) -> None:
        super().__init__(detail)
        self.name = name
        self.detail = detail


@dataclass
class _Case:
    name: str
    spec: Spec
    start_state: Any


def _perturbed_spec() -> Spec:
    return spec_from_dict(PERTURBED_SPEC)


def _perturbed_start(spec: Spec):
    """Build the canonical work node, then an extra off-canonical work node."""
    env = DynamicEnv(spec, allow_pop=True)
    state = env.reset()
    word = bfs_minimal_word(spec)
    if word is None:
        raise CheckFailure("perturbed_form", "canonical word is None")
    # canonical first build -> box(value 3), count 4
    first = next(a for a in env.legal_actions(state) if a.key() == word[0])
    state = env.step(state, first).state
    # extra off-canonical build -> box(value 4), count 5
    extra = [a for a in env.legal_actions(state) if a.kind in ("build", "combine")]
    if not extra:
        raise CheckFailure("perturbed_form", "no build available for the perturbation")
    state = env.step(state, extra[0]).state
    if len(state.nodes) != 5:
        raise CheckFailure("perturbed_form", "expected 5 nodes, got %d" % len(state.nodes))
    return state, word


def check_perturbed_needs_pop() -> str:
    spec = _perturbed_spec()
    start, _word = _perturbed_start(spec)
    case = _Case(name="perturbed_form", spec=spec, start_state=start)
    without = classify(case, "current", node_budget=20000, time_budget=5.0)
    with_pop = classify(case, "pop", node_budget=20000, time_budget=5.0)
    if without["verdict"] == SOLVABLE:
        raise CheckFailure("perturbed_needs_pop",
                           "current unexpectedly SOLVABLE: %r" % without["witness_path"])
    if with_pop["verdict"] != SOLVABLE:
        raise CheckFailure("perturbed_needs_pop",
                           "pop did NOT solve the form: %r" % with_pop)
    return ("perturbed form: current=%s (no witness), pop=SOLVABLE in %d steps"
            % (without["verdict"], with_pop["min_steps"]))


def check_pop_replay_witness_real_engine() -> str:
    spec = _perturbed_spec()
    start, word = _perturbed_start(spec)
    env = DynamicEnv(spec, allow_pop=True)
    witness = ["pop"] * len(work_stack(spec, start)) + list(word)
    state = start
    for key in witness:
        action = next((a for a in env.legal_actions(state) if a.key() == key), None)
        if action is None:
            raise CheckFailure("pop_replay_witness",
                               "witness action %r is not legal in %s" % (key, state.identity()))
        state = env.step(state, action).state
    if not env.goal_reached(state):
        raise CheckFailure("pop_replay_witness",
                           "[pop]*len(work)+canonical word did not reach the goal")
    return ("[pop]*%d + canonical word reached the goal in the REAL env (%d steps)"
            % (len(witness) - len(word), len(witness)))


def check_shipped_spec_with_pop() -> str:
    from .size_selection import spec_by_id

    details: List[str] = []
    for spec_id in ("spec_multi_step", "spec_dynamic_group", "spec_dynamic_axiom"):
        spec = spec_by_id(spec_id)
        env = DynamicEnv(spec, allow_pop=True)
        word = bfs_minimal_word(spec)
        if word is None:
            raise CheckFailure("shipped_spec_with_pop", "%s has no canonical word" % spec_id)
        start = env.reset()
        if word:
            first = next(a for a in env.legal_actions(start) if a.key() == word[0])
            if first.kind in ("build", "combine"):
                start = env.step(start, first).state
        case = _Case(name=spec_id, spec=spec, start_state=start)
        result = classify(case, "pop", node_budget=20000, time_budget=5.0)
        if result["verdict"] != SOLVABLE:
            raise CheckFailure("shipped_spec_with_pop",
                               "%s: pop classify = %r" % (spec_id, result))
        details.append("%s=%d" % (spec_id, result["min_steps"]))
    return "real-engine pop classify SOLVABLE on shipped specs: " + ", ".join(details)


def check_prune_disabled_with_pop() -> str:
    spec = _perturbed_spec()
    start, _word = _perturbed_start(spec)
    current_env = DynamicEnv(spec, allow_pop=False)
    pop_env = DynamicEnv(spec, allow_pop=True)
    current_actions = current_env.legal_actions(start)
    pop_actions = pop_env.legal_actions(start)
    pruned_current = canonical_subtree_prune(current_env, start, current_actions)
    pruned_pop = canonical_subtree_prune(pop_env, start, pop_actions)
    if len(pruned_pop) != len(pop_actions):
        raise CheckFailure("prune_disabled_with_pop",
                           "canonical prune deleted pop-legal actions (%d -> %d)"
                           % (len(pop_actions), len(pruned_pop)))
    if not any(a.kind == "pop" for a in pruned_pop):
        raise CheckFailure("prune_disabled_with_pop", "pop action was pruned")
    return ("canonical_subtree_prune kept all %d pop actions (deleted none); "
            "current prune kept %d/%d" % (len(pruned_pop), len(pruned_current),
                                          len(current_actions)))


def check_default_engine_unchanged() -> str:
    """The current action set is unchanged and never contains pop."""
    from .size_selection import spec_by_id

    for spec_id in ("spec_minimal", "spec_multi_step", "spec_dynamic_group",
                    "spec_dynamic_axiom"):
        spec = spec_by_id(spec_id)
        state = DynamicEnv(spec).reset()
        actions = DynamicEnv(spec).legal_actions(state)
        if any(a.kind == "pop" for a in actions):
            raise CheckFailure("default_engine_unchanged", "%s leaked pop" % spec_id)
    return "all shipped specs: allow_pop=False legal set contains no pop action"


CHECKS: Tuple[Tuple[str, Any], ...] = (
    ("perturbed_needs_pop", check_perturbed_needs_pop),
    ("pop_replay_witness_real_engine", check_pop_replay_witness_real_engine),
    ("shipped_spec_with_pop", check_shipped_spec_with_pop),
    ("prune_disabled_with_pop", check_prune_disabled_with_pop),
    ("default_engine_unchanged", check_default_engine_unchanged),
)


def run() -> int:
    passed = 0
    failures: List[str] = []
    for name, fn in CHECKS:
        try:
            detail = fn()
        except CheckFailure as exc:
            failures.append(exc.name)
            print("[FAIL] %s: %s" % (name, exc.detail))
        except Exception as exc:  # noqa: BLE001
            failures.append(name)
            print("[FAIL] %s: unexpected %r" % (name, exc))
        else:
            passed += 1
            print("[PASS] %s: %s" % (name, detail))
    total = len(CHECKS)
    print("RESULT: %d/%d passed (real dynamic_env engine, pop action set)" % (passed, total))
    return 0 if not failures else 1


# pytest entry points -------------------------------------------------------

def test_perturbed_needs_pop() -> None:
    check_perturbed_needs_pop()


def test_pop_replay_witness_real_engine() -> None:
    check_pop_replay_witness_real_engine()


def test_shipped_spec_with_pop() -> None:
    check_shipped_spec_with_pop()


def test_prune_disabled_with_pop() -> None:
    check_prune_disabled_with_pop()


def test_default_engine_unchanged() -> None:
    check_default_engine_unchanged()


if __name__ == "__main__":
    sys.exit(run())
