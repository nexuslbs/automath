"""Deterministic tests for the generic dynamic-nodes environment.

Run with::

    /opt/automath/venv/bin/python -m dynamic_env.tests
    /opt/automath/venv/bin/python -m pytest dynamic_env/tests.py

The runner uses only the standard library, seeds nothing (there is no randomness
anywhere), and is bounded: every spec is tiny and every exhaustive count is over
words of a fixed small length. It prints structured PASS/FAIL feedback.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .engine import (
    Action,
    DynamicEnv,
    action_set_flag,
    bfs_minimal_word,
    count_goal_words,
    env_for_action_set,
    legal_actions,
    minimal_length,
    topmost_work_node,
    work_stack,
)
from .spec import list_spec_files, load_spec, spec_from_dict

#: Fixture stem -> expected minimal number of actions to the goal.
EXPECTED_MIN: Dict[str, int] = {
    "spec_minimal": 1,
    "spec_dynamic_group": 2,
    "spec_dynamic_axiom": 3,
    "spec_multi_step": 2,
}

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_SPEC_DIR = os.path.join(_REPO_ROOT, "data", "dynamic_env")


class CheckFailure(Exception):
    def __init__(self, name: str, detail: str, feedback: str = "") -> None:
        super().__init__(detail)
        self.name = name
        self.detail = detail
        self.feedback = feedback


def _spec_paths() -> List[str]:
    return list_spec_files(_SPEC_DIR)


def _envs() -> List[Tuple[str, DynamicEnv]]:
    out: List[Tuple[str, DynamicEnv]] = []
    for path in _spec_paths():
        env = DynamicEnv(load_spec(path))
        out.append((env.spec.spec_id, env))
    return out


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def check_specs_are_data() -> str:
    paths = _spec_paths()
    if len(paths) < 4:
        raise CheckFailure("specs_are_data", "expected >= 4 spec files, got %d" % len(paths))
    seen = set()
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if "spec_id" not in raw:
            raise CheckFailure("specs_are_data", "%s has no spec_id" % path)
        if raw.get("spec_id") in seen:
            raise CheckFailure("specs_are_data", "duplicate spec_id %r" % raw.get("spec_id"))
        seen.add(raw.get("spec_id"))
    return "%d specs load as pure JSON data: %s" % (len(paths), ", ".join(sorted(seen)))


def check_no_spec_ids_in_engine() -> str:
    """The engine source must not mention any shipped spec id."""
    engine_src = open(os.path.join(_HERE, "engine.py"), encoding="utf-8").read()
    spec_src = open(os.path.join(_HERE, "spec.py"), encoding="utf-8").read()
    for path in _spec_paths():
        with open(path, "r", encoding="utf-8") as handle:
            spec_id = json.load(handle)["spec_id"]
        for module, src in (("engine.py", engine_src), ("spec.py", spec_src)):
            if spec_id in src:
                raise CheckFailure(
                    "no_spec_ids_in_engine",
                    "spec id %r is hardcoded in %s" % (spec_id, module),
                )
    return "engine.py/spec.py mention none of the %d shipped spec ids" % len(_spec_paths())


def check_adhoc_spec() -> str:
    """A brand-new spec (different names and tag) loads and is solved as-is."""
    adhoc = {
        "spec_id": "adhoc_temp",
        "description": "inline spec never shipped in data/",
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
            {"id": "ax_twice", "tag": "twice", "arity": 1, "op": "succ", "active": True}
        ],
        "build_axioms": [
            {"id": "build_twice", "tag_node": "m_twice", "arity": 1,
             "node_type": "box", "operand_type": None, "active": True}
        ],
        "dynamic_axioms": [],
        "nodes": {
            "c2": {"type": "cell", "x": 2},
            "m_twice": {"type": "mark", "m": "twice"},
            "g": {"type": "goalbit", "v": 0},
        },
        "objectives": {"g": 1},
        "guards": {"g": {"op": "exists_value", "cmp": "eq", "value": 3,
                         "combination_only": True}},
        "dynamic_node_type": None,
        "max_combine_arity": 0,
        "step_cost": 0.01,
        "goal_reward": 1.0,
        "max_steps": 5,
    }
    env = DynamicEnv(spec_from_dict(adhoc))
    word = bfs_minimal_word(env.spec)
    if word is None:
        raise CheckFailure("adhoc_spec", "inline spec goal unreachable")
    if len(word) != 2:
        raise CheckFailure("adhoc_spec", "expected min length 2, got %d" % len(word))
    state = env.reset()
    for key in word:
        action = _action_from_key(env, state, key)
        state = env.step(state, action).state
    if not env.goal_reached(state):
        raise CheckFailure("adhoc_spec", "replay of %r did not reach the goal" % (word,))
    return "inline spec 'adhoc_temp' (new names, tag 'twice') solved identically"


def _action_from_key(env: DynamicEnv, state, key: str) -> Action:
    for action in env.legal_actions(state):
        if action.key() == key:
            return action
    raise CheckFailure("replay", "action %r is not legal in state %s" % (key, state.identity()))


def check_objective_decomposition() -> str:
    details = []
    for spec_id, env in _envs():
        target = env.target_vector()
        start = env.reset()
        vector = env.state_vector(start)
        n = len(target)
        if vector[:n] != env.objective_vector(start):
            raise CheckFailure("objective_decomposition", "%s: vector does not start with objectives" % spec_id)
        if vector[n:2 * n] != target:
            raise CheckFailure("objective_decomposition", "%s: vector does not expose the target" % spec_id)
        goal = env.goal_reached(start)
        manual = all(o == t for o, t in zip(env.objective_vector(start), target))
        if goal != manual:
            raise CheckFailure("objective_decomposition", "%s: goal != all(objective == target)" % spec_id)
        details.append("%s target=%s start=%s" % (spec_id, target, env.objective_vector(start)))
    return "objective flags + target exposed in state_vector for all specs; " + "; ".join(details)


def check_deterministic_transition() -> str:
    checked = 0
    for spec_id, env in _envs():
        start = env.reset()
        actions = env.legal_actions(start)
        if not actions:
            raise CheckFailure("deterministic_transition", "%s has no legal action" % spec_id)
        action = actions[0]
        first = env.step(start, action)
        second = env.step(start, action)
        if first.state.canonical() != second.state.canonical():
            raise CheckFailure(
                "deterministic_transition",
                "%s: same state+action gave different next states" % spec_id,
            )
        if env.state_vector(first.state) != env.state_vector(second.state):
            raise CheckFailure("deterministic_transition", "%s: state vectors differ" % spec_id)
        word = bfs_minimal_word(env.spec)
        if word is None:
            raise CheckFailure("deterministic_transition", "%s goal unreachable" % spec_id)
        run_a = _replay(env, word)
        run_b = _replay(env, word)
        if run_a.canonical() != run_b.canonical():
            raise CheckFailure("deterministic_transition", "%s: replays differ" % spec_id)
        checked += 1
    return "same state+action gives byte-identical next state for all %d specs" % checked


def _replay(env: DynamicEnv, word: Sequence[str]):
    state = env.reset()
    for key in word:
        action = _action_from_key(env, state, key)
        state = env.step(state, action).state
    return state


def check_minimal_solution_unique() -> str:
    for spec_id, env in _envs():
        expected = EXPECTED_MIN.get(spec_id)
        if expected is None:
            raise CheckFailure("minimal_solution_unique", "no expected min for %s" % spec_id)
        found = minimal_length(env.spec)
        if found != expected:
            raise CheckFailure(
                "minimal_solution_unique",
                "%s: minimal length %r != expected %d" % (spec_id, found, expected),
            )
        count, witnesses = count_goal_words(env.spec, expected, cap=2)
        if count != 1:
            raise CheckFailure(
                "minimal_solution_unique",
                "%s: %d minimal solutions, expected exactly 1: %r"
                % (spec_id, count, witnesses),
            )
    return "exactly ONE minimal action sequence per spec: " + ", ".join(
        "%s=%d" % (pid, EXPECTED_MIN[pid]) for pid, _ in _envs()
    )


def check_no_shorter_solution() -> str:
    for spec_id, env in _envs():
        minimum = EXPECTED_MIN[spec_id]
        for length in range(0, minimum):
            count, witnesses = count_goal_words(env.spec, length, cap=1)
            if count != 0:
                raise CheckFailure(
                    "no_shorter_solution",
                    "%s: found a goal word of length %d (< %d): %r"
                    % (spec_id, length, minimum, witnesses),
                )
    return "no action word shorter than the minimum reaches any goal"


def check_multi_step_multiple_paths() -> str:
    env = _envs_by_id()["spec_multi_step"]
    minimum = EXPECTED_MIN["spec_multi_step"]
    longer, witnesses = count_goal_words(env.spec, minimum + 1, cap=1)
    if longer < 1:
        raise CheckFailure(
            "multi_step_multiple_paths",
            "spec_multi_step has no longer correct path at length %d" % (minimum + 1),
        )
    return (
        "spec_multi_step: 1 minimal path of length %d plus a longer path at "
        "length %d (witness %s)" % (minimum, minimum + 1, witnesses[0])
    )


def check_dynamic_group_action() -> str:
    env = _envs_by_id()["spec_dynamic_group"]
    word = bfs_minimal_word(env.spec)
    if not word or not word[0].startswith("combine:"):
        raise CheckFailure("dynamic_group_action", "minimal word does not start with combine: %r" % (word,))
    state = env.reset()
    action = _action_from_key(env, state, word[0])
    result = env.step(state, action)
    added = result.info.get("added_node")
    if added is None:
        raise CheckFailure("dynamic_group_action", "combine action added no node")
    node = result.state.node_map()[added]
    if node.type != env.spec.dynamic_node_type:
        raise CheckFailure("dynamic_group_action", "dynamic node type %r expected" % node.type)
    return "combine built dynamic grouping node %s of type %r; word=%r" % (added, node.type, word)


def check_dynamic_axiom_fires() -> str:
    env = _envs_by_id()["spec_dynamic_axiom"]
    state = env.reset()
    before_nodes = set(state.node_map())
    if "build_sub" in state.active:
        raise CheckFailure("dynamic_axiom_fires", "build_sub active before the key objective")
    action = _action_from_key(env, state, "set:key")
    result = env.step(state, action)
    if "build_sub" not in result.state.active:
        raise CheckFailure("dynamic_axiom_fires", "dynamic axiom did not activate build_sub")
    added = set(result.state.node_map()) - before_nodes
    if added != {"n7"}:
        raise CheckFailure("dynamic_axiom_fires", "expected n7 added, got %s" % sorted(added))
    if "dyn_unlock" not in result.state.fired:
        raise CheckFailure("dynamic_axiom_fires", "dyn_unlock not recorded as fired")
    # Firing is a pure function of the state: replay gives the identical state.
    again = env.step(env.reset(), _action_from_key(env, env.reset(), "set:key"))
    if again.state.canonical() != result.state.canonical():
        raise CheckFailure("dynamic_axiom_fires", "dynamic firing is not deterministic")
    return "dynamic axiom fired once: activated build_sub, added n7, fired=%r" % (result.state.fired,)


def check_wrong_step_discarded() -> str:
    env = _envs_by_id()["spec_minimal"]
    state = env.reset()
    # clear:o1 is illegal while o1 == 0 -> must be discarded with no state change.
    discard = Action(kind="clear", objective_id="o1")
    result, reason = env.try_step(state, discard)
    if result is not None or not reason:
        raise CheckFailure("wrong_step_discarded", "illegal action was not discarded")
    if env.try_step(state, discard)[0] is not None:
        raise CheckFailure("wrong_step_discarded", "discard is not deterministic")
    # A legal step moves the state; an illegal one leaves it untouched.
    legal = _action_from_key(env, state, "set:o1")
    moved = env.step(state, legal)
    if moved.state.identity() == state.identity():
        raise CheckFailure("wrong_step_discarded", "legal action left the state unchanged")
    return "illegal %s discarded (%s); legal set:o1 reached the goal=%s" % (
        discard.key(), reason, moved.done)


def check_reward_hook() -> str:
    details = []
    for spec_id, env in _envs():
        word = bfs_minimal_word(env.spec)
        state = env.reset()
        rewards: List[float] = []
        for key in word:
            action = _action_from_key(env, state, key)
            result = env.step(state, action)
            rewards.append(result.reward)
            state = result.state
        if not env.goal_reached(state):
            raise CheckFailure("reward_hook", "%s: replay did not reach goal" % spec_id)
        expected_total = env.spec.goal_reward - env.spec.step_cost * (len(word) - 1)
        if abs(sum(rewards) - expected_total) > 1e-9:
            raise CheckFailure(
                "reward_hook",
                "%s: total reward %.6f != expected %.6f" % (spec_id, sum(rewards), expected_total),
            )
        details.append("%s=%.3f" % (spec_id, sum(rewards)))
    return "per-step cost %.2f + final goal reward %.1f exactly: %s" % (
        next(iter(_envs()))[1].spec.step_cost,
        next(iter(_envs()))[1].spec.goal_reward,
        ", ".join(details),
    )


def check_goal_reachable() -> str:
    details = []
    for spec_id, env in _envs():
        word = bfs_minimal_word(env.spec)
        if word is None:
            raise CheckFailure("goal_reachable", "%s goal unreachable" % spec_id)
        state = _replay(env, word)
        if not env.goal_reached(state):
            raise CheckFailure("goal_reachable", "%s replay failed" % spec_id)
        details.append("%s:%d" % (spec_id, len(word)))
    return "reachable goal in every shipped spec (min-steps): " + ", ".join(details)


def _envs_by_id() -> Dict[str, DynamicEnv]:
    return {spec_id: env for spec_id, env in _envs()}


# --------------------------------------------------------------------------
# Unit 2: the pop action (OFF by default; opt in with allow_pop=True)
# --------------------------------------------------------------------------

def check_pop_off_by_default() -> str:
    """``allow_pop=False`` keeps the historical four-action set byte-identical."""
    for spec_id, env in _envs():
        start = env.reset()
        base = env.legal_actions(start)
        explicit = legal_actions(env.spec, start)
        if base != explicit:
            raise CheckFailure("pop_off_by_default",
                               "%s: DynamicEnv(allow_pop=False) != legal_actions()" % spec_id)
        if any(a.kind == "pop" for a in base):
            raise CheckFailure("pop_off_by_default", "%s: pop leaked into the default set" % spec_id)
    flag = action_set_flag("current")
    if flag is not False or action_set_flag("pop") is not True:
        raise CheckFailure("pop_off_by_default", "action_set_flag mapping is wrong")
    try:
        action_set_flag("bogus")
    except ValueError:
        pass
    else:
        raise CheckFailure("pop_off_by_default", "unknown action set did not raise")
    return ("default action set has no pop for all %d specs; "
            "action_set_flag(current)=False action_set_flag(pop)=True; unknown raises"
            % len(_envs()))


def check_pop_legality_and_step() -> str:
    """Pop is legal iff the work stack is non-empty; it removes exactly its top."""
    checked = 0
    for spec_id, env in _envs():
        with_pop = DynamicEnv(env.spec, allow_pop=True)
        start = with_pop.reset()
        if topmost_work_node(env.spec, start) is not None:
            raise CheckFailure("pop_legality_and_step",
                               "%s: reset state already has a work node" % spec_id)
        if any(a.kind == "pop" for a in with_pop.legal_actions(start)):
            raise CheckFailure("pop_legality_and_step",
                               "%s: pop legal on an empty work stack" % spec_id)
        discard = Action(kind="pop")
        result, reason = with_pop.try_step(start, discard)
        if result is not None or not reason:
            raise CheckFailure("pop_legality_and_step",
                               "%s: illegal pop was not discarded" % spec_id)
        word = bfs_minimal_word(env.spec)
        if word is None:
            continue
        state = with_pop.reset()
        built = None
        for key in word:
            action = _action_from_key(with_pop, state, key)
            step_result = with_pop.step(state, action)
            if action.kind in ("build", "combine"):
                built = (state, step_result)
                break
            state = step_result.state
        if built is None:
            continue  # no build/combine on the canonical path (e.g. spec_minimal)
        before, build_result = built
        after = build_result.state
        added = build_result.info["added_node"]
        pop_action = next((a for a in with_pop.legal_actions(after) if a.kind == "pop"), None)
        if pop_action is None or not with_pop.is_legal(after, pop_action):
            raise CheckFailure("pop_legality_and_step",
                               "%s: pop not legal after a build" % spec_id)
        popped = with_pop.step(after, pop_action)
        if popped.info.get("removed_node") != added:
            raise CheckFailure("pop_legality_and_step",
                               "%s: pop removed %r, expected %r"
                               % (spec_id, popped.info.get("removed_node"), added))
        removed = set(after.node_map()) - set(popped.state.node_map())
        if removed != {added}:
            raise CheckFailure("pop_legality_and_step",
                               "%s: pop changed nodes other than %s: %s"
                               % (spec_id, added, sorted(removed)))
        if (popped.state.objectives != after.objectives
                or popped.state.active != after.active
                or popped.state.fired != after.fired):
            raise CheckFailure("pop_legality_and_step",
                               "%s: pop changed non-node state" % spec_id)
        if popped.state.step != after.step + 1 or popped.state.history[-1] != "pop":
            raise CheckFailure("pop_legality_and_step",
                               "%s: pop step/history bookkeeping is wrong" % spec_id)
        if with_pop.step(after, pop_action).state.canonical() != popped.state.canonical():
            raise CheckFailure("pop_legality_and_step", "%s: pop is not deterministic" % spec_id)
        checked += 1
    return ("pop discarded when illegal, legal after a build, removes exactly the "
            "newest work node (checked %d specs)" % checked)


def check_pop_replay_witness() -> str:
    """``[POP]*len(initial work stack) + canonical word`` reaches the goal."""
    checked = 0
    for spec_id, env in _envs():
        word = bfs_minimal_word(env.spec)
        if word is None or len(word) == 0:
            continue
        with_pop = DynamicEnv(env.spec, allow_pop=True)
        # Perturbed start: one extra legal build/combine (a work node) on top of
        # the reset state. Pop the whole work stack and replay the canonical word
        # (the stack roll-back reuses the popped node ids, so the word replays).
        start = with_pop.reset()
        filler = [a for a in with_pop.legal_actions(start) if a.kind in ("build", "combine")]
        if filler:
            start = with_pop.step(start, filler[0]).state
        top = topmost_work_node(env.spec, start)
        if top is None:
            continue
        witness = ["pop"] * len(work_stack(env.spec, start)) + list(word)
        state = start
        for key in witness:
            action = _action_from_key(with_pop, state, key)
            state = with_pop.step(state, action).state
        if not with_pop.goal_reached(state):
            raise CheckFailure("pop_replay_witness",
                               "%s: witness %r did not reach the goal" % (spec_id, witness))
        # The same witness must be illegal/absent without pop.
        without = DynamicEnv(env.spec, allow_pop=False)
        state = start
        for key in witness:
            match = next((a for a in without.legal_actions(state) if a.key() == key), None)
            if match is None:
                break
            state = without.step(state, match).state
        if without.goal_reached(state):
            raise CheckFailure("pop_replay_witness",
                               "%s: witness also solved WITHOUT pop" % spec_id)
        checked += 1
    if checked == 0:
        raise CheckFailure("pop_replay_witness", "no spec exercised the pop witness")
    return ("[pop]*len(work stack)+canonical word reaches the goal in the REAL env "
            "and needs pop (checked %d specs)" % checked)


def check_action_set_selection() -> str:
    """The env selects the action set and the action space grows by exactly pop."""
    before = 0
    after = 0
    for spec_id, env in _envs():
        state = env.reset()
        current = len(DynamicEnv(env.spec, allow_pop=False).legal_actions(state))
        popped = len(DynamicEnv(env.spec, allow_pop=True).legal_actions(state))
        if popped != current:
            raise CheckFailure("action_set_selection",
                               "%s: pop changed the reset action count %d -> %d"
                               % (spec_id, current, popped))
        word = bfs_minimal_word(env.spec)
        if word is None:
            continue
        state = DynamicEnv(env.spec, allow_pop=True).reset()
        applied = False
        for key in word:
            action = _action_from_key(DynamicEnv(env.spec, allow_pop=True), state, key)
            state = DynamicEnv(env.spec, allow_pop=True).step(state, action).state
            if action.kind in ("build", "combine"):
                applied = True
                break
        if not applied:
            continue
        current = len(DynamicEnv(env.spec, allow_pop=False).legal_actions(state))
        popped = len(DynamicEnv(env.spec, allow_pop=True).legal_actions(state))
        if popped != current + 1:
            raise CheckFailure("action_set_selection",
                               "%s: pop delta %d -> %d is not exactly +1"
                               % (spec_id, current, popped))
        before += current
        after += popped
    if not (env_for_action_set(_envs()[0][1].spec, "pop").allow_pop is True):
        raise CheckFailure("action_set_selection", "env_for_action_set did not enable pop")
    return ("pop adds exactly +1 action on a non-empty work stack; "
            "reset count unchanged (sample totals %d -> %d)" % (before, after))


CHECKS: Tuple[Tuple[str, Callable[[], str]], ...] = (
    ("specs_are_data", check_specs_are_data),
    ("no_spec_ids_in_engine", check_no_spec_ids_in_engine),
    ("adhoc_spec", check_adhoc_spec),
    ("objective_decomposition", check_objective_decomposition),
    ("deterministic_transition", check_deterministic_transition),
    ("minimal_solution_unique", check_minimal_solution_unique),
    ("no_shorter_solution", check_no_shorter_solution),
    ("multi_step_multiple_paths", check_multi_step_multiple_paths),
    ("dynamic_group_action", check_dynamic_group_action),
    ("dynamic_axiom_fires", check_dynamic_axiom_fires),
    ("wrong_step_discarded", check_wrong_step_discarded),
    ("reward_hook", check_reward_hook),
    ("goal_reachable", check_goal_reachable),
    ("pop_off_by_default", check_pop_off_by_default),
    ("pop_legality_and_step", check_pop_legality_and_step),
    ("pop_replay_witness", check_pop_replay_witness),
    ("action_set_selection", check_action_set_selection),
)


def run() -> int:
    started = time.perf_counter()
    passed = 0
    failures: List[str] = []
    for name, fn in CHECKS:
        try:
            detail = fn()
        except CheckFailure as exc:
            failures.append(exc.name)
            print("[FAIL] %s: %s" % (name, exc.detail))
            if exc.feedback:
                print("       %s" % exc.feedback)
        except Exception as exc:  # noqa: BLE001 - surface as a failure
            failures.append(name)
            print("[FAIL] %s: unexpected %r" % (name, exc))
        else:
            passed += 1
            print("[PASS] %s: %s" % (name, detail))
    elapsed = time.perf_counter() - started
    total = len(CHECKS)
    print("RESULT: %d/%d passed in %.3fs (deterministic, engine=generic, env=data)"
          % (passed, total, elapsed))
    return 0 if not failures else 1


# pytest entry points -------------------------------------------------------
def test_specs_are_data() -> None:
    check_specs_are_data()


def test_no_spec_ids_in_engine() -> None:
    check_no_spec_ids_in_engine()


def test_adhoc_spec() -> None:
    check_adhoc_spec()


def test_objective_decomposition() -> None:
    check_objective_decomposition()


def test_deterministic_transition() -> None:
    check_deterministic_transition()


def test_minimal_solution_unique() -> None:
    check_minimal_solution_unique()


def test_no_shorter_solution() -> None:
    check_no_shorter_solution()


def test_multi_step_multiple_paths() -> None:
    check_multi_step_multiple_paths()


def test_dynamic_group_action() -> None:
    check_dynamic_group_action()


def test_dynamic_axiom_fires() -> None:
    check_dynamic_axiom_fires()


def test_wrong_step_discarded() -> None:
    check_wrong_step_discarded()


def test_reward_hook() -> None:
    check_reward_hook()


def test_goal_reachable() -> None:
    check_goal_reachable()


def test_pop_off_by_default() -> None:
    check_pop_off_by_default()


def test_pop_legality_and_step() -> None:
    check_pop_legality_and_step()


def test_pop_replay_witness() -> None:
    check_pop_replay_witness()


def test_action_set_selection() -> None:
    check_action_set_selection()


if __name__ == "__main__":
    sys.exit(run())
