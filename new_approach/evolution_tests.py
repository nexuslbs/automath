"""Stage 1 extended deterministic suite: the prior 11 checks PLUS new checks
for the semantic node kinds and the more-complex scenario suite.

Run with::

    /opt/automath/venv/bin/python -m new_approach.evolution_tests

The prior suite (``new_approach.tests``) is imported and run unchanged, so a
regression there fails this run too.  Nothing here uses randomness; every
exhaustive count is over a small restricted alphabet and small target.
"""

from __future__ import annotations

import itertools
import sys
import time
from typing import Callable, Dict, List, Tuple

from .env import FixedStateGoal, MinimalEnv, plan as core_plan
from .evolution import (
    EVO_ACTION_BY_NAME,
    EVO_ACTIONS,
    EVO_AXIOMS,
    KIND_GROUPS,
    REDUCTION_TABLE,
    SEM_BY_NAME,
    SEMANTIC_OPS,
    UNIQUE_SEM_TARGETS,
    EvoEnv,
    count_evo_solutions,
    ev_add,
    ev_div,
    ev_if,
    ev_lt,
    core_reduction,
    scenario_cases,
    scenarios,
    sem_action_count,
    sem_plan,
    simulate_evo,
)
from .nodes import Group, NODE_TYPES, Node, One, Zero, nat, node_type_count
from .tests import CHECKS as PRIOR_CHECKS
from .tests import CheckFailure

EXPECTED_KINDS = ("arith", "cmp", "logic", "control", "seq")


def check_semantic_kinds_registered() -> str:
    if len(SEMANTIC_OPS) < 12:
        raise CheckFailure("semantic_kinds", "expected >=12 kinds, got %d"
                           % len(SEMANTIC_OPS))
    for kind in EXPECTED_KINDS:
        if not KIND_GROUPS.get(kind):
            raise CheckFailure("semantic_kinds", "kind group %r empty" % kind)
    for name, op in SEM_BY_NAME.items():
        if EVO_ACTION_BY_NAME[name].arity != op.arity:
            raise CheckFailure("semantic_kinds",
                               "%s arity mismatch" % name)
    return ("%d semantic kinds across %s; each has a build action and a "
            "declared arity" % (len(SEMANTIC_OPS), ",".join(EXPECTED_KINDS)))


def check_semantic_expressiveness() -> str:
    got = {}
    for sc in scenarios():
        try:
            got[sc.name] = EVO_AXIOMS.evaluate(sc.target)
        except Exception as exc:  # noqa: BLE001
            raise CheckFailure("semantic_expressiveness",
                               "%s raised %r" % (sc.name, exc),
                               feedback="state=%s expected=%r actual=raise %r"
                               % (sc.target.canonical(), sc.expected, exc))
        if got[sc.name] != sc.expected:
            raise CheckFailure(
                "semantic_expressiveness",
                "%s -> %r expected %r" % (sc.name, got[sc.name], sc.expected),
                feedback="state=%s expected=%r actual=%r"
                % (sc.target.canonical(), sc.expected, got[sc.name]))
    return ("all %d complex scenarios evaluate correctly under the extended "
            "axioms" % len(got))


def check_semantic_reduction_core() -> str:
    """Every new kind maps onto the four core types, and the semantic trees are
    still evaluable/buildable by the 4-core action vocabulary."""
    targets: List[Node] = [sc.target for sc in scenarios()]
    targets += [t for _, t, _ in UNIQUE_SEM_TARGETS]
    for t in targets:
        core_reduction(t)  # asserts classes subset of NODE_TYPES
    types = node_type_count(*targets)
    if types > 4:
        raise CheckFailure("semantic_reduction",
                           "semantic suite uses %d classes (>4)" % types)
    for sc in scenarios():
        env = MinimalEnv(FixedStateGoal(sc.target))
        state = env.run(core_plan(sc.target))
        if not env.goal_achieved(state):
            raise CheckFailure(
                "semantic_reduction",
                "%s not buildable by the core 4-action plan" % sc.name,
                feedback=env.feedback(state).render())
    return ("%d semantic trees use at most the 4 core classes (%d) and all are "
            "buildable by the core plan; per-kind mapping: %d table rows"
            % (len(targets), types, len(REDUCTION_TABLE)))


def check_semantic_planner() -> str:
    for sc in scenarios():
        word = sem_plan(sc.target)
        env = EvoEnv(FixedStateGoal(sc.target))
        state = env.run(word)
        if not env.goal_achieved(state):
            raise CheckFailure(
                "semantic_planner", "%s: planner word did not reach goal"
                % sc.name, feedback=env.feedback(state).render())
        if len(word) != sem_action_count(sc.target):
            raise CheckFailure("semantic_planner",
                               "%s length mismatch" % sc.name)
    return ("the semantic planner reaches all %d scenario targets; word length "
            "== action count (tags synthesised)" % len(scenarios()))


def check_semantic_unique_solution() -> str:
    for name, target, alphabet in UNIQUE_SEM_TARGETS:
        count, words = count_evo_solutions(target, alphabet)
        if count != 1:
            raise CheckFailure(
                "semantic_unique",
                "%s: found %d solutions, expected 1: %r" % (name, count, words))
        planner = tuple(sem_plan(target))
        if words[0] != planner:
            raise CheckFailure(
                "semantic_unique",
                "%s: unique word %r != planner %r" % (name, words[0], planner))
    return ("exhaustive enumeration over the restricted semantic alphabets "
            "finds EXACTLY ONE solution per complex target (max length %d)"
            % max(sem_action_count(t) for _, t, _ in UNIQUE_SEM_TARGETS))


def check_semantic_no_shorter() -> str:
    for name, target, alphabet in UNIQUE_SEM_TARGETS:
        length = sem_action_count(target)
        for shorter in range(0, length):
            for word in itertools.product(alphabet, repeat=shorter):
                stack, error = simulate_evo((), word)
                if error is None and stack == (target,):
                    raise CheckFailure(
                        "semantic_no_shorter",
                        "%s: %r shorter than %d reached the goal"
                        % (name, word, length))
    return "no shorter semantic action word reaches any complex target"


def check_semantic_new_initial_states() -> str:
    cases = scenario_cases(initial_prefixes=True)
    for name, target, stack, _level, expected in cases:
        env = EvoEnv(FixedStateGoal(target), initial_stack=stack)
        state = env.run(list(expected))
        if not env.goal_achieved(state):
            raise CheckFailure(
                "semantic_new_initial_states",
                "%s from %s: suffix did not reach goal" % (name, stack),
                feedback=env.feedback(state).render())
    return ("%d NEW initial states (every prefix of each scenario word) reach "
            "the same complex target" % len(cases))


def check_semantic_lazy_control() -> str:
    # The untaken branch divides by zero; lazy IF must not evaluate it.
    expr = ev_if(ev_lt(One(), nat(2)), nat(7), ev_div(nat(1), Zero()))
    value = EVO_AXIOMS.evaluate(expr)
    if value != 7:
        raise CheckFailure("semantic_lazy", "expected 7, got %r" % (value,))
    return ("semantic IF is lazy: the raising untaken branch is not evaluated; "
            "value=%r" % (value,))


def check_semantic_structured_failure() -> str:
    target = ev_add(One(), One())
    env = EvoEnv(FixedStateGoal(target))
    state = env.run(["PushZero"])
    if env.goal_achieved(state):
        raise CheckFailure("semantic_structured_failure",
                           "wrong word unexpectedly reached the goal")
    fb = env.feedback(state)
    for field in ("state", "expected", "actual", "reason"):
        if not getattr(fb, field):
            raise CheckFailure("semantic_structured_failure",
                               "feedback field %s empty" % field)
    bad = env.run(["MakeAdd"])
    if bad.last_error is None:
        raise CheckFailure("semantic_structured_failure",
                           "invalid action produced no last_error")
    return "wrong action -> %s; invalid -> last_error=%r" % (
        fb.render(), bad.last_error)


NEW_CHECKS: Tuple[Tuple[str, Callable[[], str]], ...] = (
    ("semantic_kinds_registered", check_semantic_kinds_registered),
    ("semantic_expressiveness", check_semantic_expressiveness),
    ("semantic_reduction_core", check_semantic_reduction_core),
    ("semantic_planner", check_semantic_planner),
    ("semantic_unique_solution", check_semantic_unique_solution),
    ("semantic_no_shorter", check_semantic_no_shorter),
    ("semantic_new_initial_states", check_semantic_new_initial_states),
    ("semantic_lazy_control", check_semantic_lazy_control),
    ("semantic_structured_failure", check_semantic_structured_failure),
)

ALL_CHECKS: Tuple[Tuple[str, Callable[[], str]], ...] = (
    PRIOR_CHECKS + NEW_CHECKS
)


def run() -> int:
    started = time.perf_counter()
    passed = 0
    failures: List[str] = []
    for name, fn in ALL_CHECKS:
        try:
            detail = fn()
        except CheckFailure as exc:
            failures.append(exc.name)
            print("[FAIL] %s: %s" % (name, exc.detail))
            if exc.feedback:
                print("       %s" % exc.feedback)
        except Exception as exc:  # noqa: BLE001
            failures.append(name)
            print("[FAIL] %s: unexpected %r" % (name, exc))
        else:
            passed += 1
            print("[PASS] %s: %s" % (name, detail))
    elapsed = time.perf_counter() - started
    total = len(ALL_CHECKS)
    print("RESULT: %d/%d passed in %.3fs (prior=%d, semantic=%d, "
          "node_types<=4, deterministic)"
          % (passed, total, elapsed, len(PRIOR_CHECKS), len(NEW_CHECKS)))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(run())
