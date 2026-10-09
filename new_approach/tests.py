"""Deterministic single-solution tests for the minimal-node environment.

Run with::

    /opt/automath/venv/bin/python -m new_approach.tests

The runner uses only the standard library, seeds nothing (there is no
randomness anywhere), is bounded (small trees, exhaustive words of length
<= 6), and prints structured failure feedback (state, expected, actual,
reason) for every failing check.
"""

from __future__ import annotations

import itertools
import sys
import time
from typing import Callable, Dict, List, Sequence, Tuple

from .axioms import AxiomSet
from .env import (
    ExprGoal,
    FixedStateGoal,
    MinimalEnv,
    bfs_plan,
    count_solutions,
    plan,
    simulate,
)
from .expressions import (
    SAMPLES,
    eq,
    head,
    iff,
    lt,
    nat,
    sample,
    seq,
)
from .nodes import (
    Change,
    Group,
    NODE_TYPES,
    Node,
    One,
    Zero,
    cell_kind_count,
    node_type_count,
    size,
    to_cell,
    to_min2,
    to_min3,
)
from .reduction import cell_value

AX = AxiomSet()

# Targets buildable with {PushZero, PushOne, MakeChange, MakeGroup2}, so the
# exhaustive solution count is over a 4-action alphabet and stays tiny.
BRUTE_TARGETS: Tuple[Tuple[str, Node], ...] = (
    ("zero", Zero()),
    ("one", One()),
    ("change(one)", Change(One())),
    ("change(change(one))", Change(Change(One()))),
    ("group(one;zero)", Group(One(), (Zero(),))),
    ("change(group(one;zero))", Change(Group(One(), (Zero(),)))),
    ("group(change(one);zero)", Group(Change(One()), (Zero(),))),
    (
        "group(change(one);group(one;zero))",
        Group(Change(One()), (Group(One(), (Zero(),)),)),
    ),
)

BRUTE_ACTIONS = ("PushZero", "PushOne", "MakeChange", "MakeGroup2")

EXPECTED: Dict[str, object] = {
    "add(2,3)": 5,
    "sub(5,2)": 3,
    "mul(2,3)": 6,
    "lt(2,3)": True,
    "eq(add(1,1),2)": True,
    "and(true,not(false))": True,
    "or(false,true)": True,
    "if(lt(1,2),add(1,1),0)": 2,
    "len([1,2,3])": 3,
    "head([1,2,3])": 1,
    "tail([1,2,3])": (2, 3),
}


class CheckFailure(Exception):
    def __init__(self, name: str, detail: str, feedback: str = "") -> None:
        super().__init__(detail)
        self.name = name
        self.detail = detail
        self.feedback = feedback


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def check_node_type_count() -> str:
    roots: List[Node] = [sample(k) for k in SAMPLES]
    roots += [t for _, t in BRUTE_TARGETS]
    count = node_type_count(*roots)
    if count != 4:
        raise CheckFailure("node_type_count", "expected 4 types, got %d" % count)
    registered = set(NODE_TYPES)
    if len(registered) != 4:
        raise CheckFailure("node_type_count", "NODE_TYPES registry != 4")
    return "4 node types used across the whole suite: %s" % (
        ", ".join(sorted(t.__name__ for t in registered)),)


def check_meta_not_node_types() -> str:
    from .axioms import Axiom
    from .env import ExprGoal, FixedStateGoal
    for cls in (Axiom, FixedStateGoal, ExprGoal):
        if cls in NODE_TYPES:
            raise CheckFailure("meta_not_node_types", "%s leaked into NODE_TYPES" % cls)
    return "Axiom/Goal are meta definitions, not node types (count stays 4)"


def _eval_all(transform: Callable[[Node], object], evaluate: Callable[[object], object]
              ) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for name in SAMPLES:
        out[name] = evaluate(transform(sample(name)))
    return out


def check_expressiveness_min4() -> str:
    got = {name: AX.evaluate(sample(name)) for name in SAMPLES}
    for name, expected in EXPECTED.items():
        if got[name] != expected:
            raise CheckFailure(
                "expressiveness_min4",
                "%s evaluated to %r, expected %r" % (name, got[name], expected),
                feedback="state=expression %s expected=%r actual=%r"
                % (name, expected, got[name]),
            )
    return "all %d math/logic samples evaluate correctly on the 4-node encoding" % len(SAMPLES)


def check_control_flow_is_lazy() -> str:
    # IF must NOT evaluate the untaken branch (here the untaken branch would
    # raise: HEAD of an empty sequence). This is real control flow.
    expr = iff(lt(nat(1), nat(2)), nat(7), head(seq()))
    value = AX.evaluate(expr)
    if value != 7:
        raise CheckFailure("control_flow_lazy", "expected 7, got %r" % (value,))
    return "IF is lazy: untaken (raising) branch is not evaluated; value=%r" % (value,)


def check_fixed_goal_planner() -> str:
    for name, target in BRUTE_TARGETS:
        env = MinimalEnv(FixedStateGoal(target), initial_stack=())
        word = plan(target)
        state = env.run(word)
        if not env.goal_achieved(state):
            raise CheckFailure(
                "fixed_goal_planner",
                "%s: planner word %r did not reach the fixed goal" % (name, word),
                feedback=env.feedback(state).render(),
            )
        if len(word) != size(target):
            raise CheckFailure(
                "fixed_goal_planner",
                "%s: word length %d != target size %d" % (name, len(word), size(target)),
            )
    # Expression-encoded targets (use the full action set).
    for name in SAMPLES:
        target = sample(name)
        env = MinimalEnv(FixedStateGoal(target))
        word = plan(target)
        state = env.run(word)
        if not env.goal_achieved(state):
            raise CheckFailure(
                "fixed_goal_planner",
                "expression %s: planner failed" % name,
                feedback=env.feedback(state).render(),
            )
    return "planner reaches every fixed-state goal; length == target node count"


def check_unique_solution_bruteforce() -> str:
    for name, target in BRUTE_TARGETS:
        count, words = count_solutions(target, BRUTE_ACTIONS)
        if count != 1:
            raise CheckFailure(
                "unique_solution",
                "%s: found %d solutions, expected exactly 1: %r"
                % (name, count, words),
            )
        planner = tuple(plan(target))
        if words[0] != planner:
            raise CheckFailure(
                "unique_solution",
                "%s: unique solution %r != planner %r" % (name, words[0], planner),
            )
    return (
        "exhaustive enumeration over the 4-action alphabet finds EXACTLY ONE "
        "solution per fixed target (max target size %d)" % max(size(t) for _, t in BRUTE_TARGETS)
    )


def check_no_shorter_solution() -> str:
    for name, target in BRUTE_TARGETS:
        length = size(target)
        for shorter in range(0, length):
            for word in itertools.product(BRUTE_ACTIONS, repeat=shorter):
                stack, error = simulate((), word)
                if error is None and stack == (target,):
                    raise CheckFailure(
                        "no_shorter_solution",
                        "%s: word %r shorter than %d reached the goal"
                        % (name, word, length),
                    )
    return "no action word shorter than the target size reaches any fixed goal"


def check_structured_failure() -> str:
    env = MinimalEnv(FixedStateGoal(Change(One())))
    state = env.run(["PushZero"])
    if env.goal_achieved(state):
        raise CheckFailure("structured_failure", "wrong word unexpectedly reached goal")
    fb = env.feedback(state)
    for field in ("state", "expected", "actual", "reason"):
        if not getattr(fb, field):
            raise CheckFailure("structured_failure", "feedback field %s is empty" % field)
    # An invalid action must be a structured error, not a crash.
    bad = env.run(["MakeChange"])
    if bad.last_error is None:
        raise CheckFailure("structured_failure", "invalid action produced no last_error")
    return "wrong action -> %s; invalid action -> last_error=%r" % (
        fb.render(), bad.last_error)


def check_pattern_goal() -> str:
    goal = ExprGoal(lambda cur: eq(cur, nat(3)), AX, name="value==3")
    env = MinimalEnv(goal)
    min_len, count, solutions = None, 0, []
    for length in range(1, 5):
        found = []
        for word in itertools.product(BRUTE_ACTIONS, repeat=length):
            state = env.run(word)
            if env.goal_achieved(state):
                found.append(word)
        if found:
            min_len, count, solutions = length, len(found), found
            break
    if (min_len, count) != (3, 1):
        raise CheckFailure(
            "pattern_goal",
            "expected one minimal solution of length 3, got min_len=%r count=%r sols=%r"
            % (min_len, count, solutions),
        )
    from .env import bfs_plan
    word = bfs_plan(env, max_actions=5)
    if not word or not env.goal_achieved(env.run(word)):
        raise CheckFailure("pattern_goal", "bfs_plan failed for the pattern goal")
    return "pattern goal value==3: unique shortest solution %r (BFS %r)" % (
        solutions[0], word)


def check_reductions() -> str:
    # 4 -> 3 types: One := Change(Zero)
    got3 = _eval_all(to_min3, AX.evaluate)
    for name, expected in EXPECTED.items():
        if got3[name] != expected:
            raise CheckFailure(
                "reduction_3", "%s -> %r expected %r" % (name, got3[name], expected))
    types3 = node_type_count(*(to_min3(sample(n)) for n in SAMPLES))
    if types3 != 3:
        raise CheckFailure("reduction_3", "expected 3 types, got %d" % types3)

    # 3 -> 2 types: Change(x) := Group(Zero, (x,))
    got2 = _eval_all(to_min2, AX.evaluate)
    for name, expected in EXPECTED.items():
        if got2[name] != expected:
            raise CheckFailure(
                "reduction_2", "%s -> %r expected %r" % (name, got2[name], expected))
    types2 = node_type_count(*(to_min2(sample(n)) for n in SAMPLES))
    if types2 != 2:
        raise CheckFailure("reduction_2", "expected 2 types, got %d" % types2)

    # 4 -> 1 type: every term becomes one Cell class.
    got1 = _eval_all(to_cell, cell_value)
    for name, expected in EXPECTED.items():
        if got1[name] != expected:
            raise CheckFailure(
                "reduction_1", "%s -> %r expected %r" % (name, got1[name], expected))
    if cell_kind_count(*(to_cell(sample(n)) for n in SAMPLES)) != 1:
        raise CheckFailure("reduction_1", "expected a single Cell class")

    return (
        "same %d samples evaluate identically with 4, 3, 2 and 1 node types "
        "(3: One:=Change(Zero); 2: Change(x):=Group(Zero,(x,)); 1: single Cell)"
        % len(SAMPLES)
    )


def check_generalization_support() -> str:
    """The U2 seam: a bare-minimum initial state plus optimal actions to the
    end state, then NEW initial states, verified against the same goal."""
    target = Change(Change(One()))  # value 3
    cases: Tuple[Tuple[str, Tuple[Node, ...]], ...] = (
        ("empty", ()),
        ("one", (One(),)),
        ("change(one)", (Change(One()),)),
        ("already-goal", (target,)),
    )
    seen_words = []
    for name, initial in cases:
        env = MinimalEnv(FixedStateGoal(target), initial_stack=initial)
        word = plan(target, initial)
        state = env.run(word)
        if not env.goal_achieved(state):
            raise CheckFailure(
                "generalization_support",
                "%s: plan %r did not reach the fixed goal" % (name, word),
                feedback=env.feedback(state).render(),
            )
        seen_words.append((name, tuple(word)))
    return (
        "target value=3 reached from %d different initial states; plans=%r "
        "(new-initial-state verification works for the U2 split)"
        % (len(cases), seen_words)
    )


CHECKS: Tuple[Tuple[str, Callable[[], str]], ...] = (
    ("node_type_count", check_node_type_count),
    ("meta_not_node_types", check_meta_not_node_types),
    ("expressiveness_min4", check_expressiveness_min4),
    ("control_flow_lazy", check_control_flow_is_lazy),
    ("fixed_goal_planner", check_fixed_goal_planner),
    ("unique_solution_bruteforce", check_unique_solution_bruteforce),
    ("no_shorter_solution", check_no_shorter_solution),
    ("structured_failure", check_structured_failure),
    ("pattern_goal", check_pattern_goal),
    ("reductions", check_reductions),
    ("generalization_support", check_generalization_support),
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
    print("RESULT: %d/%d passed in %.3fs (node_types=4, deterministic)"
          % (passed, total, elapsed))
    return 0 if not failures else 1


# pytest entry points
def test_node_type_count() -> None:
    check_node_type_count()


def test_expressiveness_min4() -> None:
    check_expressiveness_min4()


def test_unique_solution_bruteforce() -> None:
    check_unique_solution_bruteforce()


def test_pattern_goal() -> None:
    check_pattern_goal()


def test_reductions() -> None:
    check_reductions()


def test_generalization_support() -> None:
    check_generalization_support()


if __name__ == "__main__":
    sys.exit(run())
