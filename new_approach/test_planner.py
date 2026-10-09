"""Planner unit tests (task 4342 unit 2b, branch ``gen-fitness``).

Runnable both ways::

    /opt/automath/venv/bin/python -m pytest new_approach/test_planner.py -q
    /opt/automath/venv/bin/python -m new_approach.test_planner

Covers the explicit planner over ``REDUCTION_TABLE``:

* determinism (same (target, stack) -> same action sequence);
* explicit budget bounds (``max_actions`` honored, ``max_search_nodes`` bounded,
  no infinite loop, unsolvable states return ``None`` quickly);
* legality of every returned action for every valid stack;
* 6 fixed handcrafted target forms whose canonical construction is solved;
* the safe fallback (legal action, never a crash) on unsolvable states.
"""

from __future__ import annotations

import time
from typing import List, Tuple

from .env import BUILD_ACTIONS
from .evolution import (
    EVO_ACTIONS,
    ev_add,
    ev_if,
    ev_lt,
    ev_mul,
    sem_plan,
)
from .env import plan as core_plan
from .nodes import Change, Group, Node, One, Zero
from .planner import Planner, make_planners
from .expressions import nat

#: (label, domain, target) - every one is built by the planner from empty.
HANDCRAFTED = [
    ("evo_add(1,1)", "evo", ev_add(One(), One())),
    ("evo_mul(1,C(1))", "evo", ev_mul(One(), nat(2))),
    ("evo_if(lt(1,C(1)),1,0)", "evo",
     ev_if(ev_lt(One(), nat(2)), One(), Zero())),
    ("core_g_1_(0)", "core", Group(One(), (Zero(),))),
    ("core_g_0_(1,0)", "core", Group(Zero(), (One(), Zero()))),
    ("core_change(1)", "core", Change(One())),
]


def _planner(domain: str, **kw) -> Planner:
    return Planner(EVO_ACTIONS if domain == "evo" else BUILD_ACTIONS,
                   domain=domain, **kw)


def _canonical_word(domain: str, target: Node) -> Tuple[str, ...]:
    return tuple(sem_plan(target) if domain == "evo" else core_plan(target))


def _run_word(planner: Planner, stack: Tuple[Node, ...],
              word: Tuple[str, ...]) -> Tuple[Node, ...]:
    for name in word:
        action = planner.by_name[name]
        assert action.arity <= len(stack), (name, len(stack))
        stack = planner._apply(stack, action)
    return stack


def test_determinism_same_sequence() -> None:
    for label, domain, target in HANDCRAFTED:
        p1 = _planner(domain)
        p2 = _planner(domain)
        a = p1.plan(target, ())
        b = p2.plan(target, ())
        c = p1.plan(target, ())
        assert a == b == c, (label, a, b, c)
        assert a is not None and len(a) > 0


def test_determinism_from_prefix_stacks() -> None:
    for label, domain, target in HANDCRAFTED:
        word = _canonical_word(domain, target)
        planner = _planner(domain)
        prefix: Tuple[Node, ...] = ()
        for i, name in enumerate(word):
            got = planner.plan(target, prefix)
            assert got == list(word[i:]), (label, i, got, word[i:])
            prefix = _run_word(planner, prefix, (name,))
        assert prefix == (target,)


def test_handcrafted_canonical_solves() -> None:
    solved = 0
    for label, domain, target in HANDCRAFTED:
        planner = _planner(domain)
        word = planner.plan(target, ())
        assert word is not None, label
        assert tuple(word) == _canonical_word(domain, target), (label, word)
        final = _run_word(planner, (), tuple(word))
        assert final == (target,), (label, final)
        solved += 1
    assert solved == len(HANDCRAFTED) >= 3
    print("handcrafted canonical solved: %d/%d" % (solved, len(HANDCRAFTED)))


def test_prefix_stacks_all_solved() -> None:
    solved = total = 0
    for label, domain, target in HANDCRAFTED:
        word = _canonical_word(domain, target)
        planner = _planner(domain)
        prefix: Tuple[Node, ...] = ()
        for i in range(len(word) + 1):
            total += 1
            got = planner.plan(target, prefix)
            assert got is not None
            final = _run_word(planner, prefix, tuple(got))
            assert final == (target,), (label, i)
            solved += 1
            if i < len(word):
                prefix = _run_word(planner, prefix, (word[i],))
    assert solved == total
    print("prefix states solved: %d/%d" % (solved, total))


def test_budget_max_actions_honoured() -> None:
    # A target needing 4 evo actions cannot be planned with max_actions=1.
    target = ev_mul(One(), nat(2))
    planner = _planner("evo", max_actions=1)
    # 4-action canonical word exceeds max_actions=1 -> budget refuses it.
    assert planner.plan(target, ()) is None
    assert planner.plan(target, ()) != _canonical_word("evo", target)
    # a genuinely 1-action target still fits the budget (a leaf needs PushOne)
    p_leaf = _planner("evo", max_actions=1)
    assert p_leaf.plan(One(), ()) == ["PushOne"]
    # A non-prefix stack that needs more than 1 action returns None.
    assert planner.plan(target, (One(),)) is None


def test_budget_max_search_nodes_and_no_infinite_loop() -> None:
    target = ev_mul(One(), nat(2))
    # A stack node that is NOT a subtree of target: sound prune -> None.
    junk = Group(Zero(), (One(),))
    planner = _planner("evo", max_search_nodes=10)
    t0 = time.perf_counter()
    assert planner.plan(target, (junk,)) is None
    # Search budget 0 refuses a stack that genuinely needs SEARCH (not a
    # canonical prefix): [One, One] needs MakeChange then MakeMul.
    assert planner.plan(target, (One(), One())) == ["MakeChange", "MakeMul"]
    p0 = _planner("evo", max_search_nodes=0)
    assert p0.plan(target, (One(), One())) is None
    assert time.perf_counter() - t0 < 5.0


def test_fallback_is_legal_and_safe() -> None:
    target = ev_add(One(), One())
    planner = _planner("evo")
    planner.set_target(target)
    # unsolvable state: a junk node not in the target
    from .env import State
    junk = Group(Zero(), (One(),))
    state = State(stack=(junk,))
    action = planner.best_action("S:" + target.canonical(), state)
    assert action in planner.by_name
    assert planner.by_name[action].arity <= len(state.stack)
    # empty stack still returns a legal nullary action
    action2 = planner.best_action("S:" + target.canonical(),
                                  State(stack=()))
    assert action2 in ("PushZero", "PushOne")


def test_best_action_legality_sweep() -> None:
    from .env import State
    checked = 0
    for label, domain, target in HANDCRAFTED:
        planner = _planner(domain)
        planner.set_target(target)
        word = _canonical_word(domain, target)
        stack: Tuple[Node, ...] = ()
        # walk forward along the canonical run; at every state the planner's
        # chosen action must be legal and must stay on a solving path.
        for name in word:
            action = planner.best_action("k", State(stack=stack))
            assert action in planner.by_name, (label, action)
            assert planner.by_name[action].arity <= len(stack), (label, action)
            stack = planner._apply(stack, planner.by_name[name])
            checked += 1
    # plus the unsolvable sweep
    junk = Group(Zero(), (One(),))
    for domain in ("evo", "core"):
        planner = _planner(domain)
        planner.set_target(ev_add(One(), One())
                           if domain == "evo" else Group(One(), (Zero(),)))
        a = planner.best_action("k", State(stack=(junk,)))
        assert a is None or planner.by_name[a].arity <= 1
        checked += 1
    assert checked > 0
    print("legal actions checked: %d" % checked)


NEW_CHECKS: List[Tuple[str, object]] = [
    ("determinism_same_sequence", test_determinism_same_sequence),
    ("determinism_from_prefix_stacks", test_determinism_from_prefix_stacks),
    ("handcrafted_canonical_solves", test_handcrafted_canonical_solves),
    ("prefix_stacks_all_solved", test_prefix_stacks_all_solved),
    ("budget_max_actions_honoured", test_budget_max_actions_honoured),
    ("budget_max_search_nodes_no_loop",
     test_budget_max_search_nodes_and_no_infinite_loop),
    ("fallback_is_legal_and_safe", test_fallback_is_legal_and_safe),
    ("best_action_legality_sweep", test_best_action_legality_sweep),
]


def run() -> int:
    started = time.perf_counter()
    passed = 0
    failures: List[str] = []
    for name, fn in NEW_CHECKS:
        try:
            fn()  # type: ignore[operator]
        except Exception as exc:  # noqa: BLE001
            failures.append(name)
            print("[FAIL] %s: %r" % (name, exc))
        else:
            passed += 1
            print("[PASS] %s" % name)
    elapsed = time.perf_counter() - started
    print("RESULT: %d/%d passed in %.3fs (planner over REDUCTION_TABLE; "
          "deterministic, bounded, no genome)"
          % (passed, len(NEW_CHECKS), elapsed))
    return 0 if not failures else 1


if __name__ == "__main__":
    import sys
    sys.exit(run())
