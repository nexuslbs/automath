"""Unit D-1 curriculum checks. Standard library only, like the rest of the work.

Run with::

    /opt/automath/venv/bin/python -m evolution_trainer.curriculum_tests
    /opt/automath/venv/bin/python -m pytest -q evolution_trainer/curriculum_tests.py
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Tuple

from dynamic_env.engine import DynamicEnv, bfs_minimal_word, minimal_length
from dynamic_env.spec import load_spec

from .curriculum import (
    bfs_min_steps,
    minimal_and_max,
    run_evolution,
    shortest_path_gate,
    solution_path,
    spec_path,
)
from .heldout import HELDOUT_CASES, _HELDOUT_DIR, evaluate_case


class CheckFailure(Exception):
    def __init__(self, name: str, detail: str) -> None:
        super().__init__(detail)
        self.name = name
        self.detail = detail


SHIPPED = ["spec_minimal", "spec_dynamic_group", "spec_multi_step",
           "spec_dynamic_axiom"]


def _load(spec_id: str):
    return load_spec(spec_path(spec_id))


def check_bfs_matches_engine() -> str:
    seen = []
    for spec_id in SHIPPED:
        spec = _load(spec_id)
        mine, word = bfs_min_steps(spec)
        theirs = minimal_length(spec)
        if mine != theirs:
            raise CheckFailure("bfs_matches_engine",
                               "%s: runner BFS %r != engine %r" % (spec_id, mine, theirs))
        if word is None or len(word) != mine:
            raise CheckFailure("bfs_matches_engine",
                               "%s: witness %r length != %r" % (spec_id, word, mine))
        if word != bfs_minimal_word(spec):
            raise CheckFailure("bfs_matches_engine",
                               "%s: witness %r != engine word %r"
                               % (spec_id, word, bfs_minimal_word(spec)))
        seen.append("%s=%d" % (spec_id, mine))
    return "runner BFS == engine minimal_length for " + ", ".join(seen)


def check_max_rule() -> str:
    multi = minimal_and_max(_load("spec_multi_step"))
    axiom = minimal_and_max(_load("spec_dynamic_axiom"))
    if multi["max_steps"] != multi["min_steps"] + 2:
        raise CheckFailure("max_rule", "multi_step max wrong: %r" % multi)
    if axiom["max_steps"] != max(axiom["min_steps"] + 2, axiom["min_steps"] * 2):
        raise CheckFailure("max_rule", "axiom max wrong: %r" % axiom)
    return "multi_step %r; axiom %r" % (multi, axiom)


def check_gate_single_correct_step() -> str:
    spec = _load("spec_dynamic_group")
    path = solution_path(spec)
    minimum, word = path["min_steps"], path["word"]
    env = DynamicEnv(spec)
    start = env.reset()
    gate = shortest_path_gate(spec, path["correct"])
    accepts = [a.key() for a in env.legal_actions(start) if gate(env, start, a)]
    if len(accepts) != 1:
        raise CheckFailure("gate_single_correct_step",
                           "expected exactly 1 correct action, got %r" % accepts)
    if accepts[0] != word[0]:
        raise CheckFailure("gate_single_correct_step",
                           "gate accepts %r but BFS word starts %r"
                           % (accepts[0], word[0]))
    if len(path["correct"]) != minimum:
        raise CheckFailure("gate_single_correct_step",
                           "canonical states %d != min %d"
                           % (len(path["correct"]), minimum))
    return "exactly one correct action at the start: %r (min=%d)" % (accepts, minimum)


def check_strict_runs_solve() -> str:
    spec = _load("spec_minimal")
    minimum, word = bfs_min_steps(spec)
    result = run_evolution(spec, "test_strict", "/tmp/curriculum_tests",
                           generations=3, population=6, seed=7,
                           strict_keys=word, subagent_depth=0)
    if result["best_fitness"] <= 0 or not any(result["solved_generations"]):
        raise CheckFailure("strict_runs_solve", "strict run did not solve: %r" % result)
    return ("strict spec_minimal solved; best_fitness=%.3f solved_gens=%r in %.3fs"
            % (result["best_fitness"], result["solved_generations"],
               result["elapsed_s"]))


def check_free_discard_recovers() -> str:
    spec = _load("spec_dynamic_group")
    minimum, word = bfs_min_steps(spec)
    path = solution_path(spec)
    gate = shortest_path_gate(spec, path["correct"])
    strict = run_evolution(spec, "test_s3_strict", "/tmp/curriculum_tests",
                           generations=4, population=8, seed=7,
                           strict_keys=word, subagent_depth=0)
    free = run_evolution(spec, "test_s3_free", "/tmp/curriculum_tests",
                         generations=12, population=8, seed=7, gate=gate,
                         subagent_depth=1)
    if free["best_fitness"] <= strict["best_fitness"] - 1.0:
        # The free run must at least reach the same shaped reward ceiling.
        raise CheckFailure("free_discard_recovers",
                           "free fitness %r did not recover toward strict %r"
                           % (free["best_fitness"], strict["best_fitness"]))
    return ("strict best=%.3f, free+discard best=%.3f after %d gens"
            % (strict["best_fitness"], free["best_fitness"], free["generations"]))


def check_heldout_specs() -> str:
    if len(HELDOUT_CASES) < 2:
        raise CheckFailure("heldout_specs", "expected >= 2 held-out cases")
    details = []
    for case_id in HELDOUT_CASES:
        report = evaluate_case(None, case_id, episodes=1, seed=3)
        if report["min_steps"] is None or report["max_steps"] <= report["min_steps"]:
            raise CheckFailure("heldout_specs", "%s has no plan: %r" % (case_id, report))
        if not report["trace"]:
            raise CheckFailure("heldout_specs", "%s produced no trace" % case_id)
        item = report["trace"][0]
        for key in ("step", "action", "objective_vector", "reward", "goal"):
            if key not in item:
                raise CheckFailure("heldout_specs",
                                   "%s trace missing %r" % (case_id, key))
        details.append("%s min=%d max=%d trace_steps=%d"
                       % (case_id, report["min_steps"], report["max_steps"],
                          len(report["trace"])))
    return "; ".join(details)


def check_heldout_deep_arity() -> str:
    import os

    spec = load_spec(os.path.join(_HELDOUT_DIR, "spec_dynamic_group_deep.json"))
    if spec.max_combine_arity < 3:
        raise CheckFailure("heldout_deep_arity",
                           "expected arity >= 3, got %r" % spec.max_combine_arity)
    minimum, word = bfs_min_steps(spec)
    if minimum is None or minimum < 2 or minimum > 3:
        raise CheckFailure("heldout_deep_arity",
                           "deep min %r not in [2,3] (word %r)" % (minimum, word))
    if not word or not word[0].startswith("combine:"):
        raise CheckFailure("heldout_deep_arity",
                           "deep word does not start with combine: %r" % (word,))
    return "deep dynamic_group arity=%d min=%d word=%r" % (
        spec.max_combine_arity, minimum, word)


CHECKS: List[Tuple[str, Callable[[], str]]] = [
    ("bfs_matches_engine", check_bfs_matches_engine),
    ("max_rule", check_max_rule),
    ("gate_single_correct_step", check_gate_single_correct_step),
    ("strict_runs_solve", check_strict_runs_solve),
    ("free_discard_recovers", check_free_discard_recovers),
    ("heldout_specs", check_heldout_specs),
    ("heldout_deep_arity", check_heldout_deep_arity),
]


def run_checks() -> int:
    passed = 0
    started = time.perf_counter()
    for name, fn in CHECKS:
        try:
            detail = fn()
        except CheckFailure as exc:
            print("FAIL %s: %s" % (exc.name, exc.detail))
        except Exception as exc:  # pragma: no cover
            print("ERROR %s: %s: %s" % (name, type(exc).__name__, exc))
        else:
            passed += 1
            print("PASS %s: %s" % (name, detail))
    total = len(CHECKS)
    print("RESULT: %d/%d passed in %.3fs" % (passed, total, time.perf_counter() - started))
    return 0 if passed == total else 1


# pytest mirrors
def test_bfs_matches_engine() -> None:
    check_bfs_matches_engine()


def test_max_rule() -> None:
    check_max_rule()


def test_gate_single_correct_step() -> None:
    check_gate_single_correct_step()


def test_strict_runs_solve() -> None:
    check_strict_runs_solve()


def test_free_discard_recovers() -> None:
    check_free_discard_recovers()


def test_heldout_specs() -> None:
    check_heldout_specs()


def test_heldout_deep_arity() -> None:
    check_heldout_deep_arity()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run_checks())
