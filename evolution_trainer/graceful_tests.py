"""Unit tests for graceful unsolvable handling (task 4349 unit 3).

Run::

    /opt/automath/venv/bin/python -m evolution_trainer.graceful_tests

Covers the flag contract, the pool filter (witness short-circuit + real
UNSOLVABLE exclusion + ``unsolvable_budget`` drop), the three-way evaluator and
the explicit-planner regression numbers (core33 32/33, unseen40 pop 40/40).
Evaluation only, stdlib plus the repo modules.
"""

from __future__ import annotations

import sys
from typing import Any, List, Tuple

from . import graceful as g


class CheckFailure(Exception):
    def __init__(self, name: str, detail: str) -> None:
        super().__init__(name + ": " + detail)
        self.name = name
        self.detail = detail


class _FakeStackCase:
    """A case the stack oracle sees but cannot build -> UNSOLVABLE."""

    def __init__(self, name: str, target: Any, domain: str = "evo") -> None:
        self.name = name
        self.target = target
        self.initial_stack = ()
        self.domain = domain


def check_flag_default_on_and_off_switch() -> str:
    assert g.resolve_flag(None) is True
    assert g.resolve_flag(True) is True
    assert g.resolve_flag("on") is True
    assert g.resolve_flag(False) is False
    assert g.resolve_flag("off") is False
    assert g.resolve_flag("no") is False
    return "default ON; off switch resolves False"


def check_filter_keeps_witness_without_oracle() -> str:
    cases = [case for _name, case in g.spec_cases(["spec_minimal"])]
    active, excluded, records = g.filter_unsolvable(cases, "current")
    assert len(active) == 1 and not excluded, (len(active), excluded)
    assert records["spec_minimal"]["proof_method"] == "provided_witness_replay"
    return "witness case kept, no oracle call, proof=provided_witness_replay"


def check_split_pool_excludes_provably_unsolvable() -> str:
    good = g.spec_cases(["spec_minimal"])[0]
    bad = ("bad_unbuildable", _FakeStackCase("bad_unbuildable", object()))
    split = g.split_pool([good, bad], "current", node_budget=500,
                         time_budget=0.2)
    assert "bad_unbuildable" in split["excluded"], split["excluded"]
    assert "spec_minimal" in split["active"], split["active"]
    return "UNSOLVABLE excluded, SOLVABLE active"


def check_training_run_on_off_accounting() -> str:
    good = g.spec_cases(["spec_minimal"])[0]
    bad = ("bad_unbuildable", _FakeStackCase("bad_unbuildable", object()))
    pool = [good, bad]
    off = g.training_run(pool, enabled=False, action_set="current",
                         node_budget=500, time_budget=0.2, generations=3)
    on = g.training_run(pool, enabled=True, action_set="current",
                        node_budget=500, time_budget=0.2, generations=3)
    assert off["episodes_attempted"] == 6, off["episodes_attempted"]
    assert on["episodes_attempted"] == 3, on["episodes_attempted"]
    assert on["episodes_skipped"] == 3, on["episodes_skipped"]
    assert on["static_excluded"] == 1, on["static_excluded"]
    assert off["static_excluded"] == 0, off["static_excluded"]
    return "OFF 6/6 attempted, ON 3/6 attempted (1 case excluded)"


def check_three_way_spec_solved() -> str:
    cases = g.spec_cases(["spec_minimal"])
    rep = g.three_way(cases, "current", node_budget=2000, time_budget=0.5)
    assert rep["SOLVABLE-SOLVED"] == 1, rep
    assert rep["SOLVABLE-UNSOLVED"] == 0, rep
    assert rep["PROVABLY-UNSOLVABLE"] == 0, rep
    assert "wall_secs" in rep["per_case"]["spec_minimal"]
    return "1 SOLVABLE-SOLVED, 0 failures, 0 provably-unsolvable"


def check_three_way_never_counts_unsolvable_as_failure() -> str:
    bad = ("bad_unbuildable", _FakeStackCase("bad_unbuildable", object()))
    rep = g.three_way([bad], "current", node_budget=500, time_budget=0.2)
    assert rep["PROVABLY-UNSOLVABLE"] == 1, rep
    assert rep["SOLVABLE-UNSOLVED"] == 0, rep
    assert rep["solvable_denominator"] == 0, rep
    return "provably-unsolvable is its own bucket, never a failure"


def check_planner_core33_unchanged() -> str:
    reg = g.planner_regression(["core33"], ["current"], node_budget=20000,
                               time_budget=1.5)
    got = reg["results"]["current"]["core33"]["solved"]
    assert got == 32, got
    return "core33 explicit planner 32/33 (unchanged)"


def check_planner_unseen40_pop() -> str:
    reg = g.planner_regression(["unseen40"], ["current", "pop"],
                               node_budget=20000, time_budget=1.5,
                               step_budget=40)
    current = reg["results"]["current"]["unseen40"]
    pop = reg["results"]["pop"]["unseen40"]
    assert current["solved"] == 0, current
    assert pop["solved"] == 40, pop
    assert pop["min_steps"] is not None, pop
    return "unseen40 planner 0/40 without pop, 40/40 with pop"


CHECKS = [
    ("flag_default_on_and_off_switch", check_flag_default_on_and_off_switch),
    ("filter_keeps_witness_without_oracle", check_filter_keeps_witness_without_oracle),
    ("split_pool_excludes_provably_unsolvable", check_split_pool_excludes_provably_unsolvable),
    ("training_run_on_off_accounting", check_training_run_on_off_accounting),
    ("three_way_spec_solved", check_three_way_spec_solved),
    ("three_way_never_counts_unsolvable_as_failure", check_three_way_never_counts_unsolvable_as_failure),
    ("planner_core33_unchanged", check_planner_core33_unchanged),
    ("planner_unseen40_pop", check_planner_unseen40_pop),
]


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
    print("RESULT: %d/%d passed (graceful unsolvable handling)" % (passed, len(CHECKS)))
    return 0 if not failures else 1


def test_flag_default_on_and_off_switch() -> None:
    check_flag_default_on_and_off_switch()


def test_filter_keeps_witness_without_oracle() -> None:
    check_filter_keeps_witness_without_oracle()


def test_split_pool_excludes_provably_unsolvable() -> None:
    check_split_pool_excludes_provably_unsolvable()


def test_training_run_on_off_accounting() -> None:
    check_training_run_on_off_accounting()


def test_three_way_spec_solved() -> None:
    check_three_way_spec_solved()


def test_three_way_never_counts_unsolvable_as_failure() -> None:
    check_three_way_never_counts_unsolvable_as_failure()


def test_planner_core33_unchanged() -> None:
    check_planner_core33_unchanged()


def test_planner_unseen40_pop() -> None:
    check_planner_unseen40_pop()


if __name__ == "__main__":
    sys.exit(run())
