"""Pattern/ExprGoal planner suite (branch ``planner-pattern-goals``).

Exercises ``Planner.plan_pattern`` - the goal-directed pattern planner - on a
concrete family of ``ExprGoal`` patterns grounded in the fork's own formalism.
Every case's solvability is established INDEPENDENTLY by the fork's BFS oracle
``env.bfs_plan(env, max_actions=8)``; the BFS-optimal action count is recorded
and a case is included only when the oracle returns a word (the single negative
control is the documented exception).

Formalism quoted from the fork (``new_approach/env.py``)::

    class ExprGoal:
        \"\"\"A dynamic/pattern goal: a predicate expression over the current node.\"\"\"

        def __init__(self, build: Callable[[Node], Node], axioms: AxiomSet,
                     name: str = "pattern") -> None:
            self.build = build
            self.axioms = axioms
            self.name = name

        def achieved(self, state: State) -> bool:
            if len(state.stack) != 1:
                return False
            try:
                expr = self.build(state.stack[0])
                return self.axioms.truthy(expr)
            except Exception:
                return False

Axiom evaluator predicates used (``new_approach/axioms.py``)::

    def _rule_eq(ev, args):
        return ev(args[0]) == ev(args[1])

    def _truthy(v: Value) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, int):
            return v != 0
        ...
        return bool(v)

and the expression builder (``new_approach/expressions.py``)::

    def eq(a: Node, b: Node) -> Group:
        return app(T_EQ, a, b)

Families
--------
* ``expr_two``   : ``ExprGoal(lambda cur: eq(cur, nat(2)), AX, name="value==2")``
                   with the FULL existing ``expr_two/prefix*`` start family from
                   ``u2.validation_cases`` (including the historic failure
                   ``expr_two/prefix1``) plus two fresh non-prefix starts.
* ``expr_three`` : ``eq(cur, nat(3))`` prefix starts + two fresh non-prefix
                   starts.
* ``expr_one``   : ``eq(cur, nat(1))`` prefix starts + two fresh non-prefix
                   starts.
* ``expr_99``    : NEGATIVE CONTROL, ``eq(cur, nat(99))`` - unsatisfiable
                   within the explicit budget; must be reported UNSOLVED with
                   the legal fallback only (never crash, never illegal).

The JSON output is fully deterministic (no wall-clock values) so two runs are
byte-identical; wall time is reported in the human-readable trace file and on
stdout.  Pure standard library, no torch, no training.

CLI::

    python -m new_approach.pattern_suite \
        --out-json  /opt/automath/tmp/pattern-goals/pattern_suite.json \
        --out-trace /opt/automath/tmp/pattern-goals/pattern_suite.trace.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .axioms import AxiomSet
from .env import BUILD_ACTIONS, ExprGoal, MinimalEnv, bfs_plan, simulate
from .expressions import eq, nat
from .nodes import Change, Node, One, Zero
from .planner import (
    DEFAULT_MAX_ACTIONS,
    DEFAULT_MAX_SEARCH_NODES,
    Planner,
    planner_rollout,
)

AX = AxiomSet()

#: Explicit bounds, identical in shape to ``planner.py``'s constants.
BFS_ORACLE_MAX_ACTIONS = 8


@dataclass
class PatternCase:
    name: str
    kind: str                       # prefix | fresh | negative
    pattern: str
    env: MinimalEnv
    max_steps: int
    bfs_word: Optional[List[str]]   # None only for the negative control
    bfs_optimal_actions: Optional[int]
    level: str = "pattern"


def _pattern(value: int) -> ExprGoal:
    """``eq(cur, nat(value))`` grounded in the fork's own ExprGoal formalism."""
    return ExprGoal(lambda cur: eq(cur, nat(value)), AX, name="value==%d" % value)


def _stack_canon(stack: Sequence[Node]) -> str:
    return "[" + ", ".join(n.canonical() for n in stack) + "]"


def _prefix_stacks(word: Sequence[str]) -> List[Tuple[str, Tuple[Node, ...]]]:
    """Every non-trivial prefix of the BFS word, as (label, stack)."""
    out: List[Tuple[str, Tuple[Node, ...]]] = []
    for i in range(1, len(word) + 1):
        stack, err = simulate((), word[:i])
        if err is not None:
            raise RuntimeError("prefix simulation failed: " + err)
        out.append(("prefix%d" % i, stack))
    return out


def _oracle(goal: ExprGoal, start: Sequence[Node]) -> Optional[List[str]]:
    """The fork's own BFS oracle, used to establish solvability independently."""
    env = MinimalEnv(goal, axioms=AX, initial_stack=tuple(start),
                     max_actions=BFS_ORACLE_MAX_ACTIONS)
    return bfs_plan(env, max_actions=BFS_ORACLE_MAX_ACTIONS)


def _add(cases: List[PatternCase], name: str, kind: str, goal: ExprGoal,
         start: Sequence[Node], require_oracle: bool = True) -> bool:
    word = _oracle(goal, start)
    if word is None and require_oracle:
        return False
    max_steps = (len(word) + 2) if word is not None else 6
    env = MinimalEnv(goal, axioms=AX, initial_stack=tuple(start),
                     max_actions=BFS_ORACLE_MAX_ACTIONS)
    cases.append(PatternCase(
        name=name, kind=kind, pattern=goal.name, env=env, max_steps=max_steps,
        bfs_word=None if word is None else list(word),
        bfs_optimal_actions=None if word is None else len(word)))
    return True


def _family(cases: List[PatternCase], family: str, value: int,
            fresh: Sequence[Sequence[Node]]) -> None:
    goal = _pattern(value)
    empty = _oracle(goal, ())
    if empty is None:
        raise RuntimeError("pattern %s has no BFS plan from empty" % family)
    for label, stack in _prefix_stacks(empty):
        _add(cases, "%s/%s" % (family, label), "prefix", goal, stack)
    for i, stack in enumerate(fresh):
        _add(cases, "%s/fresh%d" % (family, i + 1), "fresh", goal, stack)


def build_cases() -> List[PatternCase]:
    cases: List[PatternCase] = []
    # (a) canonical expr_two with the FULL existing prefix family.
    _family(cases, "expr_two", 2, [(Zero(),), (Zero(), Zero())])
    # (b) two more pattern predicates from the fork's axiom set, each with its
    #     own prefix family and at least two fresh NON-prefix starts.
    _family(cases, "expr_three", 3, [(Zero(),), (Zero(), One())])
    _family(cases, "expr_one", 1, [(Zero(),), (Zero(), Zero())])
    # (c) negative control: unsatisfiable within budget (no BFS word).  Included
    #     even though the oracle returns None.
    _add(cases, "expr_99/control", "negative", _pattern(99), (),
         require_oracle=False)
    return cases


# --------------------------------------------------------------------------
# Per-case rollout (same procedure as planner.planner_rollout) + trace capture
# --------------------------------------------------------------------------

def run_case(planner: Planner, case: PatternCase) -> dict:
    env = case.env
    goal = env.goal
    state = env.reset()
    trace = [state.canonical()]
    actions: List[str] = []
    n = 0
    t0 = time.perf_counter()
    plan: List[str] = []
    if planner.pattern_goals:
        depth = min(planner.max_actions, case.max_steps)
        plan = planner.plan_pattern(env, state.stack, depth_limit=depth) or []
    fallback_used = False
    for name in plan:
        if n >= case.max_steps:
            break
        state = env.step(state, name)
        n += 1
        actions.append(name)
        trace.append(state.canonical())
        if env.goal_achieved(state):
            break
    # Safe legal fallback for any remaining budget.
    while n < case.max_steps and not env.goal_achieved(state):
        name = planner.fallback_action(state.stack)
        if name is None:
            break
        fallback_used = True
        state = env.step(state, name)
        n += 1
        actions.append(name)
        trace.append(state.canonical())
    wall = time.perf_counter() - t0
    done = bool(env.goal_achieved(state))
    return {
        "name": case.name,
        "kind": case.kind,
        "pattern": case.pattern,
        "start": _stack_canon(case.env.initial_stack),
        "max_steps": case.max_steps,
        "bfs_word": case.bfs_word,
        "bfs_optimal_actions": case.bfs_optimal_actions,
        "searched_plan": list(plan),
        "actions": actions,
        "trace": trace,
        "steps": n,
        "solved": done,
        "fallback_used": fallback_used,
        "wall_secs": round(wall, 6),
    }


def run_suite(max_search_nodes: int = DEFAULT_MAX_SEARCH_NODES,
              max_actions: int = DEFAULT_MAX_ACTIONS,
              pattern_goals: bool = True):
    planner = Planner(BUILD_ACTIONS, domain="core",
                      max_search_nodes=max_search_nodes,
                      max_actions=max_actions, pattern_goals=pattern_goals)
    cases = build_cases()
    # Cross-check: the aggregate must equal planner_rollout on the same cases,
    # so the per-case trace is the same procedure the held-out driver uses.
    aggregate = planner_rollout(planner, cases)
    details = [run_case(planner, c) for c in cases]
    solved = sum(1 for d in details if d["solved"])
    if solved != aggregate["solved"] or len(cases) != aggregate["total"]:
        raise RuntimeError(
            "suite/rollout mismatch: suite=%d/%d rollout=%d/%d"
            % (solved, len(cases), aggregate["solved"], aggregate["total"]))
    return cases, details, aggregate


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def _det_record(detail: dict) -> dict:
    """Deterministic projection (everything except wall_secs)."""
    return {k: detail[k] for k in (
        "name", "kind", "pattern", "start", "max_steps", "bfs_word",
        "bfs_optimal_actions", "searched_plan", "actions", "trace", "steps",
        "solved", "fallback_used")}


def render_trace(details: Sequence[dict], aggregate: dict,
                 max_search_nodes: int, max_actions: int,
                 pattern_goals: bool) -> str:
    lines: List[str] = []
    lines.append("PATTERN_SUITE pattern_goals=%s max_search_nodes=%d "
                 "max_actions=%d" % ("ON" if pattern_goals else "OFF",
                                     max_search_nodes, max_actions))
    for d in details:
        bfs = ("none" if d["bfs_optimal_actions"] is None
               else "%d %s" % (d["bfs_optimal_actions"], d["bfs_word"]))
        lines.append("[CASE] %s kind=%s pattern=%s" % (
            d["name"], d["kind"], d["pattern"]))
        lines.append("  start=%s max_steps=%d" % (d["start"], d["max_steps"]))
        lines.append("  bfs_oracle: %s" % bfs)
        lines.append("  searched_plan=%s" % (d["searched_plan"],))
        lines.append("  actions=%s steps=%d solved=%s fallback=%s "
                     "wall=%.6fs" % (d["actions"], d["steps"], d["solved"],
                                     d["fallback_used"], d["wall_secs"]))
        lines.append("  trace: %s" % " -> ".join(d["trace"]))
    lines.append("SUMMARY solved=%d/%d wall=%.3fs" % (
        aggregate["solved"], aggregate["total"], aggregate["wall_secs"]))
    lines.append("DETERMINISM: JSON contains no wall-clock values; sha256 of "
                 "two runs is identical.")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Pattern/ExprGoal planner suite (deterministic, no torch)")
    p.add_argument("--out-json",
                   default="/opt/automath/tmp/pattern-goals/pattern_suite.json")
    p.add_argument("--out-trace",
                   default="/opt/automath/tmp/pattern-goals/"
                           "pattern_suite.trace.txt")
    p.add_argument("--label", default="pattern-goals")
    p.add_argument("--max-search-nodes", type=int,
                   default=DEFAULT_MAX_SEARCH_NODES)
    p.add_argument("--max-actions", type=int, default=DEFAULT_MAX_ACTIONS)
    p.add_argument("--no-pattern-goals", dest="pattern_goals",
                   action="store_false", default=True)
    args = p.parse_args(argv)

    cases, details, aggregate = run_suite(
        max_search_nodes=args.max_search_nodes, max_actions=args.max_actions,
        pattern_goals=args.pattern_goals)

    res = {
        "label": args.label,
        "suite": "pattern/ExprGoal goal-directed planner",
        "bfs_oracle_max_actions": BFS_ORACLE_MAX_ACTIONS,
        "max_search_nodes": args.max_search_nodes,
        "max_actions": args.max_actions,
        "pattern_goals": bool(args.pattern_goals),
        "solved": aggregate["solved"],
        "total": aggregate["total"],
        "mean_actions_solved": aggregate["mean_actions_solved"],
        "cases": [_det_record(d) for d in details],
    }
    for dest in (args.out_json, args.out_trace):
        parent = os.path.dirname(dest)
        if parent:
            os.makedirs(parent, exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True)
        fh.write("\n")
    trace = render_trace(details, aggregate, args.max_search_nodes,
                         args.max_actions, args.pattern_goals)
    with open(args.out_trace, "w") as fh:
        fh.write(trace)
    sys.stdout.write(trace)
    print("WROTE %s" % args.out_json)
    print("WROTE %s" % args.out_trace)
    return 0


if __name__ == "__main__":
    sys.exit(main())
