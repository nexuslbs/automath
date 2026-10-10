"""Graceful unsolvable handling for training and evaluation (task 4349 unit 3).

Evaluation only: no torch, no training run.  Standard library plus the
``evolution_trainer.solvability`` oracle and the ``new_approach`` /
``dynamic_env`` worlds.

The flag ``graceful_unsolvable`` (CLI ``--graceful-unsolvable`` /
``--no-graceful-unsolvable``) defaults ON and has a documented OFF switch so a
before/after comparison is possible.

When ON:

* **training** - a pool is classified ONCE with the oracle *before* it is
  attempted.  Provably ``UNSOLVABLE`` cases are excluded.  ``UNKNOWN`` cases get
  a hard per-case step/time budget; once a case exhausts its budget it is
  recorded as ``unsolvable_budget`` and dropped from the active pool, so it is
  never re-attempted unboundedly.
* **evaluation** - the three-way report buckets every case as
  ``SOLVABLE-SOLVED`` / ``SOLVABLE-UNSOLVED`` (with a failure mode) /
  ``PROVABLY-UNSOLVABLE`` (with the proof method).  A provably-unsolvable case is
  NEVER counted as a failure and never gets a solver attempt, so it cannot burn
  the run.  ``UNKNOWN`` cases are honestly reported as ``SOLVABLE-UNSOLVED``
  with ``budget_exhausted`` as the failure mode (they are not proven unsolvable).

CLI examples::

    python -m evolution_trainer.graceful --mode train-pool \
        --family dense --graceful both --generations 2 --out out/graceful
    python -m evolution_trainer.graceful --mode three-way \
        --family core33,ext114,unseen40,val64 --action-set current
    python -m evolution_trainer.graceful --mode planner \
        --family core33,ext114,unseen40 --action-set both
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .solvability import (
    DEFAULT_NODE_BUDGET,
    DEFAULT_TIME_BUDGET,
    DYN_NODE_BUDGET,
    DYN_TIME_BUDGET,
    FAMILIES,
    POP_NODE_BUDGET,
    POP_TIME_BUDGET,
    SOLVABLE,
    UNKNOWN,
    UNSOLVABLE,
    _domain_of,
    _is_stack_case,
    classify,
)

GRACEFUL_UNSOLVABLE_DEFAULT = True
DEFAULT_CASE_STEP_BUDGET = 40
BUDGET_DROP = "unsolvable_budget"

SOLVED = "SOLVABLE-SOLVED"
SOLVED_UNSOLVED = "SOLVABLE-UNSOLVED"
PROVABLY = "PROVABLY-UNSOLVABLE"


def resolve_flag(enabled: Any) -> bool:
    """Coerce a flag (bool / \"on\"/\"off\" / None) to bool, default ON."""
    if enabled is None:
        return GRACEFUL_UNSOLVABLE_DEFAULT
    if isinstance(enabled, str):
        return enabled.strip().lower() not in ("0", "off", "false", "no", "n")
    return bool(enabled)


def action_budgets(action_set: str, node_budget: int, time_budget: float
                   ) -> Tuple[int, float]:
    """Per-case oracle budget for the action set (pop uses the tighter caps)."""
    if action_set == "pop":
        return int(node_budget), float(time_budget)
    return int(node_budget), float(time_budget)


def case_budgets(case: Any, action_set: str, node_budget: int = DEFAULT_NODE_BUDGET,
                 time_budget: float = DEFAULT_TIME_BUDGET,
                 dyn_node_budget: int = DYN_NODE_BUDGET,
                 dyn_time_budget: float = DYN_TIME_BUDGET
                 ) -> Tuple[int, float]:
    if not _is_stack_case(case):
        return int(dyn_node_budget), float(dyn_time_budget)
    if action_set == "pop":
        return min(int(node_budget), POP_NODE_BUDGET), min(
            float(time_budget), POP_TIME_BUDGET)
    return int(node_budget), float(time_budget)


def _stack_order(case: Any) -> Tuple[Tuple[str, ...], Dict[str, int]]:
    domain = _domain_of(case)
    if domain == "core":
        from new_approach.env import BUILD_ACTIONS
        actions = BUILD_ACTIONS
    else:
        from new_approach.evolution import EVO_ACTIONS
        actions = EVO_ACTIONS
    names = tuple(a.name for a in actions)
    arity = {a.name: int(a.arity) for a in actions}
    return names, arity


def _fallback_rollout(case: Any, step_budget: int, time_budget: float
                      ) -> Dict[str, Any]:
    """Bounded legal rollout for a pattern goal (no fixed target)."""
    t0 = time.perf_counter()
    env = case.env
    names, arity = _stack_order(case)
    state = env.reset()
    steps = 0
    solved = bool(env.goal_achieved(state))
    while not solved and steps < int(step_budget):
        if time.perf_counter() - t0 > float(time_budget):
            return {"solved": False, "steps": steps,
                    "failure_mode": "time_budget_exhausted",
                    "wall_secs": round(time.perf_counter() - t0, 6)}
        name = next((nm for nm in names if arity[nm] <= len(state.stack)), None)
        if name is None:
            break
        state = env.step(state, name)
        steps += 1
        solved = bool(env.goal_achieved(state))
    return {"solved": solved, "steps": steps,
            "failure_mode": None if solved else "step_limit",
            "wall_secs": round(time.perf_counter() - t0, 6)}


def attempt_case(case: Any, action_set: str = "current",
                 node_budget: int = DEFAULT_NODE_BUDGET,
                 time_budget: float = DEFAULT_TIME_BUDGET,
                 step_budget: int = DEFAULT_CASE_STEP_BUDGET,
                 dyn_node_budget: int = DYN_NODE_BUDGET,
                 dyn_time_budget: float = DYN_TIME_BUDGET) -> Dict[str, Any]:
    """One bounded planner attempt on ``case``.

    The oracle search is the explicit bounded planner (over the canonical
    reduction candidates, optionally with ``pop``).  A found witness is
    replay-verified by the oracle, so a ``SOLVABLE`` verdict means the attempt
    solved the case in ``min_steps``.  A pattern goal (no fixed target) uses the
    bounded fallback rollout.  ``UNKNOWN`` means the case is not proven
    unsolvable and the attempt did not solve it: the failure mode is the oracle
    proof method.
    """
    nb, tb = case_budgets(case, action_set, node_budget, time_budget,
                          dyn_node_budget, dyn_time_budget)
    t0 = time.perf_counter()
    res = classify(case, action_set, nb, tb)
    verdict = res["verdict"]
    method = res.get("proof_method")
    wall = round(time.perf_counter() - t0, 6)
    if method == "unadaptable_case" and _is_stack_case(case):
        fallback = _fallback_rollout(case, step_budget, tb)
        return {
            "verdict": SOLVABLE if fallback["solved"] else UNKNOWN,
            "proof_method": "fallback_rollout" if fallback["solved"]
            else fallback["failure_mode"],
            "solved": bool(fallback["solved"]),
            "steps": int(fallback["steps"]),
            "min_steps": int(fallback["steps"]) if fallback["solved"] else None,
            "failure_mode": None if fallback["solved"]
            else fallback["failure_mode"],
            "wall_secs": wall,
            "oracle_wall_secs": wall,
        }
    solved = verdict == SOLVABLE
    if solved and res.get("min_steps") is not None \
            and int(res["min_steps"]) > int(step_budget):
        return {
            "verdict": UNKNOWN, "proof_method": "step_limit",
            "solved": False, "steps": int(res["min_steps"]),
            "min_steps": int(res["min_steps"]), "failure_mode": "step_limit",
            "wall_secs": wall, "oracle_wall_secs": wall,
        }
    return {
        "verdict": verdict,
        "proof_method": method,
        "solved": solved,
        "steps": int(res["min_steps"]) if solved and res.get("min_steps") is not None
        else 0,
        "min_steps": res.get("min_steps"),
        "failure_mode": None if solved else method,
        "wall_secs": wall,
        "oracle_wall_secs": wall,
    }


# --------------------------------------------------------------------------
# Training: graceful pool filter + bounded measurement
# --------------------------------------------------------------------------

def split_pool(cases: Sequence[Tuple[str, Any]], action_set: str = "current",
               node_budget: int = DEFAULT_NODE_BUDGET,
               time_budget: float = DEFAULT_TIME_BUDGET) -> Dict[str, Any]:
    """Classify a pool ONCE and split it into active vs provably-unsolvable."""
    records: Dict[str, Dict[str, Any]] = {}
    excluded: List[str] = []
    unknown: List[str] = []
    for name, case in cases:
        res = attempt_case(case, action_set, node_budget, time_budget)
        records[name] = {"name": name, **res}
        if res["verdict"] == UNSOLVABLE:
            excluded.append(name)
        elif res["verdict"] == UNKNOWN:
            unknown.append(name)
    active = [name for name, _ in cases if name not in set(excluded)]
    return {"records": records, "excluded": excluded, "unknown": unknown,
            "active": active, "n": len(cases)}


def filter_unsolvable(cases: Sequence[Any], action_set: str = "current",
                      node_budget: int = DEFAULT_NODE_BUDGET,
                      time_budget: float = DEFAULT_TIME_BUDGET
                      ) -> Tuple[List[Any], List[str], Dict[str, Any]]:
    """Drop provably-UNSOLVABLE cases from a pool, keeping ``UNKNOWN`` ones.

    A case carrying a ``witness`` (e.g. a ``ValCase`` built by
    ``size_selection``, whose witness was replay-proven at construction) is
    trivially SOLVABLE and is kept without an oracle call.  Returns
    ``(active, excluded_names, records)``.
    """
    active: List[Any] = []
    excluded: List[str] = []
    records: Dict[str, Any] = {}
    for case in cases:
        name = getattr(case, "name", repr(case))
        if getattr(case, "witness", None) is not None:
            records[name] = {"verdict": SOLVABLE,
                             "proof_method": "provided_witness_replay",
                             "solved": True, "wall_secs": 0.0}
            active.append(case)
            continue
        res = attempt_case(case, action_set, node_budget, time_budget)
        records[name] = res
        if res["verdict"] == UNSOLVABLE:
            excluded.append(name)
        else:
            active.append(case)
    return active, excluded, records


def training_run(cases: Sequence[Tuple[str, Any]], enabled: Any = None,
                 action_set: str = "current",
                 node_budget: int = DEFAULT_NODE_BUDGET,
                 time_budget: float = DEFAULT_TIME_BUDGET,
                 generations: int = 2,
                 step_budget: int = DEFAULT_CASE_STEP_BUDGET
                 ) -> Dict[str, Any]:
    """The SAME bounded pool stage with graceful handling ON vs OFF.

    OFF: every case is attempted every generation.
    ON: classify once; exclude ``UNSOLVABLE``; drop a case the first time its
    hard budget is exhausted (``unsolvable_budget``); attempt the rest.
    Returns wall seconds, episodes attempted/skipped and the decomposition.
    """
    on = resolve_flag(enabled)
    t0 = time.perf_counter()
    classifications: Dict[str, Dict[str, Any]] = {}
    excluded: set = set()
    unknown: set = set()
    if on:
        split = split_pool(cases, action_set, node_budget, time_budget)
        classifications = split["records"]
        excluded = set(split["excluded"])
        unknown = set(split["unknown"])
    dropped: set = set()
    attempted = 0
    solved = 0
    per_case: Dict[str, Dict[str, Any]] = {}
    generations = max(1, int(generations))
    for _gen in range(generations):
        for name, case in cases:
            if on and name in excluded:
                continue
            if on and name in dropped:
                continue
            attempted += 1
            res = attempt_case(case, action_set, node_budget, time_budget,
                               step_budget)
            per_case[name] = res
            if res["solved"]:
                solved += 1
            elif on and res["verdict"] == UNKNOWN:
                dropped.add(name)
    wall = round(time.perf_counter() - t0, 6)
    total = len(cases) * generations
    return {
        "graceful_unsolvable": on,
        "action_set": action_set,
        "generations": generations,
        "n_cases": len(cases),
        "n_episodes": total,
        "episodes_attempted": attempted,
        "episodes_skipped": total - attempted,
        "episodes_solved": solved,
        "static_excluded": len(excluded),
        "static_excluded_fraction": round(len(excluded) / max(1, len(cases)), 6),
        "unknown_at_entry": len(unknown),
        "budget_dropped": sorted(dropped),
        "budget_dropped_count": len(dropped),
        "wall_seconds": wall,
        "classification_wall": (round(sum(
            classifications[n]["wall_secs"] for n in classifications), 6)
            if classifications else None),
        "per_case": per_case,
    }


# --------------------------------------------------------------------------
# Evaluation: the standard three-way report
# --------------------------------------------------------------------------

def three_way(cases: Sequence[Tuple[str, Any]], action_set: str = "current",
              node_budget: int = DEFAULT_NODE_BUDGET,
              time_budget: float = DEFAULT_TIME_BUDGET,
              step_budget: int = DEFAULT_CASE_STEP_BUDGET) -> Dict[str, Any]:
    """SOLVABLE-SOLVED / SOLVABLE-UNSOLVED / PROVABLY-UNSOLVABLE report.

    Provably-unsolvable cases are classified (proof) but NEVER attempted by the
    solver, so they cannot burn the run; they are excluded from the denominator.
    ``UNKNOWN`` cases are ``SOLVABLE-UNSOLVED`` with failure mode
    ``budget_exhausted`` (honest: not proven unsolvable).
    """
    per_case: Dict[str, Dict[str, Any]] = {}
    solved = unsolved = provably = 0
    failures: Dict[str, int] = {}
    proofs: Dict[str, int] = {}
    total_wall = 0.0
    for name, case in cases:
        t0 = time.perf_counter()
        gt = classify(case, action_set, *case_budgets(
            case, action_set, node_budget, time_budget))
        proof = gt.get("proof_method")
        if gt["verdict"] == UNSOLVABLE:
            bucket = PROVABLY
            provably += 1
            proofs[proof] = proofs.get(proof, 0) + 1
            res = {"verdict": gt["verdict"], "proof_method": proof,
                   "solved": False, "steps": None, "failure_mode": None}
        else:
            res = attempt_case(case, action_set, node_budget, time_budget,
                               step_budget)
            if res["solved"]:
                bucket = SOLVED
                solved += 1
            else:
                bucket = SOLVED_UNSOLVED
                unsolved += 1
                mode = ("budget_exhausted"
                        if res["verdict"] == UNKNOWN else
                        (res.get("failure_mode") or "no_plan"))
                failures[mode] = failures.get(mode, 0) + 1
                res["failure_mode"] = mode
        wall = round(time.perf_counter() - t0, 6)
        total_wall += wall
        res.update({"bucket": bucket, "wall_secs": wall})
        per_case[name] = res
    return {
        "action_set": action_set,
        "n": len(cases),
        "SOLVABLE-SOLVED": solved,
        "SOLVABLE-UNSOLVED": unsolved,
        "PROVABLY-UNSOLVABLE": provably,
        "solvable_denominator": solved + unsolved,
        "solve_rate_over_solvable": round(solved / max(1, solved + unsolved), 6),
        "failure_modes": failures,
        "proof_methods": proofs,
        "total_wall_secs": round(total_wall, 6),
        "per_case": per_case,
    }


# --------------------------------------------------------------------------
# Planner regression (explicit planner, current vs pop)
# --------------------------------------------------------------------------

def planner_regression(families: Sequence[str], action_sets: Sequence[str],
                       node_budget: int = DEFAULT_NODE_BUDGET,
                       time_budget: float = DEFAULT_TIME_BUDGET,
                       step_budget: int = DEFAULT_CASE_STEP_BUDGET
                       ) -> Dict[str, Any]:
    """Re-run the explicit planner on the named families for each action set."""
    out: Dict[str, Any] = {"families": list(families), "results": {}}
    for action_set in action_sets:
        block: Dict[str, Any] = {}
        for family in families:
            cases = FAMILIES[family]()
            solved = 0
            steps: List[int] = []
            per_form: Dict[str, bool] = {}
            wall0 = time.perf_counter()
            for name, case in cases:
                res = attempt_case(case, action_set, node_budget, time_budget,
                                   step_budget)
                per_form[name] = bool(res["solved"])
                if res["solved"]:
                    solved += 1
                    if res["steps"]:
                        steps.append(int(res["steps"]))
            block[family] = {
                "solved": solved,
                "total": len(cases),
                "min_steps": min(steps) if steps else None,
                "median_steps": (sorted(steps)[len(steps) // 2]
                                 if steps else None),
                "wall_secs": round(time.perf_counter() - wall0, 6),
                "per_form": per_form,
            }
        out["results"][action_set] = block
    return out


# --------------------------------------------------------------------------
# Spec cases (dynamic_env specs with a proven witness)
# --------------------------------------------------------------------------

def spec_cases(spec_ids: Sequence[str]) -> List[Tuple[str, Any]]:
    """Wrap shipped specs as solvable dynamic cases (reset start + BFS witness)."""
    from dynamic_env.engine import bfs_minimal_word
    from .size_selection import ValCase, spec_by_id

    out: List[Tuple[str, Any]] = []
    for spec_id in spec_ids:
        spec = spec_by_id(spec_id)
        from dynamic_env.engine import DynamicEnv
        env = DynamicEnv(spec)
        start = env.reset()
        word = bfs_minimal_word(spec)
        witness = tuple(word or ())
        out.append((spec_id, ValCase(
            name=spec_id, domain="prefix", spec_id=spec_id, start_state=start,
            witness=witness, max_steps=int(spec.max_steps), spec=spec)))
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _families(arg: str) -> List[str]:
    if arg == "all":
        return list(FAMILIES)
    return [f.strip() for f in arg.split(",") if f.strip()]


def _emit(payload: Dict[str, Any], out: Optional[str]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print("WROTE %s" % os.path.abspath(out))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="graceful unsolvable handling (training pool + eval)")
    parser.add_argument("--mode", default="three-way",
                        choices=["train-pool", "three-way", "planner", "specs"])
    parser.add_argument("--family", default="dense")
    parser.add_argument("--graceful", default="both", choices=["on", "off", "both"])
    parser.add_argument("--action-set", default="current",
                        choices=["current", "pop", "both"])
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--node-budget", type=int, default=DEFAULT_NODE_BUDGET)
    parser.add_argument("--time-budget", type=float, default=DEFAULT_TIME_BUDGET)
    parser.add_argument("--step-budget", type=int, default=DEFAULT_CASE_STEP_BUDGET)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    families = _families(args.family)
    action_sets = (["current", "pop"] if args.action_set == "both"
                   else [args.action_set])

    if args.mode == "specs":
        all_cases = spec_cases(families)
        payload: Dict[str, Any] = {
            "mode": "specs", "families": families,
            "three_way": three_way(all_cases, action_sets[0], args.node_budget,
                                   args.time_budget, args.step_budget)}
        _emit(payload, args.out or None)
        return 0

    if args.mode == "train-pool":
        cases: List[Tuple[str, Any]] = []
        for family in families:
            cases.extend(FAMILIES[family]())
        if args.limit:
            cases = cases[: args.limit]
        flags = (["on", "off"] if args.graceful == "both" else [args.graceful])
        blocks = {}
        for flag in flags:
            for action_set in action_sets:
                key = "graceful=%s action_set=%s" % (flag, action_set)
                blocks[key] = training_run(
                    cases, enabled=(flag == "on"), action_set=action_set,
                    node_budget=args.node_budget, time_budget=args.time_budget,
                    generations=args.generations, step_budget=args.step_budget)
        payload = {"mode": "train-pool", "families": families,
                   "n_cases": len(cases), "blocks": blocks}
        _emit(payload, args.out or None)
        return 0

    if args.mode == "planner":
        payload = {"mode": "planner",
                   "planner": planner_regression(
                       families, action_sets, args.node_budget,
                       args.time_budget, args.step_budget)}
        _emit(payload, args.out or None)
        return 0

    # three-way
    payload = {"mode": "three-way", "families": families, "blocks": {}}
    for family in families:
        cases = FAMILIES[family]()
        for action_set in action_sets:
            key = "family=%s action_set=%s" % (family, action_set)
            payload["blocks"][key] = three_way(
                cases, action_set, args.node_budget, args.time_budget,
                args.step_budget)
    _emit(payload, args.out or None)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
