"""Decidable solvability oracle for the automath action spaces (evaluation only).

Public API
----------
``classify(case, action_set="current", node_budget=..., time_budget=...)`` ->
``{verdict, proof_method, witness_path, min_steps, nodes_expanded, budget_used}``

Verdicts
--------
``SOLVABLE``   - a witness action path was found AND re-executed to the goal.
``UNSOLVABLE`` - SOUND: either the canonical non-subtree prune fires (no pop
                 action exists, so a stack node that is not a subtree of the
                 fixed target can never be consumed), or a bounded exhaustive
                 search closed the reachable space without a plan.
``UNKNOWN``    - the node or time budget was exhausted before a plan or a
                 closure proof. A budget timeout is NEVER reported as
                 UNSOLVABLE.

Soundness argument (stack machine)
----------------------------------
Each build action consumes ``arity`` top nodes and pushes one constructed node;
the fixed target is reached iff the final stack is exactly ``(target,)``.
* With no pop/discard action every node that ever appears on a run ending in
  ``(target,)`` must be a canonical subtree of ``target``, and the action that
  builds such a node must be one of the canonical post-order build actions.
  Restricting the candidate set to the distinct canonical build actions is
  therefore SOUND and COMPLETE, and a solution has length <= len(canonical
  word); an exhausted search over that finite closed space is a proof of
  unsolvability.
* With a ``pop`` action the prune is NOT sound (pop removes the offending
  node); the candidate set stays sound+complete (a minimal solution pops at
  most the whole initial stack and then replays the canonical word, so <=
  len(stack)+len(word) steps) and the search is bounded by that length.

Evaluation only: no training, no torch, no checkpoint write.
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import os
import sys
import time
from collections import Counter, deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SOLVABLE = "SOLVABLE"
UNSOLVABLE = "UNSOLVABLE"
UNKNOWN = "UNKNOWN"

DEFAULT_NODE_BUDGET = 200000
DEFAULT_TIME_BUDGET = 20.0
POP_NODE_BUDGET = 4000
POP_TIME_BUDGET = 0.5
DYN_NODE_BUDGET = 30000
DYN_TIME_BUDGET = 3.0
EVAL_SEED = 20261009

POP = "__pop__"


# --------------------------------------------------------------------------
# Stack-machine problem (new_approach Node stack domain)
# --------------------------------------------------------------------------

class _StackProblem:
    kind = "stack"
    complete_depth_bound = True

    def __init__(self, name: str, target, initial: Sequence,
                 domain: str, with_pop: bool) -> None:
        from new_approach.env import BUILD_ACTIONS
        from new_approach.env import plan as core_plan
        from new_approach.evolution import EVO_ACTIONS, sem_plan
        from new_approach.nodes import iter_nodes

        self.name = name
        self.target = target
        self.initial = tuple(initial)
        self.domain = domain
        self.with_pop = bool(with_pop)
        self.actions = tuple(EVO_ACTIONS if domain == "evo" else BUILD_ACTIONS)
        self.by_name = {a.name: a for a in self.actions}
        self.buildable = True
        self.error = ""
        try:
            self.word = tuple(sem_plan(target) if domain == "evo"
                              else core_plan(target))
        except Exception as exc:  # no build action for this target structure
            self.buildable = False
            self.error = "%s: %s" % (type(exc).__name__, exc)
            self.word = ()
        if self.buildable:
            self.subtrees = frozenset(n.canonical() for n in iter_nodes(target))
            seen: set = set()
            self.candidates: List[str] = []
            for nm in self.word:
                if nm not in seen:
                    seen.add(nm)
                    self.candidates.append(nm)
            required: Counter = Counter()
            stack: Tuple = ()
            for nm in self.word:
                stack = self._apply_raw(stack, nm)
                required[stack[-1].canonical()] += 1
            self.required = required
        else:
            self.subtrees = frozenset()
            self.candidates = []
            self.required = Counter()
        self.depth_limit = len(self.word) + (len(self.initial) if self.with_pop
                                             else 0)

    def _apply_raw(self, state: Tuple, name: str) -> Tuple:
        a = self.by_name[name]
        if a.arity == 0:
            return state + (a.build(()),)
        return state[: len(state) - a.arity] + (a.build(state[-a.arity:]),)

    def apply(self, state: Tuple, name: str) -> Optional[Tuple]:
        if name == POP:
            return state[:-1] if state else state
        a = self.by_name[name]
        if len(state) < a.arity:
            return None
        return self._apply_raw(state, name)

    def is_goal(self, state: Tuple) -> bool:
        return state == (self.target,)

    def key(self, state: Tuple) -> Any:
        return tuple(n.canonical() for n in state)

    def successors(self, state: Tuple) -> Iterable[Tuple[str, Tuple]]:
        out: List[Tuple[str, Tuple]] = []
        for nm in self.candidates:
            a = self.by_name[nm]
            if a.arity > len(state):
                continue
            nxt = self._apply_raw(state, nm)
            if not self.with_pop:
                if any(n.canonical() not in self.subtrees for n in nxt):
                    continue
            out.append((nm, nxt))
        if self.with_pop and state:
            out.append((POP, state[:-1]))
        return out

    def h(self, state: Tuple) -> int:
        if self.with_pop:
            return 0
        have: Counter = Counter(n.canonical() for n in state)
        missing = 0
        for canon, need in self.required.items():
            avail = have.get(canon, 0)
            if avail < need:
                missing += need - avail
        return missing

    def replay(self, path: Sequence[str]) -> Optional[Tuple]:
        state = self.initial
        for nm in path:
            state = self.apply(state, nm)
            if state is None:
                return None
        return state

    def fast_witness(self) -> Optional[List[str]]:
        """Canonical prefix/suffix fast path (SOUND, re-verified by replay)."""
        if not self.buildable:
            return None
        if self.initial == (self.target,):
            return []
        prefix: Tuple = ()
        if prefix == self.initial:
            return list(self.word)
        for i, nm in enumerate(self.word):
            prefix = self._apply_raw(prefix, nm)
            if prefix == self.initial:
                return list(self.word[i + 1:])
        return None

    def pop_full_witness(self) -> Optional[List[str]]:
        """SOUND pop fallback: discard the whole initial stack, rebuild the word.

        A minimal solution pops at most the whole initial stack, so this witness
        (<= len(stack)+len(word) steps) always exists for a buildable target
        under the with-pop action set. Its length is an UPPER bound unless the
        BFS found the optimum first.
        """
        if not (self.with_pop and self.buildable):
            return None
        return [POP] * len(self.initial) + list(self.word)


# --------------------------------------------------------------------------
# dynamic_env problem (Spec + State domain)
# --------------------------------------------------------------------------

class _DynProblem:
    kind = "dyn"
    complete_depth_bound = False

    def __init__(self, name: str, spec, start_state=None,
                 with_pop: bool = False) -> None:
        from dynamic_env.engine import DynamicEnv

        self.name = name
        self.spec = spec
        # Unit 2: the REAL DynamicEnv drives the action set. ``allow_pop`` turns
        # the engine's own ``pop`` action on, so the oracle no longer reimplements
        # the discard (the unit-1 ``_dyn_pop`` helper is gone).
        self.env = DynamicEnv(spec, allow_pop=bool(with_pop))
        self.initial = start_state if start_state is not None else self.env.reset()
        self.with_pop = bool(with_pop)
        self.depth_limit = int(getattr(spec, "max_steps", 0) or 0)
        if self.with_pop:
            self.depth_limit += len(self.initial.nodes)

    def apply(self, state, name: str):
        match = None
        for action in self.env.legal_actions(state):
            if action.key() == name:
                match = action
                break
        if match is None:
            return None
        return self.env.step(state, match).state

    def is_goal(self, state) -> bool:
        return bool(self.env.goal_reached(state))

    def key(self, state) -> str:
        return state.identity()

    def successors(self, state) -> Iterable[Tuple[str, Any]]:
        out: List[Tuple[str, Any]] = []
        for action in self.env.legal_actions(state):
            out.append((action.key(), self.env.step(state, action).state))
        return out

    def h(self, state) -> int:
        return 0

    def replay(self, path: Sequence[str]):
        state = self.initial
        for nm in path:
            state = self.apply(state, nm)
            if state is None:
                return None
        return state

    def fast_witness(self) -> Optional[List[str]]:
        return None

    def pop_full_witness(self) -> Optional[List[str]]:
        """Sound pop-all upper bound in the REAL engine (replay-verified).

        Discard every work node of the initial state, then replay the real
        engine's canonical word. Only returned when ``bfs_minimal_word`` finds a
        word; the caller re-executes it and accepts it only if it reaches the
        goal, so its length is an UPPER bound.
        """
        if not self.with_pop:
            return None
        from dynamic_env.engine import bfs_minimal_word, work_stack

        word = bfs_minimal_word(self.spec)
        if word is None:
            return None
        prefix = ["pop"] * len(work_stack(self.spec, self.initial))
        return prefix + list(word)


# --------------------------------------------------------------------------
# Bounded best-first search with a sound closure/exhaustion proof
# --------------------------------------------------------------------------

def _reconstruct(came: Dict[Any, Any], key: Any) -> List[str]:
    path: List[str] = []
    while came.get(key) is not None:
        prev, nm = came[key]
        path.append(nm)
        key = prev
    path.reverse()
    return path


def _result(verdict: str, proof: str, witness: Optional[List[str]],
            min_steps: Optional[int], nodes: int, t0: float,
            node_budget: int, time_budget: float) -> Dict[str, Any]:
    return {
        "verdict": verdict,
        "proof_method": proof,
        "witness_path": witness,
        "min_steps": min_steps,
        "nodes_expanded": nodes,
        "budget_used": {
            "nodes": nodes,
            "seconds": round(time.perf_counter() - t0, 6),
            "node_budget": node_budget,
            "time_budget": time_budget,
        },
    }


def _unknown_or_pop_fallback(problem, proof: str, nodes: int, t0: float,
                             node_budget: int, time_budget: float
                             ) -> Dict[str, Any]:
    """On a pop budget timeout, return the sound pop-all witness if it exists.

    The witness is re-executed and only accepted when it reaches the goal; its
    step count is an upper bound (proof method says so).
    """
    fallback = getattr(problem, "pop_full_witness", None)
    if callable(fallback):
        path = fallback()
        if path is not None:
            final = problem.replay(path)
            if final is not None and problem.is_goal(final):
                return _result(
                    SOLVABLE,
                    proof + "+pop_all_then_canonical_witness_upper_bound",
                    list(path), len(path), nodes, t0, node_budget, time_budget)
    return _result(UNKNOWN, proof, None, None, nodes, t0, node_budget,
                   time_budget)


def _search(problem, node_budget: int, time_budget: float) -> Dict[str, Any]:
    t0 = time.perf_counter()
    node_budget = int(node_budget)
    time_budget = float(time_budget)

    if problem.kind == "stack" and not problem.buildable:
        return _result(UNSOLVABLE, "target_not_buildable", None, None, 0,
                       t0, node_budget, time_budget)

    # Sound fast path: exact canonical prefix/suffix; verified by replay.
    fast = problem.fast_witness()
    if fast is not None:
        final = problem.replay(fast)
        if final is not None and problem.is_goal(final):
            return _result(SOLVABLE, "canonical_prefix_fast_path", list(fast),
                           len(fast), 0, t0, node_budget, time_budget)

    start = problem.initial
    if problem.is_goal(start):
        return _result(SOLVABLE, "initial_state_is_goal", [], 0, 0, t0,
                       node_budget, time_budget)

    seen: Dict[Any, int] = {problem.key(start): 0}
    came: Dict[Any, Any] = {problem.key(start): None}
    counter = itertools.count()
    heap: List[Tuple[int, int, int, Any]] = [
        (problem.h(start), 0, next(counter), start)]
    nodes = 0
    cut_off = False
    while heap:
        _f, g, _t, state = heapq.heappop(heap)
        key = problem.key(state)
        if g > seen.get(key, 1 << 30):
            continue
        if problem.is_goal(state):
            path = _reconstruct(came, key)
            final = problem.replay(path)
            if final is None or not problem.is_goal(final):
                raise AssertionError(
                    "witness failed replay verification for %s: %r"
                    % (problem.name, path))
            return _result(SOLVABLE, "bounded_best_first_search", path,
                           len(path), nodes, t0, node_budget, time_budget)
        nodes += 1
        if nodes > node_budget:
            return _unknown_or_pop_fallback(
                problem, "node_budget_exhausted", nodes, t0, node_budget,
                time_budget)
        if time.perf_counter() - t0 > time_budget:
            return _unknown_or_pop_fallback(
                problem, "time_budget_exhausted", nodes, t0, node_budget,
                time_budget)
        if g >= problem.depth_limit:
            if any(True for _ in problem.successors(state)):
                cut_off = True
            continue
        for nm, nxt in problem.successors(state):
            nk = problem.key(nxt)
            ng = g + 1
            if ng < seen.get(nk, 1 << 30):
                seen[nk] = ng
                came[nk] = (key, nm)
                heapq.heappush(heap, (ng + problem.h(nxt), ng,
                                      next(counter), nxt))
    proof = ("exhaustive_bounded_search" if (problem.complete_depth_bound
                                             or not cut_off)
             else "depth_limited_search_no_plan")
    verdict = UNSOLVABLE if (problem.complete_depth_bound or not cut_off) \
        else UNKNOWN
    return _result(verdict, proof, None, None, nodes, t0, node_budget,
                   time_budget)


# --------------------------------------------------------------------------
# Case adaptation
# --------------------------------------------------------------------------

def _domain_of(case, default: str = "evo") -> str:
    env = getattr(case, "env", None)
    if env is not None:
        if env.__class__.__name__ == "EvoEnv":
            return "evo"
        if env.__class__.__name__ == "MinimalEnv":
            return "core"
    dom = getattr(case, "domain", None)
    if dom in ("core", "evo"):
        return dom
    return default


def _is_stack_case(case) -> bool:
    if getattr(case, "env", None) is not None:
        return True
    return hasattr(case, "target") and hasattr(case, "initial_stack")


def _adapt(case, action_set: str, name: Optional[str] = None) -> Any:
    with_pop = action_set in ("pop", "with_pop", "current_pop") or \
        str(action_set).endswith("_pop")
    if _is_stack_case(case):
        domain = _domain_of(case)
        if str(action_set) in ("core", "evo"):
            domain = str(action_set)
        env = getattr(case, "env", None)
        if env is not None:
            target = getattr(getattr(env, "goal", None), "target", None)
            initial = tuple(env.reset().stack)
        else:
            target = getattr(case, "target")
            initial = tuple(getattr(case, "initial_stack", ()))
        if target is None:
            return None
        return _StackProblem(name or getattr(case, "name", "case"), target,
                             initial, domain, with_pop)
    spec = getattr(case, "spec", None)
    if spec is not None:
        start = getattr(case, "start_state", None)
        return _DynProblem(name or getattr(case, "name", "case"), spec,
                           start, with_pop)
    return None


def classify(case, action_set: str = "current",
             node_budget: int = DEFAULT_NODE_BUDGET,
             time_budget: float = DEFAULT_TIME_BUDGET) -> Dict[str, Any]:
    """Classify ONE case; see the module docstring for the verdict contract."""
    problem = _adapt(case, action_set)
    if problem is None:
        return {
            "verdict": UNKNOWN,
            "proof_method": "unadaptable_case",
            "witness_path": None,
            "min_steps": None,
            "nodes_expanded": 0,
            "budget_used": {"nodes": 0, "seconds": 0.0,
                            "node_budget": node_budget,
                            "time_budget": time_budget},
        }
    # A provided witness (e.g. the 4344 solvable-only pool) is re-executed and
    # only accepted when it really reaches the goal.
    provided = getattr(case, "witness", None)
    if provided is not None and problem.kind == "dyn":
        final = problem.replay(tuple(provided))
        if final is not None and problem.is_goal(final):
            t0 = time.perf_counter()
            return _result(SOLVABLE, "provided_witness_replay",
                           list(provided), len(provided), 0, t0,
                           node_budget, time_budget)
    return _search(problem, node_budget, time_budget)


# --------------------------------------------------------------------------
# Families
# --------------------------------------------------------------------------

def _row(family: str, action_set: str, case_name: str, res: Dict[str, Any]
         ) -> Dict[str, Any]:
    out = {"family": family, "action_set": action_set, "case": case_name}
    out.update(res)
    return out


def family_demo_bundle() -> List[Tuple[str, Any]]:
    from new_approach.evolution_bundle import make_demo_bundle
    return [(s.sid + ":" + s.name, s) for s in make_demo_bundle(60).states]


def family_core33() -> List[Tuple[str, Any]]:
    from new_approach import gen_common as gc
    return [(c.name, c) for c in gc.core_validation_cases()]


def family_ext114() -> List[Tuple[str, Any]]:
    from new_approach import gen_common as gc
    return [(c.name, c) for c in gc.evo_validation_cases()]


def family_unseen40() -> List[Tuple[str, Any]]:
    from new_approach import gen_common as gc
    cases, _forms = gc.unseen_cases()
    return [(c.name, c) for c in cases]


def family_val64() -> List[Tuple[str, Any]]:
    from evolution_trainer.size_selection import validation_pool
    return [(c.name, c) for c in validation_pool(64)]


def family_dense() -> List[Tuple[str, Any]]:
    from new_approach import gen_common as gc
    core_dense, evo_dense, _n = gc.dense_cases()
    return [(c.name, c) for c in core_dense] + [(c.name, c) for c in evo_dense]


def family_spec(name: str) -> List[Tuple[str, Any]]:
    from evolution_trainer.size_selection import spec_by_id
    return [(name, spec_by_id(name))]


def family_perturb() -> List[Tuple[str, Any]]:
    from evolution_trainer.heldout_eval import build_perturbations
    out: List[Tuple[str, Any]] = []
    for var in build_perturbations():
        out.append((var["variant_id"], var["spec"]))
    return out


FAMILIES: Dict[str, Any] = {
    "demo_bundle": family_demo_bundle,
    "core33": family_core33,
    "ext114": family_ext114,
    "unseen40": family_unseen40,
    "val64": family_val64,
    "dense": family_dense,
    "spec_multi_step": lambda: family_spec("spec_multi_step"),
    "spec_dynamic_axiom": lambda: family_spec("spec_dynamic_axiom"),
    "spec_dynamic_group_deep": lambda: family_spec("spec_dynamic_group_deep"),
    "spec_multi_step_heldout": lambda: family_spec("spec_multi_step_heldout"),
    "perturb_multi_step": family_perturb,
}


def _budget_for(case) -> Tuple[int, float]:
    if not _is_stack_case(case):
        return DYN_NODE_BUDGET, DYN_TIME_BUDGET
    return DEFAULT_NODE_BUDGET, DEFAULT_TIME_BUDGET


def _summarise(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["family"], r["action_set"]), []).append(r)
    table: List[Dict[str, Any]] = []
    for (family, action_set), rs in groups.items():
        steps = [r["min_steps"] for r in rs if r["min_steps"] is not None]
        table.append({
            "family": family,
            "action_set": action_set,
            "n": len(rs),
            "solvable": sum(1 for r in rs if r["verdict"] == SOLVABLE),
            "unsolvable": sum(1 for r in rs if r["verdict"] == UNSOLVABLE),
            "unknown": sum(1 for r in rs if r["verdict"] == UNKNOWN),
            "min_steps_median": (sorted(steps)[len(steps) // 2]
                                 if steps else None),
            "min_steps_min": (min(steps) if steps else None),
        })
    return table


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="decidable solvability oracle (evaluation only)")
    parser.add_argument("--families", default="all",
                        help="comma list or 'all'")
    parser.add_argument("--action-set", default="both",
                        choices=("current", "pop", "both"))
    parser.add_argument("--out", default="classification.json")
    parser.add_argument("--node-budget", type=int, default=DEFAULT_NODE_BUDGET)
    parser.add_argument("--time-budget", type=float,
                        default=DEFAULT_TIME_BUDGET)
    parser.add_argument("--pop-node-budget", type=int,
                        default=POP_NODE_BUDGET)
    parser.add_argument("--pop-time-budget", type=float,
                        default=POP_TIME_BUDGET)
    parser.add_argument("--dyn-node-budget", type=int, default=DYN_NODE_BUDGET)
    parser.add_argument("--dyn-time-budget", type=float,
                        default=DYN_TIME_BUDGET)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    wanted = (list(FAMILIES) if args.families == "all"
              else [f.strip() for f in args.families.split(",") if f.strip()])
    action_sets = (["current", "pop"] if args.action_set == "both"
                   else [args.action_set])

    rows: List[Dict[str, Any]] = []
    for family in wanted:
        builder = FAMILIES[family]
        cases = builder()
        if args.limit:
            cases = cases[: args.limit]
        for name, case in cases:
            for action_set in action_sets:
                if not _is_stack_case(case):
                    nb, tb = args.dyn_node_budget, args.dyn_time_budget
                elif action_set == "pop":
                    nb, tb = args.pop_node_budget, args.pop_time_budget
                else:
                    nb, tb = args.node_budget, args.time_budget
                res = classify(case, action_set, nb, tb)
                rows.append(_row(family, action_set, name, res))
        print("FAMILY %-26s n=%d done" % (family, len(cases)), flush=True)

    table = _summarise(rows)
    print("")
    print("%-26s %-8s %5s %9s %11s %8s %14s" %
          ("family", "actions", "n", "solvable", "unsolvable", "unknown",
           "min_steps_med"))
    for t in table:
        print("%-26s %-8s %5d %9d %11d %8d %14s" %
              (t["family"], t["action_set"], t["n"], t["solvable"],
               t["unsolvable"], t["unknown"], str(t["min_steps_median"])))

    payload = {
        "oracle": "evolution_trainer.solvability",
        "seed": EVAL_SEED,
        "action_sets": action_sets,
        "families": wanted,
        "summary": table,
        "rows": rows,
    }
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("WROTE %s" % out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
