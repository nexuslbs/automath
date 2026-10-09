"""SOLVABILITY AUDIT for the explicit planner (evaluation ONLY).

Answers ONE honest question for each held-out form:

    Is ``unseen40 = 0/40`` a METHOD failure, or a property of the EVAL SET
    under the current action set?

For every form it runs the SAME ``planner.Planner`` used by ``planner_eval``
with a RAISED search budget and classifies the form as exactly one of:

    SOLVED               - a (possibly empty) plan to ``(target,)`` exists and
                           was found (prefix fast path or search).
    PROVABLY-UNSOLVABLE  - the search space is exhausted without a plan, or the
                           sound non-subtree prune fires: NO continuation can
                           reach ``(target,)``.  The exact reason is recorded.
    TIMEOUT-BUDGET       - the explicit node budget was hit before either a
                           plan or an exhaustion proof; the form is UNKNOWN,
                           not unsolvable.
    NON-FIXED-GOAL       - a pattern goal with no fixed target (no plan to
                           audit; the planner uses its legal fallback).

The classification mirrors ``planner.Planner.plan`` / ``_search`` exactly but
instruments it, so the audited verdict is the planner's own verdict - no new
search semantics are introduced.  ``new_approach/planner.py`` is NOT touched.

Evaluation only: no genome, no torch, no training, no checkpoint write.

    /opt/automath/venv/bin/python -m new_approach.solvability_audit \
        --config config/evolution_genfitness.json \
        --out /opt/automath/tmp/solvability_audit.json \
        --max-search-nodes 2000000 --max-actions 200
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from . import gen_common as gc
from .planner import DEFAULT_MAX_ACTIONS, DEFAULT_MAX_SEARCH_NODES
from .planner import Planner, make_planners

SOLVED = "SOLVED"
UNSOLVABLE = "PROVABLY-UNSOLVABLE"
TIMEOUT = "TIMEOUT-BUDGET"
NON_FIXED = "NON-FIXED-GOAL"

_REASON_NON_SUBTREE = "stack node(s) are not subtrees of target; no pop/discard action exists, so they can never be consumed and NO continuation reaches (target,)"
_REASON_EXHAUSTED = "sound best-first search exhausted the reachable subtree state space within depth_limit without reaching (target,)"
_REASON_BUDGET = "node budget hit before a plan or an exhaustion proof; form is UNKNOWN, not unsolvable"
_REASON_LONG_WORD = "canonical build word longer than max_actions"
_REASON_STATE_CAP = "initial stack longer than max_actions"


def _search_status(planner: Planner, target, start: Tuple,
                   word: Tuple[str, ...], subtrees: frozenset
                   ) -> Tuple[Optional[List[str]], int, str]:
    """Mirror ``Planner._search`` with a status + node counter.

    Returns ``(actions_or_None, nodes_generated, status)`` where status is one
    of ``"solved"``, ``"exhausted"`` (heap emptied: unsolvable proof) or
    ``"budget"`` (``max_search_nodes`` hit: unknown).
    """
    required = planner._required(target)
    depth_limit = min(planner.max_actions, max(1, len(word)))
    seen = set()
    candidates: List[str] = []
    for name in word:
        if name not in seen:
            seen.add(name)
            candidates.append(name)
    counter = itertools.count()
    came: Dict[Tuple, Optional[Tuple[Tuple, str]]] = {start: None}
    gbest: Dict[Tuple, int] = {start: 0}
    heap = [(planner._h(start, required), 0, next(counter), start)]
    nodes = 0
    while heap:
        _f, g, _t, state = heapq.heappop(heap)
        if state == (target,):
            return planner._reconstruct(came, state), nodes, "solved"
        if g >= depth_limit:
            continue
        for name in candidates:
            action = planner.by_name[name]
            if action.arity > len(state):
                continue
            nxt = planner._apply(state, action)
            ng = g + 1
            if gbest.get(nxt, 1 << 30) <= ng:
                continue
            if any(n.canonical() not in subtrees for n in nxt):
                continue
            gbest[nxt] = ng
            came[nxt] = (state, name)
            nodes += 1
            if nxt == (target,):
                return planner._reconstruct(came, nxt), nodes, "solved"
            if nodes >= planner.max_search_nodes:
                return None, nodes, "budget"
            heapq.heappush(heap, (ng + planner._h(nxt, required), ng,
                                  next(counter), nxt))
    return None, nodes, "exhausted"


def audit_case(planner: Planner, case) -> dict:
    """Classify ONE held-out form (evaluation only)."""
    t0 = time.perf_counter()
    target = getattr(getattr(case.env, "goal", None), "target", None)
    state = case.env.reset()
    start = tuple(state.stack)
    rec = {
        "name": case.name,
        "domain": getattr(case, "domain", "evo"),
        "level": getattr(case, "level", ""),
        "max_steps": int(getattr(case, "max_steps", 0)),
    }
    if target is None:
        rec.update({"class": NON_FIXED, "reason": "pattern goal, no fixed target",
                    "word_len": None, "depth_limit": None, "prefix_index": None,
                    "non_subtree_nodes": [], "search_nodes": 0, "plan_len": None,
                    "wall_secs": round(time.perf_counter() - t0, 4)})
        return rec

    word = planner._word(target)
    rec["word_len"] = len(word)
    rec["target"] = target.canonical()
    rec["initial_stack"] = gc.stack_canon(start)
    rec["prefix_index"] = None
    rec["non_subtree_nodes"] = []

    if start == (target,):
        rec.update({"class": SOLVED, "reason": "initial stack is already the goal",
                    "depth_limit": 0, "search_nodes": 0, "plan_len": 0,
                    "wall_secs": round(time.perf_counter() - t0, 4)})
        return rec
    if len(word) > planner.max_actions:
        rec.update({"class": TIMEOUT, "reason": _REASON_LONG_WORD,
                    "depth_limit": None, "search_nodes": 0, "plan_len": None,
                    "wall_secs": round(time.perf_counter() - t0, 4)})
        return rec
    if not start:
        rec.update({"class": SOLVED, "reason": "empty stack: canonical word",
                    "depth_limit": len(word), "search_nodes": 0,
                    "plan_len": len(word),
                    "wall_secs": round(time.perf_counter() - t0, 4)})
        return rec
    if len(start) > planner.max_actions:
        rec.update({"class": TIMEOUT, "reason": _REASON_STATE_CAP,
                    "depth_limit": None, "search_nodes": 0, "plan_len": None,
                    "wall_secs": round(time.perf_counter() - t0, 4)})
        return rec

    # 1) exact prefix fast path (same as Planner.plan)
    prefix: Tuple = ()
    for i, name in enumerate(word):
        prefix = planner._apply(prefix, planner.by_name[name])
        if prefix == start:
            rec["prefix_index"] = i + 1
            rec.update({"class": SOLVED, "reason": "exact prefix fast path",
                        "depth_limit": len(word) - (i + 1), "search_nodes": 0,
                        "plan_len": len(word) - (i + 1),
                        "wall_secs": round(time.perf_counter() - t0, 4)})
            return rec

    # 2) sound non-subtree prune (same as Planner.plan)
    subtrees = planner._subtrees(target)
    bad = [n.canonical() for n in start if n.canonical() not in subtrees]
    if bad:
        rec["non_subtree_nodes"] = bad
        rec.update({"class": UNSOLVABLE, "reason": _REASON_NON_SUBTREE,
                    "depth_limit": min(planner.max_actions, max(1, len(word))),
                    "search_nodes": 0, "plan_len": None,
                    "wall_secs": round(time.perf_counter() - t0, 4)})
        return rec

    # 3) instrumented bounded best-first search
    actions, nodes, status = _search_status(planner, target, start, word, subtrees)
    depth_limit = min(planner.max_actions, max(1, len(word)))
    if status == "solved":
        rec.update({"class": SOLVED, "reason": "bounded search", "depth_limit":
                    depth_limit, "search_nodes": nodes,
                    "plan_len": len(actions or []),
                    "wall_secs": round(time.perf_counter() - t0, 4)})
    elif status == "exhausted":
        rec.update({"class": UNSOLVABLE, "reason": _REASON_EXHAUSTED,
                    "depth_limit": depth_limit, "search_nodes": nodes,
                    "plan_len": None,
                    "wall_secs": round(time.perf_counter() - t0, 4)})
    else:
        rec.update({"class": TIMEOUT, "reason": _REASON_BUDGET,
                    "depth_limit": depth_limit, "search_nodes": nodes,
                    "plan_len": None,
                    "wall_secs": round(time.perf_counter() - t0, 4)})
    return rec


def audit_set(name: str, planner: Planner, cases: Sequence,
              limit: Optional[int] = None) -> dict:
    cases = list(cases)
    if limit:
        cases = cases[:limit]
    rows = [audit_case(planner, c) for c in cases]
    counts: Counter = Counter(r["class"] for r in rows)
    return {
        "set": name,
        "total": len(rows),
        "counts": dict(counts),
        "wall_secs": round(sum(r["wall_secs"] for r in rows), 3),
        "rows": rows,
    }


def _print_rows(block: dict) -> None:
    print("AUDIT_SET %s total=%d %s wall=%.3fs"
          % (block["set"], block["total"],
             " ".join("%s=%d" % (k, block["counts"][k])
                      for k in sorted(block["counts"])), block["wall_secs"]))
    print("%-34s %-9s %-19s %8s %10s %8s %9s"
          % ("form", "domain", "class", "word", "search_n", "plan", "wall_s"))
    for i, r in enumerate(block["rows"]):
        print("%-34s %-9s %-19s %8s %10s %8s %9.4f"
              % (r["name"][:34], r["domain"], r["class"],
                 str(r["word_len"]), str(r["search_nodes"]),
                 str(r["plan_len"]), r["wall_secs"]))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="planner solvability audit (evaluation only)")
    p.add_argument("--config", default="/opt/automath/repo/config/evolution_genfitness.json")
    p.add_argument("--out", default="/opt/automath/tmp/solvability_audit.json")
    p.add_argument("--sets", default="unseen40,core33,ext114",
                   help="comma list of unseen40,core33,ext114")
    p.add_argument("--max-search-nodes", type=int, default=2000000)
    p.add_argument("--max-actions", type=int, default=DEFAULT_MAX_ACTIONS)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--names", default="",
                   help="optional comma list of case names to audit (all when empty)")
    args = p.parse_args(argv)
    names = {n.strip() for n in args.names.split(",") if n.strip()}

    def keep(cases):
        if not names:
            return list(cases)
        return [c for c in cases if c.name in names]

    cfg = {}
    try:
        with open(args.config) as fh:
            cfg = json.load(fh)
    except OSError:
        cfg = {}
    seed = int(cfg.get("seed", 20261009))

    core_planner, evo_planner = make_planners(
        max_search_nodes=args.max_search_nodes, max_actions=args.max_actions)
    print("SOLVABILITY_AUDIT seed=%d max_search_nodes=%d max_actions=%d "
          "planner=new_approach.planner.Planner (mirrored search with status)"
          % (seed, args.max_search_nodes, args.max_actions))

    wanted = {s.strip() for s in args.sets.split(",") if s.strip()}
    blocks: List[dict] = []
    unseen, forms = gc.unseen_cases()
    if "unseen40" in wanted:
        core_unseen = [c for c in unseen if getattr(c, "domain", "evo") == "core"]
        evo_unseen = [c for c in unseen if getattr(c, "domain", "evo") == "evo"]
        rows = ([audit_case(core_planner, c) for c in keep(core_unseen)]
                + [audit_case(evo_planner, c) for c in keep(evo_unseen)])
        counts: Counter = Counter(r["class"] for r in rows)
        blocks.append({"set": "unseen40", "total": len(rows),
                       "counts": dict(counts),
                       "wall_secs": round(sum(r["wall_secs"] for r in rows), 3),
                       "rows": rows})
    if "core33" in wanted:
        blocks.append(audit_set("core33", core_planner,
                                keep(gc.core_validation_cases()), args.limit or None))
    if "ext114" in wanted:
        blocks.append(audit_set("ext114", evo_planner,
                                keep(gc.evo_validation_cases()), args.limit or None))

    for b in blocks:
        _print_rows(b)
        print("")

    res = {
        "audit": "planner-solvability (evaluation only)",
        "approach": "mirrors new_approach.planner.Planner.plan/_search with status",
        "seed": seed,
        "max_search_nodes": args.max_search_nodes,
        "max_actions": args.max_actions,
        "classes": {"SOLVED": SOLVED, "UNSOLVABLE": UNSOLVABLE,
                    "TIMEOUT": TIMEOUT, "NON_FIXED": NON_FIXED},
        "blocks": blocks,
        "unseen_forms": forms,
        "wall_secs_total": round(sum(b["wall_secs"] for b in blocks), 3),
    }
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("WROTE %s" % args.out)
    print("TOTAL_WALL=%.3fs" % res["wall_secs_total"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
