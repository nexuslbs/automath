"""PLANNER held-out evaluation ONLY (no genome, no training, no torch).

Runs ``new_approach.planner.Planner`` - the explicit bounded search over the
``REDUCTION_TABLE`` (COMPARISON.md section 4 next-fix #3) - on the SAME three
held-out sets used by ``gen_eval`` for flavors A/B/C:

    core33  = evolution_run.core_validation_cases()   (u2 validation)
    ext114  = evolution_run.evo_validation_cases()    (scenario prefixes)
    unseen40 = gen_common.unseen_cases()              (22 evo + 18 core)

    /opt/automath/venv/bin/python -m new_approach.planner_eval \
        --config config/evolution_genfitness.json \
        --out /opt/automath/tmp/planner_eval.json

Evaluation-only: it writes no checkpoints, mutates no training state, and reads
the config only for the seed label.  Per-form solved flags are included.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from . import gen_common as gc
from .planner import DEFAULT_MAX_ACTIONS, DEFAULT_MAX_SEARCH_NODES, make_planners
from .planner import planner_rollout


def _set_block(roll: dict) -> dict:
    """The comparison-table fields for one held-out set."""
    return {
        "solved": roll["solved"],
        "total": roll["total"],
        "mean_actions_solved": roll["mean_actions_solved"],
        "mean_reward": roll["mean_reward"],
        "wall_secs": roll["wall_secs"],
        "per_form": roll["per_form"],
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="PLANNER over the REDUCTION_TABLE: held-out evaluation only")
    p.add_argument("--config",
                   default="/opt/automath/repo/config/evolution_genfitness.json")
    p.add_argument("--out", default="/opt/automath/tmp/planner_eval.json")
    p.add_argument("--label", default="P-explicit-search-reduction-table")
    p.add_argument("--max-search-nodes", type=int,
                   default=DEFAULT_MAX_SEARCH_NODES)
    p.add_argument("--max-actions", type=int, default=DEFAULT_MAX_ACTIONS)
    p.add_argument("--no-pattern-goals", dest="pattern_goals",
                   action="store_false", default=True,
                   help="disable the new pattern/ExprGoal planner path so the "
                        "pre-change fixed-state-only baseline is reproduced")
    args = p.parse_args(argv)

    cfg = {}
    try:
        with open(args.config) as fh:
            cfg = json.load(fh)
    except OSError:
        cfg = {}
    seed = int(cfg.get("seed", 20261009))

    core_planner, evo_planner = make_planners(
        max_search_nodes=args.max_search_nodes, max_actions=args.max_actions,
        pattern_goals=args.pattern_goals)
    print("PLANNER_EVAL label=%s seed=%d max_search_nodes=%d max_actions=%d "
          "pattern_goals=%s genome=NONE train_cases=0 (evaluation only)"
          % (args.label, seed, args.max_search_nodes, args.max_actions,
             "ON" if args.pattern_goals else "OFF"))

    t0 = time.perf_counter()
    core33 = planner_rollout(core_planner, gc.core_validation_cases())
    ext114 = planner_rollout(evo_planner, gc.evo_validation_cases())
    unseen, forms = gc.unseen_cases()
    core_unseen = [c for c in unseen if getattr(c, "domain", "evo") == "core"]
    evo_unseen = [c for c in unseen if getattr(c, "domain", "evo") == "evo"]
    cu = planner_rollout(core_planner, core_unseen)
    eu = planner_rollout(evo_planner, evo_unseen)
    wall_total = round(time.perf_counter() - t0, 3)

    unseen_block = {
        "solved": cu["solved"] + eu["solved"],
        "total": cu["total"] + eu["total"],
        "mean_actions_solved": gc._mean(
            ([cu["mean_actions_solved"]] if cu["mean_actions_solved"] else [])
            + ([eu["mean_actions_solved"]] if eu["mean_actions_solved"] else [])),
        "mean_reward": round(
            (cu["mean_reward"] * cu["total"] + eu["mean_reward"] * eu["total"])
            / max(1, cu["total"] + eu["total"]), 4),
        "wall_secs": round(cu["wall_secs"] + eu["wall_secs"], 3),
        "core_half": "%d/%d" % (cu["solved"], cu["total"]),
        "evo_half": "%d/%d" % (eu["solved"], eu["total"]),
        "per_form": dict(cu["per_form"], **eu["per_form"]),
    }

    res = {
        "label": args.label,
        "approach": "P-explicit-search-over-reduction-table (evaluation only)",
        "episodes": 0,
        "seed": seed,
        "genome": None,
        "max_search_nodes": args.max_search_nodes,
        "max_actions": args.max_actions,
        "pattern_goals": bool(args.pattern_goals),
        "train_cases_core": 0,
        "train_cases_evo": 0,
        "train_cases_total": 0,
        "train_wall_secs": 0.0,
        "wall_secs_total": wall_total,
        "core33": _set_block(core33),
        "ext114": _set_block(ext114),
        "unseen": unseen_block,
        "unseen_forms": forms,
    }
    report_result(res)
    gc.write_result(res, args.out)
    print("WROTE %s" % args.out)
    return 0


def report_result(res: dict) -> None:
    print("APPROACH=%s seed=%d genome=%s max_search_nodes=%d max_actions=%d "
          "pattern_goals=%s"
          % (res["approach"], res["seed"], res["genome"],
             res["max_search_nodes"], res["max_actions"],
             "ON" if res.get("pattern_goals") else "OFF"))
    for key in ("core33", "ext114", "unseen"):
        r = res[key]
        print("SET %-7s solved=%d/%d mean_actions_solved=%s mean_reward=%s "
              "wall=%.3fs%s"
              % (key, r["solved"], r["total"], r["mean_actions_solved"],
                 r["mean_reward"], r["wall_secs"],
                 (" (core=%s evo=%s)" % (r.get("core_half"), r.get("evo_half")))
                 if key == "unseen" else ""))
    print("TOTAL_WALL=%.3fs" % res["wall_secs_total"])
    print("UNSEEN_FORMS %d cases (canonical, reproducible)"
          % len(res["unseen_forms"]))


if __name__ == "__main__":
    sys.exit(main())
