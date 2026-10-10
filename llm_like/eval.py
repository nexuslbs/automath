"""Held-out evaluation for llm_like (design section 6).

The purpose is a LEARNING CURVE on unseen/solvable forms, not a single lucky
episode and not a masked planner. This module builds the held-out set exactly
as design section 6.2 fixes it:

* the two shipped held-out specs (DATA, never authored here),
* ``evolution_trainer.size_selection.validation_pool(64)``,
* a FRESH held-out pool with seed ``EVAL_SEED = 424242``,
* the 30 perturbations from ``heldout_eval.build_perturbations``.

Every case is classified with the three-way oracle and the report is the
three-way decomposition (``SOLVABLE-SOLVED`` / ``SOLVABLE-UNSOLVED`` /
``PROVABLY-UNSOLVABLE``). ``solve_rate = solved / SOLVABLE``. Mask ON and mask
OFF are reported SEPARATELY; the headline learned number is mask OFF.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .flywheel import (
    EVAL_SEED,
    Case,
    classify_case,
    filter_pool,
    fresh_pool,
    load_spec_by_id,
    rollout_episode,
    solve_rate,
    HELDOUT_SPEC_IDS,
)
from .model import DEFAULT_CONTEXT, DEFAULT_WIDTH, LLMAgentNet, build_model

PERTURB_COUNT = 30
HELDOUT_SEED = 12345

SOLVABLE = "SOLVABLE"
UNKNOWN = "UNKNOWN"
UNSOLVABLE = "UNSOLVABLE"


# --------------------------------------------------------------------------
# Held-out case construction
# --------------------------------------------------------------------------

def _perturbation_cases(count: int = PERTURB_COUNT,
                        seed: int = HELDOUT_SEED) -> List[Case]:
    from evolution_trainer.heldout_eval import build_perturbations

    cases: List[Case] = []
    for variant in build_perturbations(count, seed):
        cases.append(Case(name="perturb_%02d" % variant["index"],
                          spec=variant["spec"], start_state=None,
                          kind="perturb", max_steps=variant["max_steps"]))
    return cases


def heldout_cases(val_size: int = 64, fresh_size: int = 64,
                  perturb_count: int = PERTURB_COUNT,
                  fresh_seed: int = EVAL_SEED,
                  limit: Optional[int] = None) -> List[Case]:
    """The full held-out set of design section 6.2 (optionally truncated)."""
    from evolution_trainer.size_selection import validation_pool

    cases: List[Case] = []
    for spec_id in HELDOUT_SPEC_IDS:
        cases.append(Case(name="heldout_" + spec_id,
                          spec=load_spec_by_id(spec_id), start_state=None,
                          kind="heldout"))
    for val_case in validation_pool(val_size):
        cases.append(Case(name="val64_" + val_case.name, spec=val_case.spec,
                          start_state=val_case.start_state,
                          witness=tuple(val_case.witness), kind="val64"))
    cases.extend(fresh_pool(fresh_size, fresh_seed))
    cases.extend(_perturbation_cases(perturb_count))
    if limit is not None:
        cases = cases[: int(limit)]
    return cases


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def evaluate_model(net: LLMAgentNet, cases: Sequence[Case],
                   action_set: str = "current", allow_pop: bool = False,
                   seed: int = 7, node_budget: int = 30000,
                   time_budget: float = 3.0,
                   masks: Sequence[bool] = (False, True),
                   filtered: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Three-way held-out evaluation with mask OFF and mask ON separately.

    ``filtered`` may be the cached :func:`flywheel.filter_pool` result so a
    learning-curve loop does not re-classify the same fixed held-out set.
    """
    started = time.perf_counter()
    if filtered is None:
        filtered = filter_pool(cases, action_set=action_set,
                               node_budget=node_budget, time_budget=time_budget)
    counts = filtered["counts"]
    verdicts = filtered["verdicts"]

    per_mask: Dict[str, Any] = {}
    per_case_by_mask: Dict[str, List[Dict[str, Any]]] = {}
    for mask_illegal in masks:
        label = "mask_on" if mask_illegal else "mask_off"
        solved_solvable = 0
        solved_total = 0
        solved_steps: List[int] = []
        entries: List[Dict[str, Any]] = []
        wall = 0.0
        for index, case in enumerate(cases):
            rng = random.Random(int(seed) * 1000003 + index)
            result = rollout_episode(net, case, rng, epsilon=0.0,
                                     mask_illegal=mask_illegal,
                                     allow_pop=allow_pop,
                                     max_steps=case.step_budget())
            wall += float(result.get("wall_s", 0.0))
            verdict = verdicts.get(case.name, {}).get("verdict", UNKNOWN)
            solved = bool(result["solved"])
            solved_total += int(solved)
            if verdict == SOLVABLE and solved:
                solved_solvable += 1
                solved_steps.append(int(result["steps"]))
            entries.append({
                "case": case.name,
                "kind": case.kind,
                "spec_id": case.spec.spec_id,
                "verdict": verdict,
                "solved": solved,
                "steps": int(result["steps"]),
                "wall_s": result.get("wall_s"),
                "trace": [record["chosen_key"]
                          for record in result.get("records", ())],
            })
        per_mask[label] = {
            "cases": len(cases),
            "solved_total": solved_total,
            "solved_solvable": solved_solvable,
            "solvable": int(counts.get(SOLVABLE, 0)),
            "unknown": int(counts.get(UNKNOWN, 0)),
            "unsolvable": int(counts.get(UNSOLVABLE, 0)),
            "solution_rate": solve_rate(solved_solvable, counts),
            "mean_steps_solved": (sum(solved_steps) / float(len(solved_steps))
                                  if solved_steps else None),
            "mean_wall_s": (wall / float(len(cases))) if cases else 0.0,
            "wall_s": round(wall, 4),
        }
        per_case_by_mask[label] = entries

    return {
        "cases": len(cases),
        "three_way": {
            SOLVABLE: int(counts.get(SOLVABLE, 0)),
            UNKNOWN: int(counts.get(UNKNOWN, 0)),
            UNSOLVABLE: int(counts.get(UNSOLVABLE, 0)),
        },
        "mask_off": per_mask.get("mask_off"),
        "mask_on": per_mask.get("mask_on"),
        "per_case_mask_off": per_case_by_mask.get("mask_off", []),
        "per_case_mask_on": per_case_by_mask.get("mask_on", []),
        "verdicts": verdicts,
        "wall_s": round(time.perf_counter() - started, 4),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="llm_like held-out evaluation")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--d", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
    parser.add_argument("--checkpoint", default=None,
                        help="path to a .pt state dict; random net if omitted")
    parser.add_argument("--out", required=True)
    parser.add_argument("--action-set", default="current",
                        choices=["current", "pop"])
    parser.add_argument("--limit", type=int, default=0,
                        help="0 = the full held-out set")
    parser.add_argument("--fresh-size", type=int, default=64)
    parser.add_argument("--val-size", type=int, default=64)
    parser.add_argument("--perturb-count", type=int, default=PERTURB_COUNT)
    args = parser.parse_args(argv)

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:  # pragma: no cover - older torch / unsupported op
        pass

    net = build_model(args.d, args.context, seed=args.seed)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        net.load_state_dict(state)
    net.eval()
    print("[llm_like] ACTUAL parameter count = %d (target <= 300000)"
          % net.param_count())

    limit = args.limit if args.limit and args.limit > 0 else None
    cases = heldout_cases(val_size=args.val_size, fresh_size=args.fresh_size,
                          perturb_count=args.perturb_count, limit=limit)
    result = evaluate_model(net, cases, action_set=args.action_set,
                            allow_pop=(args.action_set == "pop"),
                            seed=args.seed)
    result["meta"] = {
        "seed": args.seed,
        "d": args.d,
        "context": args.context,
        "action_set": args.action_set,
        "param_count": net.param_count(),
        "checkpoint": args.checkpoint,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "heldout_%s.json" % args.seed)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    with open(os.path.join(args.out, "DONE"), "w", encoding="utf-8") as handle:
        handle.write("seed=%s param_count=%d\n" % (args.seed, net.param_count()))
    with open(os.path.join(args.out, "SHA256SUMS"), "w", encoding="utf-8") as handle:
        handle.write("%s  %s\n" % (_sha256(out_path), os.path.basename(out_path)))

    print("[llm_like] seed=%s off=%s on=%s -> %s"
          % (args.seed,
             (result["mask_off"] or {}).get("solution_rate"),
             (result["mask_on"] or {}).get("solution_rate"),
             out_path))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
