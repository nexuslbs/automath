"""Held-out / generalization harness for the dynamic-nodes curriculum (Unit D-1).

Two held-out validation cases ship as DATA under ``data/dynamic_env/heldout/``:

* ``spec_multi_step_heldout`` - a ``spec_multi_step`` variant with a DIFFERENT
  target objective state (8 instead of 4);
* ``spec_dynamic_group_deep`` - a deeper dynamic grouping with a LARGER arity
  (3 operands instead of 2).

Neither is part of training. The harness scores an optional genome (a
checkpoint) on every case and reports, per case, the state trace (the elementar
1/0 objective flags, the action, the reward and the step), solved/total and the
wall time. The held-out EVAL RUN itself belongs to Unit D-2; Unit D-1 only
builds and SMOKES this harness.

CLI::

    python -m evolution_trainer.heldout --episodes 3
    python -m evolution_trainer.heldout --genome out/stage4/checkpoints/checkpoint.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Sequence

from dynamic_env.spec import load_spec

from .curriculum import minimal_and_max, spec_path
from .harness import Harness

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_HELDOUT_DIR = os.path.join(_REPO_ROOT, "data", "dynamic_env", "heldout")

#: The held-out case ids (files carry the ``.json`` suffix).
HELDOUT_CASES: Sequence[str] = ("spec_multi_step_heldout",
                                "spec_dynamic_group_deep")


def load_heldout(case_id: str, heldout_dir: str = _HELDOUT_DIR):
    return load_spec(os.path.join(heldout_dir, case_id + ".json"))


def genome_from_checkpoint(path: str) -> List[float]:
    """Accept a full checkpoint.json or a best_gen_*.json / best_agents file."""
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if "best_agent" in raw and "genome" in raw["best_agent"]:
        return [float(g) for g in raw["best_agent"]["genome"]]
    if "agent" in raw and "genome" in raw["agent"]:
        return [float(g) for g in raw["agent"]["genome"]]
    if "all_time_best" in raw and raw["all_time_best"]:
        return [float(g) for g in raw["all_time_best"]["agent"]["genome"]]
    if "population" in raw and raw["population"]:
        best = max(raw["population"], key=lambda a: a.get("fitness") or float("-inf"))
        return [float(g) for g in best["genome"]]
    raise ValueError("no genome found in %s" % path)


def evaluate_case(genome: Optional[Sequence[float]], case_id: str,
                  episodes: int = 3, seed: int = 12345,
                  heldout_dir: str = _HELDOUT_DIR) -> Dict[str, Any]:
    spec = load_heldout(case_id, heldout_dir)
    plan = minimal_and_max(spec)
    harness = Harness(spec, max_episode_steps=plan["max_steps"])
    if genome is None:
        genome = harness.net.init(random.Random(seed), scale=0.5)
    started = time.perf_counter()
    greedy = harness.rollout(genome, explore=False, collect=True)
    rng = random.Random(seed)
    solved = 0
    steps: List[int] = []
    for _ in range(episodes):
        result = harness.rollout(genome, explore=True, rng=rng)
        if result.solved:
            solved += 1
            steps.append(result.steps)
    elapsed = time.perf_counter() - started
    return {
        "case_id": case_id,
        "spec_id": spec.spec_id,
        "min_steps": plan["min_steps"],
        "max_steps": plan["max_steps"],
        "max_rule": plan["rule"],
        "greedy_solved": bool(greedy.solved),
        "greedy_steps": greedy.steps,
        "greedy_return": round(greedy.total_return, 6),
        "stochastic_episodes": episodes,
        "stochastic_solved": solved,
        "stochastic_mean_steps": (round(sum(steps) / len(steps), 4)
                                  if steps else None),
        "wall_s": round(elapsed, 6),
        "trace": [
            {"step": item["step"], "action": item["action"],
             "objective_vector": item["objective_vector"],
             "reward": round(item["reward"], 6), "goal": item["goal"]}
            for item in greedy.trace
        ],
    }


def run(genome: Optional[Sequence[float]] = None, episodes: int = 3,
        seed: int = 12345, heldout_dir: str = _HELDOUT_DIR,
        trace_case: Optional[str] = None) -> Dict[str, Any]:
    cases = []
    solved_total = 0
    for case_id in HELDOUT_CASES:
        report = evaluate_case(genome, case_id, episodes=episodes, seed=seed,
                               heldout_dir=heldout_dir)
        if report["stochastic_solved"] >= 1 or report["greedy_solved"]:
            solved_total += 1
        cases.append(report)
    detail = trace_case or cases[0]["case_id"] if cases else None
    return {
        "heldout_cases": list(HELDOUT_CASES),
        "cases": cases,
        "cases_solved": solved_total,
        "cases_total": len(cases),
        "trace_case": detail,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Unit D-1 held-out harness")
    parser.add_argument("--genome", default=None,
                        help="checkpoint / best_agents / best_gen json")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--heldout-dir", default=_HELDOUT_DIR)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    genome = genome_from_checkpoint(args.genome) if args.genome else None
    report = run(genome, episodes=args.episodes, seed=args.seed,
                 heldout_dir=args.heldout_dir)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
