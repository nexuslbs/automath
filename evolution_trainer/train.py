"""Configurable training runner for the Unit B evolutionary process.

    python -m evolution_trainer.train \
        --spec spec_minimal --spec spec_dynamic_group \
        --generations 30 --population 16 \
        --step-cost 0.05 --reproduction-fee 1.0 --mutation-rate 0.15 \
        --subagent-depth 2 --max-episode-steps 8 \
        --out /opt/automath/tmp/unit-B/train

Every knob of :class:`EvoConfig` is a flag (a JSON config file can supply the
same values with ``--json-config``); CLI flags win. The runner writes, per spec:
``checkpoints/best_gen_XXXX.json`` (best agent of each generation),
``checkpoints/best_agents_persist.json`` (accumulated winners),
``history/history.csv`` (the generation table) and
``history/reward_trajectory.csv`` (the reward curve), then prints a run summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import fields
from typing import Dict, List, Optional, Sequence

from dynamic_env.spec import load_spec

from .evolution import EvoConfig, EvolutionTrainer
from .reporting import first_birth, render_generation_summary, render_reward_trajectory

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_DEFAULT_SPEC_DIR = os.path.join(_REPO_ROOT, "data", "dynamic_env")


def _build_config(args: argparse.Namespace) -> EvoConfig:
    config = EvoConfig()
    if args.json_config:
        with open(args.json_config, "r", encoding="utf-8") as handle:
            config = EvoConfig.from_dict(json.load(handle))
    # CLI overrides: only flags the user actually set on top of the defaults.
    for name in (f.name for f in fields(EvoConfig)):
        value = getattr(args, name, None)
        if value is not None:
            setattr(config, name, value)
    return config


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--spec-dir", default=_DEFAULT_SPEC_DIR)
    parser.add_argument("--spec", action="append", default=None,
                        help="spec stem (repeatable); overrides --spec-list")
    parser.add_argument("--spec-list", default="spec_minimal,spec_dynamic_group",
                        help="comma-separated spec stems")
    parser.add_argument("--json-config", default=None)
    parser.add_argument("--out", default="evolution_out")
    parser.add_argument("--population", type=int, default=None, dest="population_size")
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--episodes-per-agent", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--step-cost", type=float, default=None)
    parser.add_argument("--intermediate-reward", type=float, default=None)
    parser.add_argument("--guard-bonus", type=float, default=None)
    parser.add_argument("--subgoal-reward", type=float, default=None)
    parser.add_argument("--goal-reward", type=float, default=None)
    parser.add_argument("--reproduction-fee", type=float, default=None)
    parser.add_argument("--parent-frac", type=float, default=None)
    parser.add_argument("--elite-frac", type=float, default=None)
    parser.add_argument("--mutation-rate", type=float, default=None)
    parser.add_argument("--mutation-sigma", type=float, default=None)
    parser.add_argument("--mix-alpha", type=float, default=None)
    parser.add_argument("--select-prob", type=float, default=None)
    parser.add_argument("--subagent-depth", type=int, default=None)
    parser.add_argument("--subagent-spawn-fee", type=float, default=None)
    parser.add_argument("--subagent-credit-cap", type=float, default=None)
    parser.add_argument("--max-subagents-per-episode", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)
    args = parser.parse_args(argv)

    specs = args.spec or [s for s in args.spec_list.split(",") if s]
    if not specs:
        raise SystemExit("train: no specs given")
    started = time.perf_counter()
    summaries: List[Dict[str, object]] = []
    for spec_stem in specs:
        spec = load_spec(os.path.join(args.spec_dir, spec_stem + ".json"))
        config = _build_config(args)
        config.checkpoint_dir = os.path.join(args.out, spec.spec_id, "checkpoints")
        config.history_dir = os.path.join(args.out, spec.spec_id, "history")
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.history_dir, exist_ok=True)
        with open(os.path.join(config.checkpoint_dir, "config.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"spec_id": spec.spec_id, "config": config.to_dict()},
                      handle, indent=2, sort_keys=True)
        trainer = EvolutionTrainer(spec, config, spec_id=spec.spec_id)
        t0 = time.perf_counter()
        history = trainer.run()
        elapsed = time.perf_counter() - t0
        best = max(history, key=lambda r: r["best_fitness"])
        total_births = sum(len(r["births"]) for r in history)
        total_deaths = sum(len(r["deaths"]) for r in history)
        total_subagents = sum(len(r["subagents"]) for r in history)
        hull_ok = all(b["hull_ok"] for r in history for b in r["births"])
        summary = {
            "spec_id": spec.spec_id,
            "generations": config.generations,
            "population": config.population_size,
            "genome_size": trainer.net.size,
            "in_dim": trainer.in_dim,
            "best_generation": best["generation"],
            "best_agent": best["best_agent"],
            "best_fitness": best["best_fitness"],
            "final_mean_fitness": history[-1]["mean_fitness"],
            "total_births": total_births,
            "total_deaths": total_deaths,
            "total_subagents": total_subagents,
            "all_children_in_weight_hull": hull_ok,
            "seconds": round(elapsed, 4),
            "checkpoints": config.checkpoint_dir,
            "history": config.history_dir,
        }
        summaries.append(summary)
        print("=" * 96)
        print("SPEC %s" % spec.spec_id)
        print(render_generation_summary(history))
        print("REWARD TRAJECTORY")
        print(render_reward_trajectory(history))
        birth = first_birth(history)
        if birth is not None:
            print("FIRST BIRTH: %s <- %s (blended=%d selected=%d |delta|=%.4f hull_ok=%s)"
                  % (birth["child"], birth["parents"], birth["blend_count"],
                     birth["select_count"], birth["mutation_magnitude"], birth["hull_ok"]))
        print(json.dumps(summary, indent=2, sort_keys=True))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "run_summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"summaries": summaries, "wall_seconds": round(time.perf_counter() - started, 4)},
                  handle, indent=2, sort_keys=True)
    print("=" * 96)
    print("TRAIN OK: %d spec(s) in %.2fs; summary at %s"
          % (len(specs), time.perf_counter() - started,
             os.path.join(args.out, "run_summary.json")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
