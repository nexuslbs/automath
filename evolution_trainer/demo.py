"""Demo runner: a SHORT, deterministic evolutionary run on shipped specs.

    python -m evolution_trainer.demo
    python -m evolution_trainer.demo --spec spec_minimal --spec spec_dynamic_group \
        --generations 12 --population 12 --out /tmp/evo-demo

The demo prints, for every spec: the policy geometry (state dim, action dim,
genome size), the per-generation table (agents, fitness/reward, step counts,
which parents produced which offspring, mutation magnitudes, deaths, subagents),
the reward trajectory, the first reproduction with its inherited-weights note,
and a direct weights-mixing hull proof.

Nothing about a spec is hardcoded: the policy geometry is derived from the spec
DATA through ``dynamic_env``.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
from typing import List, Sequence

from dynamic_env.spec import load_spec

from .evolution import EvoConfig, EvolutionTrainer
from .features import state_dim
from .genome import mix_genomes
from .reporting import (first_birth, render_agent_table,
                        render_generation_summary, render_reward_trajectory)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_DEFAULT_SPEC_DIR = os.path.join(_REPO_ROOT, "data", "dynamic_env")
_DEFAULT_SPECS = ("spec_minimal", "spec_dynamic_group", "spec_dynamic_axiom")


def mixing_hull_proof(seed: int = 12345, length: int = 8) -> str:
    """Two synthetic parents -> blended child; print the hull/delta proof."""
    rng = random.Random(seed)
    parent_a = [rng.uniform(-2.0, 2.0) for _ in range(length)]
    parent_b = [rng.uniform(-2.0, 2.0) for _ in range(length)]
    result = mix_genomes(parent_a, parent_b, rng, select_prob=0.0,
                         per_gene_blend=True, mutation_rate=1.0,
                         mutation_sigma=0.1)
    lines = ["parent_a : %s" % [round(v, 4) for v in parent_a],
             "parent_b : %s" % [round(v, 4) for v in parent_b],
             "blend    : %s" % [round(v, 4) for v in result.blend],
             "delta    : %s" % [round(v, 4) for v in result.deltas],
             "child    : %s" % [round(v, 4) for v in result.child],
             "child == blend + delta for every gene: %s"
             % all(abs(c - (b + d)) < 1e-12
                   for c, b, d in zip(result.child, result.blend, result.deltas)),
             "every blend gene inside [min(parent_a,parent_b), max(...)]: %s"
             % all(min(a, b) - 1e-12 <= m <= max(a, b) + 1e-12
                   for a, b, m in zip(parent_a, parent_b, result.blend))]
    return "\n".join(lines)


def run_one(spec_stem: str, spec_dir: str, generations: int, population: int,
            seed: int, out_dir: str) -> int:
    path = os.path.join(spec_dir, spec_stem + ".json")
    if not os.path.exists(path):
        raise SystemExit("demo: spec not found: %s" % path)
    spec = load_spec(path)
    config = EvoConfig(
        population_size=population,
        generations=generations,
        seed=seed,
        checkpoint_dir=os.path.join(out_dir, spec.spec_id, "checkpoints"),
        history_dir=os.path.join(out_dir, spec.spec_id, "history"),
    )
    trainer = EvolutionTrainer(spec, config, spec_id=spec.spec_id)
    history = trainer.run()

    print("=" * 100)
    print("spec            : %s" % spec.spec_id)
    print("description     : %s" % spec.description)
    print("state dim       : %d (env.state_vector + %d-dim subgoal mask)"
          % (state_dim(spec), len(trainer.root_mask)))
    print("action dim      : %d (derived per spec)" % trainer.featurizer.dim)
    print("network in dim  : %d | hidden=%d | genome size=%d genes"
          % (trainer.in_dim, trainer.config.hidden, trainer.net.size))
    print("population      : %d | generations=%d | seed=%d"
          % (config.population_size, config.generations, config.seed))
    print()
    print("GENERATION TABLE")
    print(render_generation_summary(history))
    print()
    print("AGENT TABLE (first 3 generations and the first generation with births)")
    early = [g for g in (1, 2, 3) if g <= len(history)]
    birth = first_birth(history)
    if birth is not None:
        for record in history:
            if record["births"]:
                if record["generation"] not in early:
                    early.append(record["generation"])
                break
    print(render_agent_table(history, generations=early, max_rows=120))
    print()
    print("REWARD TRAJECTORY (best_fitness / mean_fitness / births)")
    print(render_reward_trajectory(history))
    print()
    if birth is not None:
        print("FIRST REPRODUCTION (two parents spend the fee, child shares both)")
        print("  generation      : %d" % next(r["generation"] for r in history
                                              if r["births"] and r["births"][0] is birth))
        print("  parents         : %s" % birth["parents"])
        print("  child           : %s" % birth["child"])
        print("  fee paid        : %.2f (half from each parent)" % birth["fee_paid"])
        print("  blended genes   : %d | selected genes: %d"
              % (birth["blend_count"], birth["select_count"]))
        print("  mutation |delta|: %.6f" % birth["mutation_magnitude"])
        print("  convex-hull ok  : %s" % birth["hull_ok"])
        print("  note            : %s" % birth["note"])
    else:
        print("FIRST REPRODUCTION: none in this short run")
    print()
    print("SUBAGENT RECURSION")
    found = False
    for record in history:
        for sub in record["subagents"]:
            found = True
            print("  gen %d: %s parents=%s depth=%d mask=%s return=%.4f credit=%.4f"
                  % (record["generation"], sub["agent_id"], sub["parents"],
                     sub["depth"], sub["subgoal_mask"], sub["return"], sub["credit"]))
            for nested in sub.get("nested", []):
                print("     nested: %s parents=%s depth=%d return=%.4f credit=%.4f"
                      % (nested["agent_id"], nested["parents"], nested["depth"],
                         nested["return"], nested["credit"]))
            break
        if found:
            break
    if not found:
        print("  no subagent spawned in this short run (the unit test forces recursion)")
    best = max(history, key=lambda r: r["best_fitness"])
    print()
    print("BEST: gen %d agent %s fitness %.4f (checkpointed under %s)"
          % (best["generation"], best["best_agent"], best["best_fitness"],
             config.checkpoint_dir))
    return 0


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-dir", default=_DEFAULT_SPEC_DIR)
    parser.add_argument("--spec", action="append", default=None,
                        help="spec stem to run (repeatable; default minimal/dynamic_group/dynamic_axiom)")
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=os.path.join(tempfile.gettempdir(),
                                                      "evolution_trainer_demo"))
    args = parser.parse_args(argv)

    specs = args.spec or list(_DEFAULT_SPECS)
    print("evolution_trainer demo: %d spec(s), %d generations each, population %d"
          % (len(specs), args.generations, args.population))
    print("output dir: %s" % args.out)
    print()
    print("WEIGHTS-MIXING HULL PROOF (synthetic parents, every blend gene in the hull)")
    print(mixing_hull_proof())
    for spec_stem in specs:
        run_one(spec_stem, args.spec_dir, args.generations, args.population,
                args.seed, args.out)
    print("=" * 100)
    print("DEMO OK: %d spec(s) evolved, artifacts under %s" % (len(specs), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
