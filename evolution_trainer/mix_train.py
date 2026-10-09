"""MIX arm - RL warm-start of founder genomes, then fixed-weight evolution.

This is arm 3 of the Unit C approach comparison. The concrete mix chosen is:

1. **Warm start (policy gradient).** Run :class:`ReinforceTrainer` for
   ``warm_generations`` generations to obtain a genome that already points the
   target-conditioned policy in the rewarding direction. Gradients are dense and
   cheap on this tiny network, so a short warm start is a good use of the budget.
2. **Fixed-weight evolution.** Seed the Unit B population with that genome plus
   ``population_size - 1`` mutated copies (diversity), then run the UNCHANGED
   evolutionary process for ``evo_generations`` generations. Evolution does the
   discrete/compositional search (two-parent weight mixing + mutation) that a
   pure gradient cannot: the reward is sparse in the action-word structure and
   episodes are short.

Justification: the two signals are complementary and the budget is split, not
doubled - warm and evolution generations always add up to the pure arms'
``generations``, so the episode budget matches across all three arms. The mix
candidate is final-selected by the SAME greedy harness evaluation as the other
arms (RL best genome vs evolution all-time best genome).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

from dynamic_env.spec import load_spec

from .evolution import EvoConfig, EvolutionTrainer
from .genome import mutate_genome, round_genome
from .harness import Harness
from .rl_train import ReinforceTrainer

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_DEFAULT_SPEC_DIR = os.path.join(_REPO_ROOT, "data", "dynamic_env")


class MixTrainer:
    """RL warm-start followed by fixed-weight evolution on the same network."""

    def __init__(self, spec, config: Optional[EvoConfig] = None,
                 warm_generations: int = 24, evo_generations: int = 36,
                 learning_rate: float = 0.10, seed: int = 7,
                 spec_id: str = "") -> None:
        self.spec = spec
        self.spec_id = spec_id or spec.spec_id
        self.config = config or EvoConfig()
        self.warm_generations = warm_generations
        self.evo_generations = evo_generations
        self.learning_rate = learning_rate
        self.seed = seed
        self.trajectory: List[Dict[str, Any]] = []
        self.warm_best_fitness = float("-inf")
        self.warm_mean_fitness = 0.0

    def run(self) -> List[Dict[str, Any]]:
        rng = random.Random(self.seed)
        # -- phase 1: RL warm start ---------------------------------------
        rl = ReinforceTrainer(self.spec, self.config,
                              learning_rate=self.learning_rate, seed=self.seed,
                              spec_id=self.spec_id)
        rl_traj = rl.run(self.warm_generations)
        self.trajectory.extend(rl_traj)
        self.warm_best_fitness = rl.best_fitness
        self.warm_mean_fitness = rl_traj[-1]["mean_fitness"] if rl_traj else 0.0

        # -- seed the population: best RL genome + mutated copies ----------
        seeds: List[List[float]] = [list(rl.best_genome)]
        while len(seeds) < self.config.population_size:
            mix = mutate_genome(rl.best_genome, rng,
                                self.config.mutation_rate, self.config.mutation_sigma)
            seeds.append(mix.child)

        # -- phase 2: fixed-weight evolution (Unit B, unchanged) -----------
        evo_config = EvoConfig(**self.config.to_dict())
        evo_config.generations = self.evo_generations
        evo_config.checkpoint_dir = os.path.join("mix_checkpoints", self.spec_id)
        evo_config.history_dir = os.path.join("mix_history", self.spec_id)
        trainer = EvolutionTrainer(self.spec, evo_config, spec_id=self.spec_id)
        history = trainer.run(warm_start=seeds)
        offset = self.warm_generations
        for row in history:
            self.trajectory.append({
                "generation": offset + row["generation"],
                "best_fitness": row["best_fitness"],
                "mean_fitness": row["mean_fitness"],
                "solved": row["solved"],
                "best_steps": row["best_steps"],
            })
        # -- pick the final candidate by the shared greedy evaluation ------
        harness = Harness(
            self.spec, hidden=self.config.hidden,
            step_cost=self.config.step_cost,
            intermediate_reward=self.config.intermediate_reward,
            guard_bonus=self.config.guard_bonus,
            subgoal_reward=self.config.subgoal_reward,
            goal_reward=self.config.goal_reward,
            max_episode_steps=self.config.max_episode_steps,
        )
        persist_path = os.path.join(evo_config.checkpoint_dir,
                                    "best_agents_persist.json")
        candidates: List[tuple] = [(rl.best_genome, "rl_warm")]
        if os.path.exists(persist_path):
            with open(persist_path, "r", encoding="utf-8") as handle:
                persist = json.load(handle)
            all_time = persist.get("all_time_best")
            if all_time is not None:
                candidates.append((all_time["agent"]["genome"], "evolution"))
        best_genome, best_source, best_greedy = None, None, float("-inf")
        for genome, source in candidates:
            greedy = harness.rollout(genome, explore=False)
            if greedy.total_return > best_greedy:
                best_greedy = greedy.total_return
                best_genome = list(genome)
                best_source = source
        self.harness = harness
        self.best_genome = best_genome
        self.best_source = best_source
        self.best_fitness = best_greedy
        return self.trajectory


def run_one(spec_path: str, out_dir: str, generations: int = 60,
            population: int = 12, seed: int = 7, hidden: int = 12,
            learning_rate: float = 0.10, warm_frac: float = 0.40,
            eval_episodes: int = 30) -> Dict[str, Any]:
    spec = load_spec(spec_path)
    config = EvoConfig(generations=generations, population_size=population,
                       hidden=hidden, seed=seed)
    warm = max(1, int(round(generations * warm_frac)))
    evo = max(1, generations - warm)
    trainer = MixTrainer(spec, config, warm_generations=warm,
                         evo_generations=evo, learning_rate=learning_rate,
                         seed=seed, spec_id=spec.spec_id)
    started = time.perf_counter()
    trainer.run()
    wall = time.perf_counter() - started
    evaluation = trainer.harness.evaluate(trainer.best_genome,
                                          episodes=eval_episodes, seed=seed + 999)
    os.makedirs(out_dir, exist_ok=True)
    traj_path = os.path.join(out_dir, "trajectory.csv")
    with open(traj_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "generation", "best_fitness", "mean_fitness", "solved", "best_steps"])
        writer.writeheader()
        writer.writerows(trainer.trajectory)
    with open(os.path.join(out_dir, "best_genome.json"), "w", encoding="utf-8") as handle:
        json.dump({"spec_id": spec.spec_id, "arm": "mix",
                   "best_fitness": round(trainer.best_fitness, 6),
                   "best_source": trainer.best_source,
                   "genome": round_genome(trainer.best_genome)}, handle, indent=2)
    summary = {
        "arm": "mix",
        "spec_id": spec.spec_id,
        "generations": generations,
        "population": population,
        "seed": seed,
        "warm_generations": warm,
        "evo_generations": evo,
        "episodes": generations * population,
        "genome_size": trainer.harness.net.size,
        "in_dim": trainer.harness.in_dim,
        "best_fitness": round(trainer.best_fitness, 6),
        "best_source": trainer.best_source,
        "warm_best_fitness": round(trainer.warm_best_fitness, 6),
        "final_mean_fitness": trainer.trajectory[-1]["mean_fitness"],
        **evaluation,
        "wall_seconds": round(wall, 4),
        "trajectory": traj_path,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--spec-dir", default=_DEFAULT_SPEC_DIR)
    parser.add_argument("--out", required=True)
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hidden", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.10)
    parser.add_argument("--warm-frac", type=float, default=0.40)
    parser.add_argument("--eval-episodes", type=int, default=30)
    args = parser.parse_args(argv)
    spec_path = args.spec
    if not os.path.exists(spec_path):
        spec_path = os.path.join(args.spec_dir, args.spec + ".json")
    run_one(spec_path, args.out, generations=args.generations,
            population=args.population, seed=args.seed, hidden=args.hidden,
            learning_rate=args.learning_rate, warm_frac=args.warm_frac,
            eval_episodes=args.eval_episodes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
