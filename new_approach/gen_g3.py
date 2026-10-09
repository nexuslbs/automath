"""G3 EVOLUTIONARY DIVERSITY / NOVELTY driver (Unit D).

A bounded, detached population run that CONTINUES from the trained Phase-A
genome:

  * population = the seed genome + mutants (higher mutation rate, 0.6 = 2.4x
    the Phase-A 0.25);
  * fitness = solve_rate + lambda * novelty(genome behaviour distance), with
    novelty the mean pairwise L1 distance between per-action preference
    vectors (lambda = cfg.novelty_lambda, default 0.5);
  * restart-on-stall: after ``restart_on_stall`` generations without a fitness
    improvement, the non-elite population is re-seeded from the best elite with
    fresh gaussian noise;
  * bounded: ``wall_clock_hours`` (1.5 h) OR ``generations`` (600) OR the
    150-generation stall auto-stop, whichever comes FIRST;
  * checkpoint + history.jsonl every generation and a timestamped snapshot
    every ``heartbeat_generations`` (200) and at the end.

Launch (detached, survives the session):

    mkdir -p /opt/automath/tmp/gen3/checkpoints
    cd /opt/automath/repo
    setsid nohup /opt/automath/venv/bin/python -m new_approach.gen_g3 \
        --config /opt/automath/repo/config/evolution_gen3.json \
        < /dev/null >> /opt/automath/tmp/gen3/gen3.log 2>&1 &
    echo $!
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import time
from typing import List

from . import gen_common as gc
from .evolution_bundle import make_demo_bundle
from .evolution_population import (
    EvoConfig,
    Genome,
    append_history_jsonl,
    atomic_write_json,
    clone_genome,
    crossover,
    mutate,
    save_generation,
)
from .evolution_semi import run_bundle_shaped, train_genome_shaped

_STOP = False
_STARTED_AT = time.time()


def _install_signals() -> None:
    def _term(signum, frame):  # noqa: ANN001
        global _STOP
        _STOP = True
    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    try:
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    except (AttributeError, ValueError):
        pass


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def emit(msg: str) -> None:
    sys.stdout.write("[%s] %s\n" % (utc_now(), msg))
    sys.stdout.flush()


def novelty_of(genome: Genome, pop: List[Genome]) -> float:
    if len(pop) < 2:
        return 0.0
    n = max(1, len(genome.pref))
    tot = 0.0
    for other in pop:
        if other is genome:
            continue
        tot += sum(abs(a - b) for a, b in zip(genome.pref, other.pref)) / n
    return tot / max(1, len(pop) - 1)


def evaluate(genome: Genome, bundle, cfg: EvoConfig, idx: int) -> None:
    agent = train_genome_shaped(genome, bundle, cfg, cfg.seed + 7919 * idx)
    res = run_bundle_shaped(agent, bundle, genome.state_pref,
                            total_budget=cfg.total_budget, training=False)
    genome._result = res
    genome.solved_feasible = res.solved_feasible
    genome.steps = res.total_steps
    genome.solver_score = res.solved_rate()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Unit D G3 novelty evolution")
    p.add_argument("--config", required=True)
    p.add_argument("--generations", type=int, default=None)
    p.add_argument("--wall-clock-hours", dest="wall_clock_hours", type=float,
                   default=None)
    p.add_argument("--checkpoint-dir", dest="checkpoint_dir", default=None)
    p.add_argument("--progress-dir", dest="progress_dir", default=None)
    args = p.parse_args(argv)

    cfg = EvoConfig.from_dict(json.load(open(args.config)))
    for name in ("generations", "wall_clock_hours", "checkpoint_dir",
                 "progress_dir"):
        val = getattr(args, name)
        if val is not None:
            setattr(cfg, name, val)
    lam = float(getattr(cfg, "novelty_lambda", 0.5))
    restart = int(getattr(cfg, "restart_on_stall", 150))
    _install_signals()
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.progress_dir, exist_ok=True)
    bundle = make_demo_bundle(cfg.total_budget)
    rng = random.Random(cfg.seed)
    seed, seed_gen = gc.load_seed_genome()
    gc.export_seed_genome(os.path.join(cfg.progress_dir, "seed_genome.json"))
    emit("G3 START config=%s seed_gid=%s source_gen=%d population=%d "
         "episodes=%d generations=%d wall_clock_hours=%.2f mutation_rate=%.2f "
         "lambda=%.2f restart_on_stall=%d bundle=%s"
         % (args.config, seed.gid, seed_gen, cfg.population, cfg.episodes,
            cfg.generations, cfg.wall_clock_hours, cfg.mutation_rate, lam,
            restart, bundle.fingerprint()[:16]))

    population: List[Genome] = [clone_genome(seed, gid="g3-0000-seed",
                                             origin="seed")]
    for i in range(max(0, cfg.population - 1)):
        c = clone_genome(seed, gid="g3-0000-m%02d" % i, origin="seed-mutant")
        mutate(c, rng, cfg)
        population.append(c)

    deadline = _STARTED_AT + cfg.wall_clock_hours * 3600.0
    best_ever = -1e308
    stall = 0
    history: List[dict] = []
    best = population[0]
    exit_code = 0
    gen = 0

    while not _STOP and gen < cfg.generations and time.time() < deadline:
        gen += 1
        try:
            for i, g in enumerate(population):
                g.gid = "g3-%06d-%s" % (gen, g.gid.split("-")[-1])
                evaluate(g, bundle, cfg, i)
            for g in population:
                g.fitness = round(g.solver_score + lam * novelty_of(g, population), 6)
            population.sort(key=lambda g: (-g.fitness, g.gid))
            best = population[0]
            feasible = len(bundle.feasible())
            record = {
                "generation": gen, "timestamp": utc_now(),
                "best_gid": best.gid,
                "best_fitness": round(best.fitness, 6),
                "best_solved": best.solved_feasible,
                "best_shaped_return": round(
                    best._result.shaped_return if best._result else 0.0, 6),
                "best_novelty": round(novelty_of(best, population), 6),
                "mean_fitness": round(
                    sum(g.fitness for g in population) / len(population), 6),
                "population": [g.to_dict() for g in population],
            }
            history.append(record)
            if len(history) > 200:
                history = history[-200:]
            append_history_jsonl(cfg, record)
            if cfg.checkpoint_every and gen % cfg.checkpoint_every == 0:
                save_generation(cfg, gen, population, history, bundle, best)
            emit("GEN gen=%d best=%.4f solved=%d/%d novelty=%.4f mean=%.4f "
                 "stall=%d best_gid=%s"
                 % (gen, best.fitness, best.solved_feasible, feasible,
                    novelty_of(best, population), record["mean_fitness"],
                    stall, best.gid))
            if best.fitness > best_ever + cfg.plateau_tol:
                best_ever = best.fitness
                stall = 0
            else:
                stall += 1
            if gen % cfg.heartbeat_generations == 0 or gen == cfg.generations:
                snap = os.path.join(
                    cfg.progress_dir, "snapshot_%s_gen%06d.json"
                    % (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()), gen))
                atomic_write_json(snap, {
                    "timestamp": utc_now(), "generation": gen,
                    "best": best.to_dict(), "best_ever": best_ever,
                    "seed_gid": seed.gid, "novelty_lambda": lam,
                    "mutation_rate": cfg.mutation_rate,
                    "per_state": best._result.per_state_rows() if best._result else [],
                })
                emit("SNAPSHOT gen=%d file=%s" % (gen, snap))
            if stall >= restart:
                emit("RESTART stall=%d reseeding non-elites from %s"
                     % (stall, best.gid))
                for i in range(cfg.elites, len(population)):
                    c = clone_genome(best, gid="g3-%06d-rst%02d" % (gen, i),
                                     origin="restart")
                    for _ in range(3):
                        mutate(c, rng, cfg)
                    population[i] = c
                stall = 0
            # reproduce
            elites = [clone_genome(g, gid=g.gid, origin="elite")
                      for g in population[:max(1, cfg.elites)]]
            parents = population[:max(2, cfg.top_k)]
            children: List[Genome] = []
            while len(elites) + len(children) < cfg.population:
                a, b = rng.sample(parents, 2)
                c = crossover(a, b, rng)
                c.gid = "g3-%06d-c%03d" % (gen, len(children))
                c.origin = "offspring"
                mutate(c, rng, cfg)
                children.append(c)
            population = elites + children
        except Exception as exc:  # noqa: BLE001 - keep the long run alive
            emit("GEN_ERROR %s: %s" % (type(exc).__name__, exc))
            time.sleep(1.0)

    elapsed = time.time() - _STARTED_AT
    try:
        save_generation(cfg, gen, population, history, bundle, best)
        atomic_write_json(os.path.join(cfg.progress_dir, "final.json"), {
            "timestamp": utc_now(), "generation": gen, "best": best.to_dict(),
            "best_ever": best_ever, "elapsed_secs": round(elapsed, 3),
            "seed_gid": seed.gid,
        })
    except Exception as exc:  # noqa: BLE001
        emit("FINAL_SAVE_ERROR %s" % exc)
        exit_code = 3
    reason = ("signal" if _STOP else
              ("stall_restart_limit" if stall >= restart else
               ("generation_cap" if gen >= cfg.generations else
                ("wall_clock" if time.time() >= deadline else "stopped"))))
    emit("G3 STOP reason=%s generation=%d elapsed=%.1fs best=%.4f exit=%d"
         % (reason, gen, elapsed, best.fitness, exit_code))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
