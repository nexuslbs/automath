"""Unit C (Phase B) long-run driver: PURE evolution with NO reward shaping.

Launch (detached, survives session teardown)::

    mkdir -p /opt/automath/tmp/pure-evo/checkpoints /opt/automath/tmp/pure-evo/progress
    cd /opt/automath/repo
    setsid nohup /opt/automath/venv/bin/python -m new_approach.evolution_pure_loop \
        --config config/evolution_pure.json \
        < /dev/null >> /opt/automath/tmp/pure-evo/phaseB.log 2>&1 &
    echo $!

The loop evolves the population generation after generation with
``evolution_pure.evolve_one_generation_pure`` (agents learn from the pure
objective return ``step_cost + goal``; parents are chosen by reward threshold on
the pure return + solved rate;
offspring come from BLX-alpha / uniform crossover + mutation).  It:

* checkpoints the population every generation (``checkpoint_every``) and appends
  one record per generation to ``history.jsonl``;
* writes a timestamped progress snapshot every ``heartbeat_secs`` seconds AND
  every ``heartbeat_generations`` generations, plus one at every validation
  milestone;
* runs the validation cadence: CORE 33-case (every ``validation_every`` gens)
  and EXTENDED 114-case (every ``validation_ext_every`` gens) PURE-evolution
  agents trained with the pure objective (no shaping), plus the greedy bundle
  report (per-state) from the generation's own evaluation;
* auto-stops on the FIRST of: best fitness plateau for
  ``plateau_generations`` (tol ``plateau_tol``), ``generations`` cap, or
  ``wall_clock_hours``; writes a final checkpoint + snapshot and exits cleanly;
* ignores SIGHUP and exits cleanly on SIGTERM/SIGINT.

Pure standard library, bounded RAM.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import resource
import signal
import sys
import time
from typing import Dict, List, Optional, Sequence

from .evolution_bundle import make_demo_bundle
from .evolution_population import (
    EvoConfig,
    Genome,
    append_history_jsonl,
    atomic_write_json,
    init_population,
    load_checkpoint,
    restore_population,
    save_generation,
)
from .evolution_pure import (
    core_validation_pure,
    evo_validation_pure,
    evolve_one_generation_pure,
)

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


def load_config(path: str) -> EvoConfig:
    with open(path, "r") as fh:
        data = json.load(fh)
    return EvoConfig.from_dict(data)


def _rss_mb() -> float:
    try:
        with open("/proc/self/status", "r") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def bucket_curve(history: Sequence[dict], buckets: int = 12) -> List[dict]:
    n = len(history)
    if n == 0:
        return []
    step = max(1, n // buckets)
    out: List[dict] = []
    for start in range(0, n, step):
        chunk = history[start:start + step]
        if not chunk:
            continue
        out.append({
            "gen_from": chunk[0]["generation"],
            "gen_to": chunk[-1]["generation"],
            "best_fitness_mean": round(
                sum(r["best_fitness"] for r in chunk) / len(chunk), 6),
            "mean_fitness_mean": round(
                sum(r["mean_fitness"] for r in chunk) / len(chunk), 6),
            "best_shaped_return_mean": round(
                sum(r.get("best_shaped_return", 0.0) for r in chunk)
                / len(chunk), 6),
        })
    return out


def write_snapshot(cfg: EvoConfig, gen: int, population: List[Genome],
                   best: Genome, mean_fitness: float, history: Sequence[dict],
                   label: str = "", validation: Optional[dict] = None) -> str:
    os.makedirs(cfg.progress_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = os.path.join(
        cfg.progress_dir,
        "snapshot_%s_gen%06d%s.json" % (stamp, gen, ("_" + label)
                                        if label else ""))
    result = best._result
    elapsed = time.time() - _STARTED_AT
    payload = {
        "timestamp": utc_now(),
        "label": label,
        "generation": gen,
        "best_gid": best.gid,
        "best_fitness": round(best.fitness, 6),
        "best_shaped_return": round(result.shaped_return, 6) if result else 0.0,
        "best_solved_feasible": best.solved_feasible,
        "best_genes": best.genes(),
        "best_parents": list(best.parents),
        "mean_fitness": round(mean_fitness, 6),
        "per_state": result.per_state_rows() if result else [],
        "per_state_reward_log": (result.reward_log[:40] if result else []),
        "elapsed_secs": round(elapsed, 3),
        "generations_per_hour": round(gen / max(elapsed, 1e-9) * 3600.0, 3),
        "rss_mb": round(_rss_mb(), 3),
        "population": len(population),
        "checkpoint_dir": cfg.checkpoint_dir,
        "reward_design": {
            "objective_only": "step_cost + goal_reward; NO shaping/subgoal",
            "step_cost": cfg.step_cost, "goal_reward": 1.0,
            "phi_scale": 0.0, "subgoal_bonus": 0.0,
            "shaping_cap": 0.0, "solved_rate_weight": cfg.solved_rate_weight,
        },
        "curve": bucket_curve(history),
    }
    if validation is not None:
        payload["validation"] = validation
    atomic_write_json(path, payload)
    emit("SNAPSHOT gen=%d best=%.4f shaped=%.4f solved=%d/%d rss=%.1fMB file=%s"
         % (gen, best.fitness,
            (result.shaped_return if result else 0.0),
            best.solved_feasible,
            (result.feasible_total if result else 0),
            payload["rss_mb"], path))
    return path


def run_validation(cfg: EvoConfig, gen: int, best: Genome,
                   bundle, do_ext: bool) -> dict:
    val: Dict[str, object] = {
        "timestamp": utc_now(), "generation": gen,
        "core33": None, "ext114": None, "bundle": None,
    }
    seed = cfg.seed + 1000003 * gen
    t0 = time.time()
    core_pass, core_total = core_validation_pure(
        best, cfg, seed, cfg.validation_episodes)
    val["core33"] = {"passed": core_pass, "total": core_total,
                     "episodes": cfg.validation_episodes,
                     "wall_secs": round(time.time() - t0, 3)}
    emit("VALIDATE core33 gen=%d %d/%d wall=%.1fs"
         % (gen, core_pass, core_total, time.time() - t0))
    if do_ext:
        t0 = time.time()
        ext_pass, ext_total = evo_validation_pure(
            best, cfg, seed, cfg.validation_episodes)
        val["ext114"] = {"passed": ext_pass, "total": ext_total,
                         "episodes": cfg.validation_episodes,
                         "wall_secs": round(time.time() - t0, 3)}
        emit("VALIDATE ext114 gen=%d %d/%d wall=%.1fs"
             % (gen, ext_pass, ext_total, time.time() - t0))
    if best._result is not None:
        val["bundle"] = {
            "solved_feasible": best.solved_feasible,
            "feasible_total": len(bundle.feasible()),
            "shaped_return": round(best._result.shaped_return, 6),
            "total_steps": best._result.total_steps,
            "per_state": best._result.per_state_rows(),
        }
    return val


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Unit C Phase B pure-evolution loop (no shaping)")
    p.add_argument("--config", required=True, help="JSON config file")
    p.add_argument("--generations", type=int, default=None)
    p.add_argument("--wall-clock-hours", dest="wall_clock_hours", type=float,
                   default=None)
    p.add_argument("--checkpoint-dir", dest="checkpoint_dir", default=None)
    p.add_argument("--progress-dir", dest="progress_dir", default=None)
    p.add_argument("--heartbeat-secs", dest="heartbeat_secs", type=float,
                   default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-ext", dest="no_ext", action="store_true",
                   help="skip the 114-case validation (core only)")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    for name in ("generations", "wall_clock_hours", "checkpoint_dir",
                 "progress_dir", "heartbeat_secs", "seed"):
        val = getattr(args, name)
        if val is not None:
            setattr(cfg, name, val)
    _install_signals()
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.progress_dir, exist_ok=True)
    bundle = make_demo_bundle(cfg.total_budget)
    emit("START config=%s population=%d episodes=%d total_budget=%d "
         "generations=%d wall_clock_hours=%.2f plateau_generations=%d "
         "validation_every=%d validation_ext_every=%d bundle=%s"
         % (args.config, cfg.population, cfg.episodes, cfg.total_budget,
            cfg.generations, cfg.wall_clock_hours, cfg.plateau_generations,
            cfg.validation_every, cfg.validation_ext_every,
            bundle.fingerprint()[:16]))

    ck_path = os.path.join(cfg.checkpoint_dir, "checkpoint.json")
    rng = random.Random(cfg.seed)
    history: List[dict] = []
    start_gen = 0
    if args.resume and os.path.exists(ck_path):
        ck = load_checkpoint(ck_path)
        assert ck is not None
        population = restore_population(ck)
        start_gen = int(ck.get("generation", 0))
        emit("RESUME loaded=%s generation=%d population=%d"
             % (ck_path, start_gen, len(population)))
    else:
        population = init_population(cfg, bundle, rng)
        emit("INIT fresh population=%d generation=0" % len(population))

    deadline = _STARTED_AT + cfg.wall_clock_hours * 3600.0
    best = max(population, key=lambda g: g.fitness)
    best_ever = -1e308
    gens_since_improve = 0
    gen = start_gen
    last_snap_time = time.time()
    last_snap_gen = start_gen
    exit_code = 0
    emit("LOOP begin generation=%d deadline=%s"
         % (gen, time.strftime("%Y-%m-%dT%H:%M:%SZ",
                               time.gmtime(deadline))))

    while not _STOP and gen < cfg.generations and time.time() < deadline:
        try:
            population, record = evolve_one_generation_pure(
                population, gen, bundle, cfg, rng, cfg.seed)
            gen += 1
            history.append(record)
            append_history_jsonl(cfg, record)
            # New offspring are NOT evaluated until the next generation, so
            # their default fitness (0.0) must never outrank an evaluated
            # genome with a negative shaped return.  Pick the generation's
            # evaluated best by gid, and average only evaluated genomes.
            evaluated = [g for g in population if g._result is not None]
            best = next((g for g in evaluated if g.gid == record["best_gid"]),
                        None)
            if best is None or best._result is None:
                best = max(evaluated, key=lambda g: g.fitness,
                           default=population[0])
            mean_fitness = (sum(g.fitness for g in evaluated) / len(evaluated)
                            if evaluated else 0.0)
            if best.fitness > best_ever + cfg.plateau_tol:
                best_ever = best.fitness
                gens_since_improve = 0
            else:
                gens_since_improve += 1
            if cfg.checkpoint_every and gen % cfg.checkpoint_every == 0:
                save_generation(cfg, gen, population, history, bundle, best)
            emit("GEN gen=%d best=%.4f shaped=%.4f mean=%.4f solved=%d/%d "
                 "steps=%d plateau=%d best_gid=%s pool=%d"
                 % (gen, best.fitness,
                    best._result.shaped_return if best._result else 0.0,
                    mean_fitness, best.solved_feasible,
                    len(bundle.feasible()), best.steps, gens_since_improve,
                    best.gid, len(record["pool"])))
            now = time.time()
            milestone = (gen % cfg.validation_every == 0)
            due = (milestone
                   or now - last_snap_time >= cfg.heartbeat_secs
                   or gen - last_snap_gen >= cfg.heartbeat_generations)
            if due:
                validation = None
                if milestone:
                    do_ext = (not args.no_ext
                              and gen % cfg.validation_ext_every == 0)
                    validation = run_validation(cfg, gen, best, bundle, do_ext)
                write_snapshot(cfg, gen, population, best, mean_fitness,
                               history, label="milestone" if milestone else "",
                               validation=validation)
                last_snap_time = now
                last_snap_gen = gen
            if gens_since_improve >= cfg.plateau_generations:
                emit("AUTOSTOP plateau gens_since_improve=%d best=%.4f"
                     % (gens_since_improve, best_ever))
                break
        except Exception as exc:  # noqa: BLE001 - keep the long run alive
            emit("GEN_ERROR %s: %s" % (type(exc).__name__, exc))
            time.sleep(1.0)

    elapsed = time.time() - _STARTED_AT
    try:
        save_generation(cfg, gen, population, history, bundle, best)
        write_snapshot(cfg, gen, population, best,
                       (sum(g.fitness for g in population) / len(population)),
                       history, label="final")
    except Exception as exc:  # noqa: BLE001
        emit("FINAL_SAVE_ERROR %s" % exc)
        exit_code = 3
    reason = ("signal" if _STOP else
              ("plateau" if gens_since_improve >= cfg.plateau_generations else
               ("generation_cap" if gen >= cfg.generations else
                ("wall_clock" if time.time() >= deadline else "stopped"))))
    emit("STOP reason=%s generation=%d elapsed=%.1fs best=%.4f exit=%d"
         % (reason, gen, elapsed, best.fitness, exit_code))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
