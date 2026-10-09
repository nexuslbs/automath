"""Unit C long-run driver: continuous generations with detachable heartbeat.

Launch (detached, survives session teardown)::

    cd /opt/automath/repo
    setsid nohup /opt/automath/venv/bin/python -m new_approach.evolution_loop \
        --config /opt/automath/tmp/evolution/longrun_config.json \
        < /dev/null >> /opt/automath/tmp/evolution/longrun.log 2>&1 &

The loop:
* evolves the population generation after generation (``evolve_one_generation``);
* writes a JSON progress snapshot every ``heartbeat_secs`` seconds AND every
  ``heartbeat_generations`` generations into ``progress/`` (timestamped) and
  prints one PROGRESS line (the detached shell appends stdout to ``longrun.log``);
* persists the population + history + per-state results after every
  ``checkpoint_every`` generations, and can WARM-START with ``--resume``;
* ignores SIGHUP and exits cleanly on SIGTERM/SIGINT after writing a final
  checkpoint + progress snapshot, so a session teardown cannot kill it.

Pure standard library, no LLM, bounded RAM: the population is small and agents
are re-created each generation, so only the best genomes (not their full
Q-tables) survive.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import threading
import time
import traceback
from typing import Dict, List, Optional, Sequence

from .evolution_agents import EVO_ORDER
from .evolution_bundle import Bundle, make_demo_bundle
from .evolution_population import (
    EvoConfig,
    Genome,
    append_history_jsonl,
    evolve_one_generation,
    init_population,
    load_checkpoint,
    restore_population,
    save_generation,
)
from .evolution_population import atomic_write_json

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
    """One flushed line on stdout (the launch redirect appends it to the log)."""
    sys.stdout.write("[%s] %s\n" % (utc_now(), msg))
    sys.stdout.flush()


def load_config(path: str) -> EvoConfig:
    with open(path, "r") as fh:
        data = json.load(fh)
    return EvoConfig.from_dict(data)


def write_progress(cfg: EvoConfig, gen: int, population: List[Genome],
                   best: Genome, mean_fitness: float,
                   elapsed_secs: float, label: str = "") -> str:
    os.makedirs(cfg.progress_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = os.path.join(cfg.progress_dir,
                        "progress_%s_gen%06d%s.json"
                        % (stamp, gen, ("_" + label) if label else ""))
    result = best._result
    hours = max(elapsed_secs, 1e-9) / 3600.0
    payload = {
        "timestamp": utc_now(),
        "label": label,
        "generation": gen,
        "best_gid": best.gid,
        "best_fitness": round(best.fitness, 6),
        "mean_fitness": round(mean_fitness, 6),
        "best_genes": best.genes(),
        "best_parents": list(best.parents),
        "solved_feasible": best.solved_feasible,
        "per_state": result.per_state_rows() if result else [],
        "elapsed_secs": round(elapsed_secs, 3),
        "generations_per_hour": round(gen / hours, 3) if gen else 0.0,
        "population": len(population),
        "checkpoint_dir": cfg.checkpoint_dir,
    }
    atomic_write_json(path, payload)
    emit("PROGRESS gen=%d best=%.4f mean=%.4f solved=%d gph=%.1f file=%s"
         % (gen, best.fitness, mean_fitness, best.solved_feasible,
            payload["generations_per_hour"], path))
    return path


def _history_records(path: str) -> List[dict]:
    ck = load_checkpoint(path)
    if not ck:
        return []
    return list(ck.get("records", []))


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Unit C continuous evolution loop")
    p.add_argument("--config", required=True, help="JSON config file")
    p.add_argument("--generations", type=int, default=None,
                   help="override max generations")
    p.add_argument("--wall-clock-hours", dest="wall_clock_hours", type=float,
                   default=None)
    p.add_argument("--resume", action="store_true",
                   help="warm-start from checkpoint.json when present")
    p.add_argument("--generations-more", dest="generations_more", type=int,
                   default=None,
                   help="run this many MORE generations from the resume point")
    p.add_argument("--checkpoint-dir", dest="checkpoint_dir", default=None)
    p.add_argument("--progress-dir", dest="progress_dir", default=None)
    p.add_argument("--heartbeat-secs", dest="heartbeat_secs", type=float,
                   default=None)
    p.add_argument("--seed", type=int, default=None)
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
    bundle: Bundle = make_demo_bundle(cfg.total_budget)
    emit("START config=%s population=%d episodes=%d total_budget=%d "
         "generations=%d wall_clock_hours=%.2f heartbeat_secs=%.0f "
         "heartbeat_generations=%d bundle=%s"
         % (args.config, cfg.population, cfg.episodes, cfg.total_budget,
            cfg.generations, cfg.wall_clock_hours, cfg.heartbeat_secs,
            cfg.heartbeat_generations, bundle.fingerprint()[:16]))

    ck_path = os.path.join(cfg.checkpoint_dir, "checkpoint.json")
    rng = random.Random(cfg.seed)
    history: List[dict] = []
    start_gen = 0
    population: List[Genome]
    if args.resume and os.path.exists(ck_path):
        ck = load_checkpoint(ck_path)
        assert ck is not None
        population = restore_population(ck)
        start_gen = int(ck.get("generation", 0))
        history = _history_records(os.path.join(cfg.checkpoint_dir,
                                                "history.json"))
        emit("RESUME loaded=%s generation=%d population=%d history=%d"
             % (ck_path, start_gen, len(population), len(history)))
    else:
        population = init_population(cfg, bundle, rng)
        emit("INIT fresh population=%d generation=0" % len(population))
    if args.generations_more is not None:
        cfg.generations = start_gen + args.generations_more
        emit("CAP generations=%d (resume start=%d + more=%d)"
             % (cfg.generations, start_gen, args.generations_more))

    lock = threading.Lock()
    shared: Dict[str, object] = {
        "gen": start_gen,
        "best": max(population, key=lambda g: g.fitness),
        "mean": (sum(g.fitness for g in population) / len(population)),
        "population": population,
        "last_progress_time": time.time(),
        "last_progress_gen": start_gen,
        "elapsed": 0.0,
    }

    def maybe_progress(force: bool = False, label: str = "") -> Optional[str]:
        with lock:
            now = time.time()
            gen = int(shared["gen"])
            due = (force
                   or now - float(shared["last_progress_time"])
                   >= cfg.heartbeat_secs
                   or gen - int(shared["last_progress_gen"])
                   >= cfg.heartbeat_generations)
            if not due:
                return None
            snapshot = (gen, list(shared["population"]), shared["best"],
                        float(shared["mean"]), now - _STARTED_AT)
            shared["last_progress_time"] = now
            shared["last_progress_gen"] = gen
        return write_progress(cfg, snapshot[0], snapshot[1], snapshot[2],
                              snapshot[3], snapshot[4], label)

    stop_event = threading.Event()

    def heartbeat() -> None:
        interval = max(5.0, min(cfg.heartbeat_secs / 10.0, 60.0))
        while not stop_event.wait(interval):
            try:
                maybe_progress(False)
            except Exception:  # noqa: BLE001 - heartbeat must never die
                emit("HEARTBEAT_ERROR %s" % traceback.format_exc().splitlines()
                     [-1])

    hb = threading.Thread(target=heartbeat, name="heartbeat", daemon=True)
    hb.start()

    deadline = _STARTED_AT + cfg.wall_clock_hours * 3600.0
    gen = start_gen
    best = shared["best"]
    consecutive_errors = 0
    exit_code = 0
    emit("LOOP begin generation=%d deadline=%s"
         % (gen, time.strftime("%Y-%m-%dT%H:%M:%SZ",
                               time.gmtime(deadline))))

    while (not _STOP and gen < cfg.generations and time.time() < deadline):
        try:
            population, record = evolve_one_generation(
                population, gen, bundle, cfg, rng, cfg.seed)
            gen += 1
            history.append(record)
            append_history_jsonl(cfg, record)
            if len(history) > cfg.history_max:
                history = history[-cfg.history_max:]
            best = max(population, key=lambda g: g.fitness)
            mean_fitness = sum(g.fitness for g in population) / len(population)
            with lock:
                shared["gen"] = gen
                shared["best"] = best
                shared["mean"] = mean_fitness
                shared["population"] = population
                shared["elapsed"] = time.time() - _STARTED_AT
            if cfg.checkpoint_every and gen % cfg.checkpoint_every == 0:
                save_generation(cfg, gen, population, history, bundle, best)
            emit("GEN gen=%d best=%.4f mean=%.4f solved=%d/%d steps=%d "
                 "best_gid=%s pool=%d"
                 % (gen, best.fitness, mean_fitness, best.solved_feasible,
                    len(bundle.feasible()), best.steps, best.gid,
                    len(record["pool"])))
            maybe_progress(False)
            consecutive_errors = 0
        except Exception:  # noqa: BLE001 - keep the long run alive
            consecutive_errors += 1
            emit("GEN_ERROR consecutive=%d %s"
                 % (consecutive_errors, traceback.format_exc().splitlines()
                    [-1]))
            if consecutive_errors >= 20:
                emit("GEN_ERROR_LIMIT reached=%d exiting" % consecutive_errors)
                exit_code = 2
                break
            time.sleep(1.0)

    stop_event.set()
    elapsed = time.time() - _STARTED_AT
    try:
        save_generation(cfg, gen, population, history, bundle, best,
                        force_history=True)
        maybe_progress(True, label="final")
    except Exception:  # noqa: BLE001
        emit("FINAL_SAVE_ERROR %s" % traceback.format_exc().splitlines()[-1])
        exit_code = exit_code or 3
    reason = ("signal" if _STOP else
              ("generation_cap" if gen >= cfg.generations else
               ("wall_clock" if time.time() >= deadline else "stopped")))
    gph = gen / max(elapsed, 1e-9) * 3600.0
    emit("STOP reason=%s generation=%d elapsed=%.1fs gen_per_hour=%.1f "
         "exit=%d" % (reason, gen, elapsed, gph, exit_code))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
