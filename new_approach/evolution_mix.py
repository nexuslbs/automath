"""Unit E: BLX-alpha instinct MIXING + PERSIST + WARM-START seeding.

Bounded, deterministic, pure standard library.  It never touches a running
long-run: it reads loop-1's frozen elite and its recorded parents, mixes them,
evaluates the mix on the same 7-state / total-budget-90 suite, persists the
offspring pool and a warm-start checkpoint, and prints a JSON report.

Steps
-----
1. Read loop-1's frozen elite genome (``g00014-c011``) from its checkpoint.
2. Recover its recorded parents (``g00013-c002``, ``g00013-c003``) by
   DETERMINISTIC REPLAY of the seeded loop-1 search to generation 13.  The
   search is fully seeded, so the replay is exact; it is VALIDATED against the
   first 13 ``history.jsonl`` records.
3. BLX-alpha crossover (alpha=0.5) over ``pref`` + ``state_pref``::

       parents x parents  -> 3 offspring
       elite   x parent A -> 3 offspring
       elite   x parent B -> 2 offspring
       --------------------------------
                            8 mixed offspring

4. Evaluate all 11 genomes (8 offspring + 2 parents + elite) with the SAME
   fitness function (train ``episodes`` episodes, evaluate greedily over the
   bundle, total budget 90).
5. PERSIST the offspring pool + a warm-start checkpoint (mixed offspring +
   loop-1 elite) as ``checkpoint.json``/``state_results.json``/
   ``offspring_pool.json``.

Run::

    /opt/automath/venv/bin/python -m new_approach.evolution_mix \
        --config config/evolution_longrun.json \
        --loop1-checkpoint-dir /opt/automath/tmp/evolution/checkpoints \
        --loop1-progress-dir /opt/automath/tmp/evolution/progress \
        --out-dir /opt/automath/tmp/evolution2/checkpoints \
        --warmstart-dir /opt/automath/tmp/evolution2/warmstart/checkpoints \
        --report /opt/automath/tmp/evolution2/mix_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Sequence, Tuple

from .evolution_agents import EVO_ORDER
from .evolution_bundle import Bundle, make_demo_bundle
from .evolution_population import (
    EvoConfig,
    Genome,
    _seed_for,
    atomic_write_json,
    evaluate_genome,
    evolve_one_generation,
    init_population,
    load_checkpoint,
)

RECORDED_ELITE = "g00014-c011"
RECORDED_PARENTS = ("g00013-c002", "g00013-c003")
REPLAY_GENERATIONS = 13  # gen 0..12 -> population seeded with g00013-c* offspring
BLX_ALPHA = 0.5
EXEMPLAR_PREF = (0, 13)
EXEMPLAR_STATE = (2,)


def load_cfg(path: str, **overrides) -> EvoConfig:
    with open(path, "r") as fh:
        data = json.load(fh)
    cfg = EvoConfig.from_dict(data)
    for key, val in overrides.items():
        setattr(cfg, key, val)
    return cfg


def load_elite(ckpt_dir: str, progress_dir: str) -> Genome:
    """Return loop-1's frozen elite genome, from the checkpoint when possible."""
    ck = load_checkpoint(os.path.join(ckpt_dir, "checkpoint.json"))
    if ck:
        for d in ck.get("population", []):
            if d.get("gid") == RECORDED_ELITE:
                return Genome.from_dict(d)
    # Fallback: newest progress snapshot carries best_genes for the elite.
    if progress_dir and os.path.isdir(progress_dir):
        for name in sorted(os.listdir(progress_dir), reverse=True):
            if not name.endswith(".json"):
                continue
            snap = load_checkpoint(os.path.join(progress_dir, name))
            if snap and snap.get("best_gid") == RECORDED_ELITE:
                genes = snap["best_genes"]
                return Genome(
                    pref=[float(x) for x in genes["pref"]],
                    state_pref=[float(x) for x in genes["state_pref"]],
                    epsilon=float(genes["epsilon"]),
                    switch_patience=int(genes["switch_patience"]),
                    gid=RECORDED_ELITE, origin="elite",
                    fitness=float(snap.get("best_fitness", 0.0)),
                    parents=tuple(snap.get("best_parents", [])),
                )
    raise SystemExit("elite %s not found in %s" % (RECORDED_ELITE, ckpt_dir))


def replay_parents(cfg: EvoConfig,
                   bundle: Bundle) -> Tuple[List[Genome], List[dict]]:
    """Deterministically replay loop-1's seeded search to generation 13."""
    rng = random.Random(cfg.seed)
    population = init_population(cfg, bundle, rng)
    records: List[dict] = []
    for gen in range(REPLAY_GENERATIONS):
        population, rec = evolve_one_generation(
            population, gen, bundle, cfg, rng, cfg.seed)
        records.append(rec)
    return population, records


def validate_replay(records: List[dict], history_jsonl: str) -> List[dict]:
    """Compare the replay records with the real history.jsonl first N lines."""
    checks: List[dict] = []
    if not os.path.exists(history_jsonl):
        return checks
    with open(history_jsonl, "r") as fh:
        real = [json.loads(line) for line in fh]
    for i, rec in enumerate(records):
        if i >= len(real):
            break
        ref = real[i]
        ok = (rec["best_gid"] == ref["best_gid"]
              and abs(rec["best_fitness"] - ref["best_fitness"]) < 1e-9
              and abs(rec["mean_fitness"] - ref["mean_fitness"]) < 1e-9)
        checks.append({
            "generation": rec["generation"],
            "replay_best_gid": rec["best_gid"],
            "real_best_gid": ref["best_gid"],
            "replay_best": rec["best_fitness"],
            "real_best": ref["best_fitness"],
            "match": bool(ok),
        })
    return checks


def blx(a_vals: Sequence[float], b_vals: Sequence[float], alpha: float,
        rng: random.Random) -> List[float]:
    """BLX-alpha blend crossover of two equal-length real vectors."""
    out: List[float] = []
    for x, y in zip(a_vals, b_vals):
        lo, hi = (x, y) if x <= y else (y, x)
        span = hi - lo
        out.append(rng.uniform(lo - alpha * span, hi + alpha * span))
    return out


def _nearer(child: float, av: float, bv: float) -> str:
    da, db = abs(child - av), abs(child - bv)
    if abs(da - db) < 1e-12:
        return "blend"
    return "A" if da < db else "B"


def build_offspring(elite: Genome, p1: Genome, p2: Genome,
                    rng: random.Random) -> List[Genome]:
    """8 BLX-alpha offspring: 3 (p1xp2) + 3 (ex p1) + 2 (ex p2)."""
    plan = [
        ("p12", p1, p2, 3),
        ("ep1", elite, p1, 3),
        ("ep2", elite, p2, 2),
    ]
    offspring: List[Genome] = []
    for tag, a, b, count in plan:
        for i in range(count):
            child = Genome(
                pref=blx(a.pref, b.pref, BLX_ALPHA, rng),
                state_pref=blx(a.state_pref, b.state_pref, BLX_ALPHA, rng),
                epsilon=(a.epsilon if rng.random() < 0.5 else b.epsilon),
                switch_patience=(a.switch_patience
                                 if rng.random() < 0.5
                                 else b.switch_patience),
                gid="mix-%s-%03d" % (tag, i),
                parents=(a.gid, b.gid),
                origin="blx-offspring",
            )
            detail: Dict[str, object] = {
                "pair": tag, "parent_a": a.gid, "parent_b": b.gid,
                "method": "BLX-alpha", "alpha": BLX_ALPHA,
            }
            for idx in EXEMPLAR_PREF:
                detail["pref[%d]" % idx] = {
                    "A": round(a.pref[idx], 6), "B": round(b.pref[idx], 6),
                    "child": round(child.pref[idx], 6),
                    "nearer": _nearer(child.pref[idx], a.pref[idx],
                                      b.pref[idx]),
                }
            for idx in EXEMPLAR_STATE:
                detail["state_pref[%d]" % idx] = {
                    "A": round(a.state_pref[idx], 6),
                    "B": round(b.state_pref[idx], 6),
                    "child": round(child.state_pref[idx], 6),
                    "nearer": _nearer(child.state_pref[idx], a.state_pref[idx],
                                      b.state_pref[idx]),
                }
            child.cross_detail = detail
            offspring.append(child)
    return offspring


def eval_row(g: Genome) -> dict:
    result = g._result
    return {
        "gid": g.gid, "parents": list(g.parents), "origin": g.origin,
        "total_reward": round(g.fitness, 6),
        "solved_feasible": g.solved_feasible,
        "solver_score": round(g.solver_score, 6),
        "chooser_score": round(g.chooser_score, 6),
        "steps": g.steps,
        "per_state": result.per_state_rows() if result else [],
        "inheritance": g.cross_detail,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Unit E BLX-alpha mix + persist")
    ap.add_argument("--config", default="config/evolution_longrun.json")
    ap.add_argument("--loop1-checkpoint-dir",
                    default="/opt/automath/tmp/evolution/checkpoints")
    ap.add_argument("--loop1-progress-dir",
                    default="/opt/automath/tmp/evolution/progress")
    ap.add_argument("--out-dir",
                    default="/opt/automath/tmp/evolution2/checkpoints")
    ap.add_argument("--warmstart-dir",
                    default="/opt/automath/tmp/evolution2/warmstart/checkpoints")
    ap.add_argument("--report",
                    default="/opt/automath/tmp/evolution2/mix_report.json")
    ap.add_argument("--seed", type=int, default=20261009)
    args = ap.parse_args(argv)

    cfg = load_cfg(args.config, restart_on_stall=0)
    bundle = make_demo_bundle(cfg.total_budget)
    print("config=%s population=%d episodes=%d total_budget=%d seed=%d"
          % (args.config, cfg.population, cfg.episodes, cfg.total_budget,
             cfg.seed))

    elite = load_elite(args.loop1_checkpoint_dir, args.loop1_progress_dir)
    print("elite=%s parents=%s fitness=%.4f solved=%d/%d"
          % (elite.gid, list(elite.parents), elite.fitness,
             elite.solved_feasible, len(bundle.feasible())))

    population, records = replay_parents(cfg, bundle)
    by_gid = {g.gid: g for g in population}
    parents = [by_gid[gid] for gid in RECORDED_PARENTS if gid in by_gid]
    missing = [gid for gid in RECORDED_PARENTS if gid not in by_gid]
    checks = validate_replay(
        records, os.path.join(args.loop1_checkpoint_dir, "history.jsonl"))
    matched = sum(1 for c in checks if c["match"])
    print("replay-to-gen13 recovered=%s missing=%s validated=%d/%d"
          % ([g.gid for g in parents], missing, matched, len(checks)))
    for c in checks:
        print("  replay gen=%(generation)d best=%(replay_best).4f "
              "gid=%(replay_best_gid)s (real %(real_best).4f "
              "%(real_best_gid)s) match=%(match)s" % c)
    if len(parents) != 2:
        raise SystemExit("could not recover both recorded parents: %s"
                         % missing)

    rng = random.Random(args.seed)
    offspring = build_offspring(elite, parents[0], parents[1], rng)

    genotypes: List[Genome] = [elite] + parents + offspring
    for g in genotypes:
        evaluate_genome(g, bundle, cfg, _seed_for(g.gid, cfg.seed))

    rows = [eval_row(g) for g in genotypes]
    print("\n%-16s %-24s %5s %6s %5s %9s" %
          ("genome", "parents", "solv", "reward", "steps", "solver_score"))
    for r in rows:
        print("%-16s %-24s %5d %6.3f %5d %9.4f"
              % (r["gid"], ",".join(r["parents"]), r["solved_feasible"],
                 r["total_reward"], r["steps"], r["solver_score"]))

    print("\ninheritance exemplars (BLX-alpha=%.1f):" % BLX_ALPHA)
    for g in offspring:
        d = g.cross_detail
        print("  %s <- %s" % (g.gid, d.get("pair")))
        for key in ("pref[0]", "pref[13]", "state_pref[2]"):
            if key in d:
                print("     %-13s A=%.4f B=%.4f child=%.4f nearer=%s"
                      % (key, d[key]["A"], d[key]["B"], d[key]["child"],
                         d[key]["nearer"]))

    ranked_off = sorted(offspring, key=lambda g: (-g.fitness, g.gid))
    best_off = ranked_off[0]
    parent_rows = [eval_row(p) for p in parents]

    # ---- persist --------------------------------------------------------
    warm_pop = [elite] + ranked_off
    best_genome = max(genotypes, key=lambda g: (-g.fitness, g.gid))
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.warmstart_dir, exist_ok=True)

    pool = {
        "source": "evolution_mix",
        "alpha": BLX_ALPHA,
        "elite": eval_row(elite),
        "parents": parent_rows,
        "offspring": [eval_row(g) for g in offspring],
        "best_offspring": best_off.gid,
        "replay_validation": checks,
        "replay_validation_passed": matched == len(checks) and len(checks) > 0,
        "bundle": bundle.to_dict(),
    }
    checkpoint = {
        "version": 1,
        "generation": 0,
        "bundle": bundle.to_dict(),
        "config": cfg.to_dict(),
        "best_gid": best_genome.gid,
        "best_fitness": round(best_genome.fitness, 6),
        "population": [g.to_dict() for g in warm_pop],
    }
    state_results = {
        "version": 1,
        "generation": 0,
        "best_gid": best_genome.gid,
        "best_fitness": round(best_genome.fitness, 6),
        "per_state": (best_genome._result.per_state_rows()
                      if best_genome._result else []),
        "trace": ([] if not best_genome._result else
                  [{"step": e.step, "kind": e.kind, "sid": e.sid,
                    "detail": e.detail} for e in best_genome._result.trace]),
    }
    history = {"version": 1, "generation": 0, "records": []}

    written: Dict[str, str] = {}
    for name, payload in (("checkpoint.json", checkpoint),
                          ("state_results.json", state_results),
                          ("offspring_pool.json", pool),
                          ("history.json", history)):
        out_path = os.path.join(args.out_dir, name)
        atomic_write_json(out_path, payload)
        written[name] = out_path
        warm_path = os.path.join(args.warmstart_dir, name)
        atomic_write_json(warm_path, payload)
        written["warmstart/" + name] = warm_path

    report = {
        "elite": elite.to_dict(),
        "recovered_parents": [p.to_dict() for p in parents],
        "replay_validation_passed": matched == len(checks) and len(checks) > 0,
        "replay_checks": checks,
        "offspring": [eval_row(g) for g in offspring],
        "best_offspring": best_off.gid,
        "best_offspring_fitness": round(best_off.fitness, 6),
        "elite_fitness": round(elite.fitness, 6),
        "improved_over_elite": bool(best_off.fitness > elite.fitness + 1e-9),
        "warmstart_population": [g.gid for g in warm_pop],
        "warmstart_population_size": len(warm_pop),
        "written": written,
    }
    atomic_write_json(args.report, report)
    atomic_write_json(os.path.join(args.out_dir, "mix_report.json"), report)

    print("\nbest_offspring=%s fitness=%.4f (elite=%.4f) improved=%s"
          % (best_off.gid, best_off.fitness, elite.fitness,
             report["improved_over_elite"]))
    print("warm-start population (%d): %s"
          % (len(warm_pop), ",".join(g.gid for g in warm_pop)))
    print("persisted: %s" % written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
