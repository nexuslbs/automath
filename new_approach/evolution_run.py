"""Pipeline runner for the evolution branch: baselines, Stage 2, Stage 3.

Commands (pure standard library, seeded, bounded)::

    /opt/automath/venv/bin/python -m new_approach.evolution_run --stage baseline
    /opt/automath/venv/bin/python -m new_approach.evolution_run --stage 2
    /opt/automath/venv/bin/python -m new_approach.evolution_run --stage 3
    /opt/automath/venv/bin/python -m new_approach.evolution_run --stage 2e
    /opt/automath/venv/bin/python -m new_approach.evolution_run --stage 3e

`baseline` reproduces the U2 supervised planner (33/33) and the pure-reward
control (14/33) on the SAME 33 new-initial-state validation cases, so the
comparison in the evidence is measured in one process, not quoted.  `2`/`3`
run the core comparison; `2e`/`3e` run the extended complex-scenario suite.
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .agent import TabularAgent
from .env import FixedStateGoal, MinimalEnv
from .evolution import EvoEnv, scenario_cases, scenarios, sem_plan
from .evolution_agents import (
    CORE_ARITY,
    CORE_ORDER,
    EVO_ARITY,
    EVO_ORDER,
    EvalResult,
    EvolutionAgent,
    Guidance,
    TrainStats,
    evaluate,
    read_reward,
    reward_node,
    train_guided,
    train_valuation,
)
from .u2 import (
    curriculum as u2_curriculum,
    training_cases as u2_training_cases,
    validation_cases as u2_validation_cases,
)
from .evolution_bundle import (
    Bundle,
    make_demo_bundle,
    render_trace,
    run_bundle,
    start_order,
)
from .evolution_population import (
    EvoConfig,
    Genome,
    evolve_one_generation,
    init_population,
    load_checkpoint,
    restore_population,
    save_generation,
    train_genome,
)

SEED = 20261009


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------

@dataclass
class SimpleCase:
    name: str
    key: str
    env: object
    max_steps: int
    expected_word: Tuple[str, ...]
    level: str = ""


def core_train_cases() -> List[SimpleCase]:
    out = []
    for c in u2_training_cases():
        out.append(SimpleCase(c.name, c.key, c.env, c.max_steps,
                              tuple(c.expected_word), c.level))
    return out


def core_validation_cases() -> List[SimpleCase]:
    out = []
    for c in u2_validation_cases(list(u2_curriculum())):
        out.append(SimpleCase(c.name, c.key, c.env, c.max_steps,
                              tuple(c.expected_word), c.level))
    return out


def evo_train_cases() -> List[SimpleCase]:
    out = []
    for name, target, stack, level, expected in scenario_cases(False):
        env = EvoEnv(FixedStateGoal(target), initial_stack=stack,
                     max_actions=len(expected) + 2)
        out.append(SimpleCase(name, "S:" + target.canonical(), env,
                              len(expected) + 2, expected, level))
    return out


def evo_validation_cases() -> List[SimpleCase]:
    out = []
    for name, target, stack, level, expected in scenario_cases(True):
        env = EvoEnv(FixedStateGoal(target), initial_stack=stack,
                     max_actions=len(expected) + 2)
        key = "S:" + target.canonical()
        out.append(SimpleCase(name, key, env, len(expected) + 2, expected, level))
    return out


# --------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------

def hdr(title: str) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)


def print_curve(label: str, stats: TrainStats) -> None:
    print("%s convergence curve (per training episode):" % label)
    for row in stats.curve(12):
        print("  " + row)


def eval_line(label: str, res: EvalResult) -> None:
    print("%s: %d/%d PASS (learned_states=%d)"
          % (label, res.passed, res.total, res.learned_states))
    if res.walls:
        print("  eval wall: min=%.6fs median=%.6fs max=%.6fs"
              % (min(res.walls), statistics.median(res.walls), max(res.walls)))


def register(guidance: Guidance, cases: Sequence[SimpleCase]) -> None:
    for c in cases:
        guidance.register(c.key, c.env, c.expected_word)


# --------------------------------------------------------------------------
# Baselines (measured in this process)
# --------------------------------------------------------------------------

def run_baseline(episodes: int, seed: int = SEED) -> int:
    train = core_train_cases()
    val = core_validation_cases()
    hdr("BASELINE (U2, same 33 new-initial-state cases, measured here)")
    print("training cases=%d validation cases=%d seed=%d"
          % (len(train), len(val), seed))
    tuples = [(c.key, c.env, c.expected_word, c.max_steps) for c in train]
    for budget in (60, episodes):
        print("--- budget episodes=%d ---" % budget)
        sup = TabularAgent(episodes=budget, seed=seed, use_demo=True)
        t0 = time.perf_counter()
        sup.train(tuples)
        sup_wall = time.perf_counter() - t0
        sup_pass = sum(int(sup.act_episode(c.key, c.env, c.max_steps,
                                           c.env.goal.describe()).passed)
                       for c in val)
        ctrl = TabularAgent(episodes=budget, seed=seed, use_demo=False)
        t0 = time.perf_counter()
        ctrl.train(tuples)
        ctrl_wall = time.perf_counter() - t0
        ctrl_pass = sum(int(ctrl.act_episode(c.key, c.env, c.max_steps,
                                             c.env.goal.describe()).passed)
                        for c in val)
        print("supervised planner (use_demo=True): train=10/10 validation=%d/%d "
              "digest=%s wall=%.6fs"
              % (sup_pass, len(val), sup.digest(), sup_wall))
        print("pure-reward control (use_demo=False): validation=%d/%d "
              "learned_states=%d wall=%.6fs"
              % (ctrl_pass, len(val), ctrl.learned_states(), ctrl_wall))
        print("BASELINE_SUPERVISED_%d=%d/%d" % (budget, sup_pass, len(val)))
        print("BASELINE_CONTROL_%d=%d/%d" % (budget, ctrl_pass, len(val)))
    return 0


# --------------------------------------------------------------------------
# Stage 2 (core) and Stage 3 (core)
# --------------------------------------------------------------------------

def _agent(use_pref: bool, action_order, arity, episodes: int, args) -> EvolutionAgent:
    return EvolutionAgent(
        action_order=action_order, arity=arity,
        episodes=episodes, seed=args.seed,
        alpha=args.alpha, gamma=args.gamma,
        epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end,
        use_pref=use_pref, pref_lr=args.pref_lr, beta=args.beta,
        pref_sigma=args.pref_sigma, pref_mode=args.pref_mode,
        pref_ctx=args.pref_ctx, pref_baseline=args.pref_baseline,
    )


def run_stage2(episodes: int, args) -> int:
    train = core_train_cases()
    val = core_validation_cases()
    hdr("STAGE 2 (core U2): artificial reward/guidance scaffolding, "
        "use_demo=False")
    print("reward node GOOD = %s" % reward_node(True).canonical())
    print("reward node BAD  = %s" % reward_node(False).canonical())
    print("read_reward(GOOD)=%+.2f read_reward(BAD)=%+.2f"
          % (read_reward(reward_node(True)), read_reward(reward_node(False))))
    print("episodes=%d seed=%d alpha=%.2f gamma=%.2f eps=%.2f->%.2f"
          % (episodes, args.seed, args.alpha, args.gamma,
             args.epsilon_start, args.epsilon_end))
    agent = _agent(False, CORE_ORDER, CORE_ARITY, episodes, args)
    gd = Guidance(evo=False)
    register(gd, train)
    t0 = time.perf_counter()
    stats = train_guided(agent, train, episodes, gd)
    wall = time.perf_counter() - t0
    print_curve("Stage 2", stats)
    tr = evaluate(agent, train)
    va = evaluate(agent, val)
    eval_line("Stage 2 final training", tr)
    eval_line("Stage 2 validation (new initial states)", va)
    print("guidance hits=%d misses=%d train_wall=%.6fs digest=%s"
          % (gd.hits, gd.misses, wall, agent.digest()))
    print("STAGE2_TRAIN=%d/%d STAGE2_VAL=%d/%d"
          % (tr.passed, tr.total, va.passed, va.total))
    print("STAGE2_DIGEST=%s" % agent.digest())
    return 0


def run_stage3(episodes: int, args) -> int:
    train = core_train_cases()
    val = core_validation_cases()
    hdr("STAGE 3 (core U2): scaffolding removed, stochastic step-type "
        "valuation")
    print("no reward node; pref_mode=%s pref_lr=%.3f beta=%.3f sigma=%.3f"
          % (args.pref_mode, args.pref_lr, args.beta, args.pref_sigma))
    agent = _agent(True, CORE_ORDER, CORE_ARITY, episodes, args)
    print("initial random per-step-type preference: %s"
          % agent.pref_snapshot())
    t0 = time.perf_counter()
    stats = train_valuation(agent, train, episodes)
    wall = time.perf_counter() - t0
    print_curve("Stage 3", stats)
    tr = evaluate(agent, train)
    va = evaluate(agent, val)
    eval_line("Stage 3 final training", tr)
    eval_line("Stage 3 validation (new initial states)", va)
    print("final shaped step-type preference: %s" % agent.pref_snapshot())
    print("train_wall=%.6fs digest=%s" % (wall, agent.digest()))
    print("STAGE3_TRAIN=%d/%d STAGE3_VAL=%d/%d"
          % (tr.passed, tr.total, va.passed, va.total))
    print("STAGE3_DIGEST=%s" % agent.digest())
    return 0


# --------------------------------------------------------------------------
# Stage 2e / 3e: the extended complex-scenario suite
# --------------------------------------------------------------------------

def run_stage_ext(stage: int, episodes: int, args) -> int:
    train = evo_train_cases()
    val = evo_validation_cases()
    label = "STAGE %d (extended %d complex scenarios)" % (
        stage, len(train))
    hdr(label)
    print("training scenarios=%d validation new-initial-states=%d "
          "actions=%d episodes=%d seed=%d"
          % (len(train), len(val), len(EVO_ORDER), episodes, args.seed))
    use_pref = (stage == 3)
    agent = _agent(use_pref, EVO_ORDER, EVO_ARITY, episodes, args)
    if use_pref:
        print("initial random per-step-type preference: %s"
              % agent.pref_snapshot())
        t0 = time.perf_counter()
        stats = train_valuation(agent, train, episodes)
    else:
        gd = Guidance(evo=True)
        register(gd, train)
        t0 = time.perf_counter()
        stats = train_guided(agent, train, episodes, gd)
        print("guidance hits=%d misses=%d" % (gd.hits, gd.misses))
    wall = time.perf_counter() - t0
    print_curve("Stage %d extended" % stage, stats)
    tr = evaluate(agent, train)
    va = evaluate(agent, val)
    eval_line("Stage %d extended training" % stage, tr)
    eval_line("Stage %d extended validation" % stage, va)
    if use_pref:
        print("final shaped step-type preference: %s" % agent.pref_snapshot())
    print("train_wall=%.6fs digest=%s" % (wall, agent.digest()))
    print("STAGE%dE_TRAIN=%d/%d STAGE%dE_VAL=%d/%d"
          % (stage, tr.passed, tr.total, stage, va.passed, va.total))
    return 0


# --------------------------------------------------------------------------
# Unit C: bounded demo of the evolutionary loop
# --------------------------------------------------------------------------

def _print_generation_record(rec: dict, fitness_by_gid: dict, bundle) -> None:
    print("-" * 72)
    print("GENERATION %02d  best=%+.4f mean=%+.4f threshold=%+.4f"
          % (rec["generation"], rec["best_fitness"], rec["mean_fitness"],
             rec["threshold"]))
    print("  reproduction pool: qualified=%s" % (rec["qualified"],))
    print("  best state-choosers=%s" % (rec["choosers"],))
    print("  best solvers      =%s" % (rec["solvers"],))
    agent_fit = {a["gid"]: a["fitness"] for a in rec["agents"]}
    print("  AGENT TABLE (population=%d, evaluated this generation):"
          % len(rec["agents"]))
    print("    %-16s %-9s %-18s %8s %8s %8s %6s %5s"
          % ("gid", "origin", "parents", "fitness", "solver", "chooser",
             "solved", "steps"))
    for a in rec["agents"]:
        print("    %-16s %-9s %-18s %+8.3f %8.3f %8.3f %6d %5d"
              % (a["gid"], a["origin"], ",".join(a["parents"]) or "-",
                 a["fitness"], a["solver_score"], a["chooser_score"],
                 a["solved_feasible"], a["steps"]))
    for a in rec["agents"]:
        fitness_by_gid[a["gid"]] = a["fitness"]
    print("  OFFSPRING CROSS TABLE (which instincts crossed; child fitness "
          "is the child row above / next generation):")
    print("    %-16s %-20s %8s %9s %9s %12s"
          % ("child", "parents(A,B)", "prefA/B", "stateA/B", "eps/pat",
             "child_fit"))
    for c in rec["offspring"]:
        d = c["cross_detail"]
        pa, pb = (list(c["parents"]) + ["-", "-"])[:2]
        print("    %-16s %-20s %4d/%-3d %4d/%-4d %5s/%-4s %+12.3f"
              % (c["gid"], "%s,%s" % (pa, pb),
                 d.get("pref_from_a", 0), d.get("pref_from_b", 0),
                 d.get("state_from_a", 0), d.get("state_from_b", 0),
                 d.get("epsilon_from", "?"), d.get("patience_from", "?"),
                 agent_fit.get(c["gid"], float("nan"))))
        print("        parent A=%s fitness=%s ; parent B=%s fitness=%s"
              % (pa, ("%+.3f" % fitness_by_gid[pa])
                 if pa in fitness_by_gid else "n/a",
                 pb, ("%+.3f" % fitness_by_gid[pb])
                 if pb in fitness_by_gid else "n/a"))


def _print_full_evolution_table(history) -> None:
    print("=" * 72)
    print("FULL EVOLUTION TABLE (every agent, every generation; parents and "
          "gene crosses shown)")
    print("  %-4s %-16s %-9s %-20s %8s %8s %8s %6s %5s %-22s"
          % ("gen", "gid", "origin", "parents", "fitness", "solver",
             "chooser", "solved", "steps", "cross pA/pB sA/sB eps/pat"))
    for rec in history:
        for a in rec["agents"]:
            d = a.get("cross_detail") or {}
            cross = "-"
            if d:
                cross = "p%d/%d s%d/%d %s/%s" % (
                    d.get("pref_from_a", 0), d.get("pref_from_b", 0),
                    d.get("state_from_a", 0), d.get("state_from_b", 0),
                    d.get("epsilon_from", "?"), d.get("patience_from", "?"))
            print("  %-4d %-16s %-9s %-20s %+8.3f %8.3f %8.3f %6d %5d %-22s"
                  % (rec["generation"], a["gid"], a["origin"],
                     ",".join(a["parents"]) or "-", a["fitness"],
                     a["solver_score"], a["chooser_score"],
                     a["solved_feasible"], a["steps"], cross))


def run_stage_c(args) -> int:
    cfg = EvoConfig(
        population=args.pop,
        episodes=args.demo_episodes,
        generations=args.gens,
        total_budget=args.total_budget,
        seed=args.seed,
        elites=max(2, args.pop // 4),
        top_k=max(2, args.pop // 4),
        checkpoint_dir=args.ckpt_dir,
        progress_dir=args.progress_dir,
    )
    bundle: Bundle = make_demo_bundle(cfg.total_budget)
    hdr("UNIT C (bounded demo): population evolution + multi-state TOTAL "
        "budget + impossible goals")
    print("bundle=%s total_budget=%d states=%d feasible=%d impossible=%d "
          "fingerprint=%s"
          % (bundle.name, bundle.total_budget, len(bundle.states),
             len(bundle.feasible()), len(bundle.impossible()),
             bundle.fingerprint()[:16]))
    for s in bundle.states:
        note = ("IMPOSSIBLE: " + s.impossible_reason) if not s.feasible else ""
        print("  state %s %-20s feasible=%-5s expected_len=%2d %s"
              % (s.sid, s.name, s.feasible, s.expected_len, note))
    print("config: population=%d episodes_per_agent=%d generations=%d "
          "elites=%d top_k=%d threshold_frac=%.2f mutation_rate=%.2f "
          "mutation_sigma=%.2f sigma_pref=%.2f sigma_state=%.2f"
          % (cfg.population, cfg.episodes, cfg.generations, cfg.elites,
             cfg.top_k, cfg.threshold_frac, cfg.mutation_rate,
             cfg.mutation_sigma, cfg.sigma_pref, cfg.sigma_state))

    rng = random.Random(cfg.seed)
    population = init_population(cfg, bundle, rng)
    print("initial random genome sample: pref[0:6]=%s state_pref=%s "
          "eps=%.3f patience=%d"
          % (["%+.2f" % v for v in population[0].pref[:6]],
             ["%+.2f" % v for v in population[0].state_pref],
             population[0].epsilon, population[0].switch_patience))

    history = []
    fitness_by_gid: dict = {}
    best = population[0]
    for gen in range(cfg.generations):
        population, rec = evolve_one_generation(population, gen, bundle, cfg,
                                                rng, cfg.seed)
        _print_generation_record(rec, fitness_by_gid, bundle)
        history.append(rec)
        best = max(population, key=lambda g: g.fitness)
        sys.stdout.flush()

    _print_full_evolution_table(history)

    print("=" * 72)
    print("CHECKPOINTS under %s" % cfg.checkpoint_dir)
    paths = save_generation(cfg, cfg.generations, population, history, bundle,
                            best)
    for name in sorted(os.listdir(cfg.checkpoint_dir)):
        p = os.path.join(cfg.checkpoint_dir, name)
        print("  %s %d bytes" % (p, os.path.getsize(p)))
    print("save_generation returned: %s"
          % ", ".join("%s=%d" % (os.path.basename(k), os.path.getsize(v))
                      for k, v in paths.items()))

    # -- multi-state impossible-goal raw trace --------------------------
    print("=" * 72)
    print("MULTI-STATE TOTAL-BUDGET TRACE (forced to start on the "
          "IMPOSSIBLE state s4)")
    best_agent = train_genome(best, bundle, cfg, cfg.seed + 777)
    forced = Genome(pref=list(best.pref),
                    state_pref=[0.0] * len(bundle.states),
                    epsilon=0.02, switch_patience=best.switch_patience,
                    gid="forced-impossible-first")
    forced.state_pref[bundle.index("s4")] = 5.0
    impossible_first = run_bundle(best_agent, bundle, forced.state_pref,
                                  total_budget=cfg.total_budget,
                                  training=False)
    print("forced state instinct order: %s"
          % [bundle.by_id(s).name for s in start_order(bundle,
                                                       forced.state_pref)])
    print(render_trace(impossible_first, bundle))
    print("PER-STATE RESULT (impossible-first run):")
    for row in impossible_first.per_state_rows():
        print("  %s" % (row,))

    print("-" * 72)
    print("BEST GENOME instinct order: %s"
          % [bundle.by_id(s).name for s in start_order(bundle,
                                                       best.state_pref)])
    print("BEST GENOME greedy trace:")
    print(render_trace(best._result, bundle))
    print("PER-STATE RESULT (best genome):")
    for row in best._result.per_state_rows():
        print("  %s" % (row,))

    # -- warm start proof ----------------------------------------------
    print("=" * 72)
    ck = load_checkpoint(os.path.join(cfg.checkpoint_dir, "checkpoint.json"))
    loaded_gen = ck["generation"]
    warm_pop = restore_population(ck)
    print("WARM_START loaded=%s generation=%d population=%d best_fitness=%+.4f"
          % (os.path.join(cfg.checkpoint_dir, "checkpoint.json"), loaded_gen,
             len(warm_pop), ck["best_fitness"]))
    rng2 = random.Random(cfg.seed + loaded_gen)
    warm_pop2, rec2 = evolve_one_generation(warm_pop, loaded_gen, bundle, cfg,
                                            rng2, cfg.seed)
    print("WARM_START continued -> generation=%d best=%+.4f (loaded "
          "generation %d + 1)"
          % (rec2["generation"], rec2["best_fitness"], loaded_gen))
    print("WARM_START new pool: qualified=%s choosers=%s solvers=%s"
          % (rec2["qualified"], rec2["choosers"], rec2["solvers"]))
    print("UNITC_CONFIG_POP=%d EPISODES=%d GENERATIONS=%d TOTAL_BUDGET=%d"
          % (cfg.population, cfg.episodes, cfg.generations, cfg.total_budget))
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="evolution pipeline runner")
    p.add_argument("--stage",
                   choices=("baseline", "2", "3", "2e", "3e", "c", "all"),
                   default="all")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--pop", type=int, default=8,
                   help="Unit C demo population size")
    p.add_argument("--gens", type=int, default=6,
                   help="Unit C demo number of generations")
    p.add_argument("--total-budget", dest="total_budget", type=int, default=60,
                   help="Unit C demo total step budget across ALL states")
    p.add_argument("--demo-episodes", dest="demo_episodes", type=int,
                   default=15, help="Unit C training episodes per agent")
    p.add_argument("--ckpt-dir", dest="ckpt_dir",
                   default="/opt/automath/tmp/evolution/checkpoints_demo")
    p.add_argument("--progress-dir", dest="progress_dir",
                   default="/opt/automath/tmp/evolution/progress_demo")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--epsilon-start", dest="epsilon_start", type=float, default=0.5)
    p.add_argument("--epsilon-end", dest="epsilon_end", type=float, default=0.02)
    p.add_argument("--pref-lr", dest="pref_lr", type=float, default=0.2)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--pref-sigma", dest="pref_sigma", type=float, default=0.5)
    p.add_argument("--pref-mode", dest="pref_mode", choices=("td", "reinforce"),
                   default="td")
    p.add_argument("--pref-ctx", dest="pref_ctx", action="store_true",
                   default=False)
    p.add_argument("--pref-baseline", dest="pref_baseline", type=float,
                   default=0.1)
    args = p.parse_args(argv)

    rc = 0
    if args.stage in ("baseline", "all"):
        rc |= run_baseline(args.episodes, args.seed)
    if args.stage in ("2", "all"):
        rc |= run_stage2(args.episodes, args)
    if args.stage in ("3", "all"):
        rc |= run_stage3(args.episodes, args)
    if args.stage in ("2e", "all"):
        rc |= run_stage_ext(2, args.episodes, args)
    if args.stage in ("3e", "all"):
        rc |= run_stage_ext(3, args.episodes, args)
    if args.stage == "c":
        rc |= run_stage_c(args)
    return rc


if __name__ == "__main__":
    sys.exit(main())
