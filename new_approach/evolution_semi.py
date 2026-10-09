"""Unit B (Phase A): semi-evolutionary path with the designed shaped reward.

This module EXTENDS the Unit C population machinery
(``evolution_population.py``) and the Unit E BLX-alpha mixing
(``evolution_mix.py``) instead of rewriting them:

* every genome is still the same heritable BASIC INSTINCT
  (``pref`` over the 21 extended actions, ``state_pref`` over the bundle states,
  ``epsilon``, ``switch_patience``);
* the difference is the LEARNING SIGNAL: the agent trains on the shaped
  per-step reward of ``evolution_rewards.py`` (step cost + PBRS + sub-goal
  bonus + final goal), and selection ranks the SHAPED episode return plus the
  solved rate;
* reproduction is crossover/mutation of the instinct genes: BLX-alpha blend
  (Unit E) or uniform per-gene crossover (Unit C), then bounded gaussian
  mutation.  The fittest survive by reward-threshold + elitism.

Everything is pure standard library, seeded and bounded.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .env import FixedStateGoal
from .evolution import EvoEnv
from .evolution_agents import EVO_ARITY, EVO_ORDER, EvolutionAgent
from .evolution_bundle import (
    BUDGET_RATIO_TRIGGER,
    Bundle,
    BundleResult,
    StateResult,
    TraceEvent,
    _progress,
    start_order,
)
from .evolution_population import (
    EvoConfig,
    Genome,
    _clamp,
    _dedup,
    _pick_pair,
    _seed_for,
    _STALL,
    _update_stall,
    append_history_jsonl,
    clone_genome,
    crossover,
    init_population,
    load_checkpoint,
    mutate,
    restore_population,
    save_generation,
)
from .evolution_rewards import EpisodeShaper, RewardConfig

# Re-export names used by the loop / tests.
__all__ = [
    "ShapedBundleResult",
    "run_bundle_shaped",
    "make_shaped_agent",
    "train_genome_shaped",
    "evaluate_genome_shaped",
    "evolve_one_generation_shaped",
    "train_shaped",
    "core_validation_shaped",
    "evo_validation_shaped",
    "bundle_validation",
    "blx",
    "BLX_ALPHA",
]


BLX_ALPHA = 0.5


# --------------------------------------------------------------------------
# Shaped multi-state total-budget controller
# --------------------------------------------------------------------------

@dataclass
class ShapedBundleResult(BundleResult):
    """A ``BundleResult`` plus the shaped return and the itemised reward log."""

    shaped_return: float = 0.0
    reward_log: List[dict] = field(default_factory=list)
    positive_shaping: float = 0.0

    def solved_rate(self) -> float:
        return self.solved_feasible / max(1, self.feasible_total)


def run_bundle_shaped(
    agent: EvolutionAgent,
    bundle: Bundle,
    state_pref: Sequence[float],
    total_budget: Optional[int] = None,
    training: bool = False,
    epsilon: float = 0.0,
    max_states: int = 4,
    reward_config: Optional[RewardConfig] = None,
) -> ShapedBundleResult:
    """The Unit C total-budget controller with the SHAPED reward.

    Identical selection/switch mechanics to ``evolution_bundle.run_bundle``
    (stagnation + budget-ratio triggers, one-shared-budget, per-sweep rule) so
    the only difference is the reward.  The solving step is charged the step
    cost AND the shaping AND the ``+1.0`` final goal (the four terms are
    additive, per the binding design).
    """
    rcfg = reward_config or RewardConfig()
    budget = total_budget if total_budget is not None else bundle.total_budget
    envs = {s.sid: EvoEnv(FixedStateGoal(s.target),
                          initial_stack=s.initial_stack,
                          max_actions=budget + 4)
            for s in bundle.states}
    per_state = {s.sid: StateResult(s.sid, s.name, s.feasible)
                 for s in bundle.states}
    trace: List[TraceEvent] = []
    trajectory: List[Tuple[str, float, float, str]] = []
    reward_log: List[dict] = []
    order = start_order(bundle, state_pref)
    order_pos = {sid: i for i, sid in enumerate(order)}
    pref_by_sid = {sid: float(state_pref[bundle.index(sid)]) for sid in order}
    swept: set = set()
    remaining = budget
    step_no = 0
    shaped_return = 0.0
    solved = 0
    switched = 0

    shaper = EpisodeShaper(rcfg)

    def choose_next(exclude: Optional[str]) -> Optional[str]:
        cands = [sid for sid in order
                 if sid != exclude and not per_state[sid].solved
                 and sid not in swept]
        if not cands:
            swept.clear()
            cands = [sid for sid in order
                     if sid != exclude and not per_state[sid].solved]
        if not cands:
            return None
        cands.sort(key=lambda sid: (-pref_by_sid[sid], order_pos[sid]))
        return cands[0]

    current = choose_next(None)
    if current is not None:
        per_state[current].started = True
        env = envs[current]
        state = env.reset()
        steps_in = 0
        entry_remaining = remaining
        last_improve = 0
        best_partial = 0.0
        trace.append(TraceEvent(step_no, "start", current,
                                "instinct order=%s" % (order,)))
    else:
        state = None  # type: ignore[assignment]
        steps_in = entry_remaining = last_improve = 0
        env = None  # type: ignore[assignment]
        best_partial = 0.0

    while remaining > 0 and current is not None and state is not None:
        env = envs[current]
        sr = per_state[current]
        swept.add(current)
        target = bundle.by_id(current).target
        if env.goal_achieved(state):
            sr.solved = True
            solved += 1
            shaped_return += rcfg.goal_reward
            reward_log.append({
                "step": step_no, "sid": current, "action": "",
                "target": target.canonical(), "step_cost": 0.0,
                "shaping": 0.0, "subgoal_bonus": 0.0, "shaping_applied": 0.0,
                "final_reward": rcfg.goal_reward, "total": rcfg.goal_reward,
                "capped": 0.0, "phi": 0.0, "phi_next": 0.0,
                "similarity": 1.0, "similarity_next": 1.0,
                "top": target.canonical(), "subgoal": False,
                "already_goal": True,
            })
            trace.append(TraceEvent(step_no, "solve", current,
                                    "goal reached in %d steps" % sr.steps))
            current = choose_next(current)
            if current is not None:
                per_state[current].started = True
                state = envs[current].reset()
                steps_in = 0
                entry_remaining = remaining
                last_improve = 0
                best_partial = 0.0
                trace.append(TraceEvent(step_no, "start", current,
                                        "after solve order=%s" % (order,)))
            continue

        key = bundle.by_id(current).key()
        if training:
            action = agent.epsilon_greedy(key, state, epsilon)
        else:
            action = agent.best_action(key, state)
            if action is None:
                action = agent.epsilon_greedy(key, state, 0.0)
        ctx = agent._ctx(state)
        nxt = env.step(state, action)
        reached = env.goal_achieved(nxt)
        rb = shaper.reward(target, state, nxt, reached)
        reward = rb.total
        if training:
            td = agent.learn(key, state, action, reward, nxt, reached)
            trajectory.append((action, reward, td, ctx))
        remaining -= 1
        step_no += 1
        steps_in += 1
        sr.steps += 1
        sr.reward += reward
        shaped_return += reward
        reward_log.append(rb.to_dict(step=step_no, sid=current,
                                     target=target.canonical(),
                                     action=action))
        prog = _progress(nxt, target)
        if prog > best_partial:
            best_partial = prog
            sr.partial = max(sr.partial, prog)
            last_improve = steps_in
        state = nxt
        if reached:
            sr.solved = True
            solved += 1
            trace.append(TraceEvent(step_no, "solve", current,
                                    "goal reached in %d steps" % sr.steps))
            current = choose_next(current)
            if current is not None:
                per_state[current].started = True
                state = envs[current].reset()
                steps_in = 0
                entry_remaining = remaining
                last_improve = 0
                best_partial = 0.0
                trace.append(TraceEvent(step_no, "start", current,
                                        "after solve order=%s" % (order,)))
            continue

        patience = int(getattr(agent, "switch_patience", 6))
        reason = None
        if steps_in - last_improve >= patience:
            reason = ("stagnation: %d steps without progress (patience=%d)"
                      % (steps_in - last_improve, patience))
        elif steps_in >= max(2, int(BUDGET_RATIO_TRIGGER * entry_remaining)):
            reason = ("budget-ratio: %d of %d steps since entry"
                      % (steps_in, entry_remaining))
        if reason is not None:
            per_state[current].switches_away += 1
            nxt_sid = choose_next(current)
            switched += 1
            trace.append(TraceEvent(step_no, "switch", current,
                                    reason + " -> " + str(nxt_sid)))
            current = nxt_sid
            if current is not None:
                per_state[current].started = True
                state = envs[current].reset()
                steps_in = 0
                entry_remaining = remaining
                last_improve = 0
                best_partial = 0.0
            continue

    trace.append(TraceEvent(step_no, "finish", current or "-",
                            "budget exhausted: %d steps total" % step_no))
    solved = sum(1 for sr in per_state.values() if sr.solved)
    return ShapedBundleResult(
        per_state=per_state,
        total_steps=step_no,
        total_reward=shaped_return,
        solved_feasible=solved,
        feasible_total=len(bundle.feasible()),
        switched=switched,
        trace=trace,
        trajectory=trajectory,
        shaped_return=shaped_return,
        reward_log=reward_log,
        positive_shaping=round(shaper.positive_used, 6),
    )


# --------------------------------------------------------------------------
# Learning + fitness with the shaped signal
# --------------------------------------------------------------------------

def make_shaped_agent(genome: Genome, cfg: EvoConfig, seed: int
                      ) -> EvolutionAgent:
    return EvolutionAgent(
        action_order=EVO_ORDER,
        arity=EVO_ARITY,
        alpha=cfg.alpha,
        gamma=cfg.reward_gamma,
        epsilon_start=max(cfg.epsilon_min, genome.epsilon),
        epsilon_end=max(cfg.epsilon_min, 0.02),
        episodes=max(1, cfg.episodes),
        seed=seed,
        use_pref=True,
        pref_lr=cfg.pref_lr,
        beta=cfg.beta,
        pref_sigma=cfg.sigma_pref,
        pref_mode=cfg.pref_mode,
        pref_init=genome.pref_map(),
        switch_patience=genome.switch_patience,
    )


def train_genome_shaped(genome: Genome, bundle: Bundle, cfg: EvoConfig,
                        seed: int,
                        reward_config: Optional[RewardConfig] = None
                        ) -> EvolutionAgent:
    """Train the genome's agent on the SHAPED reward for ``cfg.episodes``."""
    agent = make_shaped_agent(genome, cfg, seed)
    for ep in range(max(1, cfg.episodes)):
        epsilon = agent._epsilon(ep)
        result = run_bundle_shaped(
            agent, bundle, genome.state_pref,
            total_budget=cfg.total_budget, training=True, epsilon=epsilon,
            reward_config=reward_config)
        agent.update_pref(result.trajectory)
    return agent


def evaluate_genome_shaped(genome: Genome, bundle: Bundle, cfg: EvoConfig,
                           seed: int,
                           reward_config: Optional[RewardConfig] = None
                           ) -> ShapedBundleResult:
    agent = train_genome_shaped(genome, bundle, cfg, seed, reward_config)
    result = run_bundle_shaped(
        agent, bundle, genome.state_pref, total_budget=cfg.total_budget,
        training=False, reward_config=reward_config)
    genome.fitness = (result.shaped_return
                      + cfg.solved_rate_weight * result.solved_rate())
    genome.solver_score = result.solver_score()
    genome.chooser_score = result.chooser_score()
    genome.solved_feasible = result.solved_feasible
    genome.steps = result.total_steps
    genome._result = result
    return result


# --------------------------------------------------------------------------
# Reproduction: BLX-alpha blend (Unit E) or uniform crossover (Unit C)
# --------------------------------------------------------------------------

def _clamp_vec(vals: Sequence[float], clip: float) -> List[float]:
    return [(-clip if v < -clip else (clip if v > clip else v)) for v in vals]


def blx(a_vals: Sequence[float], b_vals: Sequence[float], alpha: float,
        rng: random.Random, clip: float = 5.0) -> List[float]:
    """BLX-alpha blend crossover of two equal-length real vectors (clipped)."""
    out: List[float] = []
    for x, y in zip(a_vals, b_vals):
        lo, hi = (x, y) if x <= y else (y, x)
        span = hi - lo
        out.append(rng.uniform(lo - alpha * span, hi + alpha * span))
    return _clamp_vec(out, clip)


def _make_child(a: Genome, b: Genome, rng: random.Random,
                cfg: EvoConfig) -> Genome:
    if cfg.use_blx:
        child = Genome(
            pref=blx(a.pref, b.pref, cfg.blx_alpha, rng, cfg.pref_clip),
            state_pref=blx(a.state_pref, b.state_pref, cfg.blx_alpha, rng,
                           cfg.pref_clip),
            epsilon=(a.epsilon if rng.random() < 0.5 else b.epsilon),
            switch_patience=(a.switch_patience if rng.random() < 0.5
                             else b.switch_patience),
            parents=(a.gid, b.gid), origin="blx-offspring",
            cross_detail={"method": "BLX-alpha", "alpha": cfg.blx_alpha,
                          "parent_a": a.gid, "parent_b": b.gid},
        )
    else:
        child = crossover(a, b, rng)
    child = mutate(child, rng, cfg)
    return child


# --------------------------------------------------------------------------
# One generation: shaped evaluation + reward-threshold selection
# --------------------------------------------------------------------------

def evolve_one_generation_shaped(
    population: List[Genome], gen: int, bundle: Bundle, cfg: EvoConfig,
    rng: random.Random, seed_base: int,
    reward_config: Optional[RewardConfig] = None,
) -> Tuple[List[Genome], dict]:
    """Evaluate on the shaped return, select by reward-threshold + solved rate,
    reproduce by crossover/mutation.  Returns ``(new_pop, record)``."""
    for g in population:
        evaluate_genome_shaped(g, bundle, cfg, _seed_for(g.gid, seed_base),
                               reward_config)

    ranked = sorted(population, key=lambda g: (-g.fitness, g.gid))
    best = ranked[0]
    worst = ranked[-1]
    mean_fitness = sum(g.fitness for g in population) / len(population)
    # Reward-threshold on a possibly-negative shaped return.  When the best is
    # positive we keep the Unit C rule ``threshold = frac * best``; when every
    # return is non-positive we threshold by margin from the worst so the best
    # genome is always inside the reproduction pool.
    if best.fitness > 0.0:
        threshold = cfg.threshold_frac * best.fitness
    else:
        threshold = best.fitness - cfg.threshold_frac * (best.fitness
                                                         - worst.fitness)
    if threshold > best.fitness:
        threshold = best.fitness

    qualified = [g for g in population if g.fitness >= threshold]
    choosers = sorted(population, key=lambda g: (-g.chooser_score, g.gid)
                      )[:cfg.top_k]
    solvers = sorted(population, key=lambda g: (-g.solver_score, g.gid)
                     )[:cfg.top_k]
    pool = _dedup(qualified + choosers + solvers)
    if len(pool) < 2:
        pool = list(ranked[:2])

    elites = [clone_genome(g, gid=g.gid, origin="elite")
              for g in ranked[:cfg.elites]]
    offspring: List[Genome] = []
    need = max(0, cfg.population - len(elites))
    restarted = False
    if cfg.restart_on_stall and cfg.restart_on_stall > 0:
        _update_stall(best.fitness)
        if int(_STALL["gens_since_best"]) > cfg.restart_on_stall:
            seed_parent = ranked[0]
            for i in range(need):
                child = clone_genome(
                    seed_parent, gid="g%05d-c%03d" % (gen + 1, i),
                    origin="restart")
                for j in range(len(child.pref)):
                    child.pref[j] = _clamp(
                        child.pref[j] + rng.gauss(0.0, cfg.restart_sigma),
                        -cfg.pref_clip, cfg.pref_clip)
                for j in range(len(child.state_pref)):
                    child.state_pref[j] = _clamp(
                        child.state_pref[j]
                        + rng.gauss(0.0, cfg.restart_sigma),
                        -cfg.pref_clip, cfg.pref_clip)
                child.epsilon = _clamp(
                    seed_parent.epsilon + rng.gauss(0.0, 0.1),
                    cfg.epsilon_min, cfg.epsilon_max)
                child.switch_patience = int(_clamp(
                    seed_parent.switch_patience + rng.choice((-1, 1)), 2, 20))
                child.parents = (seed_parent.gid,)
                child.cross_detail = {"restart_from": seed_parent.gid}
                offspring.append(child)
            _STALL["gens_since_best"] = 0
            _STALL["restarts"] = int(_STALL["restarts"]) + 1
            restarted = True
    if not restarted:
        while len(offspring) < need:
            a, b = _pick_pair(rng, choosers, solvers, pool)
            child = _make_child(a, b, rng, cfg)
            child.gid = "g%05d-c%03d" % (gen + 1, len(offspring))
            offspring.append(child)

    new_pop = elites + offspring
    record = {
        "generation": gen + 1,
        "best_gid": best.gid,
        "best_fitness": round(best.fitness, 6),
        "best_shaped_return": round(
            best._result.shaped_return if best._result else 0.0, 6),
        "mean_fitness": round(mean_fitness, 6),
        "threshold": round(threshold, 6),
        "restart": restarted,
        "gens_since_best": int(_STALL["gens_since_best"])
        if (cfg.restart_on_stall and cfg.restart_on_stall > 0) else 0,
        "restarts_total": int(_STALL["restarts"]),
        "qualified": [g.gid for g in qualified],
        "choosers": [g.gid for g in choosers],
        "solvers": [g.gid for g in solvers],
        "pool": [g.gid for g in pool],
        "agents": [{
            "gid": g.gid, "origin": g.origin, "parents": list(g.parents),
            "fitness": round(g.fitness, 6),
            "shaped_return": round(
                g._result.shaped_return if g._result else 0.0, 6),
            "solved_rate": round(
                g._result.solved_rate() if g._result else 0.0, 6),
            "solver_score": round(g.solver_score, 6),
            "chooser_score": round(g.chooser_score, 6),
            "solved_feasible": g.solved_feasible,
            "steps": g.steps,
            "cross_detail": g.cross_detail,
        } for g in population],
        "offspring": [{
            "gid": c.gid, "parents": list(c.parents),
            "cross_detail": c.cross_detail,
        } for c in offspring],
        "best_per_state": best._result.per_state_rows() if best._result else [],
        "best_trace": [{"step": e.step, "kind": e.kind, "sid": e.sid,
                        "detail": e.detail}
                       for e in (best._result.trace if best._result else [])],
    }
    return new_pop, record


# --------------------------------------------------------------------------
# Single-case shaped training (core U2 suite / extended complex suite)
# --------------------------------------------------------------------------

def train_shaped(agent: EvolutionAgent, cases: Sequence, episodes: int,
                 reward_config: Optional[RewardConfig] = None) -> None:
    """Train ``agent`` on ``cases`` with the shaped reward (NO scaffolding)."""
    rcfg = reward_config or RewardConfig()
    agent.rng = random.Random(agent.seed)
    for ep in range(max(1, episodes)):
        epsilon = agent._epsilon(ep)
        for case in cases:
            target = case.env.goal.target
            shaper = EpisodeShaper(rcfg)
            state = case.env.reset()
            trajectory: List[Tuple[str, float, float, str]] = []
            for _ in range(case.max_steps):
                if case.env.goal_achieved(state):
                    break
                ctx = agent._ctx(state)
                action = agent.epsilon_greedy(case.key, state, epsilon)
                nxt = case.env.step(state, action)
                reached = case.env.goal_achieved(nxt)
                rb = shaper.reward(target, state, nxt, reached)
                td = agent.learn(case.key, state, action, rb.total, nxt, reached)
                trajectory.append((action, rb.total, td, ctx))
                state = nxt
                if reached:
                    break
            agent.update_pref(trajectory)


def _make_case_agent(genome: Genome, cfg: EvoConfig, order, arity,
                     seed: int, episodes: int) -> EvolutionAgent:
    return EvolutionAgent(
        action_order=order, arity=arity, alpha=cfg.alpha,
        gamma=cfg.reward_gamma, epsilon_start=max(cfg.epsilon_min, genome.epsilon),
        epsilon_end=max(cfg.epsilon_min, 0.02), episodes=max(1, episodes),
        seed=seed, use_pref=True, pref_lr=cfg.pref_lr, beta=cfg.beta,
        pref_sigma=cfg.sigma_pref, pref_mode=cfg.pref_mode,
        pref_init=genome.pref_map(), switch_patience=genome.switch_patience,
    )


def core_validation_shaped(genome: Genome, cfg: EvoConfig, seed: int,
                           episodes: int) -> Tuple[int, int]:
    """Seed a core-U2 agent with the genome's instinct, train on the CORE train
    cases with the shaped reward, validate on the SAME 33 new-initial states as
    Stage 2/3 (directly comparable)."""
    from .evolution_agents import CORE_ARITY, CORE_ORDER, evaluate
    from .evolution_run import core_train_cases, core_validation_cases

    agent = _make_case_agent(genome, cfg, CORE_ORDER, CORE_ARITY, seed,
                             episodes)
    train_shaped(agent, core_train_cases(), episodes)
    res = evaluate(agent, core_validation_cases())
    return res.passed, res.total


def evo_validation_shaped(genome: Genome, cfg: EvoConfig, seed: int,
                          episodes: int) -> Tuple[int, int]:
    """Seed an extended agent with the genome's instinct, train on the 11 complex
    scenarios with the shaped reward, validate on the 114 new-initial states."""
    from .evolution_agents import evaluate
    from .evolution_run import evo_train_cases, evo_validation_cases

    agent = _make_case_agent(genome, cfg, EVO_ORDER, EVO_ARITY, seed, episodes)
    train_shaped(agent, evo_train_cases(), episodes)
    res = evaluate(agent, evo_validation_cases())
    return res.passed, res.total


def bundle_validation(genome: Genome, bundle: Bundle, cfg: EvoConfig,
                      seed: int,
                      reward_config: Optional[RewardConfig] = None
                      ) -> ShapedBundleResult:
    """Greedy shaped evaluation of the genome on the multi-state bundle."""
    agent = train_genome_shaped(genome, bundle, cfg, seed, reward_config)
    return run_bundle_shaped(
        agent, bundle, genome.state_pref, total_budget=cfg.total_budget,
        training=False, reward_config=reward_config)
