"""Unit C: population evolution, selection, mixing and JSON persistence.

Genome (the agent's heritable BASIC INSTINCT):
* ``net``             - flat weights of the target-conditioned PolicyNet (one
                        tanh hidden layer).  The same parameters score every
                        action for any (target, stack), and the bundle start
                        order is scored by the same net over the ``__start__``
                        pseudo action.  Replaces the retired flat ``pref`` and
                        ``state_pref`` vectors.
* ``epsilon``         - exploration temperament (initial epsilon).
* ``switch_patience`` - stagnation window before a mid-state switch.

Evolution per generation:
1. Every genome is instantiated as an ``EvolutionAgent`` whose preference vector
   is SEEDED from the genome (``pref_init``) and trained on the multi-state
   bundle under ONE total step budget; it is then evaluated greedily.
2. Selection is the operator required by the mandate:
   * ELITISM   - the top ``elites`` genomes by fitness survive unchanged;
   * THRESHOLD - every agent whose reward >= ``threshold_frac * best`` joins
                 the reproduction pool;
   * the best STATE-CHOOSERS (``chooser_score``) and best SOLVERS
     (``solver_score``) are added to the pool so their information can be MIXED;
   * REPLACEMENT - offspring fill the fixed-size population and replace the
     worst genomes.
3. Offspring = crossover of two parents' instinct genes + bounded mutation.
4. The whole population, the generation history and the best agent's per-state
   results are PERSISTED as JSON.  ``load_checkpoint`` restores them so a later
   process warm-starts from the saved generation.

Pure standard library, seeded and bounded.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .evolution_agents import EVO_ARITY, EVO_ORDER, EvolutionAgent
from .evolution_bundle import (
    Bundle,
    BundleResult,
    run_bundle,
    trace_rows,
)
from .target_features import PolicyNet, net_init

CHECKPOINT_VERSION = 1


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


@dataclass
class EvoConfig:
    population: int = 16
    episodes: int = 30
    generations: int = 100000
    total_budget: int = 60
    elites: int = 2
    threshold_frac: float = 0.6
    top_k: int = 4
    mutation_rate: float = 0.25
    mutation_sigma: float = 0.15
    alpha: float = 0.5
    sigma_pref: float = 0.5
    sigma_state: float = 0.5
    # Flavor A: the genome is the flat weight vector of the target-conditioned
    # PolicyNet; ``net_sigma`` is the initial/mutation scale of those weights.
    net_hidden: int = PolicyNet.HIDDEN
    net_sigma: float = 0.5
    pref_clip: float = 5.0
    epsilon_min: float = 0.02
    epsilon_max: float = 0.8
    pref_lr: float = 0.2
    beta: float = 1.0
    pref_mode: str = "td"
    seed: int = 20261009
    checkpoint_dir: str = "/opt/automath/tmp/evolution/checkpoints"
    progress_dir: str = "/opt/automath/tmp/evolution/progress"
    history_every: int = 1
    # Hard cap on the in-RAM (and rewritten) history TAIL.  The full generation
    # table is appended to history.jsonl once per generation, so a 10k+
    # generation run keeps O(1) RAM and O(record) incremental IO instead of
    # rewriting a history that grows without bound.
    history_max: int = 200
    checkpoint_every: int = 1
    heartbeat_secs: float = 1800.0
    heartbeat_generations: int = 200
    wall_clock_hours: float = 24.0
    log_file: str = "/opt/automath/tmp/evolution/longrun.log"
    # -- stall / plateau breaker -------------------------------------------
    # When > 0, if the best fitness does not improve for this many
    # generations, re-seed the NON-ELITE population from the best elite with
    # fresh gaussian noise (KEEPING the elites unchanged).  0 disables it, so
    # a config that omits these keys reproduces the original search exactly.
    restart_on_stall: int = 0
    restart_sigma: float = 0.5
    # -- Unit B (Phase A) shaped-reward extension --------------------------
    # These default to the BINDING reward design of docs/evolution/REWARDS.md.
    # A config that leaves them out reproduces the Unit C search exactly (the
    # semi path is opt-in by calling the evolution_semi module).
    reward_gamma: float = 0.99
    step_cost: float = -0.05
    shaping_cap: float = 0.9
    subgoal_bonus: float = 0.10
    solved_rate_weight: float = 0.5
    use_blx: bool = True
    blx_alpha: float = 0.5
    plateau_generations: int = 300
    plateau_tol: float = 1e-4
    validation_every: int = 100
    validation_ext_every: int = 500
    validation_episodes: int = 300
    # -- Flavor B (task 4333): difficulty curriculum -----------------------
    # When non-empty ({"enabled": true, "phases": [...], ...}) the loop samples
    # its per-generation training bundle from the CURRENT phase's dense-case
    # tier(s).  The policy, reward and fitness formula are unchanged; an empty
    # dict reproduces flavor A exactly.
    curriculum: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict) -> "EvoConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Genome:
    # Flavor A: the heritable policy is the flat weight vector of a
    # target-conditioned MLP (see target_features.PolicyNet).  The retired
    # ``pref`` (21) and ``state_pref`` (7) flat vectors are GONE: the same net
    # now scores every action for any (target, stack), and the bundle start
    # order is scored by the same net over the '__start__' pseudo action.
    net: List[float]
    epsilon: float
    switch_patience: int
    gid: str = ""
    parents: Tuple[str, ...] = ()
    origin: str = "random"
    cross_detail: Dict[str, object] = field(default_factory=dict)
    # runtime (not heritable) results of the last evaluation
    fitness: float = 0.0
    solver_score: float = 0.0
    chooser_score: float = 0.0
    solved_feasible: int = 0
    steps: int = 0
    _result: Optional[BundleResult] = None

    def genes(self) -> dict:
        return {
            "net": [round(v, 6) for v in self.net],
            "epsilon": round(self.epsilon, 6),
            "switch_patience": int(self.switch_patience),
        }

    def to_dict(self) -> dict:
        d = {
            "gid": self.gid,
            "parents": list(self.parents),
            "origin": self.origin,
            "cross_detail": self.cross_detail,
            "fitness": round(self.fitness, 6),
            "solver_score": round(self.solver_score, 6),
            "chooser_score": round(self.chooser_score, 6),
            "solved_feasible": self.solved_feasible,
            "steps": self.steps,
        }
        d.update(self.genes())
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Genome":
        # A legacy checkpoint (pre-flavor-A) has no ``net``; fall back to a
        # zero net of the right length so loading never crashes, and let the
        # caller warm-start from that neutral policy.
        net = ([float(x) for x in data["net"]] if "net" in data
               else [0.0] * PolicyNet.size())
        return cls(
            net=net,
            epsilon=float(data["epsilon"]),
            switch_patience=int(data["switch_patience"]),
            gid=data.get("gid", ""),
            parents=tuple(data.get("parents", [])),
            origin=data.get("origin", "loaded"),
            cross_detail=dict(data.get("cross_detail", {})),
            fitness=float(data.get("fitness", 0.0)),
            solver_score=float(data.get("solver_score", 0.0)),
            chooser_score=float(data.get("chooser_score", 0.0)),
            solved_feasible=int(data.get("solved_feasible", 0)),
            steps=int(data.get("steps", 0)),
        )

    def net_weights(self) -> List[float]:
        return list(self.net)


def random_genome(rng: random.Random, n_actions: int, n_states: int,
                  cfg: EvoConfig, gid: str) -> Genome:
    """A random flavor-A genome (flat PolicyNet weights).

    ``n_actions`` / ``n_states`` are accepted only so callers written for the
    retired ``pref`` / ``state_pref`` signature keep working; they are ignored
    because the net input/output size is fixed by ``PolicyNet``.
    """
    return Genome(
        net=net_init(rng, cfg.net_sigma),
        epsilon=rng.uniform(max(0.05, cfg.epsilon_min), 0.5),
        switch_patience=rng.randint(3, 10),
        gid=gid,
        origin="random",
    )


def clone_genome(g: Genome, gid: Optional[str] = None,
                 origin: str = "elite") -> Genome:
    return Genome(
        net=list(g.net),
        epsilon=g.epsilon, switch_patience=g.switch_patience,
        gid=gid if gid is not None else g.gid,
        parents=tuple(g.parents), origin=origin,
        cross_detail=dict(g.cross_detail),
        fitness=g.fitness, solver_score=g.solver_score,
        chooser_score=g.chooser_score, solved_feasible=g.solved_feasible,
        steps=g.steps, _result=g._result,
    )


def crossover(a: Genome, b: Genome, rng: random.Random) -> Genome:
    """Uniform per-gene crossover over the flat net; records A/B provenance."""
    net: List[float] = []
    na = nb = 0
    for i in range(len(a.net)):
        if rng.random() < 0.5:
            net.append(a.net[i])
            na += 1
        else:
            net.append(b.net[i])
            nb += 1
    if rng.random() < 0.5:
        eps, eps_src = a.epsilon, "A"
    else:
        eps, eps_src = b.epsilon, "B"
    if rng.random() < 0.5:
        pat, pat_src = a.switch_patience, "A"
    else:
        pat, pat_src = b.switch_patience, "B"
    return Genome(
        net=net, epsilon=eps,
        switch_patience=pat, parents=(a.gid, b.gid), origin="offspring",
        cross_detail={
            "net_from_a": na, "net_from_b": nb,
            "epsilon_from": eps_src, "patience_from": pat_src,
        },
    )


def mutate(g: Genome, rng: random.Random, cfg: EvoConfig) -> Genome:
    for i in range(len(g.net)):
        if rng.random() < cfg.mutation_rate:
            g.net[i] = _clamp(g.net[i] + rng.gauss(0.0, cfg.mutation_sigma),
                              -cfg.pref_clip, cfg.pref_clip)
    if rng.random() < cfg.mutation_rate:
        g.epsilon = _clamp(g.epsilon + rng.gauss(0.0, 0.05),
                           cfg.epsilon_min, cfg.epsilon_max)
    if rng.random() < cfg.mutation_rate:
        g.switch_patience = int(_clamp(
            g.switch_patience + rng.choice((-1, 1)), 2, 20))
    return g


# --------------------------------------------------------------------------
# Fitness: train an agent from the genome, evaluate it on the whole bundle
# --------------------------------------------------------------------------

def make_agent(genome: Genome, cfg: EvoConfig, seed: int) -> EvolutionAgent:
    return EvolutionAgent(
        action_order=EVO_ORDER,
        arity=EVO_ARITY,
        alpha=0.5,
        gamma=0.95,
        epsilon_start=max(cfg.epsilon_min, genome.epsilon),
        epsilon_end=max(cfg.epsilon_min, 0.02),
        episodes=max(1, cfg.episodes),
        seed=seed,
        use_pref=False,
        use_net=True,
        net=genome.net,
        pref_lr=cfg.pref_lr,
        beta=cfg.beta,
        pref_sigma=cfg.net_sigma,
        pref_mode=cfg.pref_mode,
        switch_patience=genome.switch_patience,
    )


def train_genome(genome: Genome, bundle: Bundle, cfg: EvoConfig,
                 seed: int) -> EvolutionAgent:
    agent = make_agent(genome, cfg, seed)
    for ep in range(max(1, cfg.episodes)):
        epsilon = agent._epsilon(ep)
        result = run_bundle(agent, bundle, None,
                            total_budget=cfg.total_budget, training=True,
                            epsilon=epsilon)
        agent.update_pref(result.trajectory)
    return agent


def evaluate_genome(genome: Genome, bundle: Bundle, cfg: EvoConfig,
                    seed: int) -> BundleResult:
    agent = train_genome(genome, bundle, cfg, seed)
    result = run_bundle(agent, bundle, None,
                        total_budget=cfg.total_budget, training=False)
    genome.fitness = result.total_reward
    genome.solver_score = result.solver_score()
    genome.chooser_score = result.chooser_score()
    genome.solved_feasible = result.solved_feasible
    genome.steps = result.total_steps
    genome._result = result
    return result


# --------------------------------------------------------------------------
# One generation: selection + reproduction
# --------------------------------------------------------------------------

def _dedup(seq: Sequence[Genome]) -> List[Genome]:
    seen = set()
    out = []
    for g in seq:
        if g.gid not in seen:
            seen.add(g.gid)
            out.append(g)
    return out


def _pick_pair(rng: random.Random, choosers: Sequence[Genome],
               solvers: Sequence[Genome], pool: Sequence[Genome]
               ) -> Tuple[Genome, Genome]:
    a = rng.choice(list(choosers)) if choosers else rng.choice(list(pool))
    b = rng.choice(list(solvers)) if solvers else rng.choice(list(pool))
    if b.gid == a.gid and len(pool) > 1:
        others = [g for g in pool if g.gid != a.gid]
        b = rng.choice(others) if others else b
    return a, b


def _seed_for(gid: str, seed_base: int) -> int:
    """Stable per-genome seed, so an ELITE keeps its measured fitness across
    generations (elitism would otherwise re-roll its training each generation)."""
    return seed_base + (zlib.crc32(gid.encode()) & 0x7FFFFFFF) % 1_000_000


# Module-level plateau tracker (the loop is single-threaded).  It is consulted
# ONLY when ``cfg.restart_on_stall > 0``: a config that leaves it 0 never
# touches this dict and never consumes RNG here, so the original search is
# reproduced byte-for-byte.
_STALL: Dict[str, object] = {
    "best_fitness": None,
    "gens_since_best": 0,
    "restarts": 0,
}


def _update_stall(best_fitness: float) -> None:
    prev = _STALL["best_fitness"]
    if prev is None or best_fitness > float(prev) + 1e-9:
        _STALL["best_fitness"] = best_fitness
        _STALL["gens_since_best"] = 0
    else:
        _STALL["gens_since_best"] = int(_STALL["gens_since_best"]) + 1


def evolve_one_generation(population: List[Genome], gen: int, bundle: Bundle,
                          cfg: EvoConfig, rng: random.Random,
                          seed_base: int) -> Tuple[List[Genome], dict]:
    """Evaluate, select and reproduce once.  Returns (new_pop, record)."""
    for g in population:
        evaluate_genome(g, bundle, cfg, _seed_for(g.gid, seed_base))

    ranked = sorted(population, key=lambda g: (-g.fitness, g.gid))
    best = ranked[0]
    mean_fitness = sum(g.fitness for g in population) / len(population)
    threshold = cfg.threshold_frac * best.fitness

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
                child = clone_genome(seed_parent,
                                     gid="g%05d-c%03d" % (gen + 1, i),
                                     origin="restart")
                for j in range(len(child.net)):
                    child.net[j] = _clamp(
                        child.net[j] + rng.gauss(0.0, cfg.restart_sigma),
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
            child = mutate(crossover(a, b, rng), rng, cfg)
            child.gid = "g%05d-c%03d" % (gen + 1, len(offspring))
            offspring.append(child)

    new_pop = elites + offspring
    record = {
        "generation": gen + 1,
        "best_gid": best.gid,
        "best_fitness": round(best.fitness, 6),
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
        "best_trace": trace_rows(best._result) if best._result else [],
    }
    return new_pop, record


# --------------------------------------------------------------------------
# JSON persistence + warm start
# --------------------------------------------------------------------------

def atomic_write_json(path: str, payload: dict) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def append_history_jsonl(cfg: EvoConfig, record: dict) -> str:
    """Append ONE generation record to ``history.jsonl`` (incremental, O(1) IO).

    The full generation table therefore survives a 10k+ generation run without
    ever rewriting a history file that grows without bound.
    """
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, "history.jsonl")
    with open(path, "a") as fh:
        fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        fh.flush()
    return path


def save_generation(cfg: EvoConfig, gen: int, population: List[Genome],
                    history: List[dict], bundle: Bundle,
                    best: Optional[Genome] = None,
                    force_history: bool = False) -> Dict[str, str]:
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    best = best or population[0]
    ck_path = os.path.join(cfg.checkpoint_dir, "checkpoint.json")
    hist_path = os.path.join(cfg.checkpoint_dir, "history.json")
    state_path = os.path.join(cfg.checkpoint_dir, "state_results.json")
    best_result = best._result
    atomic_write_json(ck_path, {
        "version": CHECKPOINT_VERSION,
        "generation": gen,
        "bundle": bundle.to_dict(),
        "config": cfg.to_dict(),
        "best_gid": best.gid,
        "best_fitness": round(best.fitness, 6),
        "population": [g.to_dict() for g in population],
    })
    # history.json is the bounded warm-start tail; rewrite it only every
    # ``history_every`` generations (or when forced) to keep IO O(1) per gen.
    write_hist = (force_history or cfg.history_every <= 1
                  or gen % max(1, cfg.history_every) == 0
                  or not os.path.exists(hist_path))
    if write_hist:
        atomic_write_json(hist_path, {
            "version": CHECKPOINT_VERSION,
            "generation": gen,
            "records": history[-cfg.history_max:],
        })
    atomic_write_json(state_path, {
        "version": CHECKPOINT_VERSION,
        "generation": gen,
        "best_gid": best.gid,
        "best_fitness": round(best.fitness, 6),
        "per_state": best_result.per_state_rows() if best_result else [],
        "trace": trace_rows(best_result) if best_result else [],
    })
    return {"checkpoint": ck_path, "history": hist_path,
            "state_results": state_path}


def load_checkpoint(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r") as fh:
        return json.load(fh)


def restore_population(ckpt: dict) -> List[Genome]:
    return [Genome.from_dict(d) for d in ckpt["population"]]


def init_population(cfg: EvoConfig, bundle: Bundle,
                    rng: random.Random) -> List[Genome]:
    n_states = len(bundle.states)
    return [random_genome(rng, len(EVO_ORDER), n_states, cfg,
                          "g00001-r%03d" % i)
            for i in range(cfg.population)]
