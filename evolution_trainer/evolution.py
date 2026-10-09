"""Unit B: the evolutionary process on the generic dynamic-nodes environment.

This module implements the operator mandate (thread 4328) as a deterministic,
standard-library only loop:

1. **Rewards along the way.** An episode grants an intermediate reward every time
   an elementar objective node moves to its target value, a guard bonus when a
   previously locked objective becomes settable, and a larger final reward when
   the (sub)goal is reached. Every step is charged a step cost.
2. **Reproduction by two best agents.** Agents rank by fitness; two of the best
   may SPEND a reproduction fee from their reward budgets TOGETHER to create one
   CHILD whose weights are a per-weight blend/selection of BOTH parents plus
   gaussian mutation. Parents that cannot afford the fee die childless.
3. **Subagent recursion.** An agent that reaches a helpful intermediate state
   may spend a spawn fee to create a SUBAGENT targeted at the REMAINING
   subgoal. The subagent runs the same loop (and may itself spawn a subagent),
   and a capped share of its reward credits back to the spawning agent.
4. **Step spending.** The per-step cost makes short, high-reward trajectories
   out-reproduce long, unproductive ones.
5. **Mutation.** Gaussian mutation on every child keeps the search alive.
6. **Persistence.** The best agent of every generation is checkpointed as a
   genome JSON, `best_agents_persist.json` accumulates the winners, and
   `history.csv` / `reward_trajectory.csv` carry the generation table and the
   reward curve.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dynamic_env.engine import Action, DynamicEnv, State
from dynamic_env.spec import Spec

from .features import ActionFeaturizer, objective_ids, state_dim, state_features
from .genome import PolicyNet, mix_genomes, mutate_genome, round_genome


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass
class EvoConfig:
    """Every knob of the evolutionary process (CLI-overridable)."""

    population_size: int = 12
    generations: int = 20
    episodes_per_agent: int = 1
    hidden: int = 12
    # reward shaping
    step_cost: float = 0.05
    intermediate_reward: float = 0.30
    guard_bonus: float = 0.20
    subgoal_reward: float = 0.50
    goal_reward: float = 1.00
    # reproduction
    reproduction_fee: float = 1.00
    parent_frac: float = 0.60
    elite_frac: float = 0.25
    mutation_rate: float = 0.15
    mutation_sigma: float = 0.20
    mix_alpha: float = 0.50
    select_prob: float = 0.50
    per_gene_blend: bool = True
    # subagent recursion
    subagent_depth: int = 2
    subagent_spawn_fee: float = 0.30
    subagent_credit_rate: float = 1.0
    subagent_credit_cap: float = 0.50
    max_subagents_per_episode: int = 2
    # execution
    max_episode_steps: Optional[int] = None
    seed: int = 7
    checkpoint_dir: str = "checkpoints"
    history_dir: str = "history"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "EvoConfig":
        fields = {f for f in EvoConfig.__dataclass_fields__}
        clean = {k: v for k, v in raw.items() if k in fields}
        return EvoConfig(**clean)


# --------------------------------------------------------------------------
# Agents and episode results
# --------------------------------------------------------------------------

@dataclass
class Agent:
    aid: str
    genome: List[float]
    origin: str = "founder"
    parents: Tuple[str, ...] = ()
    birth_generation: int = 0
    depth: int = 0
    mutation_magnitude: float = 0.0
    note: str = ""
    budget: float = 0.0
    fitness: float = float("-inf")
    best_return: float = float("-inf")
    steps_gen: int = 0
    solved_gen: int = 0
    total_steps: int = 0
    solved: int = 0
    episodes: int = 0
    subagents_spawned: int = 0
    alive: bool = True
    scripted_keys: Optional[Tuple[str, ...]] = None

    def genome_json(self) -> Dict[str, Any]:
        return {
            "id": self.aid,
            "origin": self.origin,
            "parents": list(self.parents),
            "birth_generation": self.birth_generation,
            "depth": self.depth,
            "mutation_magnitude": round(self.mutation_magnitude, 6),
            "fitness": None if self.fitness == float("-inf") else round(self.fitness, 6),
            "budget": round(self.budget, 6),
            "solved": self.solved,
            "total_steps": self.total_steps,
            "note": self.note,
            "genome": round_genome(self.genome),
        }


@dataclass
class EpisodeResult:
    total_return: float
    steps: int
    solved: bool
    subgoal_done: bool
    subagents: List[Dict[str, Any]] = field(default_factory=list)


# --------------------------------------------------------------------------
# The trainer
# --------------------------------------------------------------------------

class EvolutionTrainer:
    """A fixed-weight evolutionary loop over one dynamic-nodes spec."""

    def __init__(self, spec: Spec, config: Optional[EvoConfig] = None,
                 spec_id: str = "") -> None:
        self.spec = spec
        self.spec_id = spec_id or spec.spec_id
        self.config = config or EvoConfig()
        self.rng = random.Random(self.config.seed)
        self.featurizer = ActionFeaturizer(spec)
        self.in_dim = state_dim(spec) + self.featurizer.dim
        self.net = PolicyNet(self.in_dim, self.config.hidden)
        self.obj_ids = objective_ids(spec)
        self.target = tuple(int(spec.objectives[oid]) for oid in self.obj_ids)
        self.root_mask = tuple(1 for _ in self.obj_ids)
        self.population: List[Agent] = []
        self.history: List[Dict[str, Any]] = []
        self._serial = 0
        self.generation_subagents: List[Dict[str, Any]] = []
        self._next_population: List[Agent] = []

    # -- ids ---------------------------------------------------------------
    def _next_aid(self, generation: int, tag: str) -> str:
        self._serial += 1
        return "g%03d-%s%03d" % (generation, tag, self._serial)

    # -- masks -------------------------------------------------------------
    @staticmethod
    def _subgoal_reached(flags: Sequence[int], target: Sequence[int],
                         mask: Sequence[int]) -> bool:
        return all(flags[i] == target[i] for i in range(len(target)) if mask[i])

    # -- action choice -----------------------------------------------------
    def _logits(self, agent: Agent, env: DynamicEnv, state: State,
                actions: Sequence[Action], subgoal_mask: Sequence[int]) -> List[float]:
        base = state_features(env, state, subgoal_mask)
        out: List[float] = []
        for action in actions:
            x = base + self.featurizer.featurize(state, action)
            out.append(self.net.forward(agent.genome, x))
        return out

    def _choose(self, agent: Agent, env: DynamicEnv, state: State,
                actions: Sequence[Action], subgoal_mask: Sequence[int],
                explore: bool = True) -> Action:
        if agent.scripted_keys:
            remaining = list(agent.scripted_keys)
            for key in list(remaining):
                for action in actions:
                    if action.key() == key:
                        remaining.remove(key)
                        agent.scripted_keys = tuple(remaining)
                        return action
        logits = self._logits(agent, env, state, actions, subgoal_mask)
        if not explore:
            best = max(range(len(actions)), key=lambda i: (logits[i], -i))
            return actions[best]
        peak = max(logits)
        weights = [math.exp(value - peak) for value in logits]
        total = sum(weights)
        draw = self.rng.random() * total
        acc = 0.0
        for action, weight in zip(actions, weights):
            acc += weight
            if draw <= acc:
                return action
        return actions[-1]

    # -- one episode -------------------------------------------------------
    def run_episode(self, agent: Agent, spec: Spec, subgoal_mask: Sequence[int],
                    depth: int, generation: int,
                    start_state: Optional[State] = None,
                    allow_spawn: bool = True) -> EpisodeResult:
        env = DynamicEnv(spec)
        state = start_state if start_state is not None else env.reset()
        max_steps = self.config.max_episode_steps or spec.max_steps
        target = tuple(int(spec.objectives[oid]) for oid in sorted(spec.objectives))
        n_obj = len(target)
        total = 0.0
        steps = 0
        solved = False
        subgoal_done = False
        subagents: List[Dict[str, Any]] = []

        prev_flags = env.objective_vector(state)
        if self._subgoal_reached(prev_flags, target, subgoal_mask):
            return EpisodeResult(0.0, 0, env.goal_reached(state), True, [])

        actions = env.legal_actions(state)
        set_legal = frozenset(a.objective_id for a in actions if a.kind == "set")
        credited: set = set()  # each objective's progress is rewarded ONCE per episode
        unlocked_credited: set = set()  # each guard unlock is rewarded ONCE per episode
        guarded_ids = frozenset(spec.guards)

        while steps < max_steps and actions:
            action = self._choose(agent, env, state, actions, subgoal_mask)
            result = env.step(state, action)
            flags = env.objective_vector(result.state)
            reward = -self.config.step_cost
            step_flips = 0
            for i in range(n_obj):
                if (subgoal_mask[i] and i not in credited
                        and prev_flags[i] != target[i] and flags[i] == target[i]):
                    reward += self.config.intermediate_reward
                    step_flips += 1
                    credited.add(i)
            next_actions = env.legal_actions(result.state)
            next_set_legal = frozenset(a.objective_id for a in next_actions if a.kind == "set")
            new_unlocks = ((next_set_legal - set_legal) & guarded_ids) - unlocked_credited
            if new_unlocks:
                reward += self.config.guard_bonus * len(new_unlocks)
                unlocked_credited |= new_unlocks
            done_full = result.done
            done_sub = self._subgoal_reached(flags, target, subgoal_mask)
            if done_full:
                reward += self.config.goal_reward
            elif done_sub:
                reward += self.config.subgoal_reward
            total += reward
            steps += 1

            if (allow_spawn and depth < self.config.subagent_depth
                    and agent.budget >= self.config.subagent_spawn_fee
                    and len(subagents) < self.config.max_subagents_per_episode):
                remaining = tuple(1 if flags[i] != target[i] else 0 for i in range(n_obj))
                helpful = step_flips > 0 or bool(new_unlocks)
                if helpful and any(remaining) and not done_full:
                    record = self._spawn_subagent(
                        agent, spec, result.state, remaining, depth + 1, generation)
                    if record is not None:
                        subagents.append(record)

            prev_flags = flags
            state = result.state
            actions = next_actions
            set_legal = next_set_legal
            if done_full or done_sub:
                solved = done_full
                subgoal_done = done_sub
                break

        flags = env.objective_vector(state)
        solved = solved or env.goal_reached(state)
        subgoal_done = subgoal_done or self._subgoal_reached(flags, target, subgoal_mask)
        return EpisodeResult(total, steps, solved, subgoal_done, subagents)

    # -- subagent recursion ------------------------------------------------
    def _best_other(self, agent: Agent) -> Optional[Agent]:
        others = [a for a in self.population if a is not agent and a.alive]
        others.sort(key=lambda a: (-a.budget, a.aid))
        return others[0] if others else None

    def _spawn_subagent(self, parent: Agent, spec: Spec, state: State,
                        remaining_mask: Tuple[int, ...], depth: int,
                        generation: int) -> Optional[Dict[str, Any]]:
        if depth > self.config.subagent_depth or not any(remaining_mask):
            return None
        fee = self.config.subagent_spawn_fee
        if parent.budget < fee:
            return None
        parent.budget -= fee
        second = self._best_other(parent)
        half = fee / 2.0
        if second is not None and second.budget >= half:
            second.budget -= half
            mix = mix_genomes(
                parent.genome, second.genome, self.rng,
                alpha=self.config.mix_alpha, select_prob=self.config.select_prob,
                per_gene_blend=self.config.per_gene_blend,
                mutation_rate=self.config.mutation_rate,
                mutation_sigma=self.config.mutation_sigma,
            )
            parents = (parent.aid, second.aid)
            note = ("subagent of %s + %s at depth %d: %d blended / %d selected genes"
                    % (parent.aid, second.aid, depth, mix.blend_count, mix.select_count))
        else:
            mix = mutate_genome(
                parent.genome, self.rng,
                self.config.mutation_rate, self.config.mutation_sigma)
            parents = (parent.aid,)
            note = ("subagent mutated copy of %s at depth %d (no second parent affordable)"
                    % (parent.aid, depth))
        sub = Agent(
            aid=self._next_aid(generation, "s"),
            genome=mix.child,
            origin="subagent",
            parents=parents,
            birth_generation=generation,
            depth=depth,
            mutation_magnitude=mix.mutation_magnitude,
            note=note,
            budget=0.0,
            scripted_keys=parent.scripted_keys,
        )
        episode = self.run_episode(sub, spec, remaining_mask, depth, generation,
                                   start_state=state, allow_spawn=True)
        credit = min(max(0.0, episode.total_return) * self.config.subagent_credit_rate,
                     self.config.subagent_credit_cap)
        parent.budget += credit
        parent.subagents_spawned += 1
        return {
            "agent_id": sub.aid,
            "parents": list(parents),
            "depth": depth,
            "subgoal_mask": list(remaining_mask),
            "return": round(episode.total_return, 6),
            "steps": episode.steps,
            "solved": episode.solved,
            "subgoal_done": episode.subgoal_done,
            "credit": round(credit, 6),
            "mutation_magnitude": round(sub.mutation_magnitude, 6),
            "note": note,
            "nested": episode.subagents,
        }

    # -- population --------------------------------------------------------
    def _founders(self) -> List[Agent]:
        population: List[Agent] = []
        for _ in range(self.config.population_size):
            aid = self._next_aid(0, "f")
            genome = self.net.init(self.rng, scale=0.5)
            population.append(Agent(aid=aid, genome=genome, origin="founder",
                                    birth_generation=0))
        return population

    def _evaluate(self, population: List[Agent], generation: int) -> None:
        self.population = population
        self.generation_subagents = []
        for agent in population:
            agent.fitness = float("-inf")
            agent.steps_gen = 0
            agent.solved_gen = 0
            for _ in range(self.config.episodes_per_agent):
                episode = self.run_episode(agent, self.spec, self.root_mask, 0, generation)
                agent.episodes += 1
                agent.steps_gen += episode.steps
                agent.total_steps += episode.steps
                agent.budget += episode.total_return
                if episode.solved:
                    agent.solved += 1
                    agent.solved_gen += 1
                agent.best_return = max(agent.best_return, episode.total_return)
                agent.fitness = (episode.total_return if agent.fitness == float("-inf")
                                 else max(agent.fitness, episode.total_return))
                self.generation_subagents.extend(episode.subagents)

    def _reproduce(self, population: List[Agent], generation: int) -> List[Dict[str, Any]]:
        ranked = sorted(population, key=lambda a: (-a.fitness, -a.budget, a.aid))
        elite_count = max(1, int(round(len(population) * self.config.elite_frac)))
        elites = ranked[:elite_count]
        pool_size = max(2, int(math.ceil(len(population) * self.config.parent_frac)))
        pool = ranked[:pool_size]
        max_offspring = max(0, self.config.population_size - len(elites))
        offspring: List[Agent] = []
        births: List[Dict[str, Any]] = []
        pairs = [(pool[i], pool[j])
                 for i in range(len(pool)) for j in range(i + 1, len(pool))]
        half = self.config.reproduction_fee / 2.0
        for parent_a, parent_b in pairs:
            if len(offspring) >= max_offspring:
                break
            if parent_a.budget < half or parent_b.budget < half:
                continue
            parent_a.budget -= half
            parent_b.budget -= half
            mix = mix_genomes(
                parent_a.genome, parent_b.genome, self.rng,
                alpha=self.config.mix_alpha, select_prob=self.config.select_prob,
                per_gene_blend=self.config.per_gene_blend,
                mutation_rate=self.config.mutation_rate,
                mutation_sigma=self.config.mutation_sigma,
            )
            note = ("child of %s + %s: %d blended / %d selected genes, "
                    "|delta|=%.4f" % (parent_a.aid, parent_b.aid,
                                      mix.blend_count, mix.select_count,
                                      mix.mutation_magnitude))
            hull_ok = all(
                min(gene_a, gene_b) - 1e-12 <= gene <= max(gene_a, gene_b) + 1e-12
                for gene_a, gene_b, gene in zip(parent_a.genome, parent_b.genome, mix.blend)
            )
            child = Agent(
                aid=self._next_aid(generation, "o"),
                genome=mix.child,
                origin="offspring",
                parents=(parent_a.aid, parent_b.aid),
                birth_generation=generation,
                mutation_magnitude=mix.mutation_magnitude,
                note=note,
            )
            offspring.append(child)
            births.append({
                "child": child.aid,
                "parents": [parent_a.aid, parent_b.aid],
                "origin": "offspring",
                "fee_paid": round(self.config.reproduction_fee, 6),
                "mutation_magnitude": round(mix.mutation_magnitude, 6),
                "blend_count": mix.blend_count,
                "select_count": mix.select_count,
                "hull_ok": bool(hull_ok),
                "note": note,
            })

        new_population: List[Agent] = list(elites) + offspring
        while len(new_population) < self.config.population_size:
            base = elites[0]
            mix = mutate_genome(base.genome, self.rng,
                                self.config.mutation_rate, self.config.mutation_sigma)
            note = ("asexual mutated copy of %s (population refill), |delta|=%.4f"
                    % (base.aid, mix.mutation_magnitude))
            child = Agent(
                aid=self._next_aid(generation, "m"),
                genome=mix.child,
                origin="asexual",
                parents=(base.aid,),
                birth_generation=generation,
                mutation_magnitude=mix.mutation_magnitude,
                note=note,
            )
            new_population.append(child)
            births.append({
                "child": child.aid,
                "parents": [base.aid],
                "origin": "asexual",
                "fee_paid": 0.0,
                "mutation_magnitude": round(mix.mutation_magnitude, 6),
                "blend_count": 0,
                "select_count": 0,
                "hull_ok": True,
                "note": note,
            })

        survivor_ids = {a.aid for a in new_population}
        for agent in population:
            if agent.aid not in survivor_ids:
                agent.alive = False
        self._next_population = new_population
        return births

    # -- history / persistence --------------------------------------------
    def _generation_record(self, generation: int, population: List[Agent],
                           births: List[Dict[str, Any]]) -> Dict[str, Any]:
        best = max(population, key=lambda a: (a.fitness, a.budget))
        mean = sum(a.fitness for a in population) / max(1, len(population))
        record = {
            "generation": generation,
            "population": len(population),
            "best_agent": best.aid,
            "best_fitness": round(best.fitness, 6),
            "best_budget": round(best.budget, 6),
            "best_steps": best.steps_gen,
            "mean_fitness": round(mean, 6),
            "solved": sum(a.solved_gen for a in population),
            "births": births,
            "deaths": [a.aid for a in population if not a.alive],
            "subagents": self.generation_subagents,
            "agents": [
                {
                    "generation": generation,
                    "agent_id": a.aid,
                    "origin": a.origin,
                    "parents": " + ".join(a.parents),
                    "fitness": round(a.fitness, 6),
                    "budget": round(a.budget, 6),
                    "steps": a.steps_gen,
                    "total_steps": a.total_steps,
                    "solved": a.solved_gen,
                    "depth": a.depth,
                    "mutation_magnitude": round(a.mutation_magnitude, 6),
                    "alive": int(a.alive),
                    "note": a.note,
                }
                for a in population
            ],
        }
        return record

    def _ensure_dirs(self) -> Tuple[str, str]:
        checkpoint_dir = self.config.checkpoint_dir
        history_dir = self.config.history_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(history_dir, exist_ok=True)
        return checkpoint_dir, history_dir

    def _write_generation(self, generation: int, record: Dict[str, Any]) -> None:
        checkpoint_dir, history_dir = self._ensure_dirs()
        best_id = record["best_agent"]
        best = next(a for a in self.population if a.aid == best_id)
        # best agent of THIS generation as a standalone genome json
        self._atomic_write_json(
            os.path.join(checkpoint_dir, "best_gen_%04d.json" % generation),
            {"generation": generation, "spec_id": self.spec_id,
             "best_fitness": record["best_fitness"], "best_agent": best.genome_json()},
        )
        # accumulate the winners across generations
        persist_path = os.path.join(checkpoint_dir, "best_agents_persist.json")
        if os.path.exists(persist_path):
            with open(persist_path, "r", encoding="utf-8") as handle:
                persist = json.load(handle)
        else:
            persist = {"spec_id": self.spec_id, "agents": [], "all_time_best": None}
        persist["spec_id"] = self.spec_id
        persist["agents"].append({
            "generation": generation,
            "fitness": record["best_fitness"],
            "agent": best.genome_json(),
        })
        current_best = persist.get("all_time_best")
        if current_best is None or record["best_fitness"] > current_best.get("fitness", float("-inf")):
            persist["all_time_best"] = {"generation": generation,
                                        "fitness": record["best_fitness"],
                                        "agent": best.genome_json()}
        self._atomic_write_json(persist_path, persist)
        # full-population checkpoint (warm start) + config
        self._atomic_write_json(
            os.path.join(checkpoint_dir, "checkpoint.json"),
            {"generation": generation, "spec_id": self.spec_id,
             "config": self.config.to_dict(),
             "population": [a.genome_json() for a in self.population]},
        )
        # generation table (one row per evaluated agent)
        self._append_agent_rows(history_dir, record)
        self._append_trajectory_row(history_dir, record)

    def _append_agent_rows(self, history_dir: str, record: Dict[str, Any]) -> None:
        path = os.path.join(history_dir, "history.csv")
        header = ["generation", "agent_id", "origin", "parents", "fitness", "budget",
                  "steps", "total_steps", "solved", "depth", "mutation_magnitude",
                  "alive", "note"]
        write_header = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            if write_header:
                writer.writeheader()
            for row in record["agents"]:
                writer.writerow({key: row.get(key, "") for key in header})

    def _append_trajectory_row(self, history_dir: str, record: Dict[str, Any]) -> None:
        path = os.path.join(history_dir, "reward_trajectory.csv")
        header = ["generation", "population", "best_agent", "best_fitness",
                  "mean_fitness", "solved", "best_steps", "births", "deaths",
                  "subagents"]
        row = {
            "generation": record["generation"],
            "population": record["population"],
            "best_agent": record["best_agent"],
            "best_fitness": record["best_fitness"],
            "mean_fitness": record["mean_fitness"],
            "solved": record["solved"],
            "best_steps": record["best_steps"],
            "births": len(record["births"]),
            "deaths": len(record["deaths"]),
            "subagents": len(record["subagents"]),
        }
        write_header = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    # -- driver ------------------------------------------------------------
    def run(self, warm_start: Optional[Sequence[Sequence[float]]] = None
            ) -> List[Dict[str, Any]]:
        """Evolve. ``warm_start`` optionally supplies founder genomes.

        Unit C's MIX arm uses this to seed the fixed-weight evolution with the
        genome an RL warm-start already learned (plus the caller's own mutated
        copies). The default ``None`` is exactly Unit B's behaviour: random
        founders, so ``train.py`` and the pure evolution arm are unchanged.
        """
        if warm_start:
            population: List[Agent] = []
            for genome in warm_start[:self.config.population_size]:
                aid = self._next_aid(0, "w")
                population.append(Agent(aid=aid, genome=list(genome),
                                        origin="warm_start", birth_generation=0))
            while len(population) < self.config.population_size:
                aid = self._next_aid(0, "f")
                population.append(Agent(aid=aid, genome=self.net.init(self.rng, scale=0.5),
                                        origin="founder", birth_generation=0))
        else:
            population = self._founders()
        for generation in range(1, self.config.generations + 1):
            self._evaluate(population, generation)
            births = self._reproduce(population, generation)
            record = self._generation_record(generation, population, births)
            self.history.append(record)
            self._write_generation(generation, record)
            population = self._next_population
        return self.history
