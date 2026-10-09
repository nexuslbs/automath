"""Deterministic tests for the Unit B evolutionary process.

Run with::

    /opt/automath/venv/bin/python -m evolution_trainer.tests
    /opt/automath/venv/bin/python -m pytest evolution_trainer/tests.py

All randomness is seeded; the checks are cheap (tiny specs, a handful of
generations). The weights-mixing check is the literal proof that an offspring
shares BOTH parents' weights: every noiseless blend gene lies inside the convex
hull of the two parents and ``child == blend + gaussian delta`` exactly.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from typing import Callable, Dict, List, Tuple

from dynamic_env.spec import load_spec, spec_from_dict

from .evolution import EvoConfig, EvolutionTrainer
from .features import ActionFeaturizer, state_dim, state_features
from .genome import PolicyNet, mix_genomes, mutate_genome

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_SPEC_DIR = os.path.join(_REPO_ROOT, "data", "dynamic_env")


class CheckFailure(Exception):
    def __init__(self, name: str, detail: str) -> None:
        super().__init__(detail)
        self.name = name
        self.detail = detail


def _spec_path(stem: str) -> str:
    return os.path.join(_SPEC_DIR, stem + ".json")


def _adhoc_three_objective_spec():
    """A 3-objective spec with no guards: forces depth-2 subagent recursion."""
    return spec_from_dict({
        "spec_id": "adhoc_three_objectives",
        "description": "three elementar objectives, recursion test only",
        "node_types": {
            "flag": {"semantics": "literal", "role": "objective", "value_field": "v",
                     "fields": [{"name": "v", "type": "int"}]},
        },
        "combine_axioms": [],
        "build_axioms": [],
        "dynamic_axioms": [],
        "nodes": {
            "o1": {"type": "flag", "v": 0},
            "o2": {"type": "flag", "v": 0},
            "o3": {"type": "flag", "v": 0},
        },
        "objectives": {"o1": 1, "o2": 1, "o3": 1},
        "guards": {},
        "dynamic_node_type": None,
        "max_combine_arity": 0,
        "step_cost": 0.01,
        "goal_reward": 1.0,
        "max_steps": 6,
    })


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def check_weight_mixing_hull() -> str:
    """Every blended gene is inside the parents' convex hull; child == blend + delta."""
    exact_parent_a = [1.0, -2.0, 3.0, 0.5]
    exact_parent_b = [3.0, 2.0, -1.0, 2.5]
    exact = mix_genomes(exact_parent_a, exact_parent_b, random.Random(0),
                        select_prob=0.0, per_gene_blend=False, alpha=0.5,
                        mutation_rate=0.0, mutation_sigma=0.0)
    expected = [0.5 * (a + b) for a, b in zip(exact_parent_a, exact_parent_b)]
    for got, want in zip(exact.child, expected):
        if abs(got - want) > 1e-12:
            raise CheckFailure("weight_mixing_hull",
                               "fixed-alpha blend %.12f != midpoint %.12f" % (got, want))
    if exact.blend != exact.child:
        raise CheckFailure("weight_mixing_hull", "zero mutation changed the child")

    hull_checks = 0
    strict_genes = 0
    for seed in range(200):
        rng = random.Random(seed)
        parent_a = [rng.uniform(-3.0, 3.0) for _ in range(24)]
        parent_b = [rng.uniform(-3.0, 3.0) for _ in range(24)]
        result = mix_genomes(parent_a, parent_b, rng, select_prob=0.5,
                             per_gene_blend=True, mutation_rate=0.35,
                             mutation_sigma=0.25)
        for gene_a, gene_b, blend, delta, child in zip(
                parent_a, parent_b, result.blend, result.deltas, result.child):
            low, high = min(gene_a, gene_b), max(gene_a, gene_b)
            if not (low - 1e-12 <= blend <= high + 1e-12):
                raise CheckFailure(
                    "weight_mixing_hull",
                    "seed %d: blend %.12f outside hull [%.12f, %.12f]"
                    % (seed, blend, low, high))
            if abs(child - (blend + delta)) > 1e-12:
                raise CheckFailure("weight_mixing_hull",
                                   "seed %d: child != blend + delta" % seed)
            if low < blend < high:
                strict_genes += 1
            hull_checks += 1
        expected_mag = math.sqrt(sum(d * d for d in result.deltas))
        if abs(result.mutation_magnitude - expected_mag) > 1e-12:
            raise CheckFailure("weight_mixing_hull", "mutation magnitude mismatch")
    if strict_genes == 0:
        raise CheckFailure("weight_mixing_hull", "no gene strictly inside a hull")
    select_only = mix_genomes([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0],
                              random.Random(1), select_prob=1.0,
                              per_gene_blend=False, mutation_rate=0.0,
                              mutation_sigma=0.0)
    used_a = any(abs(c - a) < 1e-12 for c, a in zip(select_only.child, [1, 2, 3, 4]))
    used_b = any(abs(c - b) < 1e-12 for c, b in zip(select_only.child, [5, 6, 7, 8]))
    if not (used_a and used_b):
        raise CheckFailure("weight_mixing_hull", "selection did not draw from BOTH parents")
    return ("%d blend genes inside the two-parent convex hull; child==blend+delta exact; "
            "fixed-alpha child == midpoint; selection draws both parents"
            % hull_checks)


def check_mutation_is_gaussian_delta() -> str:
    rng = random.Random(5)
    parent = [0.0] * 32
    result = mutate_genome(parent, rng, mutation_rate=1.0, mutation_sigma=0.5)
    nonzero = sum(1 for d in result.deltas if d != 0.0)
    if nonzero != 32:
        raise CheckFailure("mutation_is_gaussian_delta",
                           "mutation_rate=1.0 left %d genes unchanged" % (32 - nonzero))
    for parent_gene, blend, delta, child in zip(parent, result.blend,
                                                result.deltas, result.child):
        if abs(blend - parent_gene) > 1e-12:
            raise CheckFailure("mutation_is_gaussian_delta",
                               "asexual blend changed the parent gene")
        if abs(child - (blend + delta)) > 1e-12:
            raise CheckFailure("mutation_is_gaussian_delta", "child != blend + delta")
    mag = math.sqrt(sum(d * d for d in result.deltas))
    if abs(result.mutation_magnitude - mag) > 1e-12:
        raise CheckFailure("mutation_is_gaussian_delta", "magnitude mismatch")
    return "mutation_rate=1.0 perturbs all 32 genes; blend==parent; |delta|=%.6f" % mag


def check_network_forward() -> str:
    net = PolicyNet(7, hidden=5)
    expected = 5 * 7 + 5 + 5 + 1
    if net.size != expected:
        raise CheckFailure("network_forward", "size %d != %d" % (net.size, expected))
    genome = [0.0] * net.size
    for x in ([0.0] * 7, [1.0] * 7, [-1.0, 0.5, 0.0, 2.0, -3.0, 1.0, 0.25]):
        value = net.forward(genome, x)
        if value != 0.0:
            raise CheckFailure("network_forward", "zero genome gave %r" % value)
    nonzero = [1.0] * net.size
    first = net.forward(nonzero, [1.0] * 7)
    second = net.forward(nonzero, [0.0] * 7)
    if not math.isfinite(first) or abs(first - second) < 1e-9:
        raise CheckFailure("network_forward", "network is not sensitive to its input")
    return "size=%d; zero genome -> 0.0; nonzero genome depends on the input" % net.size


def check_target_conditioning() -> str:
    spec = load_spec(_spec_path("spec_dynamic_axiom"))
    trainer = EvolutionTrainer(spec, EvoConfig(generations=1), spec_id=spec.spec_id)
    if len(trainer.obj_ids) < 2:
        raise CheckFailure("target_conditioning", "spec_dynamic_axiom needs >= 2 objectives")
    from dynamic_env.engine import DynamicEnv
    env = DynamicEnv(spec)
    state = env.reset()
    actions = env.legal_actions(state)
    genome = [0.1 * ((i % 7) - 3) for i in range(trainer.net.size)]
    agent = trainer._founders()[0]
    agent.genome = genome
    mask_all = (1, 1)
    mask_second = (0, 1)
    logits_all = trainer._logits(agent, env, state, actions, mask_all)
    logits_second = trainer._logits(agent, env, state, actions, mask_second)
    if logits_all == logits_second:
        raise CheckFailure("target_conditioning",
                           "changing the subgoal mask did not change the logits")
    features_all = state_features(env, state, mask_all)
    features_second = state_features(env, state, mask_second)
    target_slice = features_all[len(trainer.obj_ids):2 * len(trainer.obj_ids)]
    if tuple(target_slice) != tuple(float(t) for t in trainer.target):
        raise CheckFailure("target_conditioning", "target is not exposed in the features")
    return ("spec_dynamic_axiom: state features expose the target %s; two subgoal masks "
            "give different action logits (target-conditioned)" % (trainer.target,))


def check_action_space_is_data() -> str:
    sources = [os.path.join(_HERE, name) for name in
               ("features.py", "genome.py", "evolution.py", "reporting.py")]
    shipped = [os.path.splitext(name)[0] for name in os.listdir(_SPEC_DIR)
               if name.endswith(".json")]
    for path in sources:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        for spec_id in shipped:
            if spec_id in text:
                raise CheckFailure("action_space_is_data",
                                   "%s hardcodes spec id %r" % (os.path.basename(path), spec_id))
    return "features/genome/evolution/reporting mention none of the %d shipped spec ids" % len(shipped)


def check_fixed_alpha_shares_both_parents_end_to_end() -> str:
    spec = load_spec(_spec_path("spec_minimal"))
    # mutation is switched OFF here so the child genome IS the noiseless blend and
    # the convex-hull assertion below is literal; the randomized hull + noise
    # assertion lives in check_weight_mixing_hull.
    config = EvoConfig(population_size=4, generations=1, seed=3,
                       elite_frac=0.25, parent_frac=1.0, reproduction_fee=1.0,
                       mutation_rate=0.0, mutation_sigma=0.0)
    trainer = EvolutionTrainer(spec, config, spec_id=spec.spec_id)
    population = trainer._founders()
    population[0].budget = 2.0
    population[0].fitness = 1.0
    population[1].budget = 2.0
    population[1].fitness = 0.9
    population[2].budget = 0.0
    population[2].fitness = 0.0
    population[3].budget = 0.0
    population[3].fitness = -0.1
    parent_a, parent_b = population[0], population[1]
    births = trainer._reproduce(population, 1)
    sexual = [b for b in births if b["origin"] == "offspring"]
    if not sexual:
        raise CheckFailure("fixed_alpha_shares_both", "no offspring was produced")
    birth = sexual[0]
    if birth["parents"] != [parent_a.aid, parent_b.aid]:
        raise CheckFailure("fixed_alpha_shares_both", "child's parents are not the two parents")
    if not birth["hull_ok"]:
        raise CheckFailure("fixed_alpha_shares_both", "child is not in the parents' hull")
    if abs(parent_a.budget - 1.5) > 1e-12 or abs(parent_b.budget - 1.5) > 1e-12:
        raise CheckFailure("fixed_alpha_shares_both", "reproduction fee was not split")
    child = next(a for a in trainer._next_population if a.aid == birth["child"])
    strict = 0
    for gene_a, gene_b, gene_c in zip(parent_a.genome, parent_b.genome, child.genome):
        low, high = min(gene_a, gene_b), max(gene_a, gene_b)
        if not (low - 1e-9 <= gene_c <= high + 1e-9):
            raise CheckFailure("fixed_alpha_shares_both",
                               "child gene %.9f outside [%.9f, %.9f]" % (gene_c, low, high))
        if low + 1e-9 < gene_c < high - 1e-9:
            strict += 1
    if strict == 0:
        raise CheckFailure("fixed_alpha_shares_both",
                           "child inherited no strict convex combination of both parents")
    if population[2].alive or population[3].alive:
        raise CheckFailure("fixed_alpha_shares_both", "broke agents did not die childless")
    return ("child %s gene vector is in the weight hull of %s and %s (%d genes strictly "
            "between both parents); fee split 0.5/0.5; two broke agents died childless"
            % (birth["child"], parent_a.aid, parent_b.aid, strict))


def check_step_spending() -> str:
    spec = load_spec(_spec_path("spec_dynamic_group"))
    config = EvoConfig(generations=1, seed=11, step_cost=0.05, max_episode_steps=1)
    trainer = EvolutionTrainer(spec, config, spec_id=spec.spec_id)
    # One unproductive combine action: exactly one step cost, nothing else.
    useless = trainer._founders()[0]
    useless.scripted_keys = ("combine:t_sub:n4",)
    episode = trainer.run_episode(useless, spec, trainer.root_mask, 0, 1)
    if episode.steps != 1 or abs(episode.total_return + config.step_cost) > 1e-9:
        raise CheckFailure("step_spending",
                           "useless step return %.9f != -step_cost %.9f"
                           % (episode.total_return, -config.step_cost))
    # Every extra step is charged: two useless steps cost 2 * step_cost.
    config2 = EvoConfig(generations=1, seed=11, step_cost=0.05, max_episode_steps=2)
    trainer2 = EvolutionTrainer(spec, config2, spec_id=spec.spec_id)
    two_steps = trainer2._founders()[0]
    two_steps.scripted_keys = ("combine:t_sub:n4", "combine:t_sub:n4")
    episode2 = trainer2.run_episode(two_steps, spec, trainer2.root_mask, 0, 1)
    if episode2.steps != 2 or abs(episode2.total_return + 2 * config2.step_cost) > 1e-9:
        raise CheckFailure("step_spending",
                           "two useless steps returned %.9f, expected %.9f"
                           % (episode2.total_return, -2 * config2.step_cost))
    # The solving 2-step word pays two step costs, the guard bonus, the objective
    # flip reward and the final goal reward.
    config3 = EvoConfig(generations=1, seed=11, step_cost=0.05)
    trainer3 = EvolutionTrainer(spec, config3, spec_id=spec.spec_id)
    solver = trainer3._founders()[0]
    solver.scripted_keys = ("combine:t_sub:n4+n1", "set:o")
    episode3 = trainer3.run_episode(solver, spec, trainer3.root_mask, 0, 1)
    expected = (-2 * config3.step_cost + config3.guard_bonus
                + config3.intermediate_reward + config3.goal_reward)
    if not episode3.solved or abs(episode3.total_return - expected) > 1e-9:
        raise CheckFailure("step_spending",
                           "solving return %.9f != expected %.9f" % (episode3.total_return, expected))
    return ("step cost charged per step: 2 useless steps = %.3f; solving word = "
            "goal(%.2f) + flip(%.2f) + guard(%.2f) - 2*step(%.2f) = %.4f"
            % (-2 * config2.step_cost, config3.goal_reward, config3.intermediate_reward,
               config3.guard_bonus, config3.step_cost, expected))


def check_subagent_recursion_depth_limited() -> str:
    spec = _adhoc_three_objective_spec()
    config = EvoConfig(generations=1, seed=4, subagent_depth=2,
                       subagent_spawn_fee=0.0, subagent_credit_cap=0.5,
                       max_subagents_per_episode=2, max_episode_steps=6)
    trainer = EvolutionTrainer(spec, config, spec_id=spec.spec_id)
    parent = trainer._founders()[0]
    parent.budget = 10.0
    parent.scripted_keys = ("set:o1", "set:o2", "set:o3")
    trainer.population = [parent]
    episode = trainer.run_episode(parent, spec, trainer.root_mask, 0, 1)

    depths: List[int] = []

    def collect(records: List[Dict]) -> None:
        for record in records:
            depths.append(record["depth"])
            collect(record.get("nested", []))

    collect(episode.subagents)
    if not episode.subagents:
        raise CheckFailure("subagent_recursion", "no subagent was spawned")
    if 1 not in depths:
        raise CheckFailure("subagent_recursion", "no depth-1 subagent: %s" % depths)
    if max(depths) > config.subagent_depth:
        raise CheckFailure("subagent_recursion", "depth %d exceeds limit %d"
                           % (max(depths), config.subagent_depth))
    if parent.subagents_spawned < 1:
        raise CheckFailure("subagent_recursion", "parent spawned no subagent")
    credits = sum(record["credit"] for record in episode.subagents)
    if parent.budget < 10.0 - 1e-9:
        raise CheckFailure("subagent_recursion", "subagent credits did not flow back")
    if any(record["credit"] > config.subagent_credit_cap + 1e-12 for record in episode.subagents):
        raise CheckFailure("subagent_recursion", "credit cap was exceeded")
    return ("depth-limited recursion: depths=%s (limit=%d), credits flowed back to the "
            "lineage (spawner budget %.4f), cap=%.2f"
            % (sorted(set(depths)), config.subagent_depth, parent.budget,
               config.subagent_credit_cap))


def check_reproduction_and_death() -> str:
    spec = load_spec(_spec_path("spec_dynamic_group"))
    config = EvoConfig(population_size=16, generations=14, seed=7,
                       reproduction_fee=1.0, mutation_rate=0.2)
    trainer = EvolutionTrainer(spec, config, spec_id=spec.spec_id)
    history = trainer.run()
    best = max(record["best_fitness"] for record in history)
    solved = sum(record["solved"] for record in history)
    births = sum(len(record["births"]) for record in history)
    deaths = sum(len(record["deaths"]) for record in history)
    if solved == 0:
        raise CheckFailure("reproduction_and_death",
                           "no agent solved spec_dynamic_group in %d generations" % config.generations)
    if births == 0:
        raise CheckFailure("reproduction_and_death", "no offspring was ever produced")
    if deaths == 0:
        raise CheckFailure("reproduction_and_death", "no agent ever died")
    return ("spec_dynamic_group: best_fitness=%.4f, solved=%d, births=%d, deaths=%d"
            % (best, solved, births, deaths))


def check_minimal_spec_solves() -> str:
    spec = load_spec(_spec_path("spec_minimal"))
    config = EvoConfig(population_size=8, generations=3, seed=2)
    trainer = EvolutionTrainer(spec, config, spec_id=spec.spec_id)
    history = trainer.run()
    best = max(record["best_fitness"] for record in history)
    if best < 1.0:
        raise CheckFailure("minimal_spec_solves", "best fitness %.4f < goal" % best)
    return "spec_minimal solved on generation 1 (best_fitness=%.4f)" % best


def check_deterministic_run() -> str:
    spec = load_spec(_spec_path("spec_dynamic_group"))
    config = EvoConfig(population_size=8, generations=3, seed=99,
                       checkpoint_dir=os.path.join(tempfile.gettempdir(), "evo-det-a"),
                       history_dir=os.path.join(tempfile.gettempdir(), "evo-det-b"))
    config_b = EvoConfig(**config.to_dict())
    config_b.checkpoint_dir = os.path.join(tempfile.gettempdir(), "evo-det-c")
    config_b.history_dir = os.path.join(tempfile.gettempdir(), "evo-det-d")
    trainer_a = EvolutionTrainer(spec, config, spec_id=spec.spec_id)
    trainer_b = EvolutionTrainer(spec, config_b, spec_id=spec.spec_id)
    history_a = trainer_a.run()
    history_b = trainer_b.run()
    if json.dumps(history_a, sort_keys=True) != json.dumps(history_b, sort_keys=True):
        raise CheckFailure("deterministic_run", "same seed produced different histories")
    return "two runs with seed 99 produced byte-identical generation histories"


def check_persistence_layout() -> str:
    root = tempfile.mkdtemp(prefix="evo-persist-")
    try:
        spec = load_spec(_spec_path("spec_minimal"))
        config = EvoConfig(population_size=6, generations=3, seed=5,
                           checkpoint_dir=os.path.join(root, "checkpoints"),
                           history_dir=os.path.join(root, "history"))
        trainer = EvolutionTrainer(spec, config, spec_id=spec.spec_id)
        history = trainer.run()
        for name in ("best_gen_0001.json", "best_gen_0002.json", "best_gen_0003.json",
                     "best_agents_persist.json", "checkpoint.json"):
            path = os.path.join(config.checkpoint_dir, name)
            if not os.path.exists(path):
                raise CheckFailure("persistence_layout", "missing %s" % path)
        with open(os.path.join(config.checkpoint_dir, "best_agents_persist.json"),
                  encoding="utf-8") as handle:
            persist = json.load(handle)
        if len(persist["agents"]) != config.generations:
            raise CheckFailure("persistence_layout", "persist has %d agents, expected %d"
                               % (len(persist["agents"]), config.generations))
        if persist["all_time_best"] is None or "genome" not in persist["all_time_best"]["agent"]:
            raise CheckFailure("persistence_layout", "all_time_best genome missing")
        with open(os.path.join(config.history_dir, "history.csv"), encoding="utf-8") as handle:
            agent_rows = [line for line in handle.read().splitlines() if line]
        expected_rows = sum(len(record["agents"]) for record in history)
        if len(agent_rows) != expected_rows + 1:
            raise CheckFailure("persistence_layout", "history.csv has %d rows, expected %d"
                               % (len(agent_rows), expected_rows + 1))
        with open(os.path.join(config.history_dir, "reward_trajectory.csv"),
                  encoding="utf-8") as handle:
            trajectory_rows = [line for line in handle.read().splitlines() if line]
        if len(trajectory_rows) != config.generations + 1:
            raise CheckFailure("persistence_layout", "trajectory has %d rows, expected %d"
                               % (len(trajectory_rows), config.generations + 1))
        return ("best_gen_*.json + best_agents_persist.json (%d winners) + history.csv "
                "(%d rows) + reward_trajectory.csv (%d rows)"
                % (len(persist["agents"]), len(agent_rows), len(trajectory_rows)))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def check_reward_hierarchy() -> str:
    config = EvoConfig()
    if not (config.goal_reward > config.subgoal_reward > 0.0):
        raise CheckFailure("reward_hierarchy", "goal_reward must dominate subgoal_reward")
    if not (config.goal_reward > config.intermediate_reward):
        raise CheckFailure("reward_hierarchy", "goal_reward must dominate intermediate_reward")
    if config.step_cost <= 0.0:
        raise CheckFailure("reward_hierarchy", "step_cost must be positive")
    return ("goal_reward %.2f > subgoal_reward %.2f and intermediate_reward %.2f; "
            "step_cost %.2f > 0" % (config.goal_reward, config.subgoal_reward,
                                    config.intermediate_reward, config.step_cost))


CHECKS: Tuple[Tuple[str, Callable[[], str]], ...] = (
    ("weight_mixing_hull", check_weight_mixing_hull),
    ("mutation_is_gaussian_delta", check_mutation_is_gaussian_delta),
    ("network_forward", check_network_forward),
    ("target_conditioning", check_target_conditioning),
    ("action_space_is_data", check_action_space_is_data),
    ("fixed_alpha_shares_both", check_fixed_alpha_shares_both_parents_end_to_end),
    ("step_spending", check_step_spending),
    ("subagent_recursion", check_subagent_recursion_depth_limited),
    ("minimal_spec_solves", check_minimal_spec_solves),
    ("reproduction_and_death", check_reproduction_and_death),
    ("deterministic_run", check_deterministic_run),
    ("persistence_layout", check_persistence_layout),
    ("reward_hierarchy", check_reward_hierarchy),
)


def run() -> int:
    started = time.perf_counter()
    passed = 0
    failures: List[str] = []
    for name, fn in CHECKS:
        try:
            detail = fn()
        except CheckFailure as exc:
            failures.append(exc.name)
            print("[FAIL] %s: %s" % (name, exc.detail))
        except Exception as exc:  # noqa: BLE001 - surface as a failure
            failures.append(name)
            print("[FAIL] %s: unexpected %r" % (name, exc))
        else:
            passed += 1
            print("[PASS] %s: %s" % (name, detail))
    elapsed = time.perf_counter() - started
    print("RESULT: %d/%d passed in %.3fs (seeded, stdlib only, env=data)"
          % (passed, len(CHECKS), elapsed))
    return 0 if not failures else 1


# pytest entry points -------------------------------------------------------
def test_weight_mixing_hull() -> None:
    check_weight_mixing_hull()


def test_mutation_is_gaussian_delta() -> None:
    check_mutation_is_gaussian_delta()


def test_network_forward() -> None:
    check_network_forward()


def test_target_conditioning() -> None:
    check_target_conditioning()


def test_action_space_is_data() -> None:
    check_action_space_is_data()


def test_fixed_alpha_shares_both() -> None:
    check_fixed_alpha_shares_both_parents_end_to_end()


def test_step_spending() -> None:
    check_step_spending()


def test_subagent_recursion() -> None:
    check_subagent_recursion_depth_limited()


def test_minimal_spec_solves() -> None:
    check_minimal_spec_solves()


def test_reproduction_and_death() -> None:
    check_reproduction_and_death()


def test_deterministic_run() -> None:
    check_deterministic_run()


def test_persistence_layout() -> None:
    check_persistence_layout()


def test_reward_hierarchy() -> None:
    check_reward_hierarchy()


if __name__ == "__main__":
    sys.exit(run())
