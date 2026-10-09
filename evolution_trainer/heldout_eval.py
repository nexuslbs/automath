"""Unit 3 held-out / generalization EVALUATOR for the size-invariant genome.

This is a NEW evaluation tool (it changes no trainer, curriculum, feature or
backbone file). It loads a trained size-invariant genome JSON
(``best_gen_*.json`` / ``best_agents_persist.json``), instantiates
:class:`evolution_trainer.size_invariant.SizeInvariantNet` and a ``dynamic_env``
spec, and runs episodes with greedy best-action scoring. Because the
size-invariant genome has ONE constant length (1153 genes at hidden=12), the
SAME genome can be scored on every spec without any width adaptation - which is
exactly what the arity-2 -> arity-3 transfer evaluates.

Per-case result files are written IMMEDIATELY after each case finishes, so the
evidence survives even if the run is cut short.

Solver definition
-----------------
An episode is *solved* when the environment goal is reached within the spec's
BFS-derived max step budget. ``solved_at_min`` is the stricter event "goal
reached in EXACTLY the BFS-minimal number of steps" (multi_step min 2,
dynamic_axiom min 3, spec_dynamic_group_deep min 3, per curriculum_tests). Both
are reported; the headline table uses solved (goal reached) and shows at-min
counts alongside.

Episode RNG is seeded with the held-out seed 12345 (the ancestor Unit D-2
protocol) for comparability. Every case runs:

* ``greedy``       - 30 deterministic argmax episodes (best-action scoring);
* ``stochastic``   - 30 softmax episodes, seed 12345 (the ancestor protocol);
* ``random``       - 30 uniform-random legal-action episodes (fair baseline).

CLI::

    python -m evolution_trainer.heldout_eval \
        --seed-name 7 \
        --multi-step-genome out/seed_7/stage4/multi_step/checkpoints/best_gen_0045.json \
        --axiom-genome      out/seed_7/stage4/dynamic_axiom/checkpoints/best_gen_0001.json \
        --group-genome      out/seed_7/curriculum/s3_free/checkpoints/best_gen_0001.json \
        --out-dir /opt/workspace/tmp/automath/size-invariant/unit-3/raw
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dynamic_env.engine import DynamicEnv
from dynamic_env.spec import Spec, load_spec, spec_from_dict

from .curriculum import bfs_min_steps, minimal_and_max, spec_path
from .heldout import genome_from_checkpoint
from .size_invariant import SizeInvariantNet, spec_independent_size

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_HELDOUT_DIR = os.path.join(_REPO_ROOT, "data", "dynamic_env", "heldout")

HIDDEN = 12
#: Reward shaping constants, identical to evolution_trainer.harness / EvoConfig.
STEP_COST = 0.05
INTERMEDIATE_REWARD = 0.30
GUARD_BONUS = 0.20
SUBGOAL_REWARD = 0.50
GOAL_REWARD = 1.00

#: The ancestor Unit D-2 held-out protocol seed.
HELDOUT_SEED = 12345
#: Number of perturbed target variants for the stronger sweep.
PERTURB_COUNT = 30


def load_spec_any(spec_id: str) -> Spec:
    """Load a shipped spec by id, falling back to ``data/dynamic_env/heldout``."""
    path = spec_path(spec_id)
    if not os.path.isfile(path):
        path = os.path.join(_HELDOUT_DIR, spec_id + ".json")
    return load_spec(path)


def _subgoal_reached(flags: Sequence[int], target: Sequence[int],
                     mask: Sequence[int]) -> bool:
    return all(flags[i] == target[i] for i in range(len(target)) if mask[i])


def _rollout(net: SizeInvariantNet, genome: Sequence[float], spec: Spec,
             mode: str, rng: random.Random, min_steps: Optional[int],
             max_steps: int, collect: bool = False) -> Dict[str, Any]:
    """One episode. ``mode`` in {greedy, stochastic, random}."""
    started = time.perf_counter()
    env = DynamicEnv(spec)
    state = env.reset()
    obj_ids = sorted(spec.objectives)
    target = tuple(int(spec.objectives[oid]) for oid in obj_ids)
    mask = tuple(1 for _ in obj_ids)
    actions = env.legal_actions(state)
    prev_flags = env.objective_vector(state)
    set_legal = frozenset(a.objective_id for a in actions if a.kind == "set")
    guarded = frozenset(spec.guards)
    credited: set = set()
    unlocked: set = set()
    total = 0.0
    steps = 0
    trace: List[Dict[str, Any]] = []
    actions_taken: List[str] = []
    rewards: List[float] = []

    if _subgoal_reached(prev_flags, target, mask):
        solved = bool(env.goal_reached(state))
        return {"steps": 0, "solved": solved,
                "solved_at_min": bool(solved and min_steps == 0),
                "return": 0.0, "actions": [], "rewards": [],
                "objective_vector": list(prev_flags), "trace": [],
                "wall_s": round(time.perf_counter() - started, 6)}

    while steps < max_steps and actions:
        if mode == "random":
            chosen = rng.randrange(len(actions))
        else:
            logits = net.logits(genome, env, state, actions, mask)
            if mode == "greedy":
                chosen = max(range(len(actions)), key=lambda i: (logits[i], -i))
            else:
                peak = max(logits)
                weights = [math.exp(value - peak) for value in logits]
                total_w = sum(weights)
                draw = rng.random() * total_w
                acc = 0.0
                chosen = len(actions) - 1
                for i, weight in enumerate(weights):
                    acc += weight
                    if draw <= acc:
                        chosen = i
                        break
        action = actions[chosen]
        result = env.step(state, action)
        flags = env.objective_vector(result.state)
        reward = -STEP_COST
        for i in range(len(target)):
            if (mask[i] and i not in credited and prev_flags[i] != target[i]
                    and flags[i] == target[i]):
                reward += INTERMEDIATE_REWARD
                credited.add(i)
        next_actions = env.legal_actions(result.state)
        next_set_legal = frozenset(
            a.objective_id for a in next_actions if a.kind == "set")
        new_unlocks = ((next_set_legal - set_legal) & guarded) - unlocked
        if new_unlocks:
            reward += GUARD_BONUS * len(new_unlocks)
            unlocked |= new_unlocks
        done_full = bool(result.done)
        done_sub = _subgoal_reached(flags, target, mask)
        if done_full:
            reward += GOAL_REWARD
        elif done_sub:
            reward += SUBGOAL_REWARD
        total += reward
        steps += 1
        actions_taken.append(action.key())
        rewards.append(round(reward, 6))
        if collect:
            trace.append({"step": steps, "action": action.key(),
                          "objective_vector": list(flags),
                          "reward": round(reward, 6), "goal": done_full})
        prev_flags = flags
        state = result.state
        actions = next_actions
        set_legal = next_set_legal
        if done_full or done_sub:
            break

    solved = bool(env.goal_reached(state))
    return {"steps": steps, "solved": solved,
            "solved_at_min": bool(solved and min_steps is not None
                                  and steps == min_steps),
            "return": round(total, 6),
            "actions": actions_taken, "rewards": rewards,
            "objective_vector": list(env.objective_vector(state)),
            "trace": trace,
            "wall_s": round(time.perf_counter() - started, 6)}


def _episode_summary(result: Dict[str, Any], index: int) -> Dict[str, Any]:
    return {"episode": index, "steps": result["steps"],
            "solved": result["solved"], "solved_at_min": result["solved_at_min"],
            "return": result["return"], "actions": result["actions"],
            "rewards": result["rewards"], "wall_s": result["wall_s"]}


def _aggregate(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    solved = sum(1 for e in episodes if e["solved"])
    at_min = sum(1 for e in episodes if e["solved_at_min"])
    solved_steps = [e["steps"] for e in episodes if e["solved"]]
    returns = [e["return"] for e in episodes]
    return {
        "episodes": len(episodes),
        "solved": solved,
        "solved_at_min": at_min,
        "solve_rate": round(solved / float(max(1, len(episodes))), 4),
        "mean_steps_solved": (round(sum(solved_steps) / len(solved_steps), 4)
                              if solved_steps else None),
        "mean_return": round(sum(returns) / float(max(1, len(returns))), 6),
    }


def evaluate_case(net: SizeInvariantNet, genome: Sequence[float], spec: Spec,
                  case_id: str, label: str, episodes: int, seed: int,
                  with_random: bool = True) -> Dict[str, Any]:
    plan = minimal_and_max(spec)
    min_steps = plan["min_steps"]
    max_steps = plan["max_steps"]
    started = time.perf_counter()

    greedy_eps: List[Dict[str, Any]] = []
    greedy_trace: List[Dict[str, Any]] = []
    for i in range(episodes):
        res = _rollout(net, genome, spec, "greedy", random.Random(seed + i),
                       min_steps, max_steps, collect=(i == 0))
        greedy_eps.append(_episode_summary(res, i))
        if i == 0:
            greedy_trace = res["trace"]

    rng = random.Random(seed)
    stochastic_eps = []
    for i in range(episodes):
        res = _rollout(net, genome, spec, "stochastic", rng, min_steps, max_steps)
        stochastic_eps.append(_episode_summary(res, i))

    random_eps: List[Dict[str, Any]] = []
    if with_random:
        rng2 = random.Random(seed)
        for i in range(episodes):
            res = _rollout(net, genome, spec, "random", rng2, min_steps, max_steps)
            random_eps.append(_episode_summary(res, i))

    return {
        "case_id": case_id, "genome_label": label,
        "spec_id": spec.spec_id, "min_steps": min_steps, "max_steps": max_steps,
        "max_rule": plan["rule"], "bfs_witness": plan["witness"],
        "genome_genes": len(genome), "net_size": spec_independent_size(HIDDEN),
        "heldout_seed": seed, "episodes": episodes,
        "greedy": _aggregate(greedy_eps),
        "greedy_episodes": greedy_eps,
        "greedy_trace": greedy_trace,
        "stochastic": _aggregate(stochastic_eps),
        "stochastic_episodes": stochastic_eps,
        "random": _aggregate(random_eps) if with_random else None,
        "random_episodes": random_eps,
        "wall_s": round(time.perf_counter() - started, 6),
    }


# --------------------------------------------------------------------------
# Stronger target perturbation sweep (30 distinct perturbed multi_step specs)
# --------------------------------------------------------------------------

_NAMESPACES = (
    ("n9", "n5", "n3"),
    ("a9", "a5", "a3"),
    ("m9", "m5", "m3"),
    ("p9", "p5", "p3"),
)
_VALUE_POOL = ((9, 5, 3), (7, 2, 1), (8, 4, 2), (6, 5, 1),
               (9, 4, 1), (8, 3, 2), (7, 4, 3), (9, 6, 2))


def _bounded_min_steps_2(spec: Spec) -> Optional[int]:
    """Min steps but bounded to depth 2 (skips the unbounded BFS explosion).

    A perturbed spec whose guard target is unreachable would make the full
    ``bfs_min_steps`` explore the whole (combinatorially huge) reachable graph up
    to ``max_steps``. Every perturbation generated here is reachable in two
    steps by construction (build a combination node, then set the objective), so
    a depth-2 check is exact and cheap.
    """
    env = DynamicEnv(spec)
    start = env.reset()
    if env.goal_reached(start):
        return 0
    for first in env.legal_actions(start):
        mid = env.step(start, first).state
        if env.goal_reached(mid):
            return 1
        for second in env.legal_actions(mid):
            if env.step(mid, second).done:
                return 2
    return None


def build_perturbations(count: int = PERTURB_COUNT,
                        seed: int = HELDOUT_SEED) -> List[Dict[str, Any]]:
    """Build ``count`` distinct perturbed ``spec_multi_step`` variants.

    Each variant changes the TARGET PROTOTYPE: the guard value the objective
    requires (the ancestor's "target 8" change generalised to every achievable
    value) AND the numeric node ids/values (``n9,n5,n3`` renamed and re-valued).
    All variants keep the same two-step structure (build a combination node,
    then set the objective), so each has a BFS minimum of 2.
    """
    with open(spec_path("spec_multi_step"), "r", encoding="utf-8") as handle:
        base_raw = json.load(handle)
    rng = random.Random(seed)
    variants: List[Dict[str, Any]] = []
    seen: set = set()
    guard = 0
    while len(variants) < count and guard < count * 200:
        guard += 1
        ns = _NAMESPACES[len(variants) % len(_NAMESPACES)]
        values = list(rng.choice(_VALUE_POOL))
        rng.shuffle(values)
        achievable = set()
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                a, b = values[i], values[j]
                achievable.add(a + b)
                # the engine's ``sub`` primitive clamps at 0 (engine._apply_op),
                # so only non-negative differences are reachable.
                achievable.add(max(a - b, 0))
                achievable.add(a * b)
        choices = sorted(v for v in achievable if v > 0)
        value = rng.choice(choices)
        key = (ns, tuple(values), value)
        if key in seen:
            continue
        seen.add(key)

        raw = copy.deepcopy(base_raw)
        old_ids = ("n9", "n5", "n3")
        new_nodes: Dict[str, Any] = {}
        for old, new, num in zip(old_ids, ns, values):
            node = raw["nodes"].pop(old)
            node["n"] = num
            new_nodes[new] = node
        for node_id, node in raw["nodes"].items():
            new_nodes[node_id] = node
        raw["nodes"] = new_nodes
        raw["guards"]["o"]["value"] = value
        raw["spec_id"] = "spec_multi_step_perturb_%02d" % len(variants)
        raw["description"] = ("perturbed target value=%d, num ids=%s values=%s"
                              % (value, ns, values))
        wide = spec_from_dict(raw)
        minimum = _bounded_min_steps_2(wide)
        if minimum is None:
            continue
        raw["max_steps"] = minimum + 2
        spec = spec_from_dict(raw)
        variants.append({
            "index": len(variants), "variant_id": raw["spec_id"],
            "spec": spec, "guard_value": value, "node_ids": list(ns),
            "node_values": list(values), "min_steps": minimum,
            "max_steps": minimum + 2,
        })
    return variants


def evaluate_perturbations(net: SizeInvariantNet, genome: Sequence[float],
                           episodes: int, seed: int,
                           with_random: bool = True) -> Dict[str, Any]:
    """30 perturbed-target episodes (greedy), plus a random baseline."""
    started = time.perf_counter()
    variants = build_perturbations(PERTURB_COUNT, seed)
    rng = random.Random(seed)
    rng_random = random.Random(seed)
    detail: List[Dict[str, Any]] = []
    greedy_solved = greedy_at_min = 0
    random_solved = random_at_min = 0
    for var in variants:
        spec = var["spec"]
        res = _rollout(net, genome, spec, "greedy", random.Random(seed + var["index"]),
                       var["min_steps"], var["max_steps"], collect=True)
        greedy_solved += int(res["solved"])
        greedy_at_min += int(res["solved_at_min"])
        entry = {
            "episode": var["index"], "variant_id": var["variant_id"],
            "guard_value": var["guard_value"], "node_ids": var["node_ids"],
            "node_values": var["node_values"],
            "min_steps": var["min_steps"], "max_steps": var["max_steps"],
            "steps": res["steps"], "solved": res["solved"],
            "solved_at_min": res["solved_at_min"], "return": res["return"],
            "actions": res["actions"], "rewards": res["rewards"],
            "wall_s": res["wall_s"],
        }
        if with_random:
            rres = _rollout(net, genome, spec, "random", rng_random,
                            var["min_steps"], var["max_steps"])
            random_solved += int(rres["solved"])
            random_at_min += int(rres["solved_at_min"])
            entry["random"] = {"steps": rres["steps"], "solved": rres["solved"],
                               "solved_at_min": rres["solved_at_min"],
                               "return": rres["return"]}
        detail.append(entry)
    total = len(detail)
    return {
        "case_id": "perturb_multi_step",
        "spec_id": "spec_multi_step (30 perturbed target prototypes)",
        "min_steps": 2, "max_steps": None, "episodes": total,
        "heldout_seed": seed,
        "genome_genes": len(genome), "net_size": spec_independent_size(HIDDEN),
        "greedy": {"episodes": total, "solved": greedy_solved,
                   "solved_at_min": greedy_at_min,
                   "solve_rate": round(greedy_solved / float(max(1, total)), 4)},
        "random": ({"episodes": total, "solved": random_solved,
                    "solved_at_min": random_at_min,
                    "solve_rate": round(random_solved / float(max(1, total)), 4)}
                   if with_random else None),
        "greedy_episodes": detail,
        "wall_s": round(time.perf_counter() - started, 6),
    }


def _write(out_dir: str, seed_name: str, case_id: str, report: Dict[str, Any]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "seed_%s__%s.json" % (seed_name, case_id))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Unit 3 size-invariant eval")
    parser.add_argument("--seed-name", required=True)
    parser.add_argument("--multi-step-genome", required=True)
    parser.add_argument("--axiom-genome", default=None)
    parser.add_argument("--group-genome", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--heldout-seed", type=int, default=HELDOUT_SEED)
    parser.add_argument("--no-random", action="store_true")
    args = parser.parse_args(argv)

    net = SizeInvariantNet(HIDDEN)
    assert net.size == spec_independent_size(HIDDEN)

    multi = genome_from_checkpoint(args.multi_step_genome)
    axiom = (genome_from_checkpoint(args.axiom_genome)
             if args.axiom_genome else multi)
    group = (genome_from_checkpoint(args.group_genome)
             if args.group_genome else multi)
    for name, genome in (("multi_step", multi), ("axiom", axiom), ("group", group)):
        if len(genome) != net.size:
            raise SystemExit("genome %s has %d genes, expected %d"
                             % (name, len(genome), net.size))

    meta = {"seed_name": args.seed_name, "net_size": net.size,
            "genome_paths": {
                "spec_multi_step": args.multi_step_genome,
                "spec_dynamic_axiom": args.axiom_genome,
                "spec_dynamic_group": args.group_genome},
            "episodes": args.episodes, "heldout_seed": args.heldout_seed,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    plans = [
        ("spec_multi_step", multi, "spec_multi_step best checkpoint"),
        ("spec_dynamic_axiom", axiom, "spec_dynamic_axiom best checkpoint"),
        ("spec_multi_step_heldout", multi, "multi_step genome, target 8 held-out"),
        ("spec_dynamic_group_deep", group, "group arity-2 genome, arity-3 held-out"),
    ]
    for case_id, genome, label in plans:
        spec = load_spec_any(case_id)
        report = evaluate_case(net, genome, spec, case_id, label,
                               args.episodes, args.heldout_seed,
                               with_random=not args.no_random)
        report["meta"] = meta
        path = _write(args.out_dir, args.seed_name, case_id, report)
        print("[eval] seed=%s case=%-24s greedy=%d/%d@min=%d stochastic=%d/%d "
              "random=%s wall=%.4fs -> %s"
              % (args.seed_name, case_id,
                 report["greedy"]["solved"], report["greedy"]["episodes"],
                 report["greedy"]["solved_at_min"],
                 report["stochastic"]["solved"], report["stochastic"]["episodes"],
                 (report["random"]["solved"] if report["random"] else "-"),
                 report["wall_s"], path))

    perturb = evaluate_perturbations(net, multi, args.episodes, args.heldout_seed,
                                     with_random=not args.no_random)
    perturb["meta"] = meta
    path = _write(args.out_dir, args.seed_name, "perturb_multi_step", perturb)
    print("[eval] seed=%s case=%-24s greedy=%d/%d@min=%d random=%s wall=%.4fs -> %s"
          % (args.seed_name, "perturb_multi_step",
             perturb["greedy"]["solved"], perturb["greedy"]["episodes"],
             perturb["greedy"]["solved_at_min"],
             (perturb["random"]["solved"] if perturb["random"] else "-"),
             perturb["wall_s"], path))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
