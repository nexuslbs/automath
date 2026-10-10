"""Standard HELD-OUT evaluation harness for a trained checkpoint (task 4349 unit 5).

This is an EVALUATION-ONLY module. It changes no trainer, curriculum, feature,
selection or backbone file. It loads a seed's selected genome from its stage-4
``best_agents_persist.json`` (``all_time_best``) and scores it on the held-out
families with a THREE-WAY decomposition and the 4344 mean-solve-rate protocol.

What it reports, per family per seed
------------------------------------
* ``SOLVABLE-SOLVED``      - the oracle proves the case solvable AND the trained
  genome reaches the goal in at least one of the N fresh episodes.
* ``SOLVABLE-UNSOLVED``    - the oracle proves the case solvable but the genome
  never reaches the goal in N fresh episodes (failure mode recorded).
* ``PROVABLY-UNSOLVABLE``  - the oracle PROVES no plan exists (proof method from
  ``evolution_trainer/solvability.py``).  These cases are reported separately,
  excluded from the solvable denominator and NEVER counted as failures.  The
  genome is never even attempted on them, so they cannot burn the run.

The mean solve rate reuses the 4344 protocol verbatim: ``N = 30`` fresh episodes
with an explicit per-episode seed, ``epsilon = 0.1`` and deterministic argmax
otherwise.  The episode seed is FRESH (``20261011`` by default) so the metric is
NOT the recorded ``val_solved_rate`` (whose protocol used ``20261010``).

Every case is bounded twice: the oracle search gets the (node, time) pair from
``graceful.case_budgets`` and the genome rollout gets a hard step budget.  A
global wall cap stops scheduling new cases when it is exceeded and reports the
remainder as ``skipped_budget``; a single case can never burn the whole run.

Output: one JSON file plus a paste-friendly table, deterministic and stdlib
only.

CLI::

    python -m evolution_trainer.heldout_harness \
        --seed-dir /opt/automath/tmp/unsolvable-unit4-pop/seed-7 \
        --families unseen40,val64,spec_multi_step,spec_multi_step_heldout \
        --action-set both --episodes 30 --out /tmp/heldout.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dynamic_env.spec import Spec

from . import solvability as oracle
from .curriculum import minimal_and_max
from .graceful import (
    DEFAULT_CASE_STEP_BUDGET,
    PROVABLY,
    SOLVED,
    SOLVED_UNSOLVED,
    case_budgets,
)
from .heldout import genome_from_checkpoint
from .size_invariant import SizeInvariantNet, spec_independent_size
from .size_selection import (
    SELECTION_EPSILON,
    SELECTION_EPISODES,
    SELECTION_SEED,
    spec_by_id,
    validation_pool,
)

HIDDEN = 12
#: Explicit epoch/seed for the fresh held-out episodes.  Deliberately different
#: from the recorded ``SELECTION_SEED`` (20261010) so the metric is new.
FRESH_EPISODE_SEED = 20261011
DEFAULT_MAX_WALL_SECS = 900.0

#: The in-domain "unseen40" for the size-invariant net is the shipped held-out
#: spec pool; the stack-domain ``unseen40`` family is still classified by the
#: oracle (three-way) but cannot be rolled out by a DynamicEnv policy.
_STACK_FAMILIES = frozenset({"unseen40", "core33", "ext114", "demo_bundle",
                             "dense", "perturb_multi_step"})

#: Shipped specs that ``solvability.FAMILIES`` does not expose directly.
_DIRECT_SPECS: Tuple[str, ...] = ("spec_minimal", "spec_dynamic_group")

DEFAULT_FAMILIES: Tuple[str, ...] = (
    "unseen40",
    "val64",
    "spec_minimal",
    "spec_multi_step",
    "spec_dynamic_axiom",
    "spec_dynamic_group",
    "spec_multi_step_heldout",
    "spec_dynamic_group_deep",
)


# --------------------------------------------------------------------------
# Family resolution
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SpecCase:
    """Adapter so the oracle can classify a bare ``dynamic_env`` ``Spec``.

    ``solvability._adapt`` reads ``case.spec``; a bare ``Spec`` has no such
    attribute, so an unwrapped spec would be reported ``UNKNOWN``.  The wrapper
    carries the spec plus an optional start state and keeps every other oracle
    contract (``witness`` / ``max_steps``) intact.
    """

    name: str
    spec: Spec
    start_state: Any = None
    witness: Any = None
    max_steps: int = 0


def _adapt_case(name: str, case: Any) -> Any:
    if not isinstance(case, Spec):
        return case
    max_steps = int(getattr(case, "max_steps", 0) or 0)
    witness: Optional[Tuple[str, ...]] = None
    try:
        plan = minimal_and_max(case)
        witness = tuple(plan.get("witness") or ()) or None
        max_steps = int(plan["max_steps"])
    except Exception:  # pragma: no cover - a spec with no reachable goal
        witness = None
    return SpecCase(name=name, spec=case, witness=witness,
                    max_steps=max_steps)


def family_cases(family: str) -> List[Tuple[str, Any]]:
    """Return ``[(case_name, case)]`` for a family name.

    ``solvability.FAMILIES`` supplies the stack-domain and one-spec families;
    ``spec_minimal`` / ``spec_dynamic_group`` are loaded directly.  A ``case`` is
    either a ``new_approach`` stack case (oracle-only) or a ``dynamic_env``
    ``Spec`` / ``SizeSelection.ValCase`` (oracle + genome rollout).
    """
    if family in _DIRECT_SPECS:
        raw = [(family, spec_by_id(family))]
    else:
        builder = oracle.FAMILIES.get(family)
        if builder is None:
            raise KeyError("unknown family %r (known: %s)" % (
                family,
                ",".join(sorted(set(oracle.FAMILIES) | set(_DIRECT_SPECS)))))
        raw = list(builder())
    return [(name, _adapt_case(name, case)) for name, case in raw]


def case_spec(case: Any) -> Optional[Spec]:
    """The ``dynamic_env`` spec of a case, or ``None`` for a stack case."""
    spec = getattr(case, "spec", None)
    if spec is not None:
        return spec
    if isinstance(case, Spec):
        return case
    return None


def bfs_min_for(case: Any) -> Optional[int]:
    spec = case_spec(case)
    if spec is None:
        return None
    try:
        return int(minimal_and_max(spec)["min_steps"])
    except Exception:  # pragma: no cover - defensive, an unknown spec
        return None


# --------------------------------------------------------------------------
# Genome loading (a seed's stage-4 all_time_best)
# --------------------------------------------------------------------------

def find_stage_checkpoints(seed_dir: str) -> List[Tuple[str, str]]:
    """``[(spec_id, persist_path)]`` for every stage-4 spec of a seed run."""
    out: List[Tuple[str, str]] = []
    for sub in sorted(glob.glob(os.path.join(seed_dir, "stage4", "*"))):
        path = os.path.join(sub, "checkpoints", "best_agents_persist.json")
        if os.path.isfile(path):
            out.append((os.path.basename(sub), path))
    return out


def load_selected_genome(seed_dir: str) -> Dict[str, Any]:
    """Load the ``all_time_best`` genome of the first stage-4 checkpoint found."""
    ckpts = find_stage_checkpoints(seed_dir)
    if not ckpts:
        legacy = os.path.join(seed_dir, "checkpoints", "best_agents_persist.json")
        if os.path.isfile(legacy):
            ckpts = [("legacy", legacy)]
    if not ckpts:
        raise FileNotFoundError(
            "no stage4/*/checkpoints/best_agents_persist.json under %s" % seed_dir)
    spec_id, path = ckpts[0]
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    genome = genome_from_checkpoint(path)
    best = raw.get("all_time_best") or {}
    return {
        "spec_id": spec_id,
        "checkpoint": path,
        "genome": genome,
        "genome_len": len(genome),
        "all_time_best_generation": best.get("generation"),
        "all_time_best_agent_id": (best.get("agent") or {}).get("agent_id"),
        "stage_checkpoints": [{"spec_id": s, "path": p} for s, p in ckpts],
    }


# --------------------------------------------------------------------------
# The 4344 mean-solve-rate protocol on ONE case
# --------------------------------------------------------------------------

def episode_seed_for(base_seed: int, index: int) -> int:
    """Explicit per-episode seed, distinct from the recorded selection seed."""
    return int(base_seed) * 1000000 + int(index)


def evaluate_case_episodes(net: SizeInvariantNet, genome: Sequence[float],
                           case: Any, action_set: str = "current",
                           episodes: int = SELECTION_EPISODES,
                           episode_seed: int = FRESH_EPISODE_SEED,
                           epsilon: float = SELECTION_EPSILON,
                           mask_illegal: bool = True,
                           step_budget: int = DEFAULT_CASE_STEP_BUDGET
                           ) -> Dict[str, Any]:
    """N fresh, independently seeded episodes of the genome on one case.

    Returns solve counts, mean solve rate, solved-at-BFS-min count, per-episode
    steps, the explicit episode seeds, the step budget and the wall time.
    """
    from .size_selection import rollout as _rollout

    spec = case_spec(case)
    if spec is None:
        return {"applicable": False, "reason": "stack_domain",
                "episodes": 0, "solved": 0, "mean_solve_rate": None,
                "solved_at_min": 0, "step_budget": None,
                "min_steps": None, "episode_seeds": [], "episode_steps": [],
                "wall_secs": 0.0}
    start = getattr(case, "start_state", None)
    allow_pop = action_set == "pop"
    case_max = getattr(case, "max_steps", None)
    cap = int(step_budget)
    if case_max is not None:
        cap = min(cap, int(case_max))
    if cap <= 0:
        cap = int(spec.max_steps)
    root_mask = tuple(1 for _ in spec.objectives)
    min_steps = bfs_min_for(case)
    seeds: List[int] = []
    steps: List[int] = []
    solved = at_min = 0
    t0 = time.perf_counter()
    for index in range(int(episodes)):
        seed = episode_seed_for(episode_seed, index)
        seeds.append(seed)
        rng = random.Random(seed)
        result = _rollout(net, genome, spec, start, root_mask, rng,
                          epsilon=epsilon, mask_illegal=mask_illegal,
                          max_steps=cap, allow_pop=allow_pop)
        if result["solved"]:
            solved += 1
            steps.append(int(result["steps"]))
            if min_steps is not None and int(result["steps"]) == int(min_steps):
                at_min += 1
    wall = round(time.perf_counter() - t0, 6)
    n = max(1, int(episodes))
    mean_steps = (round(sum(steps) / float(len(steps)), 6) if steps else None)
    return {
        "applicable": True,
        "episodes": int(episodes),
        "solved": solved,
        "mean_solve_rate": round(solved / float(n), 6),
        "solved_at_min": at_min,
        "min_steps": min_steps,
        "mean_solved_steps": mean_steps,
        "step_budget": cap,
        "episode_seeds": seeds,
        "episode_steps": steps,
        "wall_secs": wall,
    }


# --------------------------------------------------------------------------
# Three-way decomposition for one family + action set
# --------------------------------------------------------------------------

def _failure_mode(res: Dict[str, Any], oracle_verdict: str,
                  proof: Optional[str]) -> str:
    if oracle_verdict == oracle.UNKNOWN:
        return "budget_exhausted"
    if res.get("min_steps") is not None and res.get("solved"):
        return "step_limit"
    return proof or "no_plan"


def evaluate_family(net: SizeInvariantNet, genome: Sequence[float],
                    family: str, action_set: str = "current",
                    episodes: int = SELECTION_EPISODES,
                    episode_seed: int = FRESH_EPISODE_SEED,
                    epsilon: float = SELECTION_EPSILON,
                    mask_illegal: bool = True,
                    step_budget: int = DEFAULT_CASE_STEP_BUDGET,
                    max_wall_secs: float = DEFAULT_MAX_WALL_SECS,
                    deadline: Optional[float] = None
                    ) -> Dict[str, Any]:
    """Score one family in one action set with the three-way contract."""
    cases = family_cases(family)
    per_case: Dict[str, Dict[str, Any]] = {}
    solved = unsolved = provably = skipped = 0
    failures: Dict[str, int] = {}
    proofs: Dict[str, int] = {}
    episode_rates: List[float] = []
    total_wall = 0.0
    t_family = time.perf_counter()
    for name, case in cases:
        if deadline is not None and time.perf_counter() > deadline:
            skipped += 1
            per_case[name] = {"bucket": "SKIPPED-BUDGET", "family": family,
                              "action_set": action_set}
            continue
        spec = case_spec(case)
        if spec is not None:
            nb, tb = case_budgets(case, action_set)
        else:
            nb, tb = oracle._budget_for(case)
        t0 = time.perf_counter()
        gt = oracle.classify(case, action_set, nb, tb)
        proof = gt.get("proof_method")
        if gt["verdict"] == oracle.UNSOLVABLE:
            provably += 1
            proofs[proof] = proofs.get(proof, 0) + 1
            row = {
                "bucket": PROVABLY,
                "proof_method": proof,
                "min_steps": gt.get("min_steps"),
                "genome": {"applicable": spec is not None, "solved": 0,
                           "mean_solve_rate": None},
            }
        else:
            res = evaluate_case_episodes(
                net, genome, case, action_set, episodes=episodes,
                episode_seed=episode_seed, epsilon=epsilon,
                mask_illegal=mask_illegal, step_budget=step_budget)
            rate = res.get("mean_solve_rate")
            if rate is not None:
                episode_rates.append(float(rate))
            if res.get("solved", 0) > 0:
                bucket = SOLVED
                solved += 1
            else:
                bucket = SOLVED_UNSOLVED
                unsolved += 1
                mode = _failure_mode(res, gt["verdict"], proof)
                failures[mode] = failures.get(mode, 0) + 1
            row = {
                "bucket": bucket,
                "oracle_verdict": gt["verdict"],
                "oracle_proof": proof,
                "oracle_min_steps": gt.get("min_steps"),
                "genome": res,
            }
        wall = round(time.perf_counter() - t0, 6)
        total_wall += wall
        row.update({"family": family, "action_set": action_set,
                    "case": name, "wall_secs": wall,
                    "oracle_node_budget": nb, "oracle_time_budget": tb})
        per_case[name] = row
    n = len(cases)
    solvable = solved + unsolved
    mean_case_rate = (round(sum(episode_rates) / len(episode_rates), 6)
                      if episode_rates else None)
    return {
        "family": family,
        "action_set": action_set,
        "n": n,
        "SOLVABLE-SOLVED": solved,
        "SOLVABLE-UNSOLVED": unsolved,
        "PROVABLY-UNSOLVABLE": provably,
        "SKIPPED-BUDGET": skipped,
        "solvable_denominator": solvable,
        "solve_rate_over_solvable": (round(solved / float(solvable), 6)
                                     if solvable else None),
        "mean_case_solve_rate": mean_case_rate,
        "failure_modes": failures,
        "proof_methods": proofs,
        "total_wall_secs": round(total_wall, 6),
        "family_wall_secs": round(time.perf_counter() - t_family, 6),
        "per_case": per_case,
    }


# --------------------------------------------------------------------------
# val-term firing statistics from history.csv
# --------------------------------------------------------------------------

def parse_val_term(history_path: str) -> Dict[str, Any]:
    """How often the held-out ``val_solved_rate`` term fired in a history.csv.

    A "GEN row" is one generation; the generation's value is the best (max)
    ``val_solved_rate`` over its agents, matching the ``[GEN]`` log line.  The
    min/mean/max are reported over ALL per-agent rows as well as over the
    per-generation maxima.
    """
    rows: List[Dict[str, str]] = []
    with open(history_path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    if not rows:
        return {"history": history_path, "rows": 0, "generations": 0,
                "gen_rows_val_gt0": 0, "val_min": None, "val_mean": None,
                "val_max": None, "per_generation": []}
    by_gen: Dict[int, List[float]] = {}
    all_vals: List[float] = []
    for row in rows:
        try:
            gen = int(float(row.get("generation") or 0))
            val = float(row.get("val_solved_rate") or 0.0)
        except (TypeError, ValueError):
            continue
        all_vals.append(val)
        by_gen.setdefault(gen, []).append(val)
    series = [{"generation": g, "val_solved_rate": round(max(v), 6)}
              for g, v in sorted(by_gen.items())]
    gen_vals = [item["val_solved_rate"] for item in series]
    return {
        "history": history_path,
        "rows": len(rows),
        "generations": len(series),
        "gen_rows_val_gt0": sum(1 for v in gen_vals if v > 0.0),
        "val_min": (round(min(all_vals), 6) if all_vals else None),
        "val_mean": (round(sum(all_vals) / len(all_vals), 6)
                     if all_vals else None),
        "val_max": (round(max(all_vals), 6) if all_vals else None),
        "gen_val_min": (round(min(gen_vals), 6) if gen_vals else None),
        "gen_val_mean": (round(sum(gen_vals) / len(gen_vals), 6)
                         if gen_vals else None),
        "gen_val_max": (round(max(gen_vals), 6) if gen_vals else None),
        "per_generation": series,
    }


def selected_val_rate(seed_dir: str) -> Dict[str, Any]:
    """History-derived val stats plus the selected checkpoint's val rate."""
    out: Dict[str, Any] = {}
    for spec_id, _path in find_stage_checkpoints(seed_dir):
        hist = os.path.join(seed_dir, "stage4", spec_id, "history",
                            "history.csv")
        if os.path.isfile(hist):
            stats = parse_val_term(hist)
            stats["spec_id"] = spec_id
            out[spec_id] = stats
    return out


# --------------------------------------------------------------------------
# Top-level report
# --------------------------------------------------------------------------

def run(seed_dirs: Sequence[str],
        families: Sequence[str] = DEFAULT_FAMILIES,
        action_sets: Sequence[str] = ("current", "pop"),
        episodes: int = SELECTION_EPISODES,
        episode_seed: int = FRESH_EPISODE_SEED,
        epsilon: float = SELECTION_EPSILON,
        mask_illegal: bool = True,
        step_budget: int = DEFAULT_CASE_STEP_BUDGET,
        max_wall_secs: float = DEFAULT_MAX_WALL_SECS) -> Dict[str, Any]:
    net = SizeInvariantNet(HIDDEN)
    assert net.size == spec_independent_size(HIDDEN)
    deadline = time.perf_counter() + float(max_wall_secs)
    seed_reports: List[Dict[str, Any]] = []
    for seed_dir in seed_dirs:
        loaded = load_selected_genome(seed_dir)
        genome = loaded["genome"]
        val = selected_val_rate(seed_dir)
        selected_val = None
        if loaded["spec_id"] in val:
            selected_val = val[loaded["spec_id"]].get("gen_val_max")
        family_reports: Dict[str, Any] = {}
        for family in families:
            for action_set in action_sets:
                if time.perf_counter() > deadline:
                    break
                key = "%s|%s" % (family, action_set)
                family_reports[key] = evaluate_family(
                    net, genome, family, action_set, episodes=episodes,
                    episode_seed=episode_seed, epsilon=epsilon,
                    mask_illegal=mask_illegal, step_budget=step_budget,
                    deadline=deadline)
        seed_reports.append({
            "seed_dir": seed_dir,
            "spec_id": loaded["spec_id"],
            "checkpoint": loaded["checkpoint"],
            "genome_len": loaded["genome_len"],
            "all_time_best_generation": loaded["all_time_best_generation"],
            "all_time_best_agent_id": loaded["all_time_best_agent_id"],
            "stage_checkpoints": loaded["stage_checkpoints"],
            "selected_val_solved_rate": selected_val,
            "val_term": val,
            "families": family_reports,
        })
    return {
        "harness": "evolution_trainer.heldout_harness",
        "protocol": {
            "episodes": int(episodes),
            "episode_seed": int(episode_seed),
            "recorded_selection_seed": SELECTION_SEED,
            "epsilon": epsilon,
            "mask_illegal": bool(mask_illegal),
            "step_budget": int(step_budget),
            "max_wall_secs": float(max_wall_secs),
            "action_sets": list(action_sets),
            "families": list(families),
        },
        "seeds": seed_reports,
    }


def table_rows(payload: Dict[str, Any]) -> List[str]:
    """A paste-friendly fixed-width table."""
    header = ("%-6s %-26s %-8s %5s %7s %9s %10s %8s %9s %10s" % (
        "seed", "family", "actions", "n", "solved", "unsolved", "provably",
        "skipped", "rate", "mean_ep"))
    lines = [header, "-" * len(header)]
    for seed in payload["seeds"]:
        seed_name = os.path.basename(seed["seed_dir"]) or seed["seed_dir"]
        for key in sorted(seed["families"]):
            fam = seed["families"][key]
            lines.append("%-6s %-26s %-8s %5d %7d %9d %10d %8d %9s %10s" % (
                seed_name, fam["family"], fam["action_set"], fam["n"],
                fam["SOLVABLE-SOLVED"], fam["SOLVABLE-UNSOLVED"],
                fam["PROVABLY-UNSOLVABLE"], fam["SKIPPED-BUDGET"],
                str(fam["solve_rate_over_solvable"]),
                str(fam["mean_case_solve_rate"])))
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="standard held-out evaluator for a trained checkpoint")
    parser.add_argument("--seed-dir", action="append", required=True,
                        help="seed run dir (repeatable); stage4/*/checkpoints")
    parser.add_argument("--families", default=",".join(DEFAULT_FAMILIES),
                        help="comma list or 'default'")
    parser.add_argument("--action-set", default="both",
                        choices=("current", "pop", "both"))
    parser.add_argument("--episodes", type=int, default=SELECTION_EPISODES)
    parser.add_argument("--episode-seed", type=int,
                        default=FRESH_EPISODE_SEED)
    parser.add_argument("--epsilon", type=float, default=SELECTION_EPSILON)
    parser.add_argument("--step-budget", type=int,
                        default=DEFAULT_CASE_STEP_BUDGET)
    parser.add_argument("--max-wall-secs", type=float,
                        default=DEFAULT_MAX_WALL_SECS)
    parser.add_argument("--no-illegal-mask", action="store_true")
    parser.add_argument("--out", default="heldout_harness.json")
    args = parser.parse_args(argv)

    families = (list(DEFAULT_FAMILIES) if args.families == "default"
                else [f.strip() for f in args.families.split(",") if f.strip()])
    action_sets = (["current", "pop"] if args.action_set == "both"
                   else [args.action_set])
    payload = run(list(args.seed_dir), families, action_sets,
                  episodes=args.episodes, episode_seed=args.episode_seed,
                  epsilon=args.epsilon,
                  mask_illegal=not args.no_illegal_mask,
                  step_budget=args.step_budget,
                  max_wall_secs=args.max_wall_secs)
    text = json.dumps(payload, indent=2, sort_keys=True)
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    for line in table_rows(payload):
        print(line)
    print("WROTE %s" % out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
