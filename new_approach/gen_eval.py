"""Flavor A held-out evaluation ONLY.

Loads the best genome from a flavor-A checkpoint (flat ``PolicyNet`` weights,
NO training, NO mutation) and greedily evaluates it on the SAME three held-out
sets used by G1/G2/G3/anchor:

    core33  = evolution_run.core_validation_cases()   (u2 validation)
    ext114  = evolution_run.evo_validation_cases()    (scenario prefixes)
    unseen40 = gen_common.unseen_cases()              (22 evo + 18 core)

    /opt/automath/venv/bin/python -m new_approach.gen_eval \
        --checkpoint-dir /opt/automath/tmp/target-policy-A/checkpoints \
        --config config/evolution_targetA.json \
        --out /opt/automath/tmp/target-policy-A/eval_A.json
"""

from __future__ import annotations

import argparse
import json
import sys

from . import gen_common as gc
from .evolution_agents import CORE_ARITY, CORE_ORDER, EVO_ARITY, EVO_ORDER
from .evolution_population import EvoConfig


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Flavor A held-out evaluation only")
    p.add_argument("--checkpoint-dir",
                   default="/opt/automath/tmp/target-policy-A/checkpoints")
    p.add_argument("--config",
                   default="/opt/automath/repo/config/evolution_targetA.json")
    p.add_argument("--out",
                   default="/opt/automath/tmp/target-policy-A/eval_A.json")
    p.add_argument("--label", default="A-target-conditioned-net")
    args = p.parse_args(argv)

    cfg = EvoConfig.from_dict(json.load(open(args.config)))
    genome, gen = gc.load_seed_genome(args.checkpoint_dir)
    print("A_EVAL genome gid=%s source_generation=%d fitness=%.6f solved=%d "
          "train_cases=0 (evaluation only)"
          % (genome.gid, gen, genome.fitness, genome.solved_feasible))

    core_agent = gc._make_case_agent(genome, cfg, CORE_ORDER, CORE_ARITY,
                                     cfg.seed, 1)
    evo_agent = gc._make_case_agent(genome, cfg, EVO_ORDER, EVO_ARITY,
                                    cfg.seed, 1)
    unseen, forms = gc.unseen_cases()
    core_unseen = [c for c in unseen if getattr(c, "domain", "evo") == "core"]
    evo_unseen = [c for c in unseen if getattr(c, "domain", "evo") == "evo"]
    cu = gc.rollout_eval(core_agent, core_unseen)
    eu = gc.rollout_eval(evo_agent, evo_unseen)

    res = {
        "label": args.label,
        "approach": "A-target-conditioned-net (evaluation only)",
        "episodes": 0,
        "seed": cfg.seed,
        "train_cases_core": 0,
        "train_cases_evo": 0,
        "train_cases_total": 0,
        "train_wall_secs": 0.0,
        "seed_gid": genome.gid,
        "source_generation": gen,
        "core33": gc.rollout_eval(core_agent, gc.core_validation_cases()),
        "ext114": gc.rollout_eval(evo_agent, gc.evo_validation_cases()),
        "unseen": {
            "solved": cu["solved"] + eu["solved"],
            "total": cu["total"] + eu["total"],
            "mean_actions_solved": gc._mean(
                ([cu["mean_actions_solved"]] if cu["mean_actions_solved"] else [])
                + ([eu["mean_actions_solved"]] if eu["mean_actions_solved"] else [])),
            "mean_reward": round(
                (cu["mean_reward"] * cu["total"] + eu["mean_reward"] * eu["total"])
                / max(1, cu["total"] + eu["total"]), 4),
            "wall_secs": round(cu["wall_secs"] + eu["wall_secs"], 3),
            "core_half": "%d/%d" % (cu["solved"], cu["total"]),
            "evo_half": "%d/%d" % (eu["solved"], eu["total"]),
        },
        "unseen_forms": forms,
    }
    # Report the three sets with the metric names the comparison table uses.
    report_result(res)
    gc.write_result(res, args.out)
    print("WROTE %s" % args.out)
    return 0


def report_result(res: dict) -> None:
    print("APPROACH=%s seed=%d genome=%s generation=%s"
          % (res["approach"], res["seed"], res["seed_gid"],
             res["source_generation"]))
    for key in ("core33", "ext114", "unseen"):
        r = res[key]
        print("SET %-7s solved=%d/%d mean_actions_solved=%s mean_reward=%s "
              "wall=%.3fs%s"
              % (key, r["solved"], r["total"], r["mean_actions_solved"],
                 r["mean_reward"], r["wall_secs"],
                 (" (core=%s evo=%s)" % (r.get("core_half"), r.get("evo_half")))
                 if key == "unseen" else ""))
    print("UNSEEN_FORMS %d cases (canonical, reproducible)"
          % len(res["unseen_forms"]))


if __name__ == "__main__":
    sys.exit(main())
