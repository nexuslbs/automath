"""G3 HELD-OUT EVALUATION ONLY (Unit E finalization).

Loads a G3 checkpoint best genome (default: the gen3 novelty run) and evaluates
it on the SAME three held-out sets used by G1/G2/anchor: core33, ext114 and the
new unseen set (40). No training, no mutation: pure greedy rollout, seconds.

    /opt/automath/venv/bin/python -m new_approach.gen_g3_eval \
        --checkpoint-dir /opt/automath/tmp/gen3/checkpoints \
        --out /opt/automath/tmp/gen3/g3_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys

from . import gen_common as gc
from .evolution_population import EvoConfig


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="G3 held-out evaluation only")
    p.add_argument("--checkpoint-dir", default="/opt/automath/tmp/gen3/checkpoints")
    p.add_argument("--config",
                   default="/opt/automath/repo/config/evolution_gen3.json")
    p.add_argument("--out", default="/opt/automath/tmp/gen3/g3_eval.json")
    args = p.parse_args(argv)

    cfg = EvoConfig.from_dict(json.load(open(args.config)))
    genome, gen = gc.load_seed_genome(args.checkpoint_dir)
    print("G3_EVAL genome gid=%s source_generation=%d fitness=%.6f solved=%d "
          "train_cases=0 (evaluation only)"
          % (genome.gid, gen, genome.fitness, genome.solved_feasible))

    core_agent = gc._make_case_agent(genome, cfg, gc.CORE_ORDER, gc.CORE_ARITY,
                                     cfg.seed, 1)
    evo_agent = gc._make_case_agent(genome, cfg, gc.EVO_ORDER, gc.EVO_ARITY,
                                    cfg.seed, 1)
    unseen, forms = gc.unseen_cases()
    core_unseen = [c for c in unseen if getattr(c, "domain", "evo") == "core"]
    evo_unseen = [c for c in unseen if getattr(c, "domain", "evo") == "evo"]
    cu = gc.rollout_eval(core_agent, core_unseen)
    eu = gc.rollout_eval(evo_agent, evo_unseen)

    res = {
        "label": "G3-heldout-eval-only",
        "approach": "G3-novelty",
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
    gc.report(res)
    gc.write_result(res, args.out)
    print("WROTE %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
