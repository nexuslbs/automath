"""G2 REWARD-SHAPING CURRICULUM driver (Unit D).

Retrains the trained Phase-A agent with the Phase-A PBRS design but anneals the
shaping weight linearly from 1.0 (dense) to 0.0 (sparse = pure objective) over
the episode budget, then evaluates on the core 33, extended 114 and NEW unseen
sets.

    /opt/automath/venv/bin/python -m new_approach.gen_g2 --episodes 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import gen_common as gc
from .evolution_population import EvoConfig


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Unit D G2 annealed shaping")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--config", default="/opt/automath/repo/config/evolution_semi.json")
    p.add_argument("--checkpoint-dir", default=gc.SEED_CHECKPOINT_DIR)
    p.add_argument("--out", default=os.path.join(gc.GEN_DIR, "g2_results.json"))
    args = p.parse_args(argv)

    cfg = EvoConfig.from_dict(json.load(open(args.config)))
    genome, gen = gc.load_seed_genome(args.checkpoint_dir)
    print("SEED_GENOME gid=%s source_generation=%d fitness=%.6f solved=%d"
          % (genome.gid, gen, genome.fitness, genome.solved_feasible))
    print("SHAPING_SCHEDULE episodes=%d weight(ep)=max(0,1-ep/(episodes-1)); "
          "ep0=1.000 ep%d=0.000 (sparse/pure at the end)"
          % (args.episodes, args.episodes - 1))
    for ep in (0, args.episodes // 4, args.episodes // 2,
               3 * args.episodes // 4, args.episodes - 1):
        print("  ep=%4d weight=%.4f" % (ep, gc.shaping_weight(ep, args.episodes)))
    res = gc.evaluate_sets(genome, cfg, args.episodes, dense=False, anneal=True,
                           label="G2")
    res["source_generation"] = gen
    res["shaping_schedule"] = ("linear 1.0 -> 0.0 over %d episodes; "
                               "phi_scale and subgoal_bonus scaled by weight"
                               % args.episodes)
    gc.report(res)
    gc.write_result(res, args.out)
    print("WROTE %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
