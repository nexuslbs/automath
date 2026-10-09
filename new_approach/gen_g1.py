"""G1 DIVERSE-STATE training driver (Unit D).

Retrains the trained Phase-A agent on a much denser + more diverse
initial-state set (all prefixes + structured perturbations + ~30 random
intermediate stacks, fixed RNG 20261009), then evaluates the trained agent on
the core 33, the extended 114 and the NEW unseen set.

    /opt/automath/venv/bin/python -m new_approach.gen_g1 --episodes 200
    /opt/automath/venv/bin/python -m new_approach.gen_g1 --baseline   # anchor
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import gen_common as gc
from .evolution_population import EvoConfig


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Unit D G1 dense-state training")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--config", default="/opt/automath/repo/config/evolution_semi.json")
    p.add_argument("--checkpoint-dir", default=gc.SEED_CHECKPOINT_DIR)
    p.add_argument("--out", default=os.path.join(gc.GEN_DIR, "g1_results.json"))
    p.add_argument("--baseline", action="store_true",
                   help="evaluate the unchanged trained agent (anchor)")
    p.add_argument("--dense-batch", dest="dense_batch", type=int, default=48,
                   help="dense-set cases per training episode (host bound)")
    args = p.parse_args(argv)

    cfg = EvoConfig.from_dict(json.load(open(args.config)))
    genome, gen = gc.load_seed_genome(args.checkpoint_dir)
    print("SEED_GENOME gid=%s source_generation=%d fitness=%.6f solved=%d"
          % (genome.gid, gen, genome.fitness, genome.solved_feasible))
    gc.export_seed_genome(os.path.join(gc.GEN_DIR, "seed_genome.json"),
                          args.checkpoint_dir)
    res = gc.evaluate_sets(genome, cfg, args.episodes,
                           dense=not args.baseline, anneal=False,
                           label="baseline-anchor" if args.baseline else "G1",
                           dense_batch=args.dense_batch)
    res["source_generation"] = gen
    gc.report(res)
    gc.write_result(res, args.out)
    print("WROTE %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
