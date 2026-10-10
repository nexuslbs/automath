"""Train the learned-prune value net and write the canned checkpoint.

Task 4354 unit 3A. Thin entry point over
``evolution_trainer.learned_prune.train_and_save``: build the deterministic
labeled dataset over the bundled specs + the held-out TARGET-variant pool, train
the ``d = 32`` value head with value-MSE (the llm-like value-head architecture,
see ``learned_prune`` module docstring), and write
``evolution_trainer/learned_prune_checkpoint.json``.

This is NOT the long evolution loop: the default budget is a few hundred
gradient steps on a bounded dataset (pure standard library, CPU, minutes). Part B
may retrain fresh with the same script and compare checkpoints.

Run:
    /opt/automath/venv/bin/python -m evolution_trainer.train_learned_prune
"""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Optional, Sequence

from . import learned_prune as lp


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the task-4354 unit-3A learned prune value net")
    parser.add_argument("--checkpoint", default=lp.CHECKPOINT_PATH)
    parser.add_argument("--hidden", type=int, default=lp.DEFAULT_WIDTH)
    parser.add_argument("--steps", type=int, default=lp.DEFAULT_STEPS)
    parser.add_argument("--batch", type=int, default=lp.DEFAULT_BATCH)
    parser.add_argument("--lr", type=float, default=lp.DEFAULT_LR)
    parser.add_argument("--per-spec", type=int,
                        default=lp.DEFAULT_STATES_PER_SPEC)
    parser.add_argument("--positive-weight", type=float,
                        default=lp.DEFAULT_POSITIVE_WEIGHT)
    parser.add_argument("--negatives-per-state", type=int,
                        default=lp.DEFAULT_NEGATIVES_PER_STATE,
                        help="0 keeps EVERY off-canonical action (no thinning)")
    parser.add_argument("--rank-weight", type=float,
                        default=lp.DEFAULT_RANK_WEIGHT,
                        help="per-state listwise ranking term (0 disables)")
    parser.add_argument("--seed", type=int, default=lp.TRAIN_SEED)
    args = parser.parse_args(argv)

    net, info = lp.train_and_save(
        checkpoint=args.checkpoint, hidden=args.hidden, steps=args.steps,
        batch=args.batch, lr=args.lr, per_spec=args.per_spec, seed=args.seed,
        positive_weight=args.positive_weight,
        negatives_per_state=args.negatives_per_state,
        rank_weight=args.rank_weight)
    with open(args.checkpoint, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    report = dict(info)
    report["checkpoint_sha256"] = digest
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
