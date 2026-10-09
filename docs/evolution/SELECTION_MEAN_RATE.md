# Selection by mean solve rate (task 4344)

This branch ports the three fixes from the 4339 size-invariant report onto the
size-invariant `evolution_trainer` pipeline. The 1153-gene genome, the curriculum
stages and the evolution backbone are unchanged.

## Fix 1 - MEAN-SOLVE-RATE selection and checkpoint metric

Selection and checkpointing rank a candidate by its solve rate over
`N = 30` FRESH validation episodes per candidate per generation, not by a single
best training episode.

* `SELECTION_EPISODES = 30`, `SELECTION_SEED = 20261010`,
  `SELECTION_EPSILON = 0.1`.
* Episode `i` of generation `g` has the EXPLICIT seed
  `episode_seed = selection_seed*1000000 + g*1000 + i` (e.g. `20261010000000`,
  `20261010003007`) and its own `random.Random(seed)`.
* Recorded per candidate: `sel_solved_rate` (mean solve rate), `sel_mean_steps`,
  `sel_std_steps`, the exact `sel_episode_seeds`, per-episode solved steps and
  traces, and the resulting `fitness`.
* Deterministic tie-break (higher is better): highest mean solve rate, then
  LOWER mean solved steps, then LOWER step std, then higher fitness, then higher
  budget, then agent id (`EvolutionTrainer._rank_key`).

### BEFORE vs AFTER formulas

```
BEFORE  fitness = shaped_return + solved_rate_weight * solved_rate
                  # one best training episode on the fixed 7-state bundle
AFTER   fitness = shaped_return + solved_rate_weight * mean_solved_rate_30
                  + val_weight * val_solved_rate
```

`shaped_return` is the best training-episode total return (`fitness_base`).
Defaults (`selection_episodes=0`, `val_weight=0.0`, `mask_illegal=False`) leave
Unit B/D behaviour byte-identical. Constants used by the launcher:
`solved_rate_weight = 0.5`, `val_weight = 1.0`.

## Fix 2 - Held-out validation term

Ported from the gen-fitness lineage, branch `gen-fitness` @ `1b8d24e`, unit
`add_a` (single owner, attribution, NOT a competing variant):

* `POOL_SIZE = 64`, `val_batch = 16`, `VAL_SEED = 20261010`, `val_weight = 1.0`.
* The per-generation batch is `random.Random(val_seed + 1000003*gen).sample(...)`
  so the same `(val_seed, gen)` yields the same batch in the same order; the val
  term is reproducible per `(val_seed, gen)`.

### Pool construction (add_a next-fix #2)

The 64 cases are built ONLY from SOLVABLE prefix-family forms: proper prefixes of
the canonical build words plus fresh random intermediate stacks; every case's
start state is proven solvable by replaying its witness. The gen-fitness failure
mode was a pool that shared unseen40's unsolvable-perturbation property, so
`val_solved_rate` was 0 forever; this pool has no such forms.

`docs` note: the in-domain mapping of the forbidden sources is
`bundle/core33/ext114/dense training` = the shipped training specs' canonical
paths and `unseen40` = the shipped held-out specs' canonical paths
(`TRAIN_SPEC_IDS`, `HELDOUT_SPEC_IDS`, `FORBIDDEN_CATEGORIES`). The pool reads
those forms only to EXCLUDE them. Raw report (`disjointness_report`):
`pool=64, distinct=64, overlap_bundle=0, overlap_unseen40=0, overlap_core33=0,
overlap_ext114=0, overlap_dense_train=0`, `assert_disjoint` OK. Because a proper
canonical prefix IS a forbidden canonical-path form, the disjointness filter
removes all prefix-family cases, leaving 64 fresh random intermediate stacks.

## Fix 3 - Argmax-collapse counter

See `ARGMAX_MASKING.md`.

## Behaviour when the new machinery is off

`--selection-episodes 0`, `--val-weight 0.0` and no `--mask-illegal` reproduce the
size-invariant pipeline exactly; the new CSV columns are additive.
