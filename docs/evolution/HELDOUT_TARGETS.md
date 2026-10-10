# Held-out TARGETS: the target-change validation pool (task 4354, unit 1)

This branch (`heldout-targets`, off `selection-mean-rate` @ `195e96d`) changes
the fix#2 validation POOL CONTENT only. The fitness formula and its constants
are untouched:

```
fitness = shaped_return
        + solved_rate_weight * mean_solved_rate_30
        + val_weight * val_solved_rate          # val_weight = 1.0
```

`POOL_SIZE = 64`, `val_batch = 16`, `VAL_SEED = 20261010`, selection `N = 30`,
`epsilon = 0.1` and fix#1 mean-solve-rate selection are all unchanged. Unit 1
does NO training; it builds and proves the pool the training unit will consume.

## Why a target-change family

The original pool (`selection-mean-rate`) varied only the START STATE of a fixed
set of specs. The val term therefore measured start-state generalization only:
a candidate could score `val_solved_rate = 1.0` while ignoring the TARGET, which
is exactly the failure the held-out `spec_multi_step_heldout` (target 8 instead
of the training target 4) is meant to catch. This unit adds a distribution of
held-out TARGETS to the pool so the val term exerts pressure on TARGET
generalization.

`TARGET_VARIANT_COUNT = 16` of the 64 cases are target-change cases; the other
48 are the unchanged fresh random intermediate stacks. Because the batch is a
uniform 16-of-64 sample, every generation's val term can (and usually does)
contain target-change cases.

## Family construction (mirrors the `perturb_multi_step` sweep)

`build_perturbations` in `evolution_trainer/heldout_eval.py` is the reference
sweep. The pool reproduces its scheme with an explicit, pool-local seed scheme:

* `TARGET_VARIANT_SEED = 20261011` (independent of `VAL_SEED = 20261010`);
  `random.Random(TARGET_VARIANT_SEED)`.
* Each variant is a COPY of `data/dynamic_env/spec_multi_step.json` with the
  three numeric nodes (`n9, n5, n3`) renamed into a FRESH namespace
  (`a9,a5,a3` / `m9,m5,m3` / `p9,p5,p3`) and re-valued from the proven
  `_TARGET_VALUE_POOL` triples. The guard value `guards.o.value` (the TARGET) is
  set to an achievable value from `sum` / clamped `sub` / `mul` over those
  values. The variant gets a fresh `spec_id = spec_multi_step_target_NN`.
* The FIRST case is the fixed anchor
  `TARGET_VARIANT_ANCHOR = {ids: (a9,a5,a3), values: (9,5,3), target: 8}` -
  the `spec_multi_step_heldout` "target 8" motif with fresh node ids, so it is
  not a copy of the shipped held-out EVAL spec.
* A drawn variant is skipped when its TARGET is the training target
  (`TRAIN_TARGET_VALUES = (4,)`) or when it would reproduce the shipped held-out
  eval spec (base ids + eval target).
* Each case is PROVEN SOLVABLE by construction: `bfs_minimal_word(spec)` yields
  the BFS witness, then `_replay` (and the public `replay_witness`) STEP the
  engine's own `legal_actions`/`step` from the case start state and require
  `goal_reached`. A case that fails the replay is dropped. Every shipped case's
  witness is a 2-action word (`build:<axiom>:<ops>` then `set:o`), and every
  case's `max_steps` is `len(witness) + 2`.

### Case list (deterministic, `TARGET_VARIANT_SEED = 20261011`)

| # | spec_id | node ids -> values | TARGET | witness | steps |
| --- | --- | --- | --- | --- | --- |
| 00 | spec_multi_step_target_00 | a9=9, a5=5, a3=3 | 8 (anchor) | `build:build_add:a3+a5`, `set:o` | 2 |
| 01 | spec_multi_step_target_01 | m9=9, m5=2, m3=6 | 15 | `build:build_add:m3+m9`, `set:o` | 2 |
| 02 | spec_multi_step_target_02 | p9=4, p5=1, p3=9 | 8 | `build:build_sub:p3+p5`, `set:o` | 2 |
| 03 | spec_multi_step_target_03 | a9=6, a5=2, a3=9 | 15 | `build:build_add:a3+a9`, `set:o` | 2 |
| 04 | spec_multi_step_target_04 | m9=5, m5=6, m3=1 | 7 | `build:build_add:m3+m5`, `set:o` | 2 |
| 05 | spec_multi_step_target_05 | p9=5, p5=3, p3=9 | 12 | `build:build_add:p3+p5`, `set:o` | 2 |
| 06 | spec_multi_step_target_06 | a9=2, a5=4, a3=8 | 32 | `build:build_mul:a3+a5`, `set:o` | 2 |
| 07 | spec_multi_step_target_07 | m9=9, m5=6, m3=2 | 3 | `build:build_sub:m9+m5`, `set:o` | 2 |
| 08 | spec_multi_step_target_08 | p9=8, p5=2, p3=4 | 16 | `build:build_mul:p5+p9`, `set:o` | 2 |
| 09 | spec_multi_step_target_09 | a9=3, a5=8, a3=2 | 6 | `build:build_mul:a3+a9`, `set:o` | 2 |
| 10 | spec_multi_step_target_10 | m9=1, m5=6, m3=5 | 30 | `build:build_mul:m3+m5`, `set:o` | 2 |
| 11 | spec_multi_step_target_11 | p9=8, p5=4, p3=2 | 2 | `build:build_sub:p5+p3`, `set:o` | 2 |
| 12 | spec_multi_step_target_12 | a9=9, a5=2, a3=6 | 15 | `build:build_add:a3+a9`, `set:o` | 2 |
| 13 | spec_multi_step_target_13 | m9=9, m5=1, m3=4 | 36 | `build:build_mul:m3+m9`, `set:o` | 2 |
| 14 | spec_multi_step_target_14 | p9=5, p5=6, p3=1 | 1 | `build:build_sub:p5+p9`, `set:o` | 2 |
| 15 | spec_multi_step_target_15 | a9=2, a5=7, a3=1 | 1 | `build:build_sub:a9+a3`, `set:o` | 2 |

Distinct TARGETS in the pool: `1, 2, 3, 6, 7, 8, 12, 15, 16, 30, 32, 36`.
The training target `4` is absent. The anchor keeps the `8` (held-out eval)
motif with fresh ids; that is a target-value overlap with the eval suite, NOT a
state-form overlap (see below).

### Full pool composition

`POOL_SIZE = 64 = 16 target-change + 48 fresh random intermediate stacks`.
The 48 random cases are the first 48 draws of the pre-existing
`_candidate_stream` seeded by `VAL_SEED`; observed composition
`spec_multi_step_heldout: 19`, `spec_dynamic_group: 16`, `spec_multi_step: 12`,
`spec_dynamic_group_deep: 1`. The `prefix` family is empty (a proper canonical
prefix IS a forbidden canonical-path form, so the disjointness filter removes
all of them - unchanged from `selection-mean-rate`).

## Deterministic draw

The per-generation batch is unchanged:

```python
validation_pool()                                   # cached, deterministic
random.Random(val_seed + 1000003 * gen).sample(pool, val_batch)
```

Same `(val_seed, gen)` -> same batch, same order, including the target-change
cases. The pool is built once per process and cached.

## Disjointness proof

`disjointness_report()` reports the five forbidden categories for the whole
pool AND for the target-change family alone:

| key | meaning | value |
| --- | --- | --- |
| `overlap_bundle` / `target_variant_overlap_bundle` | training specs' canonical paths | 0 / 0 |
| `overlap_unseen40` / `target_variant_overlap_unseen40` | held-out eval specs' canonical paths | 0 / 0 |
| `overlap_core33` / `target_variant_overlap_core33` | core subset | 0 / 0 |
| `overlap_ext114` / `target_variant_overlap_ext114` | ext subset | 0 / 0 |
| `overlap_dense_train` / `target_variant_overlap_dense_train` | dense training | 0 / 0 |
| `target_variants.overlap_training_targets` | variant targets equal to 4 | 0 |
| `assert_disjoint_ok` | single evidence boolean | `True` |

`assert_disjoint(pool)` returns the marker `ASSERT_DISJOINT_OK` (and raises on
any collision). Target variants cannot collide with a forbidden canonical form
because the disjointness key is `spec_id | start_state.identity()` and every
variant has a fresh `spec_id` and fresh node ids; they additionally must not
reproduce the training TARGET. The shipped `spec_multi_step_heldout` is disjoint
from the family by spec id and by node ids (`n9,n5,n3`), so the eval suite is
not reused as validation.

## How the training unit reports per-generation val values

`evaluate_validation(net, genome, cfg, generation)` returns
`(val_solved_rate, val_total, solved_case_names)` for the deterministic batch.
The training unit must record, per generation and candidate:

* `val_solved_rate` and `val_total` (the fitness term uses `val_weight = 1.0`);
* the batch identity (`val_seed`, `gen`, the ordered case names);
* for the target-change cases in the batch, the case spec id and its TARGET
  value (`case_target_value`), so target generalization is visible separately
  from start-state generalization.

The evidence script prints `disjointness_report()` and the
`assert_disjoint(...) == "ASSERT_DISJOINT_OK"` marker together with the
per-case witness replays.
