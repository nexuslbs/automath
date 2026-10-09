# GEN_FITNESS - held-out validation fitness (task 4342, branch `gen-fitness`)

Scope: this branch adds ONE thing to the flavor-B curriculum loop: the
evolutionary selection objective now carries an out-of-distribution term
measured on a held-out, freshly generated validation pool, and the ported G1
dense-state single-case training path is wired into the loop cadence.  The
target-conditioned `PolicyNet` genome, the shaped reward, the curriculum
schedule, the evolution backbone, elitism and the plateau autostop are
unchanged.

## BEFORE (still true on `target-policy`, HEAD c6be8ff)

Selection fitness was evaluated ONLY on the fixed 7-state demo bundle
(5 feasible + 2 impossible), `evolution_semi.py`:

```
genome.fitness = result.shaped_return
                 + cfg.solved_rate_weight * result.solved_rate()
```

```
fitness_before = shaped_return + solved_rate_weight * solved_rate
```

Nothing measured or rewarded performance on held-out or newly generated
targets, so evolution optimized bundle-specialized behavior and the frozen
genome scored `unseen40 = 0/40`.

## AFTER (this branch)

```
fitness_after = shaped_return
                + solved_rate_weight * solved_rate
                + val_weight * val_solved_rate
```

where `val_solved_rate` is the genome's greedy solved rate on a per-generation
`VAL_BATCH` drawn deterministically from the held-out validation pool
(`new_approach/validation_set.py`).  With `val_weight = 0.0` the computed value
is byte-identical to BEFORE (the validation rollout is not even run).

Config keys (new: `config/evolution_genfitness.json`, a copy of the chosen
flavor-B config `evolution_targetB.json`):

| key | value | meaning |
| --- | --- | --- |
| `val_weight` | `1.0` | multiplier on the validation solved rate |
| `val_batch` | `16` | validation cases scored per genome per generation |
| `val_seed` | `20261010` | RNG seed of the per-generation batch draw |
| `dense_hook` | `true` | enable the G1 dense single-case training hook |
| `dense_batch` | `16` | dense training cases per genome per generation |
| `dense_episodes` | `1` | dense training episode passes per generation |
| `dense_seed` | `20261010` | RNG seed of the dense batch draw |

Everything else (target-conditioned `PolicyNet`, byte-identical reward,
curriculum schedule, evolution backbone, elitism, plateau autostop) is copied
unchanged from `evolution_targetB.json`.

## Validation pool - construction, sizes, disjointness

`new_approach/validation_set.py` builds the pool with the SAME generator family
as `gen_common.unseen_cases` (a proper prefix of each of the 21 targets'
canonical build word, then one structured perturbation) plus fresh random
intermediate stacks over the shared node pool, driven by a DIFFERENT RNG seed
`VAL_SEED = 20261010` (unseen40 uses `20261009`).

* `POOL_SIZE = 64` cases (mix of core/evo domains), built once and cached.
* A per-generation batch is `random.Random(val_seed + 1000003 * gen).sample(...)`
  of size `val_batch` (16), so the same `(val_seed, gen)` always yields the same
  batch in the same order.

Disjointness is guaranteed by construction (every candidate whose canonical
form `target_canonical + "|" + "[stack canons]"` collides with a forbidden form
is dropped) and re-asserted at build time by `assert_disjoint`:

| forbidden source | function | overlap required |
| --- | --- | --- |
| 7-state demo bundle | `evolution_bundle.make_demo_bundle(60).states` | 0 |
| unseen40 (the true TEST set) | `gen_common.unseen_cases()` | 0 |
| core33 | `evolution_run.core_validation_cases()` | 0 |
| ext114 | `evolution_run.evo_validation_cases()` | 0 |
| G1 dense TRAINING cases | `gen_common.dense_cases()` | 0 |

`disjointness_report()` prints the exact pool size and each overlap count for
evidence.  The true `unseen40` set is never read by fitness or selection: the
validation pool only shares its generator family, never a form.

## G1 dense training hook

`evolve_one_generation_shaped` now calls `evaluate_genome_shaped(..., gen=gen)`.
When `dense_hook` is on, before the greedy bundle evaluation each genome's
trained agent runs one bounded pass of `gen_common.train_shaped_bounded` over
`cfg.dense_batch` EVO dense cases (`gen_common.dense_train_batch`, seed
`dense_seed + 1000003*gen`), the same ported single-case path G1 used.  This is
the "training-distribution fix" half of the operator's instruction: the loop's
training now includes dense intermediates, not only bundle states.  The hook is
off by default, so flavors A/B/C are reproduced byte-identically.
