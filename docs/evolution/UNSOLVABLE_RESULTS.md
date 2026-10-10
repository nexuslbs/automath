# Unsolvable handling: final results (task 4349, unit 10)

Raw-numbered final comparison and verdict for the "unsolvable cases" work.
Companion to `docs/evolution/UNSOLVABLE.md` (design + per-unit method); this
document is the end-to-end result set.

* Branch `unsolvable-handling`. Session `developer-default-20261010-060430-k92is`.
* Evidence root remote: `/opt/automath/evidence/unsolvable-handling/`
* Workstation mirror: `/opt/workspace/tmp/automath/unsolvable-handling/`
* Training run `train-4349-unit4-pop.sh` (PID 1200012) was never killed or
  restarted; the unit-8 watcher (PID 1210277) auto-ran each seed's held-out
  eval. All numbers below are read from the artifacts those processes wrote.
* Method facts are unchanged from the unit briefs: the automath host is reached
  only through the ssh plugin tools; no IP/hostname/credential is printed; the
  project venv is `/opt/automath/venv/bin/python`; the remote clone is
  `/opt/automath/tmp/unsolvable-u1`.

---

## 0. TL;DR verdict

The **explicit planner + `pop`** now covers the family that was previously
unsolvable: `unseen40` goes `0/40 -> 40/40` with **zero training** (min 2 /
median 4 steps), while `core33` (32/33) and `ext114` (114/114) are unchanged.
The **learned route** is partial and seed-dependent on the in-domain held-out
pool, and **cannot be evaluated on `unseen40` in this harness at all** (the
family resolves to the `new_approach` stack domain, so `case_spec(case) is
None` and `genome.applicable=False`; only the oracle/planner is rolled out
there). A generalizing optimal agent therefore exists **explicitly** (planner +
pop) but not yet **as a learned policy** on the previously-unsolvable family.

---

## 1. The solvability oracle and the full classification

Oracle: `evolution_trainer/solvability.py`, a decidable classifier that returns
`SOLVABLE` (with a witness / bounded best-first search), `UNSOLVABLE` (with a
proof method) or `UNKNOWN` (bounded budget exhausted - never asserted
unsolvable). Proof methods recorded: `provided_witness_replay`,
`bounded_best_first_search`, `exhaustive_bounded_search`,
`node_budget_exhausted+pop_all_then_canonical_witness_upper_bound`.

Full classification (unit-1 `tables.md`, confirmed on the real engine by unit-2):

| family | action set | n | solvable | unsolvable | unknown | min steps | median steps |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| demo_bundle | current | 7 | 5 | 2 | 0 | 2 | 5 |
| demo_bundle | pop | 7 | 5 | 2 | 0 | 2 | 5 |
| core33 | current | 33 | 31 | 0 | 2 | 0 | 1 |
| core33 | pop | 33 | 31 | 0 | 2 | 0 | 1 |
| ext114 | current | 114 | 114 | 0 | 0 | 0 | 5 |
| ext114 | pop | 114 | 114 | 0 | 0 | 0 | 5 |
| unseen40 | current | 40 | 0 | 36 | 4 | - | - |
| unseen40 | pop | 40 | 40 | 0 | 0 | 2 | 4 |
| val64 | current | 64 | 64 | 0 | 0 | 1 | 2 |
| val64 | pop | 64 | 64 | 0 | 0 | 1 | 2 |
| dense | current | 320 | 158 | 136 | 26 | 0 | 4 |
| dense | pop | 320 | 320 | 0 | 0 | 0 | 5 |
| spec_multi_step | current | 1 | 0 | 0 | 1 | - | - |
| spec_multi_step | pop | 1 | 0 | 0 | 1 | - | - |
| spec_dynamic_axiom | current | 1 | 0 | 0 | 1 | - | - |
| spec_dynamic_axiom | pop | 1 | 0 | 0 | 1 | - | - |
| spec_dynamic_group_deep | current | 1 | 0 | 0 | 1 | - | - |
| spec_dynamic_group_deep | pop | 1 | 0 | 0 | 1 | - | - |
| spec_multi_step_heldout | current | 1 | 0 | 0 | 1 | - | - |
| spec_multi_step_heldout | pop | 1 | 0 | 0 | 1 | - | - |
| perturb_multi_step | current | 30 | 0 | 0 | 30 | - | - |
| perturb_multi_step | pop | 30 | 0 | 0 | 30 | - | - |

Reading the headline families:

* `unseen40` current **0 solved / 36 unsolvable / 4 unknown** -> pop **40/40
  solvable / 0 unsolvable**.
* `dense` current **158 solved / 136 unsolvable / 26 unknown** -> pop
  **320 solvable / 0 unsolvable**.
* `val64` **64/64 solvable under both** action sets.
* `core33` **31 solvable / 0 unsolvable / 2 unknown**; `ext114`
  **114 solvable / 0 unsolvable / 0 unknown**.
* The shipped specs and `perturb_multi_step` are **UNKNOWN** (bounded budget),
  never asserted UNSOLVABLE - the honest bucket, not a failure.

The delta (unit-1 `delta.md`): **172 forms UNSOLVABLE under `current` become
SOLVABLE with `pop`** (36 in `unseen40`, 136 in `dense`) and **30 forms UNKNOWN
under `current` become SOLVABLE with `pop`** (26 in `dense`: the full tree
`dense_*`; 4 in `unseen40`). Total 202 forms move out of the
unsolvable/unknown buckets.

## 2. FIX 1 - the implemented opt-in `pop` action

File `dynamic_env/engine.py` (commit `1097288`). `DynamicEnv(spec,
allow_pop=False)`; with `allow_pop=True` the engine emits one extra
`Action(kind="pop")`, legal iff the work/partial-solution stack is non-empty.
`pop` removes the topmost **engine-constructed work node** and rolls `next_id`
back, so the work stack is a true stack and `[pop]*k` is a faithful undo.
`action_set_flag("current"|"pop")` / `env_for_action_set(...)` are the
selectors. The canonical subtree prune is **disabled when `pop` is on**
(`size_selection.canonical_subtree_prune` returns actions unchanged if
`env.allow_pop`), because the prune's "non-subtree => unsolvable" argument
assumed no action can remove a node.

Action-space delta (real engine, `unit-2/real_env_pop_report.json`; `pop` adds
0 actions on an empty work stack and exactly **+1 after a work node**):

| spec | actions current reset | actions pop reset | current after 1 work node | pop after 1 work node | min current | min pop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| spec_minimal | 1 | 1 | - | - | 1 | 1 |
| spec_multi_step | 18 | 18 | 36 | 37 | 2 | 2 |
| spec_dynamic_axiom | 1 | 1 | - | - | 3 | 3 |
| spec_dynamic_group | 4 | 4 | 9 | 10 | 2 | 2 |
| spec_multi_step_heldout | 18 | 18 | 37 | 38 | 2 | 2 |
| spec_dynamic_group_deep | 15 | 15 | 40 | 41 | 3 | 3 |

Real-engine newly-solvable counts (`real_env_pop_report.json`): `val64` 64
`dynamic_env` start states current 64/64 pop 64/64 newly 0, action-count sum
`3084 -> 3148`; the perturbed off-canonical form (k work nodes) is
`current=UNKNOWN / pop=SOLVABLE` for k=2,3,4 (pop min steps 2 / 3 / 6) and
solvable under both for k=1. The pop verdicts are reproduced by
`[pop]*len(work)+canonical word` replaying to the goal.

Delta attributable to FIX 1: **172 UNSOLVABLE -> SOLVABLE and 30 UNKNOWN ->
SOLVABLE** (section 1). The implemented action is byte-identical under
`allow_pop=False`: the 4-kind `KIND_ORDER`, action-vector width and the
size-invariant genome length (1153) are unchanged, and the `pop` feature vector
is all-zero (unique), so the network can score it distinctly.

## 3. FIX 2 - the solvable-only validation pool

`evolution_trainer/size_selection.py`. Constants `POOL_SIZE=64`,
`val_batch=16`, `VAL_SEED=20261010`, `val_weight=1.0`; fitness
`= shaped_return + solved_rate_weight*mean_solved_rate_30 + val_weight*val_solved_rate`.
Disjointness re-verified on the unit-3 tree (`unit-3/disjointness_report.log`):
`pool=64`, `pool_distinct_forms=64`, `by_domain` prefix 0 / random 64, and
`overlap_bundle=0`, `overlap_unseen40=0`, `overlap_core33=0`, `overlap_ext114=0`,
`overlap_dense_train=0` - **all-zero overlap**.

The val term **fires**: in the seed-7 `pop` run it fired in **43/43
generations** (`unit-6/heldout_seed7.json`, `val_term.spec_multi_step`:
`rows=516`, `generations=43`, `gen_rows_val_gt0=43`, `val_mean=0.52047`,
`val_max=1.0`, `gen_val_mean=0.742733`). Task 4342's gen-fitness pool scored
`0.000/16` on the same term. This is the regression the fix targets: a single
validation pool that actually resolves candidates.

## 4. FIX 3 - the explicit planner regression (kept and extended)

Explicit planner (`REDUCTION_TABLE` bounded best-first, pattern-goal fallback)
on `current` vs `pop` (`unit-3/planner_eval.json`, sha256
`c54792970bbd1411e4fb4eba0dacdfa3de5e2bb5581932d186cc651016f8a28c`):

| family | current | pop | pop min steps | pop median steps |
| --- | ---: | ---: | ---: | ---: |
| core33 | 32/33 | 32/33 | 1 | 2 |
| ext114 | 114/114 | 114/114 | 1 | 5 |
| unseen40 | 0/40 | 40/40 | 2 | 4 |

`core33` and `ext114` are **unchanged** by `pop`; `unseen40` goes `0/40` (4
unknown) to `40/40` with **zero training**. This is the regression that keeps
the explicit route honest and extends it.

## 5. GRACEFUL handling of unsolvable cases

`evolution_trainer/graceful.py` (commit `0dc889c`); CLI
`--graceful-unsolvable` defaults **ON** with a documented OFF switch. Training
classifies a pool ONCE with the oracle before attempting it, excludes
`UNSOLVABLE` cases and drops `UNKNOWN` cases after a hard per-case budget.
Evaluation decomposes every case three ways: `SOLVABLE-SOLVED` /
`SOLVABLE-UNSOLVED` (+failure mode) / `PROVABLY-UNSOLVABLE` (+proof method); a
provably-unsolvable case is **never** counted as a failure and never attempted.

Same dense stage, 320 cases, 3 generations, oracle budgets 20000 nodes / 1.5 s
(`unit-3/SUMMARY.json` -> `train_pool_dense`):

| flag | wall s | episodes attempted | skipped | solved | static excluded | unknown at entry | budget-dropped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OFF | 143.229 | 960 | 0 | 474 | 0 | 0 | 0 |
| ON | 91.619 | 503 | 457 | 474 | 135 | 27 | 26 |

Time saved `143.229 - 91.619 = 51.610 s` (**36.0%**). Dense static exclusion
`135 UNSOLVABLE + 27 UNKNOWN = 162/320 = 50.6%` (unit-1 recorded the same total
as `136 + 26`; the `UNSOLVABLE`/`UNKNOWN` split is budget-relative while the 162
total is stable). `val64 = 0/64` excluded. `unseen40` under `current =
36 UNSOLVABLE + 4 UNKNOWN` (all 40 unreachable), under `pop = 0/40` excluded.

Three-way evaluation (`current`), `unit-3/SUMMARY.json` -> `three_way`:

| family | n | SOLVABLE-SOLVED | SOLVABLE-UNSOLVED | PROVABLY-UNSOLVABLE | solve rate | failure modes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| core33 | 33 | 32 | 1 | 0 | 0.9697 | budget_exhausted 1 |
| ext114 | 114 | 114 | 0 | 0 | 1.0000 | - |
| unseen40 | 40 | 0 | 4 | 36 | 0.0000 | budget_exhausted 4 |
| val64 | 64 | 64 | 0 | 0 | 1.0000 | - |

Shipped specs (wrapped with their BFS witness): 4/4 `SOLVABLE-SOLVED`.

## 6. The three-seed learned-route results with `pop`

Protocol (identical for every seed; `evolution_trainer/heldout_harness.py`,
commit `45fdbac`): 30 fresh episodes per case, explicit episode seed
`20261011`, epsilon 0.1, step budget 40, `--action-set both`,
`--mask-illegal`, families `unseen40, val64` + the shipped specs. Each seed's
selected genome is the seed's stage-4 `all_time_best`; the numbers are the
harness's authoritative tables.

### 6.1 seed 7

```
seed   family                     actions      n  solved  unsolved   provably  skipped      rate    mean_ep
-----------------------------------------------------------------------------------------------------------
seed-7 spec_dynamic_axiom         current      1       1         0          0        0       1.0        1.0
seed-7 spec_dynamic_axiom         pop          1       1         0          0        0       1.0   0.133333
seed-7 spec_dynamic_group_deep    current      1       1         0          0        0       1.0        1.0
seed-7 spec_dynamic_group_deep    pop          1       1         0          0        0       1.0   0.033333
seed-7 spec_dynamic_group         current      1       1         0          0        0       1.0        1.0
seed-7 spec_dynamic_group         pop          1       1         0          0        0       1.0   0.066667
seed-7 spec_minimal               current      1       1         0          0        0       1.0        1.0
seed-7 spec_minimal               pop          1       1         0          0        0       1.0        1.0
seed-7 spec_multi_step_heldout    current      1       1         0          0        0       1.0        1.0
seed-7 spec_multi_step_heldout    pop          1       1         0          0        0       1.0   0.066667
seed-7 spec_multi_step            current      1       1         0          0        0       1.0        1.0
seed-7 spec_multi_step            pop          1       1         0          0        0       1.0   0.033333
seed-7 unseen40                   current     40       0         4         36        0       0.0       None
seed-7 unseen40                   pop         40       0        40          0        0       0.0       None
seed-7 val64                      current     64      49        15          0        0  0.765625   0.497917
seed-7 val64                      pop         64      64         0          0        0       1.0   0.621875
```

* **HEADLINE val64 pop-vs-current:** current `49/64` (coverage `0.765625`,
  mean per-case `0.497917`) -> pop `64/64` (coverage `1.0`, mean per-case
  `0.621875`). **+15 cases**.
* Selected checkpoint: `spec_id=spec_multi_step`, `genome_len=1153`,
  `all_time_best_generation=21`, `agent_id=None`, history-row `fitness=2.466667`
  (best-in-file was gen 25 at 2.483333); harness `selected_val_solved_rate=1.0`
  (the task-4344 `gen_val_max` field). Checkpoint
  `.../seed-7/stage4/spec_multi_step/checkpoints/best_agents_persist.json`.
* val-term firing: `rows=516`, `generations=43`, `gen_rows_val_gt0=43/43`,
  `val_min=0.0`, `val_mean=0.52047`, `val_max=1.0`, `gen_val_mean=0.742733`.
* Shipped specs: **6/6 solved 1/1 under BOTH action sets**; the target-change
  held-out spec `spec_multi_step_heldout` is **1/1** for seed 7.
* JSON sha256: `f2ed52ab37b8949520ac958ba4baf7ecd292f7cc7e1dce05e9020c05b756c1e7`
  (`/opt/automath/evidence/unsolvable-handling/unit-6/heldout_seed7.json`).

### 6.2 seed 13

```
seed   family                     actions      n  solved  unsolved   provably  skipped      rate    mean_ep
-----------------------------------------------------------------------------------------------------------
seed-13 spec_dynamic_axiom        current      1       1         0          0        0       1.0        1.0
seed-13 spec_dynamic_axiom        pop          1       1         0          0        0       1.0   0.866667
seed-13 spec_dynamic_group_deep   current      1       1         0          0        0       1.0        1.0
seed-13 spec_dynamic_group_deep   pop          1       0         1          0        0       0.0        0.0
seed-13 spec_dynamic_group        current      1       1         0          0        0       1.0        1.0
seed-13 spec_dynamic_group        pop          1       1         0          0        0       1.0   0.066667
seed-13 spec_minimal              current      1       1         0          0        0       1.0        1.0
seed-13 spec_minimal              pop          1       1         0          0        0       1.0        1.0
seed-13 spec_multi_step_heldout   current      1       1         0          0        0       1.0        1.0
seed-13 spec_multi_step_heldout   pop          1       1         0          0        0       1.0   0.066667
seed-13 spec_multi_step           current      1       1         0          0        0       1.0        1.0
seed-13 spec_multi_step           pop          1       0         1          0        0       0.0        0.0
seed-13 unseen40                  current     40       0         4         36        0       0.0       None
seed-13 unseen40                  pop         40       0        40          0        0       0.0       None
seed-13 val64                     current     64      46        18          0        0   0.71875   0.481771
seed-13 val64                     pop         64      46        18          0        0   0.71875   0.481771
```

* **HEADLINE val64 pop-vs-current:** current `46/64` (coverage `0.71875`, mean
  `0.481771`) **identical** to pop `46/64` / `0.481771`. **+0 cases.** The
  per-case comparison (`unit-9/seed13_val64_percase_compare.txt`) shows the 14
  spec-shaped cases have identical `(solved, episodes, mean)`: the only
  per-case differences are the `action_set` label and wall times
  (`ALL_14_IDENTICAL_ON_(solved,episodes,mean_solve_rate)=True`).
* Selected checkpoint: `spec_id=spec_multi_step`, `genome_len=1153`,
  `all_time_best_generation=24`, `agent_id=None`, history-row `fitness=0.879167`
  (best-in-file was gen 14 at 1.879167); `selected_val_solved_rate=0.875`.
* val-term firing: `rows=312`, `generations=26`, `gen_rows_val_gt0=26/26`,
  `val_min=0.0`, `val_mean=0.397837`, `val_max=0.875`, `gen_val_mean=0.632212`.
* Shipped specs: 5/6 solved under `pop` (`spec_multi_step` and
  `spec_dynamic_group_deep` regress to 0/1); all 6 solved under `current`.
* JSON sha256: `154450c5828f54122400308be6d76fc1de30689465b29c04f70e1552034c11b6`
  (`/opt/automath/evidence/unsolvable-handling/unit-8/heldout_seed13.json`).

### 6.3 seed 42

PENDING: raw value supplied by the follow-up commit of this same unit.

Seed 42 held-out eval launched by the unit-8 watcher at 2026-10-10T06:04:05Z (30 episodes, both action sets); its JSON was not yet written at first commit time. Values are added by the follow-up commit; the raw JSON is unit-8/heldout_seed42.json and the raw sha256 is in the DONE marker.

### 6.4 The `unseen40` finding (applies to every seed)

`unseen40` resolves through `oracle.FAMILIES` to the **stack-domain**
`new_approach` cases (`ncases=40, rollable=0`). `case_spec(case)` returns
`None`, so `genome.applicable=False` on every case and **the LEARNED policy is
never rolled out there** - only the oracle/planner is. Every seed shows the
same shape:

* `unseen40 | current`: 0 solved, 4 SOLVABLE-UNSOLVED (`budget_exhausted`),
  36 PROVABLY-UNSOLVABLE (`exhaustive_bounded_search`).
* `unseen40 | pop`: oracle SOLVABLE on all 40, but the genome is N/A, so all 40
  are counted SOLVABLE-UNSOLVED with **no policy evidence** (family wall is
  oracle classification only).

This is a structural harness limitation, not a learned-policy result; the
concrete next fix is in section 9.

## 7. HONEST verdict: is there a generalizing optimal agent per Axioms/Goal?

**Explicitly, yes - the planner + `pop` reaches the previously-unsolvable
family.** `unseen40 40/40` with zero training and `dense 320/320` solvable;
`core33` 32/33 and `ext114` 114/114 are unchanged (FIX 3 kept and extended).

**As a learned route, not yet.** On the in-domain held-out pool the gain is
partial and seed-dependent: seed 7 gains `+15` val64 cases (`49/64 -> 64/64`),
seed 13 gains `+0` (identical per-case results), and seed 42 is reported in
section 6.3. The learned policy **cannot be evaluated on `unseen40`** in this
harness because the family is stack-domain and the genome is not applicable
there. So: a generalizing optimal agent exists **explicitly (planner + pop)**;
it does **not** yet exist as a learned policy on the previously-unsolvable
family. No result below is masked or hand-patched.

## 8. Provably-unsolvable and solvable-but-unsolved

**PROVABLY-UNSOLVABLE (current action set), with proof method:**

| family | count | proof method |
| --- | ---: | --- |
| unseen40 | 36 | `exhaustive_bounded_search` |
| dense | 136 | `exhaustive_bounded_search` |
| core33 | 0 | - |
| ext114 | 0 | - |
| spec_* / perturb_multi_step | 0 (all UNKNOWN) | - |

With `pop`, provably-unsolvable drops to **0** in every family above - every
such form has a `pop` witness (`bounded_best_first_search` for the short ones,
`node_budget_exhausted+pop_all_then_canonical_witness_upper_bound` for the long
ones; see `unit-1/delta.md`).

**SOLVABLE-BUT-UNSOLVED, with exact failure mode:**

| family | action set | count | failure mode |
| --- | --- | ---: | --- |
| unseen40 | current | 4 | `budget_exhausted` |
| core33 | current | 1 | `budget_exhausted` |
| dense | current | 26 | `budget_exhausted` (UNKNOWN at classification; not separately eval'd in the three-way report) |
| val64 | current | 0 | - |

Under `pop` the classification has **0 unsolvable and 0 unknown** in
`unseen40`/`dense`, so those buckets are empty there.

## 9. Concrete next fixes

1. **Wire the `new_approach` stack env into the held-out harness** so the
   learned policy (not just the oracle) can be evaluated on `unseen40` with
   `pop`. Today `case_spec(case)=None` => `genome.applicable=False`; the harness
   needs a stack-domain adapter that builds the `new_approach` state and can
   roll the policy out, so the "learned route on the previously-unsolvable
   family" claim becomes testable.
2. **The mask remains load-bearing for the learned route.** The canonical prune
   auto-disables under `pop`; the visited-state `--mask-illegal` mask is what
   keeps the learned route legal. Do NOT remove it while wiring (1).
3. **Seed variance of the learned route.** Seed 7 gains `+15` val64 cases,
   seed 13 `+0`. Re-run more seeds / a longer wall cap and report the
   distribution before claiming any learned generalization; the current 3-seed
   sample is what it is.
4. **Close the UNKNOWN bucket honestly.** The shipped specs and
   `perturb_multi_step` remain UNKNOWN under both action sets (bounded budget).
   Raise the oracle node/time budget on just those families, or prove them,
   rather than labelling them.

## 10. Evidence index

Remote root `/opt/automath/evidence/unsolvable-handling/` (workstation mirror
`/opt/workspace/tmp/automath/unsolvable-handling/`):

| unit | key artifacts |
| --- | --- |
| unit-1 | `tables.md`, `delta.md`, `classification.json`, `tests.log` |
| unit-2 | `real_env_pop_report.json`, `dynamic_env_tests.log`, `solvability_tests.log` |
| unit-3 | `SUMMARY.json`, `disjointness_report.log`, `planner_eval.json`, `three_way.json` |
| unit-4 | `LAUNCH_MANIFEST.md`, `launch_cmd.txt`, `train-4349-unit4-pop.sh` |
| unit-5 | `UNIT5_MANIFEST.md`, `smoke_harness.json` |
| unit-6 | `heldout_seed7.json` (+ `.log`), `MANIFEST.md` |
| unit-7 | `heldout_seed7_table.txt`, `heldout_seed7_fast.json`, `heldout_seed7_rollable.json`, `EVIDENCE.md` |
| unit-8 | `heldout_seed13.json`, `heldout_seed42.json`, `watcher.log`, `DONE` |
| unit-9 | `heldout_seed13_table.txt`, `seed13_meta.txt`, `seed13_val64_percase_compare.txt`, `heldout_seed13.sha256` |
| root | `FINAL_EVIDENCE.md`, `COMPARISON.md`, `SHA256SUMS` |

Per-seed JSON sha256 (authoritative 30-episode held-out results):

| seed | path | sha256 |
| --- | --- | --- |
| 7 | unit-6/heldout_seed7.json | `f2ed52ab37b8949520ac958ba4baf7ecd292f7cc7e1dce05e9020c05b756c1e7` |
| 13 | unit-8/heldout_seed13.json | `154450c5828f54122400308be6d76fc1de30689465b29c04f70e1552034c11b6` |
| 42 | unit-8/heldout_seed42.json | PENDING |
