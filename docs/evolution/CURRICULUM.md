# Unit D-1 - operator curriculum on the fixed-weight evolution backbone

Branch **`dynamic-nodes`**. Module `evolution_trainer/curriculum.py` (runner) +
`evolution_trainer/heldout.py` (held-out harness). Standard library only.

Unit C selected the **fixed-weight evolutionary process** (`evolution_trainer`
`evolution` arm) as the training backbone: fastest arm, best sample efficiency
and the only arm whose best training genome solved both hard combinatorial specs.
Unit D-1 therefore drives the SAME `EvolutionTrainer` through the operator's
four-stage curriculum instead of building another trainer.

## The operator mandate, mapped

1. **Start minimal.** S1 on `spec_minimal`; the strict stage is sub-second.
2. **Dynamic nodes with deterministic tests, strict steps.** S2 runs the
   deterministic single-solution tests shipped in `dynamic_env/tests.py` as the
   curriculum items and trains `spec_dynamic_group` under the exact BFS word.
3. **Little by little let the agent choose.** S3: strict sequence first, then
   free choice where a WRONG step is DISCARDED (deterministic, no state change,
   one step spent). The strict-stage winner is re-scored in the free stage to
   expose overfitting, and free training recovers.
4. **Multi-step goals.** S4: `spec_multi_step` and `spec_dynamic_axiom` with the
   BFS-exact MINIMAL step count, an operator MAX budget and terminal failure at
   MAX; the per-step cost makes fewer steps score higher.

## Exact planning

`bfs_min_steps(spec)` is an independent BFS in the runner; it is asserted equal
to the engine's `minimal_length` and `bfs_minimal_word` for every shipped spec
(`curriculum_tests.check_bfs_matches_engine`). `solution_path(spec)` replays
that unique witness into a `state -> single correct action` map; the S3 gate
accepts ONLY that action, which is what makes "only one step is correct per
state" mechanical and FINITE (an unbounded forward reachability search over the
grouping specs does not terminate, because every combine mints a fresh node id).
Asserted: `check_gate_single_correct_step`.

| spec | BFS min | MAX rule | MAX |
| --- | --- | --- | --- |
| `spec_minimal` | 1 | (S1 fixed) | - |
| `spec_dynamic_group` | 2 | (S3 fixed) | - |
| `spec_multi_step` | 2 | min + 2 | 4 |
| `spec_dynamic_axiom` | 3 | max(min+2, min*2) | 6 |

The table is regenerated at run time by `minimal_and_max` and written into the
stage summaries, so it can never drift from the engine.

## Backbone changes (default-path preserving)

* `EvolutionTrainer.__init__` gains `self.step_gate = None`; `run_episode`
  discards a rejected action (`-step_cost`, one step, SAME state). `None` is
  exactly Unit B behaviour, so `train.py` and the Unit C arms are unchanged.
* `EvolutionTrainer.run(warm_start=None, on_generation=None)`: the optional
  `on_generation(generation, record) -> bool` hook stops after writing that
  generation's checkpoints. It is the checkpoint-based early stop used by the
  long S4 run. Default `None` is unchanged.
* `Harness.rollout(collect=True)` trace entries gain `step`, `action`,
  `state_vector`, `objective_vector` and `goal` (extra keys only; the reward
  shaping and the existing `inputs`/`chosen`/`reward` keys are untouched).

## Held-out design (built here, evaluated in Unit D-2)

Two cases ship as DATA, not code, and are NOT used in training:

* `data/dynamic_env/heldout/spec_multi_step_heldout.json` - the same shape as
  `spec_multi_step` with a DIFFERENT target objective state (8 instead of 4,
  reachable as `add(5,3)`).
* `data/dynamic_env/heldout/spec_dynamic_group_deep.json` - a deeper dynamic
  grouping with a LARGER arity (3 operands: `sum(1,2,5)=8`).

They live in a `heldout/` subdirectory so the shipped `dynamic_env/tests.py`
suite (which expects exactly one minimal solution per top-level spec) is
unaffected. `evolution_trainer/heldout.py` scores an optional checkpoint on both
cases and reports, per case, the elementar 1/0 objective vector, the action, the
reward and the step, plus solved/total and the wall time. Unit D-1 only builds
and SMOKES it; the real held-out eval runs in Unit D-2 after stage 4 has
progressed.

## Reproducing

```sh
/opt/automath/venv/bin/python -m evolution_trainer.curriculum --stage all --out out/curriculum
/opt/automath/venv/bin/python -m evolution_trainer.curriculum --stage s4 \
    --spec spec_multi_step --generations 1500 --out out/stage4
/opt/automath/venv/bin/python -m evolution_trainer.heldout --episodes 3
/opt/automath/venv/bin/python -m evolution_trainer.curriculum_tests
```
