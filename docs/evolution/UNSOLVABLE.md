# Unsolvable / non-solvable cases: decidable oracle and graceful handling

Unit 1b, task 4349. Evaluation only: no training run, no torch, no checkpoint
write. Code: `evolution_trainer/solvability.py`. Raw classification:
`/opt/automath/evidence/unsolvable-handling/unit-1/classification.json` (canonical,
on the compute host) with per-family JSON under `by_family/`, logs
`run_reducedA.log` / `run_reducedB.log` and tables `tables.md` / `delta.md`.

## 1. Definition of solvable

A case is a fixed start state plus a target in one of two action spaces.

* **Stack domain** (`new_approach` build actions, `core` and `evo` variants):
  the state is a stack of nodes; each build action consumes `arity` top nodes
  and pushes one constructed node. The case is **SOLVABLE** iff some finite
  sequence of actions maps the initial stack to exactly `(target,)`.
* **Dynamic domain** (`dynamic_env` Spec + State, used by the held-out specs):
  the state is a set of nodes with objectives/axioms; the case is **SOLVABLE**
  iff the engine's `goal_reached(state)` is true after some finite sequence of
  legal actions within `spec.max_steps`.

A verdict is one of exactly three values:

| verdict | meaning |
| --- | --- |
| `SOLVABLE` | a concrete action path was found and **re-executed** to the goal |
| `UNSOLVABLE` | a **sound** proof that no plan exists (see section 3) |
| `UNKNOWN` | a budget was exhausted before either proof; never reported as `UNSOLVABLE` |

## 2. Oracle algorithm

`classify(case, action_set, node_budget, time_budget)` adapts the case to a
small problem object (`_StackProblem` / `_DynProblem`) and then:

1. **Not-buildable short circuit** (stack): if the fixed target has no
   canonical build word, the case is `UNSOLVABLE` with proof
   `target_not_buildable`.
2. **Canonical fast path**: if the current stack is a prefix of the canonical
   build word, the suffix is a witness; it is re-executed before being
   accepted (`canonical_prefix_fast_path`).
3. **Bounded best-first search** over the *closed* candidate action set (the
   distinct canonical build actions of the target), deduplicated by the
   canonical stack key, with a depth limit. A reconstructed path is always
   replayed before it is reported (`bounded_best_first_search`).
4. **Budget handling**: when the node or time budget is exhausted the verdict
   stays `UNKNOWN` (`node_budget_exhausted` / `time_budget_exhausted`), except
   that under the `pop` action set a sound pop-all-then-rebuild witness is
   tried first and, if it replays to the goal, reported as `SOLVABLE` with the
   proof suffix `pop_all_then_canonical_witness_upper_bound`.

The search is bounded, so the oracle always terminates.

## 3. Soundness argument

**SOLVABLE is sound by construction.** Every reported witness is executed from
the initial state by `replay()`; the verdict is only emitted when the final
state equals `(target,)` (stack) or `goal_reached` is true (dynamic). If a
witness fails replay verification, the search raises rather than returning a
bogus verdict.

**UNSOLVABLE is sound.** Two independent closure proofs exist:

* *Canonical non-subtree prune (stack, no pop).* If no action removes a top
  node (no `pop`), then every node that ever appears on a run ending in
  `(target,)` must be a canonical subtree of `target`, and the action that
  builds such a node must be one of the canonical post-order build actions of
  `target`. Restricting the candidate set to those distinct canonical actions
  is therefore sound and complete, so every node the oracle ever pushes is a
  subtree of the target and the reachable state space is finite. A solution
  then has length `<= len(canonical word)`. When the search closes that finite
  space without a plan (`exhaustive_bounded_search`), no plan can exist.
* *Not-buildable target.* If the canonical planner cannot build the target at
  all, no build sequence can, regardless of the start stack.

**A timeout is never UNSOLVABLE.** When a budget fires, the oracle returns
`UNKNOWN`; the only `UNSOLVABLE` verdicts come from the two closure proofs
above. Under the `pop` action set the prune is *not* sound (pop removes the
offending node), so the candidate set remains sound and complete but the
depth limit is widened to `len(word) + len(initial)` and a budget timeout
still yields `UNKNOWN`, optionally upgraded to a replay-verified
**upper-bound** witness by the pop-all fallback (a minimal solution pops at
most the whole initial stack, so `[pop]*len(stack) + canonical word` always
reaches the goal for a buildable target).

## 4. Classification

Budgets: stack `current` 20000 nodes / 1.5 s, stack `pop` 3000 nodes / 0.5 s,
dynamic 8000 nodes / 1.0 s (reduced from the 200000/20 s first pass so the
whole sweep is bounded; `UNKNOWN` rows are budget artefacts, not
unsolvability). `min steps` / `median steps` are computed over `SOLVABLE`
rows; a step count carries `upper bound` when the proof method says so.

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

## 5. current vs with-pop delta

Total forms UNSOLVABLE under current that become SOLVABLE with pop: 172

| family | case | min steps (with pop) | step count is upper bound | proof method |
| --- | --- | ---: | --- | --- |
| unseen40 | unseen_e_add_mul_0 | 3 | no | bounded_best_first_search |
| unseen40 | unseen_e_add_mul_1 | 5 | no | bounded_best_first_search |
| unseen40 | unseen_e_cmp_0 | 15 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| unseen40 | unseen_e_cmp_1 | 15 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| unseen40 | unseen_e_div_0 | 13 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| unseen40 | unseen_e_div_1 | 3 | no | bounded_best_first_search |
| unseen40 | unseen_e_eq_gt_0 | 15 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| unseen40 | unseen_e_eq_gt_1 | 15 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| unseen40 | unseen_e_logic_0 | 4 | no | bounded_best_first_search |
| unseen40 | unseen_e_logic_1 | 5 | no | bounded_best_first_search |
| unseen40 | unseen_e_mod_0 | 13 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| unseen40 | unseen_e_mod_1 | 3 | no | bounded_best_first_search |
| unseen40 | unseen_e_neg_0 | 5 | no | bounded_best_first_search |
| unseen40 | unseen_e_neg_1 | 3 | no | bounded_best_first_search |
| unseen40 | unseen_e_nested_0 | 5 | no | bounded_best_first_search |
| unseen40 | unseen_e_nested_1 | 10 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| unseen40 | unseen_e_seq4_0 | 3 | no | bounded_best_first_search |
| unseen40 | unseen_e_seq_1 | 12 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| unseen40 | unseen_g2_0 | 3 | no | bounded_best_first_search |
| unseen40 | unseen_g2_1 | 4 | no | bounded_best_first_search |
| unseen40 | unseen_g3_0 | 3 | no | bounded_best_first_search |
| unseen40 | unseen_g3_1 | 3 | no | bounded_best_first_search |
| unseen40 | unseen_g4_0 | 6 | no | bounded_best_first_search |
| unseen40 | unseen_g4_1 | 6 | no | bounded_best_first_search |
| unseen40 | unseen_g_add_0 | 3 | no | bounded_best_first_search |
| unseen40 | unseen_g_add_1 | 4 | no | bounded_best_first_search |
| unseen40 | unseen_g_grouped_0 | 3 | no | bounded_best_first_search |
| unseen40 | unseen_g_grouped_1 | 9 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| unseen40 | unseen_one_0 | 2 | no | bounded_best_first_search |
| unseen40 | unseen_one_1 | 2 | no | bounded_best_first_search |
| unseen40 | unseen_three_0 | 3 | no | bounded_best_first_search |
| unseen40 | unseen_three_1 | 2 | no | bounded_best_first_search |
| unseen40 | unseen_two_0 | 3 | no | bounded_best_first_search |
| unseen40 | unseen_two_1 | 3 | no | bounded_best_first_search |
| unseen40 | unseen_zero_0 | 2 | no | bounded_best_first_search |
| unseen40 | unseen_zero_1 | 2 | no | bounded_best_first_search |
| dense | dense_e_add_mul/pert1 | 10 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_add_mul/pert2 | 9 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_add_mul/pert3 | 5 | no | bounded_best_first_search |
| dense | dense_e_add_mul/pert4 | 5 | no | bounded_best_first_search |
| dense | dense_e_add_mul/pert5 | 3 | no | bounded_best_first_search |
| dense | dense_e_add_mul/pert6 | 9 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_add_mul/pert7 | 2 | no | bounded_best_first_search |
| dense | dense_e_branch/pert12 | 18 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_branch/pert14 | 2 | no | bounded_best_first_search |
| dense | dense_e_branch/pert2 | 18 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_branch/pert5 | 17 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_cmp/pert1 | 15 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_cmp/pert12 | 15 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_cmp/pert13 | 2 | no | bounded_best_first_search |
| dense | dense_e_cmp/pert2 | 17 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_cmp/pert3 | 15 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_cmp/pert4 | 17 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_cmp/pert5 | 15 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_cmp/pert6 | 17 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_cmp/pert7 | 15 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_cmp/pert8 | 16 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_cmp/pert9 | 17 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_div/pert1 | 14 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_div/pert10 | 2 | no | bounded_best_first_search |
| dense | dense_e_div/pert11 | 1 | no | bounded_best_first_search |
| dense | dense_e_div/pert2 | 12 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_div/pert3 | 14 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_div/pert4 | 15 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_div/pert5 | 14 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_div/pert6 | 15 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_div/pert7 | 2 | no | bounded_best_first_search |
| dense | dense_e_div/pert8 | 5 | no | bounded_best_first_search |
| dense | dense_e_div/pert9 | 5 | no | bounded_best_first_search |
| dense | dense_e_eq_gt/pert1 | 13 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_eq_gt/pert10 | 2 | no | bounded_best_first_search |
| dense | dense_e_eq_gt/pert11 | 1 | no | bounded_best_first_search |
| dense | dense_e_eq_gt/pert2 | 14 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_eq_gt/pert3 | 15 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_eq_gt/pert4 | 15 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_eq_gt/pert5 | 14 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_eq_gt/pert6 | 13 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_eq_gt/pert7 | 15 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_eq_gt/pert8 | 16 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_eq_gt/pert9 | 13 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_logic/pert1 | 7 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_logic/pert2 | 9 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_logic/pert3 | 5 | no | bounded_best_first_search |
| dense | dense_e_logic/pert4 | 4 | no | bounded_best_first_search |
| dense | dense_e_logic/pert5 | 3 | no | bounded_best_first_search |
| dense | dense_e_logic/pert6 | 2 | no | bounded_best_first_search |
| dense | dense_e_mod/pert1 | 14 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_mod/pert10 | 3 | no | bounded_best_first_search |
| dense | dense_e_mod/pert11 | 2 | no | bounded_best_first_search |
| dense | dense_e_mod/pert2 | 12 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_mod/pert3 | 14 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_mod/pert4 | 13 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_mod/pert5 | 12 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_mod/pert6 | 6 | no | bounded_best_first_search |
| dense | dense_e_mod/pert7 | 6 | no | bounded_best_first_search |
| dense | dense_e_mod/pert8 | 5 | no | bounded_best_first_search |
| dense | dense_e_mod/pert9 | 3 | no | bounded_best_first_search |
| dense | dense_e_neg/pert1 | 4 | no | bounded_best_first_search |
| dense | dense_e_neg/pert2 | 4 | no | bounded_best_first_search |
| dense | dense_e_neg/pert3 | 2 | no | bounded_best_first_search |
| dense | dense_e_neg/pert4 | 2 | no | bounded_best_first_search |
| dense | dense_e_nested/pert1 | 11 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_nested/pert2 | 13 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_nested/pert3 | 12 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_nested/pert4 | 12 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_nested/pert5 | 5 | no | bounded_best_first_search |
| dense | dense_e_nested/pert6 | 4 | no | bounded_best_first_search |
| dense | dense_e_nested/pert7 | 3 | no | bounded_best_first_search |
| dense | dense_e_nested/pert8 | 3 | no | bounded_best_first_search |
| dense | dense_e_nested/pert9 | 1 | no | bounded_best_first_search |
| dense | dense_e_seq/pert10 | 2 | no | bounded_best_first_search |
| dense | dense_e_seq/pert11 | 2 | no | bounded_best_first_search |
| dense | dense_e_seq/pert2 | 15 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_seq/pert5 | 15 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_seq/pert7 | 16 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_seq4/pert10 | 19 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_seq4/pert11 | 20 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_seq4/pert12 | 21 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_seq4/pert16 | 3 | no | bounded_best_first_search |
| dense | dense_e_seq4/pert17 | 1 | no | bounded_best_first_search |
| dense | dense_e_seq4/pert3 | 20 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_seq4/pert4 | 21 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_seq4/pert5 | 20 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_seq4/pert7 | 19 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_seq4/pert8 | 21 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_e_seq4/pert9 | 20 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_g2/pert1 | 4 | no | bounded_best_first_search |
| dense | dense_g2/pert2 | 3 | no | bounded_best_first_search |
| dense | dense_g2/pert3 | 2 | no | bounded_best_first_search |
| dense | dense_g3/pert1 | 5 | no | bounded_best_first_search |
| dense | dense_g3/pert2 | 4 | no | bounded_best_first_search |
| dense | dense_g3/pert3 | 3 | no | bounded_best_first_search |
| dense | dense_g3/pert4 | 5 | no | bounded_best_first_search |
| dense | dense_g4/pert1 | 6 | no | bounded_best_first_search |
| dense | dense_g4/pert2 | 5 | no | bounded_best_first_search |
| dense | dense_g4/pert3 | 2 | no | bounded_best_first_search |
| dense | dense_g4/pert4 | 3 | no | bounded_best_first_search |
| dense | dense_g4/pert5 | 6 | no | bounded_best_first_search |
| dense | dense_g_add/pert1 | 6 | no | bounded_best_first_search |
| dense | dense_g_add/pert3 | 2 | no | bounded_best_first_search |
| dense | dense_g_add/pert5 | 2 | no | bounded_best_first_search |
| dense | dense_g_add/pert6 | 1 | no | bounded_best_first_search |
| dense | dense_g_grouped/pert1 | 8 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_g_grouped/pert3 | 5 | no | bounded_best_first_search |
| dense | dense_g_grouped/pert4 | 4 | no | bounded_best_first_search |
| dense | dense_g_grouped/pert5 | 3 | no | bounded_best_first_search |
| dense | dense_g_grouped/pert6 | 1 | no | bounded_best_first_search |
| dense | dense_one/pert1 | 2 | no | bounded_best_first_search |
| dense | dense_random03_e_neg | 6 | no | bounded_best_first_search |
| dense | dense_random04_g2 | 5 | no | bounded_best_first_search |
| dense | dense_random05_e_eq_gt | 13 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_random07_e_branch | 16 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_random08_g_grouped | 9 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_random09_e_seq | 14 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_random10_one | 4 | no | bounded_best_first_search |
| dense | dense_random11_e_add_mul | 10 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_random17_e_logic | 8 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_random18_zero | 4 | no | bounded_best_first_search |
| dense | dense_random19_e_nested | 12 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_random20_two | 1 | no | bounded_best_first_search |
| dense | dense_random21_e_seq4 | 19 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_random22_g2 | 4 | no | bounded_best_first_search |
| dense | dense_random23_e_div | 14 | yes | node_budget_exhausted+pop_all_then_canonical_witness_upper_bound |
| dense | dense_random24_g4 | 6 | no | bounded_best_first_search |
| dense | dense_random25_e_neg | 5 | no | bounded_best_first_search |
| dense | dense_random26_g_grouped | 5 | no | bounded_best_first_search |
| dense | dense_three/pert1 | 4 | no | bounded_best_first_search |
| dense | dense_three/pert2 | 3 | no | bounded_best_first_search |
| dense | dense_three/pert3 | 1 | no | bounded_best_first_search |
| dense | dense_two/pert1 | 2 | no | bounded_best_first_search |
| dense | dense_two/pert2 | 3 | no | bounded_best_first_search |
| dense | dense_zero/pert1 | 2 | no | bounded_best_first_search |

Forms UNKNOWN under current that become SOLVABLE with pop: 30
- dense / dense_e_branch/pert1 -> 15 steps (node_budget_exhausted+pop_all_then_canonical_witness_upper_bound)
- dense / dense_e_branch/pert10 -> 19 steps (node_budget_exhausted+pop_all_then_canonical_witness_upper_bound)
- dense / dense_e_branch/pert11 -> 20 steps (node_budget_exhausted+pop_all_then_canonical_witness_upper_bound)
- dense / dense_e_branch/pert13 -> 2 steps (bounded_best_first_search)
- dense / dense_e_branch/pert3 -> 18 steps (node_budget_exhausted+pop_all_then_canonical_witness_upper_bound)
- dense / dense_e_branch/pert4 -> 18 steps (node_budget_exhausted+pop_all_then_canonical_witness_upper_bound)
- dense / dense_e_branch/pert6 -> 16 steps (node_budget_exhausted+pop_all_then_canonical_witness_upper_bound)
- dense / dense_e_branch/pert7 -> 17 steps (node_budget_exhausted+pop_all_then_canonical_witness_upper_bound)
- dense / dense_e_branch/pert8 -> 18 steps (node_budget_exhausted+pop_all_then_canonical_witness_upper_bound)
- dense / dense_e_branch/pert9 -> 18 steps (node_budget_exhausted+pop_all_then_canonical_witness_upper_bound)
- dense / dense_e_cmp/pert10 -> 4 steps (bounded_best_first_search)
- dense / dense_e_cmp/pert11 -> 4 steps (bounded_best_first_search)
- dense / dense_e_seq/pert1 -> 13 steps (node_budget_exhausted+pop_all_then_canonical_witness_upper_bound)
- dense / dense_e_seq/pert3 -> 13 steps (node_budget_exhausted+pop_all_then_canonical_witness_upper_bound)
- dense / dense_e_seq/pert4 -> 14 steps (node_budget_exhausted+pop_all_then_canonical_witness_upper_bound)
- dense / dense_e_seq/pert6 -> 14 steps (node_budget_exhausted+pop_all_then_canonical_witness_upper_bound)
- dense / dense_e_seq/pert8 -> 3 steps (bounded_best_first_search)
- dense / dense_e_seq/pert9 -> 4 steps (bounded_best_first_search)
- dense / dense_e_seq4/pert1 -> 20 steps (node_budget_exhausted+pop_all_then_canonical_witness_upper_bound)
- dense / dense_e_seq4/pert13 -> 3 steps (bounded_best_first_search)
- dense / dense_e_seq4/pert14 -> 4 steps (bounded_best_first_search)
- dense / dense_e_seq4/pert15 -> 3 steps (bounded_best_first_search)
- dense / dense_e_seq4/pert2 -> 18 steps (node_budget_exhausted+pop_all_then_canonical_witness_upper_bound)
- dense / dense_e_seq4/pert6 -> 20 steps (node_budget_exhausted+pop_all_then_canonical_witness_upper_bound)
- dense / dense_random15_e_cmp -> 14 steps (node_budget_exhausted+pop_all_then_canonical_witness_upper_bound)
- dense / dense_random29_e_branch -> 16 steps (node_budget_exhausted+pop_all_then_canonical_witness_upper_bound)
- unseen40 / unseen_e_branch_0 -> 17 steps (node_budget_exhausted+pop_all_then_canonical_witness_upper_bound)
- unseen40 / unseen_e_branch_1 -> 18 steps (node_budget_exhausted+pop_all_then_canonical_witness_upper_bound)
- unseen40 / unseen_e_seq4_1 -> 20 steps (node_budget_exhausted+pop_all_then_canonical_witness_upper_bound)
- unseen40 / unseen_e_seq_0 -> 12 steps (node_budget_exhausted+pop_all_then_canonical_witness_upper_bound)

Rows total: 1224

## 6. Graceful handling design

### Training

* **Skip provably-unsolvable forms.** Forms classified `UNSOLVABLE` cannot be
  solved by construction; feeding them to a planner trains against an
  impossible target and dilutes the reward. They are removed from the training
  mix by default.
* **Bounded attempt for `UNKNOWN`.** Any form the oracle could not settle is
  still trainable, but each attempt gets a hard step budget; on exhaustion the
  episode yields the shaped partial reward and ends, so one hard form cannot
  stall an epoch.
* **Measure the time saved.** Log `oracle_nodes`, `oracle_seconds` and the
  per-form wall time with and without the skip so the saving is a measured
  number, not a claim. The classification run itself records nodes/seconds per
  row in `budget_used`.

### Evaluation

* **Per-case budget.** Every eval case is capped by node and time budgets; a
  case that exhausts its budget is `UNKNOWN`.
* **Provably-unsolvable reported separately.** `UNSOLVABLE` cases are pooled
  into their own bucket and are **never counted as failures** (a planner
  cannot solve them, so counting them as misses would punish it for the
  problem, not the solver). They are additionally used as a negative control:
  a planner that "solves" them is a red flag.
* **Solve rate is over solvable forms.** `solve_rate = solved / SOLVABLE`, with
  `UNKNOWN` and `UNSOLVABLE` counts always reported next to it so the
  denominator is auditable.

## 7. Reproduction

```
cd /opt/automath/tmp/unsolvable-u1
/opt/automath/venv/bin/python -m evolution_trainer.solvability \
  --families all --action-set both \
  --node-budget 20000 --time-budget 1.5 \
  --pop-node-budget 3000 --pop-time-budget 0.5 \
  --dyn-node-budget 8000 --dyn-time-budget 1.0 \
  --out classification.json
/opt/automath/venv/bin/python analyze.py . by_family/*.json   # tables.md + delta.md
```

## 8. Implemented action space: the real `pop` action (unit 2, task 4349 FIX 1)

Unit 1b modelled `pop` in the oracle. Unit 2 implements it in the engine and
re-runs the classification with the REAL `DynamicEnv`.

### 8.1 The change

* `dynamic_env/engine.py`: `DynamicEnv(spec, allow_pop=False)`. With
  `allow_pop=True` the engine emits a new `Action(kind="pop")`, legal iff the
  work/partial-solution stack is non-empty. The work stack is exactly the
  combination nodes the interpreter itself constructed (`g<next_id>` ids that are
  NOT in `spec.nodes`); initial nodes and `add_node` dynamic-axiom nodes are not
  part of it. `pop` removes the topmost work node and ROLLS `next_id` back, so
  the work stack is a true stack and `[pop]*k` is a faithful undo.
* Wiring: module functions `legal_actions(spec, state, allow_pop)` /
  `is_legal` / `step` / `try_step`; `DynamicEnv.legal_actions/is_legal/step/
  try_step` forward `self.allow_pop`; `action_set_flag("current"|"pop")` and
  `env_for_action_set(spec, name)` are the config/CLI selectors.
* `allow_pop=False` is byte-identical: the 4-kind `KIND_ORDER` is unchanged, the
  action vector width is unchanged and the size-invariant genome stays 1153
  genes (proved by `size_invariant_tests`, "one genome length 1153"). A `pop`
  action (only present under the pop action set) carries no kind bit and no main
  key, so it is distinguishable without resizing anything.

### 8.2 Soundness update (the prune is no longer a proof of unsolvability)

The unit-1b `canonical non-subtree => unsolvable` argument assumed NO action can
remove a node. With `pop` that assumption is false. `size_selection.
canonical_subtree_prune` now returns the actions UNCHANGED whenever
`env.allow_pop` is true, so the canonical mask never deletes a pop-legal state.
Under the `current` action set the prune is unchanged.

### 8.3 Action-space delta (real engine, one state)

`pop` adds 0 actions on an empty work stack and exactly +1 after a work node.

| spec | actions current (reset) | actions pop (reset) | actions current (1 work node) | actions pop (1 work node) | min current | min pop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| spec_minimal | 1 | 1 | - | - | 1 | 1 |
| spec_multi_step | 18 | 18 | 36 | 37 | 2 | 2 |
| spec_dynamic_axiom | 1 | 1 | - | - | 3 | 3 |
| spec_dynamic_group | 4 | 4 | 9 | 10 | 2 | 2 |
| spec_multi_step_heldout | 18 | 18 | 37 | 38 | 2 | 2 |
| spec_dynamic_group_deep | 15 | 15 | 40 | 41 | 3 | 3 |

### 8.4 Real-engine re-classification (same budgets as section 4)

The `_DynProblem` oracle now builds `DynamicEnv(spec, allow_pop=with_pop)` and
iterates the ENGINE'S legal actions (the unit-1 `_dyn_pop` helper is deleted).
The stack families still run on their own `new_approach` action engine (they
have no `dynamic_env` counterpart; see 8.5). The table is byte-for-byte the
unit-1 table, so the oracle model is CONFIRMED on the real engine for every
family the engine can represent.

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

Real-engine newly-solvable counts (pop minus current), raw from
`real_env_pop_report.json`:

* `val64` (64 `dynamic_env` start states, real engine): current 64/64,
  pop 64/64, newly 0; action-count sum over the 64 start states 3084 -> 3148.
* unseen40-type perturbed form, real engine (`evolution_trainer/
  solvability_tests.PERTURBED_SPEC`, an off-canonical work node blocks the goal
  guard): with 2 work nodes current=UNKNOWN and pop=SOLVABLE in 2 steps; with 3
  work nodes current=UNKNOWN and pop=SOLVABLE in 3 steps; with 4 work nodes
  current=UNKNOWN and pop=SOLVABLE in 6 steps. `[pop]*len(work)+canonical word`
  replays to the goal in the real engine.

### 8.5 Exact disagreement with the unit-1 oracle model

* The named families `demo_bundle`, `core33`, `ext114`, `unseen40` and `dense`
  are `new_approach` STACK-domain cases (`.env` present); they have no
  `dynamic_env` Spec, so `DynamicEnv` cannot represent them. Their pop evidence
  is the stack engine (`new_approach` build actions), which reproduces
  unseen40 40/40 and dense 320/320 exactly. This is a domain mismatch in the
  unit-1 CLI, not a behavioural disagreement.
* The unit-1 `_DynProblem` pop deleted the node with the largest numeric id
  among ALL nodes, including the spec's initial value nodes (e.g. `n9`). The
  real engine's `pop` deletes only the topmost ENGINE-CONSTRUCTED work node and
  rolls `next_id` back. On the dynamic families this changes no verdict in the
  table above; it is the difference that makes `[pop]*k + canonical word`
  replayable.

### 8.6 Unit tests (unit 2)

* `dynamic_env/tests.py`: 17/17 PASS: pop off-by-default, legality + exact step
  semantics (newest work node, bookkeeping, determinism), pop-all-and-replay
  witness in the real env, action-set selection (+1 after a build). Tail:
  `RESULT: 17/17 passed in 1.535s`.
* `evolution_trainer/solvability_tests.py` (NEW): 5/5 PASS: perturbed form
  current=UNKNOWN / pop=SOLVABLE, pop-all replay witness, shipped specs solve
  with pop, canonical prune deletes no pop action, default engine unchanged.
  Tail: `RESULT: 5/5 passed`.
* Regression on the touched trainer modules: `size_selection_tests` 5/5 PASS,
  `size_invariant_tests` 6/6 PASS (genome length still 1153).

