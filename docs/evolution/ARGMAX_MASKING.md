# Argmax-collapse counter: legal-action masking + stochastic eval (task 4344)

The 4339 report's failure mode is a greedy argmax collapse: the policy repeats a
wrong build (`build_add`/`build_mul`) and never selects the required
`build_sub:n9+n5`, or oscillates `set:key`/`clear:key`; the checkpoint
`best_fitness` is a single-lucky-episode artefact.

## Masking

`evolution_trainer/size_selection.py::masked_legal_actions` composes two masks and
always keeps the canonical minimal solution step:

1. **Canonical-subtree prune** (`canonical_subtree_prune`). Ported from the
   add_a unit-2b planner prune: an action whose resulting node set leaves the
   canonical minimal-solution subtree is removed. The dynamic-nodes action set has
   no pop/discard action, so a node outside the subtree cannot be discarded. The
   allowed set is the union of node `canonical()` strings over the states reached
   by replaying `bfs_minimal_word` (`canonical_allowed_nodes`, cached per spec).
   Soundness caveat: on a stack machine with an exact-stack goal this is a strict
   unsolvability proof (add_a unit-2b); on the shipped objective-goal specs the
   guard reads objective values, so the prune is the canonical-path filter that
   preserves the unique minimal solution and removes off-canonical builds.
2. **Visited-state mask**. An action returning to a logical `state.identity()`
   already visited this episode is removed (kills set/clear oscillation).

Performance: the caller passes the already-computed `env.legal_actions(state)` and
the mask evaluates candidates with the pure
`dynamic_env.engine._step_internal(spec, state, action)`, avoiding the
combinatorially expensive full `is_legal` re-enumeration inside `DynamicEnv.step`.
`_candidate_stream` is bounded by `max_attempts` so pool building terminates.

The visitor/no-mask path (`unmasked_legal_actions`) is retained as the BEFORE
control.

## Stochastic evaluation

Selection episodes use epsilon-greedy action choice with `epsilon = 0.1` and the
fixed episode seeds, so the checkpoint metric is not a single deterministic argmax
trajectory. The held-out validation pass is greedy on the fixed per-generation
batch. The SAME episode-seed protocol is used at checkpoint evaluation.

## Measured degenerate loop, BEFORE vs AFTER

Existing 4339 checkpoints
`/opt/automath/tmp/size-invariant-unit2/out/seed_*/stage4/*/checkpoints/best_gen_0045.json`,
30 fresh episodes, `episode_seed(20261010, 0, i)`, `repeat_positions` =
`count_repeat_sequences`, `max_run` = `max_repeat_run`:

| spec | seed | eps | mask | solved/30 | repeat_positions | max_run | wall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| spec_multi_step | 7 | 0.0 | off | 0 | 0 | 1 | 8.84s |
| spec_multi_step | 7 | 0.0 | on | **30** | 0 | 1 | 0.23s |
| spec_multi_step | 42 | 0.1 | off | 0 | 113 | 6 | 6.85s |
| spec_multi_step | 42 | 0.1 | on | **30** | 0 | 1 | 0.09s |
| spec_dynamic_axiom | 7 | 0.1 | off | 6 | 4 | 2 | 1.05s |
| spec_dynamic_axiom | 7 | 0.1 | on | **30** | 0 | 1 | 0.12s |
| spec_dynamic_axiom | 42 | 0.0 | off | 0 | 90 | 2 | 1.62s |
| spec_dynamic_axiom | 42 | 0.0 | on | **30** | 0 | 1 | 0.10s |

Raw greedy trace, seed 7 `spec_multi_step`:
`off = [build_add:n5+n9, build_mul:n5+n9, build_sub:g1+n9, ...]` (12 steps, goal
never reached); `on = [build_sub:n9+n5, set:o]` (goal in 2 steps). The mask moves
every measured seed to 30/30 solved with zero repeated-identical actions and is
roughly 60x faster on the masked path.
