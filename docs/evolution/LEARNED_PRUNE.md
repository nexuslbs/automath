# LEARNED_PRUNE - explicit trained planner / value function over REDUCTION_TABLE

Task 4354 UNIT 3A, branch `heldout-targets`. This unit turns the task-4344
fix#3 eval-time hand mask into an EXPLICIT TRAINED planner/value function over
the reduction/state space, so the evolution/validation rollouts can run in
three modes: `hand` (the incumbent filter), `learned` (the trained value
function only), `none` (no mask at all).

## 1. Why this exists (the load-bearing mask)

The task-4344 report measured the legal-action mask as load-bearing: with the
canonical-subtree prune ON the BFS-minimal policy reaches 30/30, with it OFF it
reaches 0/30 (the greedy policy repeats an off-canonical build forever). That
mask lived inside the rollout as an eval-time filter
(`evolution_trainer/size_selection.py`,
`canonical_subtree_prune` / `masked_legal_actions`). The mandate for this unit
is to replace that filter with a TRAINED function
`learned_prune(state, actions, net) -> mask` that predicts, for every legal
action, whether the successor state stays on the canonical minimal-solution
subtree of the reduction table.

## 2. Reuse (coordinate, do not duplicate)

* VALUE HEAD + OBJECTIVE: the llm-like small net of task 4353, branch
  `llm-like-learning` @ `8f108a7`, file `llm_like/model.py`
  (`LLMAgentNet.value`: `v(s, s') = w_v . tanh(W_v [s ; s'] + b_v)`) and the
  frozen layer table in `docs/evolution/LLM_LIKE.md` sections 1.4 and 3.2. Here
  the same one-hidden-layer `tanh` scalar head and the same value-MSE loss are
  used at width `d = 32` (the design allows 32 or 64). The value HEAD is reused;
  the llm-like torch training stack is NOT required by the default path.
* FEATURES: `evolution_trainer/size_invariant.py` (`node_features`,
  `hash_vector`, `KIND_ORDER`, `action_main_key`): the same spec-agnostic
  per-node/per-action encoder the evolution policy genome already uses, so the
  value net is size-invariant and spec-agnostic.
* LABELS / ORACLE: `evolution_trainer/size_selection.py`
  (`canonical_allowed_nodes`, `canonical_subtree_prune`), whose concept comes
  from the gen-fitness planner over the REDUCTION_TABLE
  (`new_approach/planner.py`, branch `gen-fitness` @ `1b8d24e`, unit `add_a`)
  and its pattern-goals extension (`planner-pattern-goals` @ `c69edd7`).

## 3. Data labeling (deterministic, stdlib)

`learned_prune.label_action(env, state, action)` is the training oracle: it
returns 1 when every node of the successor state is in
`canonical_allowed_nodes(spec)` and 0 otherwise. This is exactly the membership
predicate of `canonical_subtree_prune`, so the label is the hand prune's own
decision, computed from the REDUCTION_TABLE canonical word / BFS oracle.

### 3.1 The contradictory-pair bug (found and fixed in unit 3A-FINISH)

The first learned-prune dataset was CONTRADICTORY. Two successor states,
`build:build_sub:n9+n5` (label 1, on the canonical word) and
`build:build_sub:n5+n9` (label 0, off it), produced the IDENTICAL feature vector
(shared `feature_sha bf8e2ebb5b7a01ca`). They differ only in the declared
`canonical()` order of the composed items, which the original `node_features`
pooling does NOT encode (it sums/reduces node features, so node order is lost).
Identical input with opposite labels means no value function can separate the
pair: the net learned to keep the off-canonical successor, and `mode=learned`
kept 18/18 off-canonical builds at reset on `spec_multi_step` (off-canonical
value 0.982 > canonical 0.965), reaching 0/30.

The fix adds a CANON BAG block to the state encoder: every node's
`canonical()` string is hashed into a fixed `CANON_BAG_DIM = 32` bag
(`hash_vector(node.canonical(), CANON_BAG_DIM, salt=7)`) and folded into
`_state_vector`. `STATE_DIM` becomes 89 and the pair vector `PAIR_DIM` becomes
224; the contradictory pair now gets different feature shas.

`build_dataset(...)` samples a bounded set of states per spec
(`DEFAULT_STATES_PER_SPEC = 96`): the reset state, every canonical-prefix
state from `size_selection._canonical_states`, the target-variant start states,
and deterministic random walks from those. Every legal action of every sampled
state is labeled and featurized. The bundled training specs
(`spec_minimal`, `spec_multi_step`, `spec_dynamic_axiom`,
`spec_dynamic_group`), the shipped held-out specs, and the 16 new
target-variant cases of the task-4354 pool are all included.

Class balance. On a canonical state at most one action is on-path while dozens
of off-canonical builds are legal, so keeping every negative would make the
dataset about 0.8 percent positive and a net could score 99 percent agreement
by always predicting 0 (useless as a planner). The routine therefore keeps ALL
positives and thins the negatives to `DEFAULT_NEGATIVES_PER_STATE = 2` per
state with a deterministic stride. Training additionally applies a positive
class weight (`DEFAULT_POSITIVE_WEIGHT = 4.0`).

Determinism. Features are pure hashing (no RNG); the state walk uses
`FEATURE_SEED`; the train/check split uses `SPLIT_SEED`; the batch order uses
`TRAIN_SEED + 1`; the initialization uses `TRAIN_SEED`. Same
`(seed, spec, per_spec, negatives_per_state)` produces the same labels and the
same features, byte for byte.

## 4. Model

`learned_prune.ValueNet` is a pure-standard-library one-hidden-layer `tanh`
scalar head:

```
v(x) = w2 . tanh(W1 x + b1) + b2,   x = [s ; a ; s']  in R^224
```

with `s` and `s'` the 89-dim size-invariant state blocks (pooled node features +
global scalars + hashed objective values + guard-target slots + the 32-dim canon
bag of section 3.1), `a` the 46-dim action block, width `d = 32`. This mirrors
the llm-like value head (`value_proj` + `w_v_head`) with a shared-width hidden
layer. There is NO torch import in `learned_prune.py`, so the default
`evolution_trainer` path stays standard library only.

Training is weighted value-MSE with Adam plus an OPTIONAL per-state listwise
ranking term (`rank_weight`, default 1.0) that pushes every positive above every
negative in the same state; the default budget is `DEFAULT_STEPS = 240`
gradient steps at `batch = 32`, learning rate 0.02. This is a minute-scale CPU
job, explicitly NOT the long evolution loop. The script
`evolution_trainer/train_learned_prune.py` writes the canned checkpoint
`evolution_trainer/learned_prune_checkpoint.json` (sorted-key JSON, rounded
floats, deterministic); the shipped checkpoint was trained with
`--per-spec 64 --negatives-per-state 8 --steps 800 --positive-weight 4.0
--rank-weight 3.0`.

## 5. Mode wiring

`learned_prune.prune_actions(env, state, actions, seen, mode, net)` dispatches:

* `hand`    -> `size_selection.masked_legal_actions` (canonical subtree +
  visited-state mask);
* `learned` -> `learned_prune.learned_prune` ONLY, threshold 0.5, with NO
  hand-written filter;
* `none`    -> the actions unchanged.

`size_selection.rollout` gained `prune_mode` / `prune_net` keyword arguments.
`prune_mode=None` resolves to `"hand"` when `mask_illegal` is set and `"none"`
otherwise, so the pre-3A default behaviour is unchanged. `EvoConfig` gained
`prune_mode: str = "hand"` and `prune_checkpoint: str = ""`; the trainer loads
the checkpoint once when `prune_mode == "learned"` and passes it through
`evaluate_candidate` / `evaluate_validation`. `curriculum.py` gained
`--prune-mode {hand,learned,none}` and `--prune-checkpoint`.

Standalone harness:

```sh
/opt/automath/venv/bin/python -m evolution_trainer.learned_prune \
    --spec spec_multi_step --episodes 30 --modes hand,learned,none
/opt/automath/venv/bin/python -m evolution_trainer.train_learned_prune
/opt/automath/venv/bin/python -m evolution_trainer.learned_prune --metrics
```

## 6. How Part B runs the mask-REMOVED before/after

1. BEFORE (incumbent mask, load-bearing): train/evaluate with
   `--mask-illegal` and the default `--prune-mode hand`. This reproduces the
   task-4344 protocol (canonical-subtree + visited mask ON).
2. AFTER (mask removed, learned planner): evaluate with
   `--prune-mode learned --prune-checkpoint <checkpoint>` and NO hand-written
   filter. The same 30 explicit-seed selection episodes and the same 16-case
   per-generation validation batch are used, so the only difference is the
   mask source.
3. CONTROL (no mask): `--prune-mode none`.
4. Report `solve_rate`, `mean_solved_steps` and `val_solved_rate` per mode over
   the same seeds, plus the learned-vs-hand agreement rate on the labeled
   distribution (section 3 of the EVIDENCE pack). If the learned mask does not
   agree with the hand mask, the agreement rate and the confusion matrix are
   the signal: it is reported raw, never rounded up.

## 7. Honesty notes

* The learned mask is trained against the hand prune's OWN decision, so a high
  training-case agreement is expected by construction; the informative number
  for Part B is the agreement on the held-in split and on the held-out
  TARGET-variant specs, plus the 30-episode solve rate under `mode=learned`.
* The learned prune is a function of `(state, action)` only. The hand
  `masked_legal_actions` also applies the visited-state mask, which is
  episode-dependent; `mode=learned` deliberately applies NO such hand filter,
  which is exactly the "mask-removed" condition Part B must measure.
* No host address, credential or secret appears in this document or its code.
