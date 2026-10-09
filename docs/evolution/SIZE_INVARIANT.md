# Size-invariant genome / architecture (Unit 1)

Branch `size-invariant-genome`, created from `origin/dynamic-nodes`
(`fa06e177cdc79d3322a7aea883684c016030b75e`).

This unit replaces the genome/architecture of the dynamic-nodes evolution, and
nothing else. The evolutionary process (`evolution_trainer/evolution.py`) and the
curriculum pipeline (`config/evolution_dynamic.json`, `evolution_trainer/curriculum.py`)
are untouched except for the two wiring lines the new genome interface forces.

## 1. Why the Unit B genome was spec-sized

A Unit B genome is the flat weight vector of a one-hidden-layer `tanh` MLP whose
input is `state_features(env, state, mask) + action_features(state, action)`
(`docs/evolution/EVOLUTION.md` section 1). Both terms are functions of the spec:

```
state_dim(spec)          = 3*|objectives| + |node_types| + 3
ActionFeaturizer(spec).dim = 4 + |vocab| + 1 + 2*max_arity
```

so a genome trained on one spec has a length no other spec can use. The Unit D-2
verdict is exactly this: the direct-transfer case (same width) solved 4/30 while
the deeper arity-3 grouping was **0/30 (structural dimension mismatch)** and the
network width was tied to the spec size.

## 2. Choice: graph encoder over the node graph + masked pooling (hybrid)

**Chosen: (a) a graph encoder over the target+state node graph with fixed hidden
dims and masked pooling, hybridised with fixed-width signed feature hashing.**
Option (b), padded width with masking, is rejected as the primary mechanism: a pad
cap still fails the moment a spec exceeds it, and even masked padding keeps a
spec-sized tensor, which is what we are removing. Hashing the spec's own id
strings removes the vocabulary cap; masked mean pooling removes the node/arity
cap; fixed hidden dims remove the matrix-shape dependence entirely.

Invariant statement (what this unit proves):

> One genome JSON (the same weights, no re-allocation, no re-shaping) runs on any
> spec built by `dynamic_env`. The genome parameter count and the input/output
> shapes are INDEPENDENT of the spec's node count, arity and action count. The
> input trait vectors are per-node (size-invariant) and every matrix is
> fixed-size.

Mechanism, three independent "no width" rules:

1. **One action, one scalar.** Every legal action is scored independently by the
   same network, so the output is always one logit per action; the action count
   is not a width.
2. **A state is a set of nodes.** Each node becomes a fixed-length feature vector
   (role, semantics, hashed type, hashed id, normalised value, objective
   current/target/subgoal-mask flags, in/out degree); one message-passing round
   over the graph edges read from the spec's own `node`/`nodes` fields produces a
   per-node hidden vector; a **masked mean pool** condenses the set to a fixed
   `hidden` latent. The node count and the combination arity are therefore not
   widths.
3. **Names are hashed, not indexed.** Axiom / tag / objective / node-type / node
   id strings enter through a deterministic **signed feature hash** of fixed width
   (`hash_vector`), so there is no per-spec vocabulary table and no vocabulary
   width.

Genome layout (all shapes fixed; `H = hidden = 12` by default, `NODE_DIM = 25`,
`ACTION_DIM = 4 + 8 + 1 + H = 25`, `GLOBAL_DIM = 4`):

```
W_node   H x NODE_DIM    b_node H
W_self   H x H           W_neigh H x H        b_msg H
W_g      H x GLOBAL_DIM  b_g    H
W_a      H x ACTION_DIM  b_a    H
W_c      H x H           b_c    H
w_out    H               b_out  1
```

With `H=12` the genome length is the constant **1153** for every spec. The state
encoder runs once per state and the same latent scores every action, so the
trainer also does less work per step than the old per-action feature vector.

## 3. Reference reuse: `origin/target-policy` (READ-ONLY, not merged)

`origin/target-policy` (`new_approach/target_features.py`) implemented a
target-conditioned `PolicyNet` that scores `features(target, stack, action)` with
the same parameters for any target. **Reused idea:** the same net scores every
action and the target enters only through the input as data. **Built fresh here:**
its feature blocks are fixed-length only because they summarise a FIXED hardcoded
token vocabulary (`SEMANTIC_OPS`, `TOKEN_NAMES`), which still ties the encoder to
a known op set. This unit instead reads the graph structure from the spec's own
declared fields and hashes the spec's own names, so even the vocabulary is not an
assumption. Nothing from the `target-policy` tree is copied or merged.

## 4. Wiring (the only non-architecture edit)

`evolution_trainer/evolution.py`, three edits, each the minimum the new interface
forces:

1. import `SizeInvariantNet` instead of `PolicyNet`/`state_dim`/`state_features`
   (`ActionFeaturizer` and `objective_ids` are kept because the trainer still
   exposes `self.featurizer`/`self.obj_ids` to existing callers);
2. `__init__`: `self.net = SizeInvariantNet(self.config.hidden)` and
   `self.in_dim = self.net.size` (was `state_dim(spec) + featurizer.dim`);
3. `_logits`: `return self.net.logits(agent.genome, env, state, actions, subgoal_mask)`
   (was building a spec-sized input vector per action).

The reward law, step spending, two-parent crossover + gaussian mutation, subagent
recursion, death of non-reproducers and the curriculum stages are byte-identical
to `origin/dynamic-nodes`.

**Out of scope / known gap:** `evolution_trainer/harness.py` (and the Unit C arms
`mix_train.py`, `rl_train.py`, `compare_arms.py`, `arms_tests.py`) still bind the
old spec-sized `PolicyNet`. They are not part of this unit's architecture module
or its test command; the held-out harness migration belongs to the evaluation
unit. `curriculum.py` depends only on `EvolutionTrainer` and stays green
(`pytest -q evolution_trainer/curriculum_tests.py`: 7 passed).

## 5. Proof sketch

`evolution_trainer/size_invariant_tests.py` (6 checks, stdlib only):

* `genome_size_is_spec_invariant` - one length (1153) across `spec_minimal`,
  `spec_dynamic_group`, `spec_dynamic_axiom`, `spec_multi_step`,
  `spec_multi_step_heldout`, `spec_dynamic_group_deep` (arity 3) and an INLINE
  adhoc spec with 6 values and combine arity 5 (no padded cap).
* `same_genome_scores_every_spec` - one genome list, never rebuilt; finite logits
  on all 7 specs; the genome JSON is byte-identical before and after.
* `transfer_small_to_large_arity` - the same weights score arity-2 and arity-3
  actions (the exact case that was 0/30).
* `no_reallocation_when_node_count_grows` - the same genome scores a state before
  and after the node count grows inside one episode.
* `trained_genome_transfers` - a genome EVOLVED on `spec_dynamic_group` scores
  `spec_multi_step`, `spec_dynamic_group_deep` and the arity-5 adhoc spec.
* `logits_are_deterministic_and_sensitive` - reproducible, sensitive to a
  single-gene change, and target-conditioned by the subgoal mask.

Existing batteries stay green: `python -m evolution_trainer.tests` 13/13,
`pytest -q evolution_trainer/tests.py` 13 passed, `python -m dynamic_env.tests`
13/13, `pytest -q dynamic_env/tests.py` 13 passed.

## 6. Honest limits

* The feature hash can collide; the signed, magnitude-varied embedding keeps
  colliding names distinguishable on average, but a collision remains possible.
* One message-passing round is enough for the shipped graph depths; deeper
  reasoning is left to more rounds (a fixed change, no new parameters per spec).
* This unit proves the architecture invariant; it does not yet claim a trained
  transfer solve-rate. That experiment (2nd/3rd seed, held-out eval on the new
  genome) is the next unit and needs the held-out harness migrated to this net.
* No new third-party dependency: the module is standard library only, like
  `dynamic_env` and the rest of `evolution_trainer`.
