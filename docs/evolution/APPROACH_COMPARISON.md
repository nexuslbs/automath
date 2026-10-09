# Approach comparison for the dynamic-nodes environment (Unit C)

Unit C asks the operator's question: **which way of training an agent on the
dynamic-nodes environment is best** - pure reinforcement learning, evolution
with fixed weights, or a mix of both? This document fixes the three arms, the
fair-comparison contract and the decision rule. The measured results live in
`/opt/workspace/tmp/automath/dynamic-nodes/unit-C/EVIDENCE.md`.

## The three arms

| arm | runner | signal used |
| --- | --- | --- |
| `evolution` | `evolution_trainer.train` / `EvolutionTrainer` (Unit B, unchanged) | fixed-weight two-parent mixing + mutation, shaped reward |
| `rl` | `evolution_trainer.rl_train` | policy gradient (REINFORCE with baseline), shaped reward |
| `mix` | `evolution_trainer.mix_train` | RL warm-start, then fixed-weight evolution |

## The fair-comparison contract

`evolution_trainer/harness.py` is the shared surface. Every arm uses:

* the SAME network: `genome.PolicyNet` (one tanh hidden layer, `hidden=12`);
* the SAME input features: `features.state_features(env, state, mask)` +
  `ActionFeaturizer.featurize(state, action)`; only the spec DATA differs;
* the SAME shaped reward, byte-for-byte the Unit B shaping: per-step cost,
  once-per-objective intermediate reward, once-per-guard unlock bonus, subgoal
  reward and goal reward (the environment's own reward hook supplies the
  goal/step part);
* the SAME episode budget: `generations x population` episodes (RL consumes one
  episode per "individual slot" and one SGD step per generation; mix splits its
  generations between warm start and evolution);
* the SAME evaluation: a GREEDY harness episode (the reported `best_fitness`)
  plus `eval_episodes` stochastic episodes (the reported `solved/total`);
* the SAME seed.

Subagent recursion is an evolution-internal training device; it is disabled in
the shared evaluation so no arm is scored on an advantage it did not train.

## Why this mix

The mix is **RL warm-start of the founders, then fixed-weight evolution** (not
the other order). A short policy gradient is a dense, cheap signal: on this tiny
network a few hundred episodes already move the target-conditioned logits in the
rewarding direction, which gives evolution a useful basin to start from.
Evolution then does the discrete/compositional search over the action word
(two-parent weight mixing + mutation) that a policy gradient alone does not
explore well on short, sparsely-rewarded episodes. The budget is split, never
doubled: `warm_generations + evo_generations = generations`.

## Decision rule

Rank by: (1) solves the most specs, **including** `spec_multi_step`; (2) sample
efficiency (episodes to first solve / best generation); (3) wall time; (4)
robustness across seeds. If no arm solves `spec_dynamic_axiom` or
`spec_multi_step`, that is reported plainly - Unit C is approach-selection
evidence, and Unit D runs the full curriculum.
