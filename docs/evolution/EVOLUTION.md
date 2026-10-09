# The evolutionary process (Unit B)

Branch `dynamic-nodes`. Module `evolution_trainer/`, built on the generic
environment of `dynamic_env/` (`docs/evolution/DYNAMIC_NODES.md`,
`docs/evolution/STRICT_LAWS.md`). This document specifies the OPERATOR MANDATE of
thread 4328 as implemented and tested: an evolutionary process that rewards
agents along the way, lets the best two spend reward together to create a child
that shares BOTH parents' weights, lets agents recursively spawn subagents on
simpler subgoals, charges a cost for every step (so unproductive agents die
childless), mutates, and checkpoints the winners.

Everything is standard library only, exactly like `dynamic_env`: the environment
is stdlib-only and the trainer keeps that boundary. No new dependency is added.

---

## 1. Genome = weights of a TARGET-CONDITIONED policy network

A genome is the flat weight vector of a one-hidden-layer `tanh` MLP
(`evolution_trainer/genome.py`). The network scores EVERY legal action; the
softmax over those scores is the action distribution.

```
input  = state_features(env, state, subgoal_mask)   # len = 3*|objectives| + |node_types| + 3
       + action_features(state, action)             # len = 4 + |vocab| + 1 + 2*max_arity
hidden = tanh(W1 x + b1),  hidden default 12
output = W2 hidden + b2                             # ONE logit per action
```

* `state_features` starts from `env.state_vector(state)`, which ALWAYS exposes
  the elementar 1/0 objective flags AND the target assignment (the Unit A
  interface), normalises the count/step/fingerprint tail, and appends the
  `subgoal_mask`. The mask is 1 for objectives THIS episode must reach and 0 for
  already-satisfied objectives that are pinned, so the SAME genome can be
  conditioned on a full goal or a simpler subgoal.
* `action_features` (`evolution_trainer/features.py`) is derived from spec DATA
  only: a 4-way kind one-hot (`build`/`combine`/`set`/`clear`), a one-hot over
  the spec's own role-prefixed id vocabulary (`ax:`, `tag:`, `obj:`), the arity,
  and per operand slot the operand's normalised position in the current
  `evaluable_operands` plus a hashed node-type bucket. No node name, axiom id or
  spec id is hardcoded; the test `action_space_is_data` enforces that.

Because the target is part of the state vector and the mask is an explicit
input, the policy is target-conditioned from the first observation. The
predecessor verdict (thread 4327) was that a flat per-target preference vector is
the ceiling; this genome is the recommended fix.

## 2. Reward law (reward granted ALONG THE WAY, anti-farming)

For every transition the trainer charges `step_cost` and grants:

| term | when | value (default) |
| ---- | ---- | --------------- |
| step cost | every action | `-0.05` |
| intermediate reward | an elementar objective node moves to its target value, ONCE per objective per episode | `+0.30` |
| guard bonus | `set:<objective>` becomes newly legal for an objective that DECLARES a guard, ONCE per guarded objective per episode | `+0.20` |
| subgoal reward | the episode's active subgoal mask is fully satisfied (and the full goal is not) | `+0.50` |
| final goal | the full objective conjunction is reached | `+1.00` |

The two `credited` sets (`credited`, `unlocked_credited`) are the anti-farming
rule: toggling an objective away and back, or clearing and re-setting a guarded
objective, cannot re-earn the same progress reward. The reward hierarchy is
asserted: `goal_reward > subgoal_reward > 0` and
`goal_reward > intermediate_reward > 0`, `step_cost > 0`. In the shipped specs
the best reachable episode return is exactly the analytic maximum
(`spec_minimal` 1.25, `spec_dynamic_group` 1.40, `spec_dynamic_axiom` 1.65), which
is the raw evidence that no farming path inflates fitness.

## 3. Population, budgets and step spending

An `Agent` carries its genome, its `parents`, a spendable `budget` (the running
sum of episode returns), a per-generation `fitness` (the best episode return of
the generation), cumulative `solved`/`total_steps`, its `depth`, and the
mutation magnitude it was born with.

A generation is one evaluation pass over the population. Each agent plays
`episodes_per_agent` episodes from the reset state with the full goal mask. The
step cost is what makes evolution favour short trajectories: an agent that
reaches more reward in fewer steps accumulates a larger budget, and budget is
what buys reproduction and subagents. An agent that never earns enough budget
cannot reproduce and DIES childless (its id is recorded in the generation
record's `deaths`).

## 4. Reproduction by two best agents, sharing BOTH weights

Selection ranks the population by `(-fitness, -budget, id)`. The top
`elite_frac` are elites and survive; the top `parent_frac` form the parent pool.
Every pair in the pool is tried in rank order. If BOTH parents can each pay half
the `reproduction_fee` from their own budget, they pay it TOGETHER and produce
ONE child:

```
child_gene = a_gene          with probability select_prob/2
           | b_gene          with probability select_prob/2
           | w*a + (1-w)*b   otherwise, w = uniform(0,1) per gene (or mix_alpha fixed)
child_gene += N(0, mutation_sigma)   with probability mutation_rate
```

Both branches keep a blended gene inside `[min(a,b), max(a,b)]`, so the noiseless
blend is in the two-parent convex hull; mutation is added on top. This is the
literal meaning of "shares BOTH agents' existing weights". Every birth records
the parents, the number of blended and selected genes, the mutation magnitude
`|delta|`, and `hull_ok` (the convex-hull assertion computed at reproduction
time). If the population falls below `population_size`, the shortfall is filled
with ASEXUAL mutated copies of the best elite (mutation keeps the search alive).
Parents that reproduce but are not elites are not carried over: they die after
passing on their weights.

## 5. Subagent recursion

During an episode, after a helpful step (an objective progress flip or a guarded
objective becoming settable) the agent may spawn a SUBAGENT when all of:

* recursion depth is below `subagent_depth`;
* the agent can pay `subagent_spawn_fee`;
* fewer than `max_subagents_per_episode` spawns happened this episode;
* there remains at least one objective not at its target.

The subagent targets the REMAINING subgoal: its `subgoal_mask` is 1 exactly for
the objectives still off target. It is the child of two eligible agents (the
spawner plus the best other agent that can pay half the fee) or, when no second
parent is affordable, a mutated copy of the spawner alone. The subagent runs the
SAME loop from the state the parent reached, earns its own rewards, and may
itself spawn a subagent (depth-limited). A capped share of the subagent's return
credits back to the spawning agent's budget:

```
credit = min( max(0, subagent_return) * subagent_credit_rate, subagent_credit_cap )
```

The unit test `subagent_recursion` uses a three-objective spec and a scripted
parent to force depth 1 AND depth 2, asserts the depth limit is never exceeded,
and asserts the credits flow back and respect the cap.

## 6. Persistence

Per generation, under `checkpoint_dir`:

* `best_gen_XXXX.json` - the best agent of that generation as a genome JSON
  (id, origin, parents, depth, mutation magnitude, fitness, budget, genome);
* `best_agents_persist.json` - accumulates one winner per generation plus
  `all_time_best` (the predecessor's shape: a list of agent records with a
  `genome`);
* `checkpoint.json` - the whole population (warm start) and the config.

Under `history_dir`:

* `history.csv` - the GENERATION TABLE, one row per evaluated agent:
  `generation,agent_id,origin,parents,fitness,budget,steps,total_steps,solved,depth,mutation_magnitude,alive,note`;
* `reward_trajectory.csv` - the reward curve:
  `generation,population,best_agent,best_fitness,mean_fitness,solved,best_steps,births,deaths,subagents`.

Every run is reproducible: a single seeded `random.Random` drives all sampling,
mixing and mutation; `deterministic_run` asserts two runs of the same seed give
byte-identical generation histories.

## 7. Commands

```sh
# deterministic tests (13 checks) and the pytest mirror
/opt/automath/venv/bin/python -m evolution_trainer.tests
/opt/automath/venv/bin/python -m pytest -q evolution_trainer/tests.py

# short deterministic demo on minimal + dynamic_group + dynamic_axiom
/opt/automath/venv/bin/python -m evolution_trainer.demo \
    --generations 12 --population 12 --out /opt/automath/tmp/unit-B/out/demo

# configurable short training run (every EvoConfig knob is a flag)
/opt/automath/venv/bin/python -m evolution_trainer.train \
    --spec spec_minimal --spec spec_dynamic_group --spec spec_dynamic_axiom \
    --generations 30 --population 16 --step-cost 0.05 \
    --reproduction-fee 1.0 --mutation-rate 0.15 --subagent-depth 2 \
    --out /opt/automath/tmp/unit-B/out/train
# a JSON config can supply the same values: --json-config config/evolution_dynamic.json
```

Long curriculum runs are explicitly NOT this unit: this unit is short, seeded and
deterministic; the long-run contract belongs to later units.

## 8. Boundary and honest limits

* The genome is spec-specific and target-conditioned WITHIN a spec. Generalising
  one genome across specs (a spec-invariant action encoder) is not claimed here;
  the action vocabulary is rebuilt per spec from spec DATA.
* A subgoal is a MASK over the same spec (the environment law is untouched, per
  `STRICT_LAWS.md` L2), not a derived spec; already-satisfied objectives are
  pinned so the subgoal is genuinely simpler.
* The policy is a linear scorer over derived features, not a deep network; on
  this 2 vCPU / 3.8 GB host that is deliberate, and the shipped specs are solved
  by evolution without gradients.
* The trainer is deterministic only for a fixed seed, spec and config; changing
  the population size changes the RNG stream (as expected for a seeded GA).
