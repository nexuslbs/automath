# Unit B (Phase A): the designed shaped reward for the semi-evolutionary path

Branch `semi-evo` (from `origin/evolution` @
`1915b5a2d8d45c3963cd35a55c64dd77f03184d3`).

This is the PRIMARY deliverable of Unit B (Phase A): a reward scheme designed,
documented and implemented BEFORE the long run, with the reward granted ALONG
THE WAY.  The scheme is implemented in `new_approach/evolution_rewards.py` and
consumed by `new_approach/evolution_semi.py` (learning + selection) and
`new_approach/evolution_semi_loop.py` (the detached Phase A run).

It is explicitly NOT the Stage 2 scaffolding: there is **no artificial
GOOD/BAD reward node**, no `T_REWARD` observation, and no planner action-word
supervision anywhere on this path.  The only signals are four numeric terms,
each a function of the real environment state and the real goal.

---

## 1. What is rewarded, when, with what value, and why

| # | Component | When | Value | Why |
| - | --------- | ---- | ----- | --- |
| 1 | **per-step cost** | every action, always | `-0.05` | fewer actions is better; the original `-0.01` was too weak against a long budget |
| 2 | **PBRS progress shaping** | every action | `F(s,a,s') = gamma*Phi(s') - Phi(s)` with `gamma = 0.99`, `Phi(s) = 0.5 * goal_similarity(s)` | Ng, Harada & Russell (1999) potential-based shaping gives a dense, policy-invariant progress signal without inventing a fake step reward |
| 3 | **sub-goal bonus** | an action produces a top node that is a PROPER, non-goal, non-trivial sub-structure of the target and the matched target size strictly grows | `+0.10` | rewards completing a real intermediate milestone (e.g. `add(1,1)` on the way to `mul(add(1,1), sub(3,1))`) |
| 4 | **final goal** | a feasible state is solved | `+1.0` | the unchanged real objective |
| 5 | **per-episode cap** | on the sum of POSITIVE `(shaping + bonus)` | `<= 0.9` | the true `+1.0` goal always dominates the shaping; negative shaping is never capped because it only punishes unproductive moves |

All four terms are ADDITIVE on a step: a solving step is charged the step cost,
earns the shaping and any bonus, and earns `+1.0` at the same time.  (The
existing Unit C controller replaced the step penalty with the goal reward; the
designed scheme deliberately charges both, per term 1 + term 4 above.)

### Why this shape

* The Unit C / Stage 3 signal is sparse (`+1.0` per solved state, `-0.01` per
  step) and the pure evolution loop converged to a local optimum of `2/5`
  feasible states.  Dense progress shaping plus milestone bonuses give the
  population a usable gradient early, while the `+1.0` goal (and the cap that
  keeps shaping below it) keeps the true objective dominant.
* PBRS is the standard way to add dense shaping without changing the optimal
  policy: for the undiscounted / properly discounted case the shaping telescopes
  to a potential difference, so it cannot create a spurious optimum that beats
  the goal.
* The sub-goal bonus is the one non-potential term.  It is deliberately small
  (`+0.10`, at most a handful per episode) and capped, so it biases exploration
  toward real sub-structures without ever outweighing the goal.

---

## 2. `goal_similarity` - the exact formula

The stack TOP node is the unit of measurement, because every build action pops
its operands and pushes exactly one new top node.

```
size(x)          = number of nodes in tree x        (new_approach.nodes.size)
match_count(t,c) = number of nodes of the largest target sub-structure of t
                   that is canonically EQUAL to c; size(t) when c IS t; 0 otherwise
n_target         = size(target)
n_top            = size(top)                         (0 when the stack is empty)

size_prox  = 0.0                                     if n_top == 0
           = 1 - |n_top - n_target| / max(n_top, n_target)   otherwise

match_frac = match_count(target, top) / n_target     (0.0 if n_target == 0)

goal_similarity(s) = 0.5 * size_prox + 0.5 * match_frac        in [0, 1]
Phi(s)             = 0.5 * goal_similarity(s)                  in [0, 0.5]
```

* `size_prox` is the size-proximity to the target node count.
* `match_frac` is the prefix/sub-structure match fraction: the share of the
  target's nodes covered by the largest target sub-structure equal to the top.
* `goal_similarity` reaches `1.0` exactly when the top node IS the target
  (`match_count = n_target`, `size_prox = 1`).

Worked example, `target = mul(add(1,1), sub(3,1))` (`n_target = 15`), after
building `add(1,1)` (its subtree has `size = 4`):

```
size_prox  = 1 - |4 - 15| / 15 = 4/15 = 0.266667
match_frac = 4 / 15                 = 0.266667
goal_similarity = 0.5*4/15 + 0.5*4/15 = 4/15 = 0.266667
Phi = 0.133333
```

A `PushOne` from the empty stack gives `size_prox = 1 - 14/15 = 1/15` and
`match_frac = 1/15`, i.e. `goal_similarity = 1/15`.  An empty stack gives `0`.

### Sub-goal bonus predicate

```
is_proper_subgoal(target, top):
    top is not None
    AND canonical(top) != canonical(target)      # the goal earns +1.0, not +0.10
    AND size(top) >= 2                            # ignore bare leaf pushes
    AND size(top) <  size(target)
    AND match_count(target, top) == size(top)     # top IS a target sub-structure
```

The bonus is granted only when `match_count(target, new_top) >
match_count(target, old_top)`, so standing on the same sub-structure does not
farm the bonus, and the running `positive_used` accumulator enforces the cap.

---

## 3. Bound / policy-invariance note (honest reading)

PBRS alone (`F = gamma*Phi(s') - Phi(s)`) is policy-invariant for the discounted
objective and its per-step value is bounded by ~`0.5*(1+gamma)` because
`Phi` is in `[0, 0.5]`.  The sub-goal bonus and the per-episode cap are extra
terms that the operator's design mandates; they break exact policy invariance,
so the honest statement is:

* the shaping term is standard PBRS;
* the bonus is a small, capped exploration bias toward real sub-structures;
* the cap guarantees `sum(positive shaping + bonus) <= 0.9 < 1.0`, so one
  solved feasible state (`+1.0`) always outweighs all shaping earned in the
  episode, and the true objective dominates the search.

No artificial GOOD/BAD step node exists to reintroduce Stage 2 behaviour.

---

## 4. How the scheme drives the semi-evolution

* **Learning**: `evolution_semi.train_genome_shaped` runs
  `run_bundle_shaped`, which is the Unit C multi-state total-budget controller
  (one shared budget, stagnation + budget-ratio switching, per-sweep rule) with
  the per-step reward replaced by the four terms above.  The agent's tabular
  Q-learning and its per-step-type preference are both updated from the shaped
  return.
* **Selection**: `evolution_semi.evolve_one_generation_shaped` ranks genomes by
  `fitness = shaped_episode_return + 0.5 * solved_rate`; parents are the
  reward-threshold survivors (`threshold_frac * best`, with a margin fallback
  for non-positive returns), plus the best state-choosers and best solvers, and
  the top `elites` survive unchanged.
* **Reproduction**: offspring are BLX-alpha blends (`use_blx=true`, Unit E) or
  uniform per-gene crossover (Unit C) of the two parents' instinct genes,
  followed by bounded gaussian mutation.  Several agents evolve across
  generations; the fittest are automatically selected over time.
* **Everything else** is reused unchanged: the 4 core node types
  (`Zero`/`One`/`Change` + dynamic `Group`), the 18-entry `REDUCTION_TABLE`
  semantic kinds, the 11-target complex suite and the 114 new-initial-state
  validation.

---

## 5. Phase A commands

```sh
# deterministic unit tests for the scheme + generation machinery
/opt/automath/venv/bin/python -m new_approach.evolution_semi_tests

# the detached long run (auto-stops on plateau / generation cap / wall clock)
mkdir -p /opt/automath/tmp/semi-evo/checkpoints /opt/automath/tmp/semi-evo/progress
cd /opt/automath/repo
setsid nohup /opt/automath/venv/bin/python -m new_approach.evolution_semi_loop \
    --config config/evolution_semi.json \
    < /dev/null >> /opt/automath/tmp/semi-evo/phaseA.log 2>&1 &
echo $!
```

Validation cadence inside the loop: CORE 33-case every
`validation_every = 100` generations and EXTENDED 114-case every
`validation_ext_every = 500` generations (both train a fresh agent seeded with
the best genome's instinct under the shaped reward, then validate on the SAME
cases as Stage 2/3), plus the greedy multi-state bundle report per-state from
the generation's own evaluation.
