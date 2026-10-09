# Evolution pipeline: node extension, reward scaffolding, stochastic valuation

Branch `evolution` (from `new` @ `b78cc29c1843e5ee17c9abeb132cd91100c269fc`).
Units: Stage 1 (more semantic node kinds), Stage 2 (artificial reward/guidance
scaffolding), Stage 3 (remove scaffolding -> stochastic step-type valuation).

The four CORE node types are unchanged: `Zero`, `One`, `Change`, `Group`. All
new vocabulary is DERIVED, so the former 4 -> 3 -> 2 -> 1 reduction proof stays
literally true.

---

## Stage 1 - more semantic node kinds (derived)

### 1. Design

The operator's core directive is to add more nodes so agents can solve more
complex scenarios. Adding new node *classes* would break the reduction theorem
(the whole point of the `new` branch), so this unit adds new semantic node
KINDS instead:

* A kind `K` with reserved integer tag `t` and arity `k` is the core node
  `Group(nat(t), (a1..ak))` - exactly the same construction the existing
  `ADD`/`LT`/... operators already use.
* Each kind gets a BUILD ACTION (`MakeAdd`, ...) that consumes `k` already
  built operands and pushes that `Group`. The action vocabulary grows by 18
  kinds; the node-class count does not.
* New axiom tags introduced: `DIV=14`, `GT=15`, `LE=16`, `GE=17`, `MOD=18`,
  `NEG=19`. Existing tags are reused for `ADD/SUB/MUL/EQ/LT/AND/OR/NOT/IF/SEQ`.

The environment is `EvoEnv`, the same stack machine as `MinimalEnv` over the
extended action vocabulary. Each action pops `arity` nodes and pushes one, so a
target tree of `N` nodes is reached in exactly `N` actions and the unique build
is still the post-order word (`sem_plan`).

### 2. New kinds, build actions, arities and their reduction onto the 4 cores

`nat(t)` is the core numeral `One`/`Change` chain; every mapping below uses
only `Zero`, `One`, `Change`, `Group`.

| kind | action | tag | arity | explicit core encoding |
| ---- | ------ | --- | ----: | ---------------------- |
| arithmetic | `MakeAdd` | 1 | 2 | `Group(nat(1),(a,b))` |
| arithmetic | `MakeSub` | 2 | 2 | `Group(nat(2),(a,b))` |
| arithmetic | `MakeMul` | 3 | 2 | `Group(nat(3),(a,b))` |
| arithmetic | `MakeDiv` | 14 | 2 | `Group(nat(14),(a,b))` |
| arithmetic | `MakeMod` | 18 | 2 | `Group(nat(18),(a,b))` |
| arithmetic | `MakeNeg` | 19 | 1 | `Group(nat(19),(a,))` |
| comparison | `MakeEq` | 5 | 2 | `Group(nat(5),(a,b))` |
| comparison | `MakeLt` | 4 | 2 | `Group(nat(4),(a,b))` |
| comparison | `MakeGt` | 15 | 2 | `Group(nat(15),(a,b))` |
| comparison | `MakeLe` | 16 | 2 | `Group(nat(16),(a,b))` |
| comparison | `MakeGe` | 17 | 2 | `Group(nat(17),(a,b))` |
| logic | `MakeAnd` | 7 | 2 | `Group(nat(7),(a,b))` |
| logic | `MakeOr` | 8 | 2 | `Group(nat(8),(a,b))` |
| logic | `MakeNot` | 6 | 1 | `Group(nat(6),(a,))` |
| control | `MakeIf` | 9 | 3 | `Group(nat(9),(c,t,e))` (lazy) |
| sequencing | `MakeSeq2` | 10 | 2 | `Group(nat(10),(a,b))` |
| sequencing | `MakeSeq3` | 10 | 3 | `Group(nat(10),(a,b,c))` |
| sequencing | `MakeSeq4` | 10 | 4 | `Group(nat(10),(a,b,c,d))` |

**Reduction theorem (kept).** Each new kind is a `Group`, and every `Group` is
one of the four core classes; `nat` only emits `Zero`/`One`/`Change`. The test
`semantic_reduction_core` asserts that all 17 distinct semantic trees in the
suite use at most the 4 core classes and that every scenario target is still
buildable by the core 4-action plan (`PushZero/PushOne/MakeChange/MakeGroupK`).
`node_type_count` stays 4; `to_min3`/`to_min2`/`to_cell` therefore apply to the
new kinds unchanged (an identity structural decomposition for the kind itself).

Honest boundary: `MakeSeq4` has 4 items, so a core build would need a
`MakeGroup5` action that the minimal core vocabulary does not ship. The NODE is
still a legal core `Group`; only that one build action goes beyond the core
action set. This is stated rather than hidden, and no test target depends on it.

### 3. Complex scenario suite (requires the new kinds)

`evolution.scenarios()` is deterministic and single-solution where possible:

| scenario | expression | expected |
| -------- | ---------- | -------- |
| `e_add_mul` | `add(1, mul(1, 3))` | 4 |
| `e_div` | `div(mul(3,3), 3)` | 3 |
| `e_mod` | `mod(7,3)` | 1 |
| `e_neg` | `neg(3)` | -3 |
| `e_cmp` | `and(lt(1,3), ge(3,3))` | true |
| `e_eq_gt` | `and(eq(2,2), not(gt(1,2)))` | true |
| `e_logic` | `or(not(0), and(1,0))` | true |
| `e_branch` | `if(lt(1,3), add(1,1), mul(0,3))` | 2 |
| `e_nested` | `mul(add(1,1), sub(3,1))` | 4 |
| `e_seq` | `seq(add(1,1), not(0), mul(3,1))` | (2, true, 3) |
| `e_seq4` | `seq(add(2,2), div(8,2))` | (4, 4) |

`UNIQUE_SEM_TARGETS` adds six small targets whose restricted alphabets make an
exhaustive single-solution proof cheap. `semantic_new_initial_states` verifies
all 114 non-trivial prefixes of the 11 scenario words reach the same target.

### 4. Test command

```sh
cd /opt/automath/repo
/opt/automath/venv/bin/python -m new_approach.tests            # prior 11/11, unchanged
/opt/automath/venv/bin/python -m new_approach.evolution_tests  # prior 11 + 9 new
```

The extended runner imports the prior `CHECKS` unchanged, so a regression in
the original semantics fails the extended run too.

---

## Stage 2 - artificial reward/guidance scaffolding

### 5. Design

Training-only scaffolding, no planner action-word supervision
(`use_demo=False`):

* After every step the environment emits a REWARD NODE into the observation:
  `Group(nat(20), (One(),))` for GOOD, `Group(nat(20), (Zero(),))` for BAD.
  `read_reward` turns it into `+0.5` / `-0.5`.
* A state on the unique optimal trajectory has exactly one intended step; that
  action earns GOOD, every other action earns BAD. A state off the trajectory
  has no intended step, so every action there earns BAD. Terminal success adds
  `+1.0`.
* The agent (`GuidedQAgent`) is target-conditioned tabular Q-learning; its ONLY
  learning signal is the scalar read from the reward node plus the goal reward.
  No action word from the planner is ever fed to the learner.
* Evaluation removes the reward node entirely and acts greedily on the learned
  Q table (`evaluate`), so the learned policy is tested without scaffolding.

Command (core U2 set and the extended complex-scenario set):

```sh
/opt/automath/venv/bin/python -m new_approach.evolution_run --stage baseline
/opt/automath/venv/bin/python -m new_approach.evolution_run --stage 2  --episodes 600
/opt/automath/venv/bin/python -m new_approach.evolution_run --stage 2e --episodes 200
```

Measured convergence, final success and validation are in
`/opt/workspace/tmp/automath/evolution/unit-B/EVIDENCE.md` (Stage 2 section).

---

## Stage 3 - remove the artificial steps: stochastic step-type valuation

### 6. Design

The reward-node scaffolding is deleted. The agent now carries a RANDOM
per-action-type preference, a priori, and that distribution is shaped by the
REAL reward signal observed during its own runs:

* `pref[a]` is drawn from `N(0, sigma)` per action type at construction (seeded).
* Action choice is greedy/epsilon-greedy over `q(s,a) + beta * pref[a]`; the
  preference is ALSO the fallback policy for a `(goal, state)` with no Q entry,
  so the step-type valuation generalises across states the tabular Q never saw.
* After each episode `update_pref` shapes the preference from the real reward
  only (never from a guidance node):
  - `td`: bump each step type by its reward prediction error
    `r + gamma*max Q(s') - Q(s,a)`;
  - `reinforce`: bump it by the discounted return-to-go minus a PER-STEP-TYPE
    running baseline (so no global sign saturates).
* Reward is the ordinary environment reward: `+1.0` at the goal, `-0.01` per
  step. No reward node is ever constructed on this path.

Command:

```sh
/opt/automath/venv/bin/python -m new_approach.evolution_run --stage 3  --episodes 60
/opt/automath/venv/bin/python -m new_approach.evolution_run --stage 3  --episodes 2000
/opt/automath/venv/bin/python -m new_approach.evolution_run --stage 3e --episodes 200
```

Headline measured outcomes (config `pref_mode=td, pref_lr=0.2, beta=1.0,
sigma=0.5`; raw output in the evidence file):

| run | training | validation |
| --- | -------- | ---------- |
| Stage 2 (guidance), 60 ep | 10/10 | 33/33 |
| Stage 2 extended, 200 ep | 11/11 | 114/114 |
| Stage 3 (valuation), 60 ep, seed 20261009 | 5/10 | 14/33 |
| Stage 3, 2000 ep, seed 20261009 | 8/10 | 24/33 |
| Stage 3, 2000 ep, 6-seed range | 1-8/10 | 10-26/33 (mean 21.7, median 24) |
| Stage 3 extended, 200 ep | 1/11 | 16/114 |
| supervised planner | 10/10 | 33/33 |
| pure-reward control, 60 ep | - | 14/33 |
| pure-reward control, 2000 ep | - | 19/33 |

Honest reading: the stochastic step-type valuation DOES converge on the core
U2 set (a rising success/reward curve that stabilises), and at the longer
budget it exceeds the pure-reward control on average (median 24/33 vs 19/33),
but it is seed-sensitive (one of six seeds collapses to 10/33) and it does NOT
reach the supervisor or Stage 2. On the long-horizon extended complex-scenario
suite (21 actions, plans up to 17 steps) it does NOT converge at this budget
(1/11 training, 16/114 validation), which is reported as a negative result.
Using `reinforce` instead of `td`, or context-conditioned preferences, did not
improve these numbers.

---

# Unit C - the evolutionary loop, multi-state total budget, persistence

Unit C turns the Stage 1-3 single-agent machinery into a POPULATION that evolves
over generations, evaluated on a BUNDLE of states under ONE TOTAL step budget,
with impossible goals in the mix and a persisted, warm-startable checkpoint.

Modules: `new_approach/evolution_bundle.py` (bundle + total-budget controller),
`new_approach/evolution_population.py` (genome, selection, reproduction,
persistence), `new_approach/evolution_loop.py` (detached long run),
`new_approach/evolution_c_tests.py` (12 deterministic checks). Stage 3's
`EvolutionAgent` is unchanged except for an optional `pref_init` seed and the
`switch_patience` instinct (both default off, so the Unit B results reproduce).

## 1. Genome (the heritable BASIC INSTINCT)

| gene | meaning |
| ---- | ------- |
| `pref[21]` | per-step-type preference over the extended action alphabet (the Stage 3 instinct), seeded into the agent's `pref` via `pref_init` |
| `state_pref[S]` | per-state preference over the bundle's `S` states: which state to start first and which to prefer on a switch (the BEST STATE-CHOOSER instinct) |
| `epsilon` | exploration temperament (initial epsilon, clipped to `[epsilon_min, epsilon_max]`) |
| `switch_patience` | stagnation window (steps without reward progress) before a mid-state switch, `[2, 20]` |

## 2. Multi-state TOTAL-budget mode (`evolution_bundle.py`)

A `Bundle` is a tuple of `BundleState`s plus ONE `total_budget` shared by all of
them. Feasibility is DERIVED, not declared: a target is feasible iff every
`Group` in it has a build action in `SEM_BY_KEY` (`target_feasible`). The demo
bundle has 5 feasible states (optimal words of 2/3/5/7/11 actions) and 2
structurally impossible ones:

| state | target | reason impossible |
| ----- | ------ | ----------------- |
| `s4` | `Group(nat(99),(1,0))` | no build action has tag=99 arity=2 |
| `s6` | `Group(nat(10),(1,0,1,0,1))` | no build action has tag=10 arity=5 |

`run_bundle` is the controller:

* start state = highest `state_pref`, then the rest of the instinct order;
* ONE shared counter `remaining`; every step decrements it and is charged
  `-0.01`; solving a feasible state adds `+1.0` (the solving step earns the goal
  reward INSTEAD of the step penalty). The sum of per-state rewards always
  equals the run total;
* mid-state switch triggers (both carry an explicit reason):
  - **stagnation**: `switch_patience` steps without structural-progress
    improvement (`size(top)/size(target)`);
  - **budget-ratio**: the state has spent `BUDGET_RATIO_TRIGGER = 0.5` of the
    budget it had when entered.
* a per-sweep rule: within one sweep every unsolved state is tried once (ranked
  by `state_pref`) before an abandoned state may be revisited, so an impossible
  state cannot immediately eat the budget again.

Fitness = total reward of the greedy evaluation. Two extra scores are tracked
for selection: `solver_score` = sum over feasible states of `1.0` if solved else
the best partial fraction; `chooser_score` = `solved_feasible - impossible_steps
/ total_steps`.

## 3. Population evolution and the selection operator

Generation step (`evolve_one_generation`): every genome is instantiated as an
`EvolutionAgent` seeded with its `pref` and `switch_patience`, trained for
`episodes` full-bundle episodes under the shared budget, then evaluated greedily
(stable per-genome seed, so an elite keeps its measured fitness and best fitness
is monotone under elitism). Reproduction:

1. **ELITISM** - the top `elites` genomes by fitness survive unchanged.
2. **THRESHOLD** - every agent with `reward >= threshold_frac * best_reward`
   joins the reproduction pool.
3. **BEST CHOOSERS + BEST SOLVERS** - the top `top_k` by `chooser_score` and the
   top `top_k` by `solver_score` are added to the pool, so the two instincts can
   be mixed.
4. **CROSSOVER + MUTATION** - a chooser and a solver are paired; each child
   inherits each gene from A or B (uniform per-gene) and is then perturbed
   (`mutation_rate`, `mutation_sigma`, clipped). The cross detail
   (`pref_from_a/b`, `state_from_a/b`, `epsilon_from`, `patience_from`) is
   recorded for every child.
5. **REPLACEMENT** - elites + offspring fill the fixed population size.

## 4. Persistence and warm start

`save_generation` writes three JSON files (atomic tmp+rename) under the
checkpoint dir: `checkpoint.json` (population + generation + best), `history.json`
(all generation records) and `state_results.json` (the best agent's per-state
table and trace). `load_checkpoint` + `restore_population` reproduce the genes
exactly; `evolution_loop --resume` and the warm-start phase of `--stage c` load
generation `G` and continue at `G+1`.

## 5. Long run and convergence target

`evolution_loop --config <cfg>` evolves continuously. It ignores `SIGHUP`,
exits cleanly on SIGTERM/SIGINT after a final checkpoint, writes a timestamped
progress snapshot every `heartbeat_secs` AND every `heartbeat_generations` to
`progress/`, and prints one `PROGRESS` line (the detached shell appends stdout to
`longrun.log`). `config/evolution_longrun.json` is the long-run config. Stop
conditions: a generation cap, a wall-clock cap, or a signal.

The convergence TARGET is an optimal agent that solves every feasible state
inside the shared budget, skips the impossible ones fast and uses minimal steps.
The bounded demo measures how far the loop gets and reports it honestly: the
population MEAN fitness rises (e.g. `+0.347 -> +0.936` over 8 generations at
pop=12/episodes=60/budget=100) and elitism holds the best at `+1.020`
(2 of 5 feasible states); it does NOT reach 5/5 at this agent/budget, which is
consistent with the Unit B Stage 3 result that the reward-only tabular agent
does not solve the long-horizon scenarios. The long run keeps evolving.

## 6. Unit C commands

```sh
/opt/automath/venv/bin/python -m new_approach.evolution_c_tests        # 12 checks
/opt/automath/venv/bin/python -m new_approach.evolution_run --stage c \
    --pop 12 --gens 8 --demo-episodes 60 --total-budget 100 \
    --ckpt-dir /opt/automath/tmp/evolution/checkpoints_demo
/opt/automath/venv/bin/python -m new_approach.evolution_loop \
    --config config/evolution_longrun.json
```

