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
