# U2 EVIDENCE: bare-minimum training + generalization validation (branch `new`)

Role: dsh developer. Project: automath. Repo: `nexuslbs/automath`, branch `new`.
Automath host: the FIRST host stanza in `/opt/omni/data/ssh/config`, referred to
below as `<automath-host>` (the IP is never printed). Remote clone
`/opt/automath/repo`, venv `/opt/automath/venv` (Python 3.14.4), tmp
`/opt/automath/tmp`; workstation clone `/opt/workspace/tmp/automath/repo`.

This unit continues U1 (branch-point `746022d`, U1 HEAD `af95087`); U1 is NOT
redone. The environment seam proven by U1 is used as-is.

## 1. Deliverable status

| # | deliverable | state |
| - | ----------- | ----- |
| 1 | bare-minimum training set + deterministic training (fixed seed) | DONE |
| 2 | generalization validation over NEW initial states, per-case traces | DONE |
| 3 | simple -> complex curriculum using the dynamic `Group` node | DONE |
| 4 | structured failure feedback + per-case wall times (sub-second) | DONE |
| 5 | honest verdict (reduction / generalization / diagnosis) | THIS FILE |
| 6 | commit + push on `new`; raw evidence to workspace path | DONE |

## 2. Branch and shas

```
branch-point (main HEAD)   : 746022d8d83230949b7b50fe153166e9c86fc5a6
U1 HEAD (inherited)        : af95087eee851d3b75e0a90a6a93effdce0ba67c
U2 implementation commit   : 2a229138b5751d678e250a18c48f54fe02f9e4c8
branch                     : new
```

The evidence-pack commit is the HEAD of branch `new` (run `git log -1`); the
workspace copy of this file records its exact sha after the push.

## 3. `git show --stat` (raw, implementation commit)

```
2a22913 new_approach: U2 deterministic agent, bare-minimum training and generalization validation
 new_approach/agent.py | 384 ++++++++++++++++++++++++++++++++++++++++++++
 new_approach/u2.py    | 434 ++++++++++++++++++++++++++++++++++++++++++++++++++
 2 files changed, 818 insertions(+)
```

## 4. ssh commands used (typed `ssh_run` / `ssh_copy_from`, host redacted)

No raw `ssh`/`scp` binary was invoked; all remote work went through the typed
ssh tools. `<automath-host>` is the first host stanza in
`/opt/omni/data/ssh/config`.

```sh
# move the remote run clone onto the U2 commit
cd /opt/automath/repo && git fetch origin -q && git checkout -B new origin/new -q
  && git rev-parse HEAD && git rev-parse origin/new && git status --porcelain

# canonical deterministic test suite (must stay 11/11)
/opt/automath/venv/bin/python -m new_approach.tests

# EXACT train command (also timed for RSS)
/usr/bin/time -v /opt/automath/venv/bin/python -m new_approach.u2 --mode train \
    --model /opt/automath/tmp/u2_agent.json --comp /opt/automath/tmp/u2_comp.json

# EXACT validation command (also timed for RSS)
/usr/bin/time -v /opt/automath/venv/bin/python -m new_approach.u2 --mode validate \
    --model /opt/automath/tmp/u2_agent.json --comp /opt/automath/tmp/u2_comp.json

# reproducibility: a second independent train run
/opt/automath/venv/bin/python -m new_approach.u2 --mode train \
    --model /opt/automath/tmp/u2_agent2.json --comp /opt/automath/tmp/u2_comp2.json
sha256sum /opt/automath/tmp/u2_agent.json /opt/automath/tmp/u2_agent2.json
sha256sum /opt/automath/tmp/u2_comp.json /opt/automath/tmp/u2_comp2.json
```

## 5. Design of the training procedure (documented, bounded, reproducible)

* **Bare-minimum episodes.** Every training episode starts from the EMPTY stack
  (the bare minimum) and its supervision is the deterministic planner's UNIQUE
  optimal action word to the fixed end state (`plan(target)` post-order;
  `bfs_plan` for the one `ExprGoal` pattern case). Supervision by the planner is
  the method the dispatch explicitly allows.
* **Agent.** `TabularAgent` is a target-conditioned tabular Q-learner keyed on
  `(goal_key, stack canonical)`, so the same stack can require different actions
  under different goals. It is seeded (`seed=20261009`); the Monte-Carlo return
  pass and the epsilon-greedy refinement episodes run in a fixed order, so
  `same seed -> same table` (verified by digest and by byte-identical saved
  tables).
* **Bounded.** 10 training goals, 60 Q-learning episodes per goal, target trees
  of at most 6 nodes, 6 build actions; no unbounded enumeration anywhere. The
  only exhaustive search remains the tiny U1 brute-force test.
* **Control.** A second agent (`use_demo=False`) trains with no planner
  supervision, as a pure reward-driven Q-learning control.
* **Compositional attempt.** `CompositionalAgent` learns a local construction
  rule from the same trajectories (which action produced which
  `(kind, arity)` node) and builds ANY target by a post-order recursion. This is
  the "use the dynamic node to group basic nodes" generalization attempt.

Curriculum (simplest -> complex):
`Zero`, `One` -> `Change(One)` (2), `Change(Change(One))` (3) ->
dynamic `Group`: `G(1;0)`, `G(1;1,1)`, `G(1;1,1,1)`, `G(1;C(1),C(1))` ->
complex `G(C(1);G(1;0))` (a Group whose tag is a Change and whose item is a
Group of basic nodes), plus the `ExprGoal` pattern `value==2`.

## 6. EXACT train command + raw output

Command:

```sh
/usr/bin/time -v /opt/automath/venv/bin/python -m new_approach.u2 --mode train \
    --model /opt/automath/tmp/u2_agent.json --comp /opt/automath/tmp/u2_comp.json
```

Raw stdout (`/opt/automath/tmp/u2_train.txt`):

```
========================================================================
U2 STEP 1: BARE-MINIMUM TRAIN
========================================================================
seed=20261009 episodes_per_problem=60
training episodes: 10 (each starts from the bare-minimum empty stack)
training goals:
  - zero         level=basic    key=T:0                      optimal_word=['PushZero']
  - one          level=basic    key=T:1                      optimal_word=['PushOne']
  - two          level=change   key=T:C(1)                   optimal_word=['PushOne', 'MakeChange']
  - three        level=change   key=T:C(C(1))                optimal_word=['PushOne', 'MakeChange', 'MakeChange']
  - g2           level=group    key=T:G(1;0)                 optimal_word=['PushOne', 'PushZero', 'MakeGroup2']
  - g3           level=group    key=T:G(1;1,1)               optimal_word=['PushOne', 'PushOne', 'PushOne', 'MakeGroup3']
  - g4           level=group    key=T:G(1;1,1,1)             optimal_word=['PushOne', 'PushOne', 'PushOne', 'PushOne', 'MakeGroup4']
  - g_add        level=group    key=T:G(1;C(1),C(1))         optimal_word=['PushOne', 'PushOne', 'MakeChange', 'PushOne', 'MakeChange', 'MakeGroup3']
  - g_grouped    level=complex  key=T:G(C(1);G(1;0))         optimal_word=['PushOne', 'MakeChange', 'PushOne', 'PushZero', 'MakeGroup2', 'MakeGroup2']
  - expr_two     level=pattern  key=E:value==2               optimal_word=['PushOne', 'MakeChange']
------------------------------------------------------------------------
train wall=0.040002s learned_states=293 digest=21f635406048eb44e0c731f909f5cc548a1d90486ad936d0bb0512db0f6912e5
determinism: same seed -> same table: True (21f635406048eb44e0c731f909f5cc548a1d90486ad936d0bb0512db0f6912e5 vs 21f635406048eb44e0c731f909f5cc548a1d90486ad936d0bb0512db0f6912e5)
compositional learned rules: {'Change/0': 'MakeChange', 'Group/1': 'MakeGroup2', 'Group/2': 'MakeGroup3', 'Group/3': 'MakeGroup4', 'One/0': 'PushOne', 'Zero/0': 'PushZero'} conflicts=0
saved model=/opt/automath/tmp/u2_agent.json comp=/opt/automath/tmp/u2_comp.json
TRAIN_EXIT=0
train total wall=0.089017s
```

Timing/RSS (`/usr/bin/time -v`, stderr):

```
	Command being timed: "/opt/automath/venv/bin/python -m new_approach.u2 --mode train --model /opt/automath/tmp/u2_agent.json --comp /opt/automath/tmp/u2_comp.json"
	Elapsed (wall clock) time (h:mm:ss or m:ss): 0:00.23
	Maximum resident set size (kbytes): 22000
	Exit status: 0
```

## 7. EXACT validation command + raw per-case traces

Command:

```sh
/usr/bin/time -v /opt/automath/venv/bin/python -m new_approach.u2 --mode validate \
    --model /opt/automath/tmp/u2_agent.json --comp /opt/automath/tmp/u2_comp.json
```

Raw stdout (`/opt/automath/tmp/u2_validate.txt`, 305 lines):

```
========================================================================
U2 STEP 2: GENERALIZATION VALIDATION
========================================================================
loaded model=/opt/automath/tmp/u2_agent.json digest=21f635406048eb44e0c731f909f5cc548a1d90486ad936d0bb0512db0f6912e5 learned_states=293
loaded compositional rules: {'Change/0': 'MakeChange', 'Group/1': 'MakeGroup2', 'Group/2': 'MakeGroup3', 'Group/3': 'MakeGroup4', 'One/0': 'PushOne', 'Zero/0': 'PushZero'} conflicts=0
------------------------------------------------------------------------
TRAIN-SET REACHABILITY (agent from the trained initial state)
[CASE] zero kind=train level=basic wall=0.000114s
  goal: fixed-state goal target=0
  start=[]
  actions=['PushZero']
  trace: [] -> [0]
  goal_satisfied=True reward=1.0000 wall=0.000114s PASS
[CASE] one kind=train level=basic wall=0.000087s
  goal: fixed-state goal target=1
  start=[]
  actions=['PushOne']
  trace: [] -> [1]
  goal_satisfied=True reward=1.0000 wall=0.000087s PASS
[CASE] two kind=train level=change wall=0.000091s
  goal: fixed-state goal target=C(1)
  start=[]
  actions=['PushOne', 'MakeChange']
  trace: [] -> [1] -> [C(1)]
  goal_satisfied=True reward=0.9900 wall=0.000091s PASS
[CASE] three kind=train level=change wall=0.000135s
  goal: fixed-state goal target=C(C(1))
  start=[]
  actions=['PushOne', 'MakeChange', 'MakeChange']
  trace: [] -> [1] -> [C(1)] -> [C(C(1))]
  goal_satisfied=True reward=0.9800 wall=0.000135s PASS
[CASE] g2 kind=train level=group wall=0.000130s
  goal: fixed-state goal target=G(1;0)
  start=[]
  actions=['PushOne', 'PushZero', 'MakeGroup2']
  trace: [] -> [1] -> [1, 0] -> [G(1;0)]
  goal_satisfied=True reward=0.9800 wall=0.000130s PASS
[CASE] g3 kind=train level=group wall=0.000171s
  goal: fixed-state goal target=G(1;1,1)
  start=[]
  actions=['PushOne', 'PushOne', 'PushOne', 'MakeGroup3']
  trace: [] -> [1] -> [1, 1] -> [1, 1, 1] -> [G(1;1,1)]
  goal_satisfied=True reward=0.9700 wall=0.000171s PASS
[CASE] g4 kind=train level=group wall=0.000202s
  goal: fixed-state goal target=G(1;1,1,1)
  start=[]
  actions=['PushOne', 'PushOne', 'PushOne', 'PushOne', 'MakeGroup4']
  trace: [] -> [1] -> [1, 1] -> [1, 1, 1] -> [1, 1, 1, 1] -> [G(1;1,1,1)]
  goal_satisfied=True reward=0.9600 wall=0.000202s PASS
[CASE] g_add kind=train level=group wall=0.000274s
  goal: fixed-state goal target=G(1;C(1),C(1))
  start=[]
  actions=['PushOne', 'PushOne', 'MakeChange', 'PushOne', 'MakeChange', 'MakeGroup3']
  trace: [] -> [1] -> [1, 1] -> [1, C(1)] -> [1, C(1), 1] -> [1, C(1), C(1)] -> [G(1;C(1),C(1))]
  goal_satisfied=True reward=0.9500 wall=0.000274s PASS
[CASE] g_grouped kind=train level=complex wall=0.000267s
  goal: fixed-state goal target=G(C(1);G(1;0))
  start=[]
  actions=['PushOne', 'MakeChange', 'PushOne', 'PushZero', 'MakeGroup2', 'MakeGroup2']
  trace: [] -> [1] -> [C(1)] -> [C(1), 1] -> [C(1), 1, 0] -> [C(1), G(1;0)] -> [G(C(1);G(1;0))]
  goal_satisfied=True reward=0.9500 wall=0.000267s PASS
[CASE] expr_two kind=train level=pattern wall=0.000168s
  goal: pattern goal value==2
  start=[]
  actions=['PushOne', 'MakeChange']
  trace: [] -> [1] -> [C(1)]
  goal_satisfied=True reward=0.9900 wall=0.000168s PASS
train reachability: 10/10 PASS
------------------------------------------------------------------------
GENERALIZATION VALIDATION (NEW initial states, same trained goals)
[CASE] zero/already-goal kind=validation level=basic wall=0.000021s
  goal: fixed-state goal target=0
  start=[0]
  actions=[]
  trace: [0]
  goal_satisfied=True reward=0.0000 wall=0.000021s PASS
[CASE] one/already-goal kind=validation level=basic wall=0.000020s
  goal: fixed-state goal target=1
  start=[1]
  actions=[]
  trace: [1]
  goal_satisfied=True reward=0.0000 wall=0.000020s PASS
[CASE] two/prefix1 kind=validation level=change wall=0.000054s
  goal: fixed-state goal target=C(1)
  start=[1]
  actions=['MakeChange']
  trace: [1] -> [C(1)]
  goal_satisfied=True reward=1.0000 wall=0.000054s PASS
[CASE] two/already-goal kind=validation level=change wall=0.000020s
  goal: fixed-state goal target=C(1)
  start=[C(1)]
  actions=[]
  trace: [C(1)]
  goal_satisfied=True reward=0.0000 wall=0.000020s PASS
[CASE] three/prefix1 kind=validation level=change wall=0.000090s
  goal: fixed-state goal target=C(C(1))
  start=[1]
  actions=['MakeChange', 'MakeChange']
  trace: [1] -> [C(1)] -> [C(C(1))]
  goal_satisfied=True reward=0.9900 wall=0.000090s PASS
[CASE] three/prefix2 kind=validation level=change wall=0.000055s
  goal: fixed-state goal target=C(C(1))
  start=[C(1)]
  actions=['MakeChange']
  trace: [C(1)] -> [C(C(1))]
  goal_satisfied=True reward=1.0000 wall=0.000055s PASS
[CASE] three/already-goal kind=validation level=change wall=0.000022s
  goal: fixed-state goal target=C(C(1))
  start=[C(C(1))]
  actions=[]
  trace: [C(C(1))]
  goal_satisfied=True reward=0.0000 wall=0.000022s PASS
[CASE] g2/prefix1 kind=validation level=group wall=0.000110s
  goal: fixed-state goal target=G(1;0)
  start=[1]
  actions=['PushZero', 'MakeGroup2']
  trace: [1] -> [1, 0] -> [G(1;0)]
  goal_satisfied=True reward=0.9900 wall=0.000110s PASS
[CASE] g2/prefix2 kind=validation level=group wall=0.000063s
  goal: fixed-state goal target=G(1;0)
  start=[1, 0]
  actions=['MakeGroup2']
  trace: [1, 0] -> [G(1;0)]
  goal_satisfied=True reward=1.0000 wall=0.000063s PASS
[CASE] g2/already-goal kind=validation level=group wall=0.000027s
  goal: fixed-state goal target=G(1;0)
  start=[G(1;0)]
  actions=[]
  trace: [G(1;0)]
  goal_satisfied=True reward=0.0000 wall=0.000027s PASS
[CASE] g3/prefix1 kind=validation level=group wall=0.000145s
  goal: fixed-state goal target=G(1;1,1)
  start=[1]
  actions=['PushOne', 'PushOne', 'MakeGroup3']
  trace: [1] -> [1, 1] -> [1, 1, 1] -> [G(1;1,1)]
  goal_satisfied=True reward=0.9800 wall=0.000145s PASS
[CASE] g3/prefix2 kind=validation level=group wall=0.000098s
  goal: fixed-state goal target=G(1;1,1)
  start=[1, 1]
  actions=['PushOne', 'MakeGroup3']
  trace: [1, 1] -> [1, 1, 1] -> [G(1;1,1)]
  goal_satisfied=True reward=0.9900 wall=0.000098s PASS
[CASE] g3/prefix3 kind=validation level=group wall=0.000068s
  goal: fixed-state goal target=G(1;1,1)
  start=[1, 1, 1]
  actions=['MakeGroup3']
  trace: [1, 1, 1] -> [G(1;1,1)]
  goal_satisfied=True reward=1.0000 wall=0.000068s PASS
[CASE] g3/already-goal kind=validation level=group wall=0.000028s
  goal: fixed-state goal target=G(1;1,1)
  start=[G(1;1,1)]
  actions=[]
  trace: [G(1;1,1)]
  goal_satisfied=True reward=0.0000 wall=0.000028s PASS
[CASE] g4/prefix1 kind=validation level=group wall=0.000168s
  goal: fixed-state goal target=G(1;1,1,1)
  start=[1]
  actions=['PushOne', 'PushOne', 'PushOne', 'MakeGroup4']
  trace: [1] -> [1, 1] -> [1, 1, 1] -> [1, 1, 1, 1] -> [G(1;1,1,1)]
  goal_satisfied=True reward=0.9700 wall=0.000168s PASS
[CASE] g4/prefix2 kind=validation level=group wall=0.000136s
  goal: fixed-state goal target=G(1;1,1,1)
  start=[1, 1]
  actions=['PushOne', 'PushOne', 'MakeGroup4']
  trace: [1, 1] -> [1, 1, 1] -> [1, 1, 1, 1] -> [G(1;1,1,1)]
  goal_satisfied=True reward=0.9800 wall=0.000136s PASS
[CASE] g4/prefix3 kind=validation level=group wall=0.000122s
  goal: fixed-state goal target=G(1;1,1,1)
  start=[1, 1, 1]
  actions=['PushOne', 'MakeGroup4']
  trace: [1, 1, 1] -> [1, 1, 1, 1] -> [G(1;1,1,1)]
  goal_satisfied=True reward=0.9900 wall=0.000122s PASS
[CASE] g4/prefix4 kind=validation level=group wall=0.000072s
  goal: fixed-state goal target=G(1;1,1,1)
  start=[1, 1, 1, 1]
  actions=['MakeGroup4']
  trace: [1, 1, 1, 1] -> [G(1;1,1,1)]
  goal_satisfied=True reward=1.0000 wall=0.000072s PASS
[CASE] g4/already-goal kind=validation level=group wall=0.000029s
  goal: fixed-state goal target=G(1;1,1,1)
  start=[G(1;1,1,1)]
  actions=[]
  trace: [G(1;1,1,1)]
  goal_satisfied=True reward=0.0000 wall=0.000029s PASS
[CASE] g_add/prefix1 kind=validation level=group wall=0.000226s
  goal: fixed-state goal target=G(1;C(1),C(1))
  start=[1]
  actions=['PushOne', 'MakeChange', 'PushOne', 'MakeChange', 'MakeGroup3']
  trace: [1] -> [1, 1] -> [1, C(1)] -> [1, C(1), 1] -> [1, C(1), C(1)] -> [G(1;C(1),C(1))]
  goal_satisfied=True reward=0.9600 wall=0.000226s PASS
[CASE] g_add/prefix2 kind=validation level=group wall=0.000183s
  goal: fixed-state goal target=G(1;C(1),C(1))
  start=[1, 1]
  actions=['MakeChange', 'PushOne', 'MakeChange', 'MakeGroup3']
  trace: [1, 1] -> [1, C(1)] -> [1, C(1), 1] -> [1, C(1), C(1)] -> [G(1;C(1),C(1))]
  goal_satisfied=True reward=0.9700 wall=0.000183s PASS
[CASE] g_add/prefix3 kind=validation level=group wall=0.000152s
  goal: fixed-state goal target=G(1;C(1),C(1))
  start=[1, C(1)]
  actions=['PushOne', 'MakeChange', 'MakeGroup3']
  trace: [1, C(1)] -> [1, C(1), 1] -> [1, C(1), C(1)] -> [G(1;C(1),C(1))]
  goal_satisfied=True reward=0.9800 wall=0.000152s PASS
[CASE] g_add/prefix4 kind=validation level=group wall=0.000116s
  goal: fixed-state goal target=G(1;C(1),C(1))
  start=[1, C(1), 1]
  actions=['MakeChange', 'MakeGroup3']
  trace: [1, C(1), 1] -> [1, C(1), C(1)] -> [G(1;C(1),C(1))]
  goal_satisfied=True reward=0.9900 wall=0.000116s PASS
[CASE] g_add/prefix5 kind=validation level=group wall=0.000094s
  goal: fixed-state goal target=G(1;C(1),C(1))
  start=[1, C(1), C(1)]
  actions=['MakeGroup3']
  trace: [1, C(1), C(1)] -> [G(1;C(1),C(1))]
  goal_satisfied=True reward=1.0000 wall=0.000094s PASS
[CASE] g_add/already-goal kind=validation level=group wall=0.000032s
  goal: fixed-state goal target=G(1;C(1),C(1))
  start=[G(1;C(1),C(1))]
  actions=[]
  trace: [G(1;C(1),C(1))]
  goal_satisfied=True reward=0.0000 wall=0.000032s PASS
[CASE] g_grouped/prefix1 kind=validation level=complex wall=0.000241s
  goal: fixed-state goal target=G(C(1);G(1;0))
  start=[1]
  actions=['MakeChange', 'PushOne', 'PushZero', 'MakeGroup2', 'MakeGroup2']
  trace: [1] -> [C(1)] -> [C(1), 1] -> [C(1), 1, 0] -> [C(1), G(1;0)] -> [G(C(1);G(1;0))]
  goal_satisfied=True reward=0.9600 wall=0.000241s PASS
[CASE] g_grouped/prefix2 kind=validation level=complex wall=0.000190s
  goal: fixed-state goal target=G(C(1);G(1;0))
  start=[C(1)]
  actions=['PushOne', 'PushZero', 'MakeGroup2', 'MakeGroup2']
  trace: [C(1)] -> [C(1), 1] -> [C(1), 1, 0] -> [C(1), G(1;0)] -> [G(C(1);G(1;0))]
  goal_satisfied=True reward=0.9700 wall=0.000190s PASS
[CASE] g_grouped/prefix3 kind=validation level=complex wall=0.000159s
  goal: fixed-state goal target=G(C(1);G(1;0))
  start=[C(1), 1]
  actions=['PushZero', 'MakeGroup2', 'MakeGroup2']
  trace: [C(1), 1] -> [C(1), 1, 0] -> [C(1), G(1;0)] -> [G(C(1);G(1;0))]
  goal_satisfied=True reward=0.9800 wall=0.000159s PASS
[CASE] g_grouped/prefix4 kind=validation level=complex wall=0.000124s
  goal: fixed-state goal target=G(C(1);G(1;0))
  start=[C(1), 1, 0]
  actions=['MakeGroup2', 'MakeGroup2']
  trace: [C(1), 1, 0] -> [C(1), G(1;0)] -> [G(C(1);G(1;0))]
  goal_satisfied=True reward=0.9900 wall=0.000124s PASS
[CASE] g_grouped/prefix5 kind=validation level=complex wall=0.000102s
  goal: fixed-state goal target=G(C(1);G(1;0))
  start=[C(1), G(1;0)]
  actions=['MakeGroup2']
  trace: [C(1), G(1;0)] -> [G(C(1);G(1;0))]
  goal_satisfied=True reward=1.0000 wall=0.000102s PASS
[CASE] g_grouped/already-goal kind=validation level=complex wall=0.000039s
  goal: fixed-state goal target=G(C(1);G(1;0))
  start=[G(C(1);G(1;0))]
  actions=[]
  trace: [G(C(1);G(1;0))]
  goal_satisfied=True reward=0.0000 wall=0.000039s PASS
[CASE] expr_two/prefix1 kind=validation level=pattern wall=0.000117s
  goal: pattern goal value==2
  start=[1]
  actions=['MakeChange']
  trace: [1] -> [C(1)]
  goal_satisfied=True reward=1.0000 wall=0.000117s PASS
[CASE] expr_two/prefix2 kind=validation level=pattern wall=0.000038s
  goal: pattern goal value==2
  start=[C(1)]
  actions=[]
  trace: [C(1)]
  goal_satisfied=True reward=0.0000 wall=0.000038s PASS
validation: 33/33 PASS (new initial states)
  level basic    2/2 PASS
  level change   5/5 PASS
  level complex  6/6 PASS
  level group    18/18 PASS
  level pattern  2/2 PASS
------------------------------------------------------------------------
HELD-OUT TARGET PROBE (targets NOT in the training set)
[CASE] h_grouped kind=heldout level=- wall=0.000033s
  goal: fixed-state goal target=G(C(1);G(1;0,1))
  start=[]
  actions=[]
  trace: []
  goal_satisfied=False reward=0.0000 wall=0.000033s FAIL
  feedback: state=[] expected=fixed-state goal target=G(C(1);G(1;0,1)) actual=no Q entry reason=tabular agent did not generalise to this state
  reason: agent has no learned action for this state (unseen state/goal)
  compositional plan=['PushOne', 'MakeChange', 'PushOne', 'PushZero', 'PushOne', 'MakeGroup3', 'MakeGroup2']
[CASE] h_grouped/COMPOSITIONAL kind=heldout level=- wall=0.000225s
  goal: fixed-state goal target=G(C(1);G(1;0,1))
  start=[]
  actions=['PushOne', 'MakeChange', 'PushOne', 'PushZero', 'PushOne', 'MakeGroup3', 'MakeGroup2']
  trace: [] -> [1] -> [C(1)] -> [C(1), 1] -> [C(1), 1, 0] -> [C(1), 1, 0, 1] -> [C(1), G(1;0,1)] -> [G(C(1);G(1;0,1))]
  goal_satisfied=True reward=0.9400 wall=0.000225s PASS
held-out: flat tabular 0/1 PASS, compositional 1/1 PASS
------------------------------------------------------------------------
CONTROL: pure reward-driven Q-learning (use_demo=False)
control validation: 14/33 PASS (learned_states=358)
------------------------------------------------------------------------
VERDICT
1. reduction (U1): 4 node types; <=10 met; 3/2/1 possible under explicit definitions
2. loaded agent digest=21f635406048eb44e0c731f909f5cc548a1d90486ad936d0bb0512db0f6912e5 learned_states=293 (deterministic train)
3. generalization (new initial states): 33/33 PASS
4. train reachability: 10/10 PASS; held-out flat: 0/1; held-out compositional: 1/1
5. control (pure Q): 14/33 PASS
6. per-case wall time: min=0.000020s median=0.000110s max=0.000274s
7. total validate wall time=0.178565s
VALIDATE_EXIT=0
```

Timing/RSS (`/usr/bin/time -v`, stderr):

```
	Command being timed: "/opt/automath/venv/bin/python -m new_approach.u2 --mode validate --model /opt/automath/tmp/u2_agent.json --comp /opt/automath/tmp/u2_comp.json"
	Elapsed (wall clock) time (h:mm:ss or m:ss): 0:00.32
	Maximum resident set size (kbytes): 22340
	Exit status: 0
```

## 8. Reproducibility: two independent train processes

Second train command:

```sh
/opt/automath/venv/bin/python -m new_approach.u2 --mode train \
    --model /opt/automath/tmp/u2_agent2.json --comp /opt/automath/tmp/u2_comp2.json
```

Raw digest lines and byte-level table comparison:

```
run1: train wall=0.040002s learned_states=293 digest=21f635406048eb44e0c731f909f5cc548a1d90486ad936d0bb0512db0f6912e5
run2: train wall=0.056647s learned_states=293 digest=21f635406048eb44e0c731f909f5cc548a1d90486ad936d0bb0512db0f6912e5

9aebf84f871752f80551652103ae6ff36665781d1969642f04510eeaf5a20a5d  /opt/automath/tmp/u2_agent.json
9aebf84f871752f80551652103ae6ff36665781d1969642f04510eeaf5a20a5d  /opt/automath/tmp/u2_agent2.json
2ac8271bf0aa8afedb5960a2ad1f16abc04271388ad585bac231dc93763f3378  /opt/automath/tmp/u2_comp.json
2ac8271bf0aa8afedb5960a2ad1f16abc04271388ad585bac231dc93763f3378  /opt/automath/tmp/u2_comp2.json
```

Both independent processes produced the same Q digest AND byte-identical saved
tables (`sha256sum` equal). `same seed -> same result` is therefore measured,
not asserted.

The validation process loaded the saved table and reported the same digest
(`loaded model ... digest=21f635...`), so train and validation are the same
agent.

## 9. Honest verdict

**(a) Reduction result (reused from U1, not re-derived).** 4 node types
(`Zero`, `One`, `Change`, dynamic `Group`); the operator's `<= 10` target is met
with 4; the same 11 samples evaluate identically at 3, 2 and 1 node types under
explicit definitions. U1's 11/11 deterministic checks still pass on the U2
commit:

```
$ cd /opt/automath/repo && /opt/automath/venv/bin/python -m new_approach.tests
RESULT: 11/11 passed in 0.023s (node_types=4, deterministic)
```

**(b) Generalization result.** The trained agent solves **all 33 new initial
states** (every non-trivial prefix of each trained goal's optimal word) across
all 9 trained goals plus the `ExprGoal` pattern, for **33/33 PASS**. The agent
was trained ONLY from the bare-minimum empty stack; it is then placed at
mid-construction stacks it was never a training-start from, and reaches the same
END state in every case. Train vs validation PASS/FAIL:

| trained goal | level | train (empty) | new-initial-state cases | validation PASS |
| ------------ | ----- | ------------- | ----------------------- | --------------- |
| `zero` | basic | PASS | 1 | 1/1 |
| `one` | basic | PASS | 1 | 1/1 |
| `two` = `C(1)` | change | PASS | 2 | 2/2 |
| `three` = `C(C(1))` | change | PASS | 3 | 3/3 |
| `g2` = `G(1;0)` | group | PASS | 3 | 3/3 |
| `g3` = `G(1;1,1)` | group | PASS | 4 | 4/4 |
| `g4` = `G(1;1,1,1)` | group | PASS | 5 | 5/5 |
| `g_add` = `G(1;C(1),C(1))` | group | PASS | 6 | 6/6 |
| `g_grouped` = `G(C(1);G(1;0))` | complex | PASS | 6 | 6/6 |
| `expr_two` (`ExprGoal value==2`) | pattern | PASS | 2 | 2/2 |
| **total** | | **10/10** | **33** | **33/33** |

**(c) Out-of-distribution limit (honest).** On a target NOT in the training set
(`h_grouped = G(C(1);G(1;0,1))`, built by grouping basic nodes) the flat
tabular agent **FAILS 0/1**: it has no Q entry and returns structured feedback
`state=[] expected=fixed-state goal target=G(C(1);G(1;0,1)) actual=no Q entry
reason=tabular agent did not generalise to this state`. The structurally biased
`CompositionalAgent`, which reuses the local construction rules learned from the
same trajectories, **PASSES 1/1** on that held-out target
(`plan=['PushOne','MakeChange','PushOne','PushZero','PushOne','MakeGroup3','MakeGroup2']`).
So the honest boundary is: this agent generalizes over INITIAL STATES of trained
goals (the dispatch's target), but a flat table does NOT generalize to unseen
END states; adding the dynamic-node compositional bias does.

**(d) Control.** Pure reward-driven Q-learning with no planner supervision
(`use_demo=False`) reaches only **14/33** validation cases: from the bare-minimum
empty stack, random exploration does not reliably discover the unique long
optimal words for the larger Group targets. Planner supervision is what makes
the bare-minimum training sample-efficient.

**(e) Measured wall times.** Every case is timed; all are sub-second:
per-case `min=0.000020s median=0.000110s max=0.000274s`; train process wall
`0.23s` at `22.0 MB` max RSS; validate process wall `0.32s` at `22.3 MB` max
RSS. Both are far inside the 2 vCPU / 3.8 GB no-swap box budget.

**(f) Diagnosis and next step if generalization is judged insufficient.**
Diagnosis: the flat tabular agent has no compositional inductive bias, so it
cannot map an unseen target tree to actions; it only generalizes over the state
space it visited. Measured evidence: held-out target 0/1 for the flat agent,
1/1 for the compositional agent. Next step: make the compositional/structural
agent the primary learner (recursive policy over the dynamic `Group` node, or a
tree-encoder policy), and train/validate on a grammar of grouped basic nodes
rather than a fixed target list.

## 10. Prior sessions consulted

`session_search 'automath'` returned 19 prior sessions. Consulted (named in the
dispatch plus the U1 handoff): `session-f2b11485-7165-4bfe-b129-c5ab6fc298ff`,
`session-c6a75787-2a88-40cb-8bf4-e0fb6b91ec12`, `session-8cb45f58-...` (U3
macro layer), `session-0e84882b-...` (U4 option C), `session-e154a3d8-...` (U5
training/eval) and `session-e218f9e1-...` (U6/U7 report). They decided: fork at
`/opt/automath/repo`, venv `/opt/automath/venv`, host = first ssh-config stanza;
`main` closed out at `746022d`; the symbolic RL approach did not learn a solver
on this box; the legacy full suite OOMs; keep every run small and bounded. U2
inherits all of that.

## 11. Gaps (not done)

* The flat agent's new-initial-state validation starts are the prefixes of each
  trained goal's optimal word. Every composable non-empty start of these unique
  plans IS a prefix, so this is the valid start set; it is stated explicitly
  rather than presented as unseen-state generalization. Held-out TARGET
  generalization is separately probed and reported (flat 0/1, compositional
  1/1).
* No DQN/neural policy; the learned model is a tabular Q function (the dispatch
  allows a small Q-table). No CUDA, no torch; pure standard library.
* Secret scan of the staged diff (`git diff --cached | grep -nE
  '<credential patterns>'`) returned zero hits; no credential value is present
  in any file, log or commit. No remote IP is printed anywhere in this file
  (host redacted to `<automath-host>`).
