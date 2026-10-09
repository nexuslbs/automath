# U3 evidence: macro-action layer (FIX 2) + torch SmartAgent (FIX 4)

Remote host: automath host (the reachable one in /opt/omni/data/ssh/config).
Remote clone /opt/automath/repo, venv /opt/automath/venv.
Start commit 69ece251cf8579f0a1aa34a30ff9aaf8ab00793a.
Workstation push clone: /opt/workspace/tmp/automath/repo.

## 1. Remote fetch before running (HEAD at start)

```
$ cd /opt/automath/repo && git status --short && git rev-parse HEAD && git fetch origin && git rev-parse origin/main
(clean)
69ece251cf8579f0a1aa34a30ff9aaf8ab00793a
From https://github.com/nexuslbs/automath
   69ece25..9142b60  main       -> origin/main
9142b606486be2bd45c784478c29c90287a9cd4a
Python 3.14.4
```

## 2. FIX 2 catalogue: scripts/run_case.py --list-macros

```
$ /opt/automath/venv/bin/python scripts/run_case.py --list-macros
== case binary_int family=result macros=4 primitive_basic_actions=24 ==
  [1] result_true (family=result): Create scratch 1, write IntBoolean(true), verify it
      expansion: CreateScratch(0, 0, 0) ; DefineScratchFromInt(1, 3, 1) ; VerifyGoal(0, 38, 1)
  [2] result_false (family=result): Create scratch 1, write IntBoolean(false), verify it
      expansion: CreateScratch(0, 0, 0) ; DefineScratchFromInt(1, 3, 0) ; VerifyGoal(0, 38, 1)
  [3] result_write_true (family=result): Create scratch 1 and write IntBoolean(true), no verify
      expansion: CreateScratch(0, 0, 0) ; DefineScratchFromInt(1, 3, 1)
  [4] result_check (family=result): Verify scratch 1 against the goal, no write
      expansion: VerifyGoal(0, 38, 1)
```

Old primitive space (for contrast), same run:

```
$ /opt/automath/venv/bin/python scripts/run_case.py --list
case           goal                                           allowed  basic  source
binary_int     HaveResultScratch(Eq[BinaryInt])               22       24     arithmetic_test.test_binary_int_basic
signed_int     HaveResultScratch(Eq[SignedInt])               22       24     arithmetic_test.test_signed_int_basic
int_to_binary  HaveResultScratch(Eq[IntToBinary])             22       24     arithmetic_test.test_int_to_binary
boolean_lt     HaveResultScratch(LessThan(Integer,Integer))   22       24     boolean_test.test_boolean
indices        HaveScratch(Void (index/args-group initial state)) 34       24     indices_test.test_indices
control_flow   HaveScratch(Void (If/Loop/function initial state)) 34       24     control_flow_test.test_control_flow
```

Action-space size: 4 macro-actions vs 22 allowed (24 basic) on binary_int.

## 3. Macro-action unit test (fixed tolerance)

```
$ /opt/automath/venv/bin/python -m pytest test_suite/macro_action_test.py -q
.....                                                                    [100%]
5 passed in 4.14s
```

First run (before tolerance fix) failed only on float comparison scale:

```
E       assert 1.4610668586101383e-07 < 1e-09
E        +  where 1.4610668586101383e-07 = abs((9960.319147146107 - 9960.319147))
```

## 4. FIX 4: torch install (detached) and version proof

```
$ nohup /opt/automath/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu > /opt/automath/logs/torch-install.log 2>&1 &
$ cat /opt/automath/logs/torch-install.log
Looking in indexes: https://download.pytorch.org/whl/cpu
Requirement already satisfied: torch in /opt/automath/venv/lib/python3.14/site-packages (2.14.1+cpu)
... (deps already satisfied)
$ /opt/automath/venv/bin/python -c "import torch; print(torch.__version__)"
2.14.1+cpu
$ /opt/automath/venv/bin/pip freeze | grep -iE "^torch|^numpy"
numpy==2.5.3
torch @ file:///opt/automath/torchdl/torch-2.14.1%2Bcpu-cp314-cp314-manylinux_2_28_x86_64.whl#sha256=35103e180214793207f95d7e12e100ecba3ace6f5fb37e1fcefbb07e096ed180
```

Note: the prior dispatch had already downloaded the cp314 CPU wheel into
/opt/automath/torchdl and installed it, so the pip run is idempotent
("already satisfied"). Version recorded: 2.14.1+cpu on Python 3.14.4.

Smart branch wiring proof:

```
$ /opt/automath/venv/bin/python -c "import train; print('macro_action_space_size', train.get_action_space_size()); import agent.smart_agent as sa; print('SmartAgent import OK')"
macro_action_space_size 3
SmartAgent import OK
```

## 5. Smart-agent macro episode (untrained DQN, raw trace)

```
$ /opt/automath/venv/bin/python scripts/run_case.py --case binary_int --agent smart --max-steps 20
===== case binary_int (arithmetic) =====
MACRO SELECTION SPACE (4): result_true, result_false, result_write_true, result_check
03:06:20 > [Explore] Action: 2, Args: (13, 8, 8) State size: torch.Size([1, 7059, 8])
step 1: action=VerifyGoal macro=result_false ok=False cost=0 reward=-139.680853 goal=False wall_ms=24.696 err=BooleanExceptionInfo
  primitive 1: action=CreateScratch ok=True cost=91845900 reward=-19.335623 goal=False
  primitive 2: action=DefineScratchFromInt ok=True cost=92732550 reward=-19.34523 goal=False
  primitive 3: action=VerifyGoal ok=False cost=0 reward=-101.0 goal=False
03:06:21 > [Explore] Action: 4, Args: (2, 0, 14) State size: torch.Size([1, 7231, 8])
step 2: action=VerifyGoal macro=result_check ok=False cost=0 reward=-101.0 goal=False wall_ms=20.905 err=BooleanExceptionInfo
  primitive 1: action=VerifyGoal ok=False cost=0 reward=-101.0 goal=False
03:06:21 > [Explore] Action: 1, Args: (0, 0, 0) State size: torch.Size([1, 7285, 8])
step 3: action=VerifyGoal macro=result_true ok=True cost=1931586 reward=9960.25384 goal=True wall_ms=84.844 err=none
  primitive 1: action=CreateScratch ok=True cost=94960100 reward=-19.368967 goal=False
  primitive 2: action=DefineScratchFromInt ok=True cost=95744400 reward=-19.377193 goal=False
  primitive 3: action=VerifyGoal ok=True cost=1931586 reward=9999.0 goal=True
SUMMARY actions=3 reward=9719.572987 goal=True total_wall_s=1.796525 max_history=7 dropped_history=0 action_space=4 macro_actions=4
```

The DQN action space is 4 (the macro set); the macro rebuilds reward from the
ordinary primitive reward path; history is bounded (max_history=7).

## 6. Simple-agent macro episode (deterministic catalogue proof)

```
$ /opt/automath/venv/bin/python scripts/run_case.py --case binary_int --macros --agent simple --max-steps 5
===== case binary_int (arithmetic) =====
MACRO SELECTION SPACE (4): result_true, result_false, result_write_true, result_check
step 1: action=VerifyGoal macro=result_true ok=True cost=1871096 reward=9960.319147 goal=True wall_ms=154.553 err=none
  primitive 1: action=CreateScratch ok=True cost=91845900 reward=-19.335623 goal=False
  primitive 2: action=DefineScratchFromInt ok=True cost=92732550 reward=-19.34523 goal=False
  primitive 3: action=VerifyGoal ok=True cost=1871096 reward=9999.0 goal=True
SUMMARY actions=1 reward=9960.319147 goal=True total_wall_s=1.059651 max_history=3 dropped_history=0 action_space=4 macro_actions=4
```

## 7. Existing test entry point

```
$ /opt/automath/venv/bin/python -m pytest test_suite/design_test.py -q
...                                                                      [100%]
3 passed in 2.61s
```

## 8. Final commit verification (remote, final sha 0bb1196)

```
$ cd /opt/automath/repo && git fetch origin && git reset --hard origin/main
   5abdc1e..0bb1196  main       -> origin/main
### HEAD=0bb1196c0bd34ad47228e146e88234ac167794ca
### origin/main=0bb1196c0bd34ad47228e146e88234ac167794ca
### git show --stat HEAD
commit 0bb1196c0bd34ad47228e146e88234ac167794ca
    requirements-train: record the installed CPU torch wheel and version
 requirements-train.txt | 9 +++++++--
 1 file changed, 7 insertions(+), 2 deletions(-)
### secret scan of HEAD diff
SCAN CLEAN (0 hits)
### pytest macro on final sha
.....                                                                    [100%]
5 passed in 3.52s
### smart smoke final sha 0bb1196 (max-steps 20)
step 1: action=VerifyGoal macro=result_false ok=False cost=0 reward=-139.680853 goal=False wall_ms=36.927 err=BooleanExceptionInfo
step 2: action=VerifyGoal macro=result_check ok=False cost=0 reward=-101.0 goal=False wall_ms=11.023 err=BooleanExceptionInfo
step 3: action=VerifyGoal macro=result_true ok=True cost=1931586 reward=9960.25384 goal=True wall_ms=80.755 err=none
SUMMARY actions=3 reward=9719.572987 goal=True total_wall_s=1.668471 max_history=7 dropped_history=0 action_space=4 macro_actions=4
```

Commits on main (newest first):
* 0bb1196c0bd34ad47228e146e88234ac167794ca  requirements-train: record the installed CPU torch wheel and version
* 5abdc1e70d083549f8a095362b26bd20d68d8184  test_suite: compare macro reward at the 6-decimal trace scale
* 9142b606486be2bd45c784478c29c90287a9cd4a  env: add bounded macro-action layer and enable torch SmartAgent over it

## 9. --macro-prior injection seam end to end

```
$ cat /tmp/u3-macro-prior.json
{"family": "result", "macros": [{"name": "llm_candidate_true", "description": "injected via --macro-prior", "family": "result", "steps": [{"action": "CreateScratch", "args": [0,0,0]}, {"action": "DefineScratchFromInt", "args": [1, "from_int:IntBoolean", 1]}, {"action": "VerifyGoal", "args": [0, "from_int:StateScratchIndex", 1]}]}]}
$ /opt/automath/venv/bin/python scripts/run_case.py --list-macros --macro-prior /tmp/u3-macro-prior.json
== case binary_int family=result macros=1 primitive_basic_actions=24 ==
  [1] llm_candidate_true (family=result): injected via --macro-prior
      expansion: CreateScratch(0, 0, 0) ; DefineScratchFromInt(1, 3, 1) ; VerifyGoal(0, 38, 1)
$ /opt/automath/venv/bin/python scripts/run_case.py --case binary_int --macros --macro-prior /tmp/u3-macro-prior.json --max-steps 5
===== case binary_int (arithmetic) =====
MACRO SELECTION SPACE (1): llm_candidate_true
step 1: action=VerifyGoal macro=llm_candidate_true ok=True cost=1871096 reward=9960.319147 goal=True wall_ms=91.352 err=none
  primitive 1: action=CreateScratch ok=True cost=91845900 reward=-19.335623 goal=False
  primitive 2: action=DefineScratchFromInt ok=True cost=92732550 reward=-19.34523 goal=False
  primitive 3: action=VerifyGoal ok=True cost=1871096 reward=9999.0 goal=True
SUMMARY actions=1 reward=9960.319147 goal=True total_wall_s=0.948912 max_history=3 dropped_history=0 action_space=1 macro_actions=1
```
