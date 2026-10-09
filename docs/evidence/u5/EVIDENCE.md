# U5 - bounded training + held-out test with per-case raw evidence

Inherited sha: `cd40cb68f60cf84b6b2079e20bd3c9d47d4ac113` (fix 5, macro-action
layer + torch SmartAgent over it, offline prior). Branch `main`.

Every block below is a raw command and its raw output, appended as measured.

## 0. Host probe (both hosts in /opt/omni/data/ssh/config)

```
ssh_status(root@<automath-host>) -> {"reachable": true, "code": 0, "durationMs": 3299}
ssh_status(root@<redacted-host>) -> {"reachable": false, "code": 255,
  "stderr": "Permission denied, please try again.\r\nPermission denied, please try again.\r\nroot@<redacted-host>: Permission denied (publickey,password).\r\n"}
```
The reachable automath host is <automath-host> (IP never printed in the report body;
raw stdout above is the tool answer).

```
=== free -m BEFORE ===
               total        used        free      shared  buff/cache   available
Mem:            3814        2065         526         150        1650        1749
Swap:              0           0           0
```
The Discourse `app` + `nginx-proxy` template-case containers were already up and
stay up (never touched); no `llm-deploy` container is running (kept down).

## 1. DELIVERABLE 1 - SPLIT

TRAIN  = `binary_int`, `int_to_binary`, `boolean_lt`
HELD-OUT = `signed_int`, `indices`, `control_flow`

Justification:
* The three TRAIN cases are exactly the `HaveResultScratch` registry cases that
  share the built-in 4-macro result catalogue (`result_true`, `result_false`,
  `result_write_true`, `result_check`). The agent's action space is built from
  the FIRST train case only (`_train_catalogue(train_specs)`), so no held-out
  case, goal value or catalogue entry can influence training.
* `signed_int` is a same-family unseen case (a genuine generalisation test).
* `indices` and `control_flow` are the other family (`HaveScratch`, 3-macro
  built-in catalogue). The DQN has a FIXED output dimension = training
  catalogue size (4), so evaluation replays the TRAIN catalogue on every case;
  the held-out cases never build it. This is stated openly as a limitation: the
  scratch-family cases are tested for transfer under the train macro set, not
  with their own catalogue (which would need a differently sized network).

## 2. DELIVERABLE 2 - BOUNDED TRAIN

New committed script: `scripts/train_macro_smart.py` (train + eval subcommands).

Raw command launched detached on the remote:

```
cd /opt/automath/repo && PYTHONUNBUFFERED=1 nohup /opt/automath/venv/bin/python -u \
  scripts/train_macro_smart.py train \
  --train-cases binary_int,int_to_binary,boolean_lt \
  --max-episodes 400 --max-wall-s 480 --max-steps 12 --seed 1 \
  --checkpoint /opt/automath/tmp/macro_smart.pt \
  --jsonl /opt/automath/logs/u5_train.jsonl \
  --summary-json /opt/automath/logs/u5_train_summary.json \
  > /opt/automath/logs/u5_train.log 2>&1 &
PID=990508
```
The hard caps are `--max-episodes 400` and `--max-wall-s 480` (<= 600 s target).

## 3. DELIVERABLE 4 (PART) - unshare -n availability and outbound failure

```
command -v unshare -> /usr/bin/unshare
unshare -n -- /opt/automath/venv/bin/python -c "import urllib.request;
  urllib.request.urlopen('http://example.com', timeout=5)"
NETWORK: request FAILED as expected -> URLError <urlopen error [Errno -3] Temporary failure in name resolution>
```
So `unshare -n` is available and gives a namespace with NO network.

## 4. DELIVERABLE 5(a) - 4307 structured error

Raw probe (`/opt/automath/logs/u5_4307_probe.txt`): a deliberately wrong macro
`wrong_verify_missing_scratch = [VerifyGoal(0, from_int:StateScratchIndex, 1)]`
(verify scratch 1 before it exists) on `binary_int`.

```
PROBE A: macro=wrong_verify_missing_scratch on case=binary_int
last_action_error (kind, message) = ('VerifyGoal', 'BooleanExceptionInfo',
    'BooleanExceptionInfo<39>{1}(IsEmpty<139>{1}(Optional<44>{0}))', 1)
macro reward = -101.0 terminated=False truncated=False
last_primitive_trace = [{'primitive': 1, 'action': 'VerifyGoal', 'reward': -101.0,
    'ok': False, 'cost': 0, 'goal': False}]
--- render_observation ---
goal: HaveResultScratch
goal_achieved: False
...
last_error: action=VerifyGoal class=BooleanExceptionInfo index=1
    message=BooleanExceptionInfo<39>{1}(IsEmpty<139>{1}(Optional<44>{0}))
dropped_history_count: 0
```
The error is a STRUCTURED tuple surfaced in the state (`last_error:` line of the
observation), not an opaque traceback.

The U4 offline prior macros are also wrong for `binary_int` and each yields a
structured error row:
```
PRIOR catalogue (3): ['DefineScratchFromFunctionWithIntArg',
  'DefineScratchFromFunctionWithIntArg_2', 'CreateArgsGroup']
  DefineScratchFromFunctionWithIntArg    err=('DefineScratchFromFunctionWithIntArg', 'BooleanExceptionInfo', ...) reward=-303.0 trace_ok=[False, False, False]
  DefineScratchFromFunctionWithIntArg_2  err=(...) reward=-303.0 trace_ok=[False, False, False]
  CreateArgsGroup                        err=('<unknown>', 'BooleanExceptionInfo', ...) reward=-101.0 trace_ok=[False]
```

## 5. DELIVERABLE 2 (cont) - training log tail, curve, checkpoint

```
=== TRAIN DONE (wall) ===
EPISODE {"episode": 42, "case": "boolean_lt", "macro_steps": 2, "primitive_steps": 5,
  "reward": 9921.602938, "goal": true, "cost": 1903756, "epsilon": 0.89238204,
  "loss": 4693.2451, "updates": 2, "macros": ["result_write_true", "result_true"]}
WALL CAP reached after 42 episodes
TRAIN_SUMMARY {"mode":"train","train_cases":["binary_int","int_to_binary","boolean_lt"],
  "action_space_size":4,"macros":["result_true","result_false","result_write_true","result_check"],
  "episodes_run":42,"max_episodes":400,"max_wall_s":480.0,"wall_s":484.948602,
  "updates":85,"epsilon_final":0.89238204,"checkpoint":"/opt/automath/tmp/macro_smart.pt"}
=== CHECKPOINT ===
-rw-r--r-- 1 root root 3117079 Oct  9 03:32 /opt/automath/tmp/macro_smart.pt
9102ec88f9bc367bd24a616d00802780f5c783792a13a282c22369b6b5b576dc  /opt/automath/tmp/macro_smart.pt
```
Curve first 10 (raw):
```
ep1  binary_int    macro=3 prim=7 reward=9882.849441 goal=True  eps=0.9        loss=null macs=[result_write_true,result_write_true,result_true]
ep2  int_to_binary macro=2 prim=6 reward=9820.587367 goal=True  eps=0.9        loss=null macs=[result_false,result_true]
ep3  boolean_lt    macro=1 prim=3 reward=9960.319696 goal=True  eps=0.9        loss=null macs=[result_true]
ep4  binary_int    macro=3 prim=7 reward=9719.559851 goal=True  eps=0.9        loss=null macs=[result_check,result_false,result_true]
ep5  int_to_binary macro=3 prim=9 reward=9680.806332 goal=True  eps=0.9        loss=null macs=[result_false,result_false,result_true]
ep6  boolean_lt    macro=4 prim=11 reward=9642.007021 goal=True eps=0.9        loss=null macs=[result_false,result_write_true,result_false,result_true]
ep7  binary_int    macro=5 prim=12 reward=-395.845371 goal=False eps=0.9       loss=null macs=[result_false,result_write_true,result_false,result_write_true,result_false]
ep8  int_to_binary macro=2 prim=5 reward=9921.601851 goal=True  eps=0.9        loss=null macs=[result_write_true,result_true]
ep9  boolean_lt    macro=2 prim=3 reward=9960.319696 goal=True  eps=0.9        loss=null macs=[result_write_true,result_check]
ep10 binary_int    macro=1 prim=3 reward=9960.319147 goal=True  eps=0.9        loss=null macs=[result_true]
```
Curve last 10 (raw):
```
ep33 boolean_lt    macro=3 prim=6 reward=9820.588450 goal=True  eps=0.89479485 loss=4806.0225 macs=[result_false,result_write_true,result_check]
ep34 binary_int    macro=4 prim=9 reward=9680.806332 goal=True  eps=0.89443699 loss=3472.9421 macs=[result_false,result_false,result_write_true,result_check]
ep35 int_to_binary macro=2 prim=5 reward=9921.601851 goal=True  eps=0.89425811 loss=3359.8359 macs=[result_write_true,result_true]
ep36 boolean_lt    macro=3 prim=4 reward=9859.305759 goal=True  eps=0.89398986 loss=4297.7798 macs=[result_check,result_write_true,result_check]
ep37 binary_int    macro=2 prim=6 reward=9820.587367 goal=True  eps=0.89381107 loss=3423.8918 macs=[result_false,result_true]
ep38 int_to_binary macro=4 prim=8 reward=9618.532522 goal=True  eps=0.89345360 loss=2830.2837 macs=[result_check,result_check,result_false,result_true]
ep39 boolean_lt    macro=1 prim=3 reward=9960.319696 goal=True  eps=0.89336425 loss=4150.4375 macs=[result_true]
ep40 binary_int    macro=6 prim=12 reward=-395.817671 goal=False eps=0.89282837 loss=3224.1575 macs=[result_write_true,result_write_true,result_false,result_check,result_write_true,result_false]
ep41 int_to_binary macro=3 prim=5 reward=9758.291378 goal=True  eps=0.89256054 loss=2779.5803 macs=[result_check,result_check,result_true]
ep42 boolean_lt    macro=2 prim=5 reward=9921.602938 goal=True  eps=0.89238204 loss=4693.2451 macs=[result_write_true,result_true]
```
Verdict: DQN did NOT converge (epsilon ~0.892; 2/42 train episodes still fail;
success is 4-macro catalogue coverage, not a learned policy).

## 6. DELIVERABLE 3 - held-out evaluation (raw, from the network-on run)

```
CASE binary_int     end goal=true  cost=1871096   history=3  dropped=0  macro=1 prim=3 reward=9960.319147 wall=1.242348 sub_second=false macros=[result_true]
CASE signed_int     end goal=true  cost=1871855   history=3  dropped=0  macro=1 prim=3 reward=9960.318325 wall=0.579325 sub_second=true  macros=[result_true]
CASE int_to_binary  end goal=true  cost=1871096   history=3  dropped=0  macro=1 prim=3 reward=9960.319147 wall=0.584151 sub_second=true  macros=[result_true]
CASE boolean_lt     end goal=true  cost=1870590   history=3  dropped=0  macro=1 prim=3 reward=9960.319696 wall=0.607861 sub_second=true  macros=[result_true]
CASE indices        end goal=false cost=125170600 history=12 dropped=0  macro=5 prim=12 reward=-397.625196 wall=2.235541 sub_second=false macros=[result_false,result_false,result_write_true,result_write_true,result_false]
CASE control_flow   end goal=false cost=623797950 history=12 dropped=0  macro=5 prim=12 reward=-412.020303 wall=6.715888 sub_second=false macros=[result_false,result_write_true,result_true,result_write_true,result_false]
```
All start states: `{goal=false, cost=0, history_amount=0, dropped_history=0, last_action_error=null}`.

The committed JSON `results/heldout_macro_smart.json` is the LATER run made
under `unshare -n` (section 7); it differs slightly because dropout stays active
during `select_action` (`DQN.forward(x, training=True)`), so inference is not
bit-deterministic:

```
CASE binary_int     goal=true  macro=1 prim=3 reward=9960.319147 wall=1.251696 maxhist=3
CASE signed_int     goal=true  macro=2 prim=5 reward=9921.600221 wall=0.894761 maxhist=5
CASE int_to_binary  goal=true  macro=2 prim=5 reward=9921.601851 wall=0.813572 maxhist=5
CASE boolean_lt     goal=true  macro=2 prim=5 reward=9921.602938 wall=0.835290 maxhist=5
CASE indices        goal=false macro=4 prim=12 reward=-560.379837 wall=1.902683 maxhist=12
    last_action_error=["VerifyGoal","BooleanExceptionInfo","BooleanExceptionInfo<39>{1}(Eq<126>{2}( IntBoolean<155>[1] Void<57>{0} ))",12]
CASE control_flow   goal=false macro=5 prim=12 reward=-491.990808 wall=6.792219 maxhist=12
```

## 7. DELIVERABLE 4 - NO-NETWORK PROOF (raw, same namespace)

```
=== inside netns: ip addr ===     (empty: no interfaces)
=== outbound curl ===
curl: (6) Could not resolve host: example.com
CURL_FAIL
=== outbound python urllib ===
URLLIB_FAIL URLError <urlopen error [Errno -3] Temporary failure in name resolution>
=== bounded TRAINING under netns ===
TRAIN cases: ['binary_int', 'int_to_binary', 'boolean_lt']
EPISODE {"episode": 1, "case": "binary_int", "macro_steps": 2, "primitive_steps": 4, "reward": -77.398149, "goal": false, "epsilon": 0.9, "loss": null, "macros": ["result_write_true", "result_true"]}
...
EPISODE {"episode": 6, "case": "boolean_lt", "macro_steps": 2, "primitive_steps": 4, "reward": -240.694241, "goal": false, "epsilon": 0.9, "loss": null, "macros": ["result_check", "result_false"]}
TRAIN_SUMMARY {"mode": "train", "episodes_run": 6, "max_wall_s": 90.0, "wall_s": 4.690755, "updates": 0, "checkpoint": "/opt/automath/tmp/u5_noneth.pt"}
=== FULL EVALUATION under netns (main trained checkpoint) ===
CASE binary_int ... goal_reached true ; CASE signed_int ... true ; CASE int_to_binary ... true ;
CASE boolean_lt ... true ; CASE indices ... false ; CASE control_flow ... false
=== done inside netns, rc=0 ===
```
Re-runnable:
```
cd /opt/automath/repo && unshare -n -- bash -c '
  /opt/automath/venv/bin/python -c "import urllib.request; urllib.request.urlopen(\"http://example.com\", timeout=5)"
  /opt/automath/venv/bin/python scripts/train_macro_smart.py train --max-episodes 6 --max-wall-s 90 --checkpoint /tmp/u5_noneth.pt
  /opt/automath/venv/bin/python scripts/train_macro_smart.py eval --checkpoint /opt/automath/tmp/macro_smart.pt'
```

## 8. DELIVERABLE 5(b) - 4308 bounded state

Documented bound:
```
$ grep -n DEFAULT_MAX_HISTORY_STATE_SIZE config/settings.py
11:DEFAULT_MAX_HISTORY_STATE_SIZE = 128
```
Run counters (all six cases, both eval runs): `max_history_size` = 3 (binary_int
network-on), 5 (signed_int/int_to_binary/boolean_lt network-on run was 3), 12
(indices/control_flow), `dropped_history = 0` throughout. The run never reached
the 128 bound because `--max-steps 12` caps the episode; the bound itself is
regression-tested by `scripts/check_bounded_memory.sh` under a 1400000 kB
`ulimit -v`.

## 9. Box RAM before/after

```
BEFORE (training launch):  total 3814  used 2065  free 526  buff/cache 1650  available 1749  Swap 0
DURING (peak observed):    total 3814  used 3326  free 267  buff/cache  634  available  487  Swap 0
AFTER (eval done):         total 3814  used 2111  free 1478 buff/cache 639  available 1703  Swap 0
```
Training RSS oscillated 777-1349 MB (torch caching allocator); no OOM, no swap,
the `app` + `nginx-proxy` template containers stayed up and untouched.

## 10. COMMIT + PUSH (raw)

Staged diff secret scan:
```
git add scripts/train_macro_smart.py results/heldout_macro_smart.json results/u5_train_curve.jsonl docs/REPORT.md
git diff --cached | grep -nE 'PRIVATE-KEY|ghp-|ghs-|sk_|AKI-A|api_key-|password-'
scan rc=1 (1 = zero hits)
```
```
$ git show --stat HEAD
commit 99752be5edd8eff56c2cec29c3e0eaa8b1180c9b
 docs/REPORT.md                   | 132 +++++++
 results/heldout_macro_smart.json | 753 +++++++++++++++++++++++++++++++++++++++
 results/u5_train_curve.jsonl     |  42 +++
 scripts/train_macro_smart.py     | 335 +++++++++++++++++
 4 files changed, 1262 insertions(+)

$ git push origin main
To https://github.com/nexuslbs/automath
   cd40cb6..99752be  main -> main

workstation HEAD      = 99752be5edd8eff56c2cec29c3e0eaa8b1180c9b
workstation origin/main = 99752be5edd8eff56c2cec29c3e0eaa8b1180c9b
remote origin/main    = 99752be5edd8eff56c2cec29c3e0eaa8b1180c9b
remote HEAD           = 99752be5edd8eff56c2cec29c3e0eaa8b1180c9b
remote git status --short = (clean)
remote script sha256  = 1527d3fa394bf0aadfec948083c3db1ada59e06bebf29aa701987dad30954518 (== committed blob)
```

## 11. Box state after (raw)

```
$ docker ps --format '{{.Names}} {{.Status}}'
app Up 12 hours
nginx-proxy Up 12 hours
$ docker ps -a | grep -i llm  -> no llm-deploy container
$ free -m
               total        used        free      shared  buff/cache   available
Mem:            3814        2068        1473         150         687        1746
Swap:              0           0           0
```

## 12. my usage (raw) - UNAVAILABLE on this workstation

Two cheap checks:
```
$ command -v python3 python      -> (no output, exit 1)
$ ls /usr/bin/python*            -> (no output, exit 2)
```
`dsh-usage.py` is a Python script and no Python interpreter exists in the
workstation image, so the TSV accounting row cannot be produced. The session log
is `/opt/omni/data/workstation/dsh-home/sessions/--var-lib-workstation-projects-default--/session-e218f9e1-6c6d-4cf1-bf22-be15207f1652/session.v4.jsonl.zstd`
(334501 bytes, zstd-compressed); no python/zstd tooling was used to decode it.


