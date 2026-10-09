# U6 INDEPENDENT VERIFICATION (tester)

Automath host = the reachable host in /opt/omni/data/ssh/config (IP NOT printed here; second host <redacted-host> -> reachable=false, Permission denied). Remote clone /opt/automath/repo, venv /opt/automath/venv, logs /opt/automath/logs. Workstation clone /opt/workspace/tmp/automath/repo. No repo writes; eval --out pointed at /tmp.

## CHECK 1 - REPO STATE

Remote (raw ssh_run on automath host):
```
=== HEAD ===
99752be5edd8eff56c2cec29c3e0eaa8b1180c9b
=== fetch ===
=== origin/main ===
99752be5edd8eff56c2cec29c3e0eaa8b1180c9b
=== status --porcelain ===
=== status rc=0 ===
=== log -6 ===
99752be scripts: bounded macro SmartAgent training and held-out eval; docs: U5 split and no-network proof
cd40cb6 prior: add offline minicpm5-1b macro-action candidates (option C)
c2b7d45 scripts: disable model thinking and force JSON in the macro prior pass; docs: option C protocol
b0c19f5 scripts: add offline LLM macro-action prior builder (option C)
0bb1196 requirements-train: record the installed CPU torch wheel and version
5abdc1e test_suite: compare macro reward at the 6-decimal trace scale
=== branch ===
main
=== nproc ===
2
```
Workstation clone (raw bash):
```
99752be5edd8eff56c2cec29c3e0eaa8b1180c9b
--- status ---
(empty)
--- log ---
99752be scripts: bounded macro SmartAgent training and held-out eval; docs: U5 split and no-network proof
cd40cb6 ...
0bb1196 ...
5abdc1e ...
```
VERDICT CHECK 1: PASS - both HEADs and origin/main == 99752be, both trees clean, log -6 contains all named commits.

## CHECK 3 - MACRO LAYER + TORCH

`--list-macros` (built-in) raw:
```
== case binary_int family=result macros=4 primitive_basic_actions=24 ==
  [1] result_true ... expansion: CreateScratch(0, 0, 0) ; DefineScratchFromInt(1, 3, 1) ; VerifyGoal(0, 38, 1)
  [2] result_false ...
  [3] result_write_true ...
  [4] result_check ...
```
`test_suite/macro_action_test.py` with the venv:
```
.....                                                                    [100%]
5 passed in 4.13s
```
torch:
```
2.14.1+cpu
```
Injected prior `prior/macro_prior_minicpm5-1b.json` replaces the built-in catalogue (raw):
```
== case binary_int family=result macros=3 primitive_basic_actions=24 ==
  [1] DefineScratchFromFunctionWithIntArg (family=result): LLM proposed candidate for family result
  [2] DefineScratchFromFunctionWithIntArg_2 (family=result): LLM proposed candidate for family result
  [3] CreateArgsGroup (family=scratch): LLM proposed candidate for family scratch
MACRO SELECTION SPACE (3): DefineScratchFromFunctionWithIntArg, DefineScratchFromFunctionWithIntArg_2, CreateArgsGroup
SUMMARY actions=1 reward=-303.0 goal=False total_wall_s=0.055797 max_history=3 dropped_history=0 action_space=3 macro_actions=3
```
VERDICT CHECK 3: PASS - 4 built-in macros; 5/5 tests pass; torch 2.14.1+cpu; the injected prior yields exactly 3 macros (built-in 4 gone) and the selection space is the injected set.

## CHECK 6 (part) - 4307 / 4308 from MY run

My own probe `/tmp/u6_probe_4307.py` (wrong macro = VerifyGoal before scratch exists) raw:
```
last_action_error = ('VerifyGoal', 'BooleanExceptionInfo', 'BooleanExceptionInfo<39>{1}(IsEmpty<139>{1}(Optional<44>{0}))', 1)
macro reward = -101.0 terminated = False truncated = False
last_primitive_trace = [{'primitive': 1, 'action': 'VerifyGoal', 'reward': -101.0, 'ok': False, 'cost': 0, 'goal': False}]
--- render_observation ---
...
last_error: action=VerifyGoal class=BooleanExceptionInfo index=1 message=BooleanExceptionInfo<39>{1}(IsEmpty<139>{1}(Optional<44>{0}))
dropped_history_count: 0
max_history_size = 1 dropped_history = 0
```
Wrong-macro run via run_case also exited without traceback (`... err=BooleanExceptionInfo ... SUMMARY ... dropped_history=0`).
VERDICT 4307: PASS - structured tuple + `last_error:` line in the observation, not a traceback.
VERDICT 4308: PASS - max_history_size / dropped_history counters present and bounded (0 dropped); DEFAULT_MAX_HISTORY_STATE_SIZE=128 bound not reached under --max-steps 12.

## CHECK 2 - FIX 5 (bounded memory, retention dropped)

### 2a. committed regression `bash scripts/check_bounded_memory.sh` (detached, /usr/bin/time -v)
Raw /usr/bin/time block (log /opt/automath/logs/u6_check.log):
```
check_bounded_memory: cap=1400000 kB, python=/opt/automath/venv/bin/python
[>>test_binary_int_basic] Time taken: 12.73 seconds <Completed tests: 22 (62 actions)>
...
[>test_suite.arithmetic_test] Time taken: 454.55 seconds
check_bounded_memory: exit 0
	Command being timed: "bash scripts/check_bounded_memory.sh"
	User time (seconds): 429.49
	Percent of CPU this job got: 94%
	Elapsed (wall clock) time (h:mm:ss or m:ss): 7:36.05
	Maximum resident set size (kbytes): 355128
	Swaps: 0
	Exit status: 0
```
VERDICT 2a: PASS. Exit 0 under `ulimit -v 1400000`; peak RSS 355128 kB (347 MiB), well under the 1.4 GB cap. NOTE: my wall 7:36.05 > developer 6:51.07 because a second arithmetic job ran concurrently (2-core box).

### 2b. module run `/usr/bin/time -v timeout 300 python -u -c "..."` (detached)
Raw /usr/bin/time block (log /opt/automath/logs/u6_module.log):
```
Command exited with non-zero status 124
	Elapsed (wall clock) time (h:mm:ss or m:ss): 5:00.08
	Maximum resident set size (kbytes): 260424
	Swaps: 0
	Exit status: 124
```
At the 300 s cap it had completed 12 of the module sub-cases (log: binary_int_basic .. rational_basic) with peak RSS 260424 kB (254 MiB).
VERDICT 2b: PASS for the MEMORY claim (max RSS 260 MB, bounded, << 600 MB). Exit 124 under the 300 s wall cap is EXPECTED and consistent with the CPU-bound ~411 s workload (the developer explicitly recorded "exit 0, wall < 300 s" as NOT met). The developer's own u2 note states this gap; I confirm the residual cost is CPU, not memory.

## CHECK 4 - HELD-OUT EVAL vs committed results/heldout_macro_smart.json

Command (documented eval form, --out redirected to /tmp so the repo is NOT written):
```
/opt/automath/venv/bin/python scripts/train_macro_smart.py eval \
  --checkpoint /opt/automath/tmp/macro_smart.pt \
  --cases binary_int,signed_int,int_to_binary,boolean_lt,indices,control_flow \
  --max-steps 12 --out /tmp/u6_evalN.json
```
CHECKPOINT: /opt/automath/tmp/macro_smart.pt EXISTS (3117079 bytes, sha256 9102ec88f9bc367bd24a616d00802780f5c783792a13a282c22369b6b5b576dc) but is NOT committed (`git ls-files | grep -c macro_smart.pt` = 0). The committed JSON records `"model": "/opt/automath/tmp/macro_smart.pt"`. Its sha is not committed, so its provenance is the developer's word; I re-ran against the on-disk file. (No re-train needed.)

Per-case goal_reached vs committed (raw CASE lines in /tmp/u6_eval1.log, u6_eval2.log, u6_eval_netns.log):
```
committed      binary_int T  signed_int T  int_to_binary T  boolean_lt T  indices F  control_flow F
my run 1       binary_int T  signed_int T  int_to_binary T  boolean_lt T  indices F  control_flow F
my run 2       binary_int T  signed_int T  int_to_binary T  boolean_lt T  indices F  control_flow F
my netns run   binary_int T  signed_int F  int_to_binary T  boolean_lt T  indices F  control_flow F
```
Per-case reward / macro_steps vs committed (committed -> my run1 -> my run2 -> netns):
```
binary_int     comm r=9960.319147 m=1  | r1 r=9960.319147 m=1  | r2 r=9960.319147 m=1  | netns r=9960.319147 m=1   MATCH
signed_int     comm r=9921.600221 m=2  | r1 r=9743.030074 m=4  | r2 r=9960.318325 m=1  | netns r=-395.835163 m=5 (goal FALSE)
int_to_binary  comm r=9921.601851 m=2  | r1 r=9704.225129 m=5  | r2 r=9960.319147 m=1  | netns r=9882.849442 m=3
boolean_lt     comm r=9921.602938 m=2  | r1 r=9921.602938 m=2  | r2 r=9820.588450 m=2  | netns r=9921.602938 m=2
indices        comm r=-560.379837 m=4  | r1 r=-397.625196 m=5  | r2 r=-478.932333 m=5  | netns r=-478.964019 m=5
control_flow   comm r=-491.990808 m=5  | r1 r=-491.808834 m=5  | r2 r=-571.739674 m=4  | netns r=-411.931746 m=5
```
Non-determinism test (same checkpoint, run twice): PASS/FAIL set was identical in run1 and run2, but the action sequences, macro counts and rewards DIFFER (e.g. signed_int m=4 vs m=1; control_flow r=-491.81 vs r=-571.74). So dropout in select_action is real. HOWEVER the third same-checkpoint run (netns) FLIPPED signed_int to goal_reached=false. Therefore the PASS/FAIL set is NOT guaranteed stable across repeats.
VERDICT 4: DISCREPANCY. The 4/6-vs-2/6 headline (4 HaveResultScratch true, indices/control_flow false) held in my run1/run2 and matches committed, but (a) the exact committed rewards/macro counts are a single non-reproducible sample and mismatch mine, and (b) the PASS/FAIL set is not stable: signed_int flipped true->false in my netns re-run with the same checkpoint.

## CHECK 5 - NO-NETWORK (`unshare -n`) and outbound failure in the SAME namespace
Raw:
```
--- ip addr ---
1: lo: <LOOPBACK> mtu 65536 qdisc noop state DOWN ...   (no other interface)
--- outbound curl http://example.com ---
curl rc=6
curl: (6) Could not resolve host: example.com
--- outbound urllib http://example.com ---
urllib rc=1
urllib.error.URLError: <urlopen error [Errno -3] Temporary failure in name resolution>
--- eval under netns (trained checkpoint) ---
netns eval rc=0
CASE {"case": "binary_int", ... "goal_reached": true ...}
...
CASE {"case": "control_flow", ... "goal_reached": false ...}
```
VERDICT 5: PASS. The eval (rc=0, all 6 cases ran) and the failing outbound curl/urllib are proven inside the SAME `unshare -n` namespace; no network available.

## CHECK 6 (cont) - 4309 per-case wall times I measured (sub_second = wall < 1.0 s)
```
run1  binary_int 1.309760 F | signed_int 2.034518 F | int_to_binary 2.479858 F | boolean_lt 1.227291 F | indices 3.585154 F | control_flow 7.572446 F
run2  binary_int 1.297644 F | signed_int 0.628573 T | int_to_binary 0.649430 T | boolean_lt 1.088995 F | indices 2.643293 F | control_flow 5.765473 F
netns binary_int 1.245304 F | signed_int 2.539212 F | int_to_binary 1.706995 F | boolean_lt 1.094603 F | indices 2.413171 F | control_flow 6.959367 F
committed values: binary_int 1.251696 F | signed_int 0.894761 T | int_to_binary 0.813572 T | boolean_lt 0.835290 T | indices 1.902683 F | control_flow 6.792219 F
```
VERDICT 4309: PASS that the sub_second flag is computed and reported per case, but the flag is NOT stable: binary_int/indices/control_flow are always >1 s; signed_int/int_to_binary/boolean_lt flip >=1 s depending on the sampled macro sequence. NOTE run1 wall times were measured with the check_bounded regression running concurrently, which inflates them.

## CHECK 7 - HOST STATE
```
$ docker ps -a --format '{{.Names}} {{.Image}} {{.Status}}'
app local_discourse/app Up 12 hours
nginx-proxy nginx:alpine Up 12 hours
mail-relay boky/postfix:latest Exited (143) 14 hours ago
```
No model/llm container running (no llm-deploy). The app + nginx-proxy containers belong to other work and were NOT touched (I ran no docker lifecycle verb). nproc = 2.
```
free -m BEFORE my runs:  total 3814 used 2077 free 1463 buff/cache 687 available 1736 Swap 0
free -m DURING (2 arithmetic jobs): total 3814 used 2417 free 1119 buff/cache 692 available 1397 Swap 0
free -m AFTER my runs:   total 3814 used 2062 free 1472 buff/cache 695 available 1752 Swap 0
```
VERDICT 7: PASS. RAM returned to baseline; no swap; no model container.

## CHECK 8 - ADVERSARIAL

### 8(a) trained checkpoint vs fresh random checkpoint
I built an UNTRAINED checkpoint with the same constructor/seed path (`_make_agent(4, 12345).save('/tmp/u6_rand.pt')`, sha256 1faaf46e82bde1ffb0603319558af1062a4532a50b25b370c136f3171fa4281c) and ran the exact eval command against it:
```
UNTRAINED  binary_int T m=1 | signed_int T m=1 | int_to_binary T m=1 | boolean_lt T m=1 | indices F m=4 r=-560.379837 | control_flow F m=4 r=-571.739674
```
VERDICT 8(a): FAIL (the check was designed to show trained != untrained). The untrained network yields the SAME 4/6 PASS/FAIL set (all four HaveResultScratch cases reach the goal, both HaveScratch cases fail) and even reproduces the committed `indices` row EXACTLY (m=4, reward=-560.379837). So the held-out evaluation does NOT distinguish the trained from an untrained network and does not evidence learning. The mechanism does load the trained checkpoint (`agent.load`), but the outcome does not depend on it.

### 8(b) training/held-out leakage
Raw: `scripts/train_macro_smart.py` TRAIN_CASES_DEFAULT = binary_int,int_to_binary,boolean_lt; train loop `spec = train_specs[(episode-1) % len(train_specs)]` (only train specs); `_train_catalogue(specs)` builds the action space from `specs[0]` (binary_int) ONLY; eval builds the SAME train catalogue and never uses the evaluated case to build it. Registry `--list` shows the 6 cases and their goals.
VERDICT 8(b): PASS - no held-out case, goal or macro catalogue is an input to training. (The held-out set includes one same-family case `signed_int` and two other-family cases forced to replay the train catalogue; this is a stated design limitation, not leakage.)

### 8(c) is PASS by construction rather than learning?
Evidence: `env/macro_action.py::_RESULT_MACROS` defines `result_true` as literally `CreateScratch(0,0,0); DefineScratchFromInt(1, from_int:IntBoolean, 1); VerifyGoal(0, from_int:StateScratchIndex, 1)` - the same recipe the repo's own arithmetic test uses. A greedy (`epsilon=0.0`) untrained net picks it and reaches the goal on all four HaveResultScratch cases; untrained reproduces the committed `indices` numbers. The developer's own u5 EVIDENCE already admits "DQN did NOT converge" and "success is 4-macro catalogue coverage, not a learned policy".
VERDICT 8(c): the "PASS" on the 4 result cases is BY CONSTRUCTION (the catalogue contains the answer), NOT learning. The developer DID state this honestly, so it is not a false claim, but the held-out eval must not be read as evidence of a learned policy.

## VERDICT LIST
```
CHECK 1  repo state (99752be, clean, both clones) .......... PASS
CHECK 2  fix 5: bounded memory / retention drop ........... PASS (2a exit 0, RSS 355 MB; 2b exit 124 CPU-bound, RSS 260 MB)
CHECK 3  macro layer + torch + prior injection ............ PASS
CHECK 4  held-out eval vs committed ....................... DISCREPANCY (goals 4/6 stable in run1/run2 but matched committed only at goal level; signed_int flipped false in netns; rewards/macros not reproducible)
CHECK 5  no-network under unshare -n ...................... PASS
CHECK 6  4307 structured observation error ................ PASS
CHECK 6  4308 bounded history/dropped counters ............ PASS
CHECK 6  4309 per-case wall + sub_second .................. DISCREPANCY (flag not stable across samples)
CHECK 7  host state / RAM / no model container ............ PASS
CHECK 8a trained vs random checkpoint ..................... FAIL (untrained gives same 4/6; eval does not evidence training)
CHECK 8b leakage .......................................... PASS
CHECK 8c PASS by construction ............................. PASS-by-construction (catalogue solves the case; developer admitted it)
```

## INDEPENDENTLY CONFIRMED
- The artifact exists and is clean at HEAD 99752be on the remote and on the workstation clone.
- FIX 5 memory bounding is real and reproducible: fixed module peak RSS 260 MB vs the developer's recorded pre-fix 1736 MB; the committed regression exits 0 under `ulimit -v 1400000` with 355 MB peak.
- Macro layer, 5/5 macro tests, torch 2.14.1+cpu, and `--macro-prior` injection (built-in 4 replaced by the injected prior) all work.
- 4307 (structured error in the observation, no traceback) and 4308 (max_history_size/dropped_history bounded) hold in my own run.
- Everything (eval) runs with NO network under `unshare -n`; outbound curl/urllib fail in the same namespace.
- No held-out leakage into training; no model/llm container; other-work containers untouched.

## NOT VERIFIED / DISCREPANCY
- The committed per-case rewards/macro counts/wall times are NOT reproducible: every eval run samples a different macro sequence (dropout in select_action). The PASS/FAIL set itself is not stable (signed_int flipped in my netns run). The committed `results/heldout_macro_smart.json` cannot be treated as a deterministic result.
- The held-out eval does NOT demonstrate learning: an untrained random checkpoint reaches the same 4/6 goals and reproduces the committed `indices` row exactly.
- The checkpoint /opt/automath/tmp/macro_smart.pt is NOT committed; only its presence and sha (9102ec88...) are recorded by me, provenance is the developer's word.
- The developer's "full suite" (tests.py) was never run by them and I did not run it (same ~411 s arithmetic cost); reported as a gap by the developer, closed only for the two module/regression commands above.
- 8(b) leak check is a code inspection plus registry listing, not an exhaustive data-flow proof.

## my usage (raw)
dsh-usage.py could NOT be run on this workstation: no Python interpreter exists in the image.
```
$ command -v python3 python ; echo "py rc=$?"
py rc=1
$ python3 /opt/omni/services/workstation/dsh-usage.py .../session-5366d238-...
bash: python3: command not found
```
Session log dir (resolved): /opt/omni/data/workstation/dsh-home/sessions/--var-lib-workstation-projects-default--/session-5366d238-5ec0-4aa0-b741-3bad196b7b4e . TSV usage row UNAVAILABLE (same limitation the U5 developer reported).
