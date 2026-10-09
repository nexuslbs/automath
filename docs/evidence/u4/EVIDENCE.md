# U4 option C evidence (offline LLM macro-action prior)

Role: dsh developer. Repo: nexuslbs/automath, branch main, inherited sha
0bb1196c0bd34ad47228e146e88234ac167794ca. Automath host = the reachable host in
/opt/omni/data/ssh/config (IP redacted; the other host is unreachable).

## STEP 1 - model state (raw)

Host probe: the reachable host returned `reachable: true`; the other host
returned `reachable: false, code 255, Permission denied (publickey,password)`.

Remote `hostname`: `small-llm-ubuntu-4gb-nbg1-1`.

Remote `docker ps -a` (before deploy):
```
CONTAINER ID   IMAGE                 COMMAND                  CREATED      STATUS                      PORTS                    NAMES
6c84fd5c93c3   local_discourse/app   "/sbin/boot"             2 days ago   Up 12 hours                 127.0.0.1:8088->80/tcp   app
76e9ab1d1a00   nginx:alpine          "/docker-entrypoint.…"   2 days ago   Up 12 hours                                          nginx-proxy
bcd7707eda5f   boky/postfix:latest   "/tini -- /bin/sh -c…"   2 days ago   Exited (143) 13 hours ago                            mail-relay
```
=> the small llama.cpp model container was NOT running: it was only present as
a built image + GGUF.

Remote `free -m` (before deploy):
```
               total        used        free      shared  buff/cache   available
Mem:            3814        2075        1502         146         651        1739
Swap:              0           0           0
```

Remote model artifacts present:
```
=== docker images ===
llm-deploy/minicpm5-1b:llamacpp-8345f33      c9ec1bdac130        173MB         45.8MB
llm-deploy/minicpm5-2b:llamacpp-8345f33      3154a279ab4d        173MB         45.8MB
llm-deploy/qwen3-4b:llamacpp-8345f33         cec0c0e72df9        173MB         45.8MB
llm-deploy/ternary-bonsai-8b:prism-eaecb50   ed457509192b        172MB         45.4MB
=== /root/llm-deploy/models ===
-rw-r--r-- 1 root root 1153529216 Oct  5 19:30 MiniCPM5-1B-Q8_0.gguf
...
=== port 8080 ===
8080 free
=== compose ps (llm-deploy-minicpm5-1b) ===
NAME      IMAGE     COMMAND   SERVICE   CREATED   STATUS    PORTS
```

Deploy (detached, one model at a time) log tail:
```
SKIP build (image llm-deploy/minicpm5-1b:llamacpp-8345f33 already present)
[llm-deploy] container started
[llm-deploy] health OK after 3 attempt(s)
[llm-deploy] model:         minicpm5-1b (Q8_0)
[llm-deploy] openai base:   http://127.0.0.1:8080/v1
```
`docker ps` after deploy: `llm-deploy-minicpm5-1b-llama-1 Up (healthy) 127.0.0.1:8080->8080/tcp`;
`curl -sS http://127.0.0.1:8080/health` -> `{"status":"ok"}`.

## STEP 2 - offline prior pass

### First pass (8 calls) all rejected: model spent the budget in <think>, content empty

Command (detached, on remote):
```
/opt/automath/venv/bin/python scripts/build_macro_prior.py --families result,scratch --candidates-per-family 4
```
Raw log:
```
[error] no candidate survived validation
[family result] case=binary_int allowed=24 types=50
[family scratch] case=indices allowed=24 types=50
[call 1] family=result REJECT latency=53.231s reason=no JSON object in response
[call 2] family=result REJECT latency=31.718s reason=no JSON object in response
[call 3] family=result REJECT latency=30.096s reason=no JSON object in response
[call 4] family=result REJECT latency=33.866s reason=no JSON object in response
[call 5] family=scratch REJECT latency=50.924s reason=no JSON object in response
[call 6] family=scratch REJECT latency=28.988s reason=no JSON object in response
[call 7] family=scratch REJECT latency=28.462s reason=no JSON object in response
[call 8] family=scratch REJECT latency=28.418s reason=no JSON object in response
[write] /opt/automath/repo/prior/macro_prior_minicpm5-1b.json calls=8 accepted=0
```
Raw stored responses were all empty strings with `completion_tokens=256`:
```
--- call 1 result tokens= 256 ---
''
... (all 8 identical empty content)
```
Cause (free, no generation): `GET /props` chat template contains
`{%- if enable_thinking is defined %}{%- if enable_thinking is false %}{{- '<think>\n\n</think>\n\n' }}`.
MiniCPM5 is a reasoning model; without the toggle all 256 tokens go to the
hidden think channel and `message.content` is empty. Fix committed in
`c2b7d45`: send `chat_template_kwargs={"enable_thinking": false}` and
`response_format={"type":"json_object"}`, and run 2 candidates per family so the
total stays at the 12-call budget (8 + 4 = 12).

### Second pass (4 calls, total 12): 3 accepted, 1 rejected

Command (detached, on remote):
```
/opt/automath/venv/bin/python -u scripts/build_macro_prior.py --families result,scratch --candidates-per-family 2
```
Raw log:
```
[family result] case=binary_int allowed=24 types=50
[family scratch] case=indices allowed=24 types=50
[call 1] family=result ACCEPT DefineScratchFromFunctionWithIntArg latency=19.425s
[call 2] family=result ACCEPT DefineScratchFromFunctionWithIntArg_2 latency=18.581s
[call 3] family=scratch REJECT latency=28.868s reason=JSON decode error: Expecting ',' delimiter: line 1 column 890 (char 889)
[call 4] family=scratch ACCEPT CreateArgsGroup latency=8.893s
[write] /opt/automath/repo/prior/macro_prior_minicpm5-1b.json calls=4 accepted=3
```

Per-call raw response, latency and usage recorded in the artifact
(`prior/macro_prior_minicpm5-1b.json`, 6863 bytes):
```
call 1 result  ok latency=19.425s prompt=767 completion=145 accepted
  {"name": "DefineScratchFromFunctionWithIntArg", "steps": [{"action": "DefineScratchFromFunctionWithIntArg", "args": ["from_int:IntBoolean", "from_int:FullStateArgIndex", "from_int:NodeMainIndex"]}, ...]}
call 2 result  ok latency=18.581s prompt=767 completion=140 accepted
call 3 scratch ok latency=28.868s prompt=767 completion=256 REJECT (JSON truncated at 256 tokens)
call 4 scratch ok latency=8.893s  prompt=767 completion=44  accepted
```
Validation table (from the artifact):
```
call 1 family=result  DefineScratchFromFunctionWithIntArg   accepted
call 2 family=result  DefineScratchFromFunctionWithIntArg_2 accepted
call 3 family=scratch <none>                                rejected: JSON decode error: Expecting ',' delimiter
call 4 family=scratch CreateArgsGroup                       accepted
```
Honest note: the accepted candidates are structurally valid (actions and
`from_int:` tokens resolve and the macro expands) but they are NOT good binary
recipes; the 1B model did not propose the CreateScratch + DefineScratchFromInt +
VerifyGoal shape, so the injected run earns negative reward. The seam and the
protocol are proven; the proposal quality of a 1B model is the limitation.

## STEP 3 - the prior is consumed (raw)

Built-in (no prior), then injected prior, on the remote:
```
=== BUILT-IN (no prior) ===
== case binary_int family=result macros=4 primitive_basic_actions=24 ==
  [1] result_true ... [2] result_false ... [3] result_write_true ... [4] result_check ...
=== INJECTED PRIOR ===
== case binary_int family=result macros=3 primitive_basic_actions=24 ==
  [1] DefineScratchFromFunctionWithIntArg (family=result): LLM proposed candidate for family result
      expansion: DefineScratchFromFunctionWithIntArg(3, 9, 10) ; DefineScratchFromFunctionWithIntArg(9, 10, 38) ; DefineScratchFromFunctionWithIntArg(10, 38, 16)
  [2] DefineScratchFromFunctionWithIntArg_2 (family=result) ...
  [3] CreateArgsGroup (family=scratch) ...
```
Run with the injected catalogue (the selection space is the LLM set, replacing
the built-in 4):
```
===== case binary_int (arithmetic) =====
MACRO SELECTION SPACE (3): DefineScratchFromFunctionWithIntArg, DefineScratchFromFunctionWithIntArg_2, CreateArgsGroup
step 1: action=DefineScratchFromFunctionWithIntArg macro=DefineScratchFromFunctionWithIntArg ok=False ... reward=-303.0 ...
...
SUMMARY actions=4 reward=-606.0 goal=False total_wall_s=0.127 max_history=6 dropped_history=0 action_space=3 macro_actions=3
```

## STEP 4 - protocol documented

`docs/REPORT.md` section 8 "U4 option C: offline LLM macro-action prior
protocol" describes what the LLM gets, the required JSON, the validation rules,
the `--macro-prior` injection and the explicit statement that the training loop
uses NO LLM and NO network. Committed in c2b7d45.

## STEP 5 - box freed

```
=== compose down ===
 Container llm-deploy-minicpm5-1b-llama-1  Stopped / Removed
 Network llm-deploy-minicpm5-1b_default  Removed
=== docker ps ===
app Up 12 hours 127.0.0.1:8088->80/tcp
nginx-proxy Up 12 hours
=== free -m ===
               total        used        free      shared  buff/cache   available
Mem:            3814        2095         609         146        1532        1719
```
The discourse `app` + `nginx-proxy` containers were already running before this
unit and are NOT ours to stop (not the llm-deploy project); they are the reason
~2 GB stays in use. Every llm-deploy container/image process is gone.

## Commit, secret scan, shas (raw)

```
combined scan git diff 0bb1196..HEAD | grep -nE 'PRIVATE-KEY|ghp-|ghs-|x-access-tok|sk_|AKI-A|api_key-|password-'
scan rc=1 (1 = zero hits, on every staged diff too)
HEAD (workstation)      = cd40cb68f60cf84b6b2079e20bd3c9d47d4ac113
origin/main (workstation)= cd40cb68f60cf84b6b2079e20bd3c9d47d4ac113
remote HEAD             = cd40cb68f60cf84b6b2079e20bd3c9d47d4ac113
remote origin/main      = cd40cb68f60cf84b6b2079e20bd3c9d47d4ac113
commits: 0bb1196 -> b0c19f5 -> c2b7d45 -> cd40cb6
```


