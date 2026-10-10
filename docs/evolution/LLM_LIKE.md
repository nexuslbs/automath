# LLM-like learning module for the automath math-agent (DESIGN, Unit D1)

Status: **DESIGN ONLY. No implementation is written or committed by this unit.**
This file is the complete deliverable; the commit that adds it contains this file
and nothing else. Implementation is a later unit.

Provenance:

* Task: UNIT D1, orchestrator task 4353 (automath LLM-like learning).
* Repo: `nexuslbs/automath`. Branch: `llm-like-learning`.
* Base: `origin/HEAD` = `origin/main` = `746022d8d83230949b7b50fe153166e9c86fc5a6`
  (the branch was created off this exact commit; nothing on any other branch was
  modified).
* The training stack surveyed in section 2 lives on the evolution_trainer line;
  the survey was taken from
  `origin/unsolvable-handling` = `134d6cde8c10f8602cb637b4ae58dbe79870d6af`
  (the current tip of `dynamic-nodes` -> `size-invariant-genome` ->
  `selection-mean-rate` -> `unsolvable-handling`). Quoted files are pinned to
  that sha.
* Prior evidence read (not re-derived): `gen-fitness @ 1b8d24ee`, `planner-
  pattern-goals @ c69edd71`, `selection-mean-rate @ 195e96dc` (training HEAD
  `70fcb0093f0d286ba5e6173e27fdef34cbe81ba6`), `unsolvable-handling` evidence
  pack, and the wiki page `Projects/Automath/index.md`.

Recommended implementation base (for the follow-up unit): rebase or copy this
file onto the evolution_trainer tip `134d6cde` before implementing, because the
module below imports `dynamic_env` and reads `evolution_trainer` constants that
do not exist on `main`.

---

## 1. What is borrowed from LLMs, and what is NOT

The point of this module is to take the mechanisms that make a language model
learn from its own output and re-express them at a size the automath host can
run: a CPU-only box (2 vCPU, 3814 MB, no swap, no GPU) with a target of at most
about 300k parameters. There is no deployed transformer or pretrained LLM.

### 1.1 (a) Next-step / next-token prediction

An LLM is trained to predict the next token given the preceding tokens. Here the
"tokens" are the environment's own DERIVED actions (`build`, `combine`, `set`,
`clear`, and the opt-in `pop`), and the "sequence" is an episode's solved action
trajectory. At step `t` the small net receives the node/stack state (the same
graph the existing size-invariant genome encodes) plus the recent context, and
emits one logit per legal action. The training target is the action that actually
solved that episode (the next action in its witness), so the net is trained
self-supervised over solved trajectories with a next-action cross-entropy loss.
This is the direct analogue of next-token cross-entropy. It is NOT language
modelling: there is no text, no tokenizer, no vocabulary file, no pretrained
weights. The "vocabulary" is the spec-derived action set at the current state.

### 1.2 (b) Attention / context over the last K states and actions

A transformer's useful property here is in-context conditioning: the prediction
can attend over recent experience instead of relying only on weights. The
existing graph encoder pools a single state to a fixed latent, so it has no
memory of the last few decisions. This module adds exactly ONE small attention
head (single head, single layer, no feed-forward block, no transformer stack)
over the last `K = 8` `(state, chosen action)` context tokens. The attention
output plus the current state latent feeds the policy and value heads. This lets
the agent learn from its own recent trajectory within an episode and is cheap
(`4 * d * d` parameters at head width `d`). It is explicitly NOT a transformer
stack: no multi-head, no FFN, no layer-norm block stack, no residual depth.

### 1.3 (d) Self-generated curriculum / data flywheel

The training data is the agent's OWN solved episodes, not an external corpus.
Loop: roll out the current policy, keep only episodes that reach the goal,
filter the form pool with the existing decidable solvability oracle (section
2.6), add the kept trajectories to a bounded replay buffer, retrain, and repeat.
Because the buffer only keeps what the agent can currently solve, the training
distribution grows with the policy: the "curriculum" is self-generated (the
agent's own successes define the next training set). The explicit planner and the
BFS witness are used only as a solvability filter and as an optional one-time
seed; they are not the training signal.

### 1.4 (e) Small value model over (state, next-state) transitions

A small value head scores a transition `(s, s')` with a scalar. This is the
"reward-model / critic" analogue and is trained with an MSE loss. It is optional
in the mandate, and is specified here so the implementation unit has an exact
contract. When enabled it can rank candidate reductions at inference (a
continuation is scored, not just the immediate action), which is the mechanism
that made search-and-critic pipelines work for LLMs.

### 1.5 Explicitly NOT borrowed

* No transformer stack, no multi-head attention, no positional transformer
  encoder/decoder.
* No pretrained model, no text corpus, no tokenizer, no network call in the
  training loop (the operator constraint: "no network in the training loop").
* No gradient-through-environment or differentiable engine; the environment
  stays the frozen stdlib `dynamic_env` engine and trajectories are data.
* No change to the stdlib-only boundary of `dynamic_env` / `evolution_trainer`.
  The new code is a separate package and is the only torch consumer besides the
  pre-existing `agent/smart_agent.py`.

---

## 2. Survey of the existing training stack (verbatim quotes)

All quotes below are from `origin/unsolvable-handling @ 134d6cde` unless a
different sha is named. File paths are relative to the repo root.

### 2.1 Evolution backbone

`evolution_trainer/evolution.py:50-105` (the whole knob surface, including the
selection and graceful-unsolvable flags added by tasks 4344 and 4349):

```python
50: @dataclass
51: class EvoConfig:
52:     """Every knob of the evolutionary process (CLI-overridable)."""
53:
54:     population_size: int = 12
55:     generations: int = 20
56:     episodes_per_agent: int = 1
57:     hidden: int = 12
58:     # reward shaping
59:     step_cost: float = 0.05
60:     intermediate_reward: float = 0.30
61:     guard_bonus: float = 0.20
62:     subgoal_reward: float = 0.50
63:     goal_reward: float = 1.00
64:     # reproduction
65:     reproduction_fee: float = 1.00
66:     parent_frac: float = 0.60
67:     elite_frac: float = 0.25
68:     mutation_rate: float = 0.15
69:     mutation_sigma: float = 0.20
70:     mix_alpha: float = 0.50
71:     select_prob: float = 0.50
72:     per_gene_blend: bool = True
73:     # subagent recursion
74:     subagent_depth: int = 2
75:     subagent_spawn_fee: float = 0.30
76:     subagent_credit_rate: float = 1.0
77:     subagent_credit_cap: float = 0.50
78:     max_subagents_per_episode: int = 2
79:     # execution
80:     max_episode_steps: Optional[int] = None
81:     seed: int = 7
82:     checkpoint_dir: str = "checkpoints"
83:     history_dir: str = "history"
84:     # task 4344: selection by mean solve rate + held-out validation term +
85:     # legal-action masking. Defaults preserve Unit B / Unit D behaviour exactly
86:     # (selection_episodes=0 and val_weight=0.0 turn the new machinery off).
87:     selection_episodes: int = 0
88:     selection_seed: int = 20261010
89:     selection_epsilon: float = 0.1
90:     solved_rate_weight: float = 0.5
91:     val_weight: float = 0.0
92:     val_batch: int = 16
93:     val_seed: int = 20261010
94:     val_pool_size: int = 64
95:     mask_illegal: bool = False
96:     # task 4349 unit 3: graceful unsolvable handling. Default ON; the
97:     # documented OFF switch (graceful_unsolvable=False) restores the previous
98:     # behaviour exactly. The held-out val pool (val64) has 0 provably-unsolvable
99:     # forms, so ON changes no 4344 selection number.
100:     graceful_unsolvable: bool = True
101:     # task 4349 unit 4: action set for the whole training loop. ``allow_pop``
102:     # False (default) keeps the historical four-action set byte-identical;
103:     # True enables the opt-in ``pop`` action in every training/selection/val
104:     # rollout (and thereby disables the canonical subtree prune).
105:     allow_pop: bool = False
```

`evolution_trainer/evolution.py:190-214` (the trainer installs the
SIZE-INVARIANT net, not the old spec-sized MLP):

```python
190: class EvolutionTrainer:
191:     """A fixed-weight evolutionary loop over one dynamic-nodes spec."""
192:
193:     def __init__(self, spec: Spec, config: Optional[EvoConfig] = None,
194:                  spec_id: str = "") -> None:
195:         self.spec = spec
196:         self.spec_id = spec_id or spec.spec_id
197:         self.config = config or EvoConfig()
198:         self.rng = random.Random(self.config.seed)
199:         self.featurizer = ActionFeaturizer(spec)
200:         self.net = SizeInvariantNet(self.config.hidden)
201:         self.in_dim = self.net.size
202:         self.obj_ids = objective_ids(spec)
203:         self.target = tuple(int(spec.objectives[oid]) for oid in self.obj_ids)
204:         self.root_mask = tuple(1 for _ in self.obj_ids)
205:         self.population: List[Agent] = []
206:         self.history: List[Dict[str, Any]] = []
207:         self._serial = 0
208:         self.generation_subagents: List[Dict[str, Any]] = []
209:         self._next_population: List[Agent] = []
...
214:         self.step_gate = None
```

`evolution_trainer/evolution.py:286-317` (the reward-along-the-way episode loop
that a flywheel rollout can reuse byte-for-byte; progress is credited ONCE per
objective and ONCE per guarded objective):

```python
286:         while steps < max_steps and actions:
287:             action = self._choose(agent, env, state, actions, subgoal_mask)
288:             if self.step_gate is not None and not self.step_gate(env, state, action):
289:                 # Wrong step DISCARDED (deterministic, no state change): the
290:                 # agent spends a step and tries again from the SAME state.
291:                 total -= self.config.step_cost
292:                 steps += 1
293:                 continue
294:             result = env.step(state, action)
295:             flags = env.objective_vector(result.state)
296:             reward = -self.config.step_cost
297:             step_flips = 0
298:             for i in range(n_obj):
299:                 if (subgoal_mask[i] and i not in credited
300:                         and prev_flags[i] != target[i] and flags[i] == target[i]):
301:                     reward += self.config.intermediate_reward
302:                     step_flips += 1
303:                     credited.add(i)
...
312:             if done_full:
313:                 reward += self.config.goal_reward
314:             elif done_sub:
315:                 reward += self.config.subgoal_reward
316:             total += reward
317:             steps += 1
```

The shared rollout surface used by every comparison arm is
`evolution_trainer/harness.py:61-96`:

```python
61: class Harness:
62:     """Shared rollout and evaluation over one dynamic-nodes spec."""
63:
64:     def __init__(self, spec: Spec, hidden: int = 12,
65:                  step_cost: float = 0.05,
66:                  intermediate_reward: float = 0.30,
67:                  guard_bonus: float = 0.20,
68:                  subgoal_reward: float = 0.50,
69:                  goal_reward: float = 1.00,
70:                  max_episode_steps: Optional[int] = None) -> None:
71:         self.spec = spec
72:         self.featurizer = ActionFeaturizer(spec)
73:         self.in_dim = state_dim(spec) + self.featurizer.dim
74:         self.net = PolicyNet(self.in_dim, hidden)
...
86:     def inputs(self, env: DynamicEnv, state: State,
87:                actions: Sequence[Action],
88:                mask: Sequence[int]) -> List[Tuple[float, ...]]:
89:         base = state_features(env, state, mask)
90:         return [base + self.featurizer.featurize(state, action)
91:                 for action in actions]
92:
93:     def logits(self, genome: Sequence[float], env: DynamicEnv, state: State,
94:                actions: Sequence[Action], mask: Sequence[int]) -> List[float]:
95:         return [self.net.forward(genome, x)
96:                 for x in self.inputs(env, state, actions, mask)]
```

Honest note: `harness.py` still binds the old spec-sized `PolicyNet` (see
`docs/evolution/SIZE_INVARIANT.md` section "Out of scope / known gap"); the
new module must NOT reuse `harness.py` as-is and should read trajectories from
`DynamicEnv` directly, or migrate the harness in the implementation unit.

`evolution_trainer/harness.py:116-131, 185-200` (trajectory collection already
exists and is exactly what the flywheel buffer needs):

```python
116:     def rollout(self, genome: Sequence[float],
117:                 mask: Optional[Sequence[int]] = None,
118:                 explore: bool = True,
119:                 rng: Optional[random.Random] = None,
120:                 collect: bool = False,
...
126:         env = DynamicEnv(self.spec)
127:         state = env.reset()
...
185:             if collect:
186:                 trace.append({"step": steps, "action": action.key(),
187:                               "state_vector": list(env.state_vector(result.state)),
188:                               "objective_vector": list(flags),
189:                               "reward": reward, "goal": bool(done_full),
190:                               "inputs": xs, "chosen": chosen})
...
199:         solved = solved or env.goal_reached(state)
200:         return RolloutResult(total, steps, solved, state, trace)
```

### 2.2 Target-conditioned policy and the graph-encoder genome

`evolution_trainer/size_invariant.py:41-48` (the fixed genome layout):

```python
41: Genome layout (all matrices fixed size; ``H = hidden``)::
42:
43:     W_node   H x NODE_DIM          b_node   H
44:     W_self   H x H                 W_neigh  H x H          b_msg  H
45:     W_g      H x GLOBAL_DIM        b_g      H
46:     W_a      H x ACTION_DIM        b_a      H
47:     W_c      H x H                 b_c      H
48:     w_out    H                     b_out    1
```

`evolution_trainer/size_invariant.py:74-77` (fixed widths):

```python
74: NODE_DIM = (len(ROLE_ORDER) + len(SEMANTICS_ORDER) + TYPE_HASH_DIM
75:             + ID_HASH_DIM + 1 + 1 + 3 + 2)
76: #: Fixed global scalars: node count, step, active-axiom count, objective count.
77: GLOBAL_DIM = 4
```

`evolution_trainer/size_invariant.py:192-231` (the invariant: one genome length,
no spec-sized matrices; the state is a SET of nodes, one pooled latent, and each
action is scored independently):

```python
192: class SizeInvariantNet:
193:     """A fixed-size graph-encoder policy; one scalar logit per legal action."""
194:
195:     def __init__(self, hidden: int = 12) -> None:
...
199:         self.node_dim = NODE_DIM
200:         self.global_dim = GLOBAL_DIM
201:         self.action_dim = len(KIND_ORDER) + ID_HASH_DIM + 1 + hidden
202:         offset = 0
203:         self.w_node = offset
204:         offset += hidden * self.node_dim
...
229:         self.size = offset
230:         #: Alias so callers that log ``net.in_dim`` keep working.
231:         self.in_dim = self.size
```

`evolution_trainer/size_invariant.py:323-331` (masked mean pooling, so the node
count is never a width):

```python
323:         # masked mean pooling: the node count is never a width
324:         pooled = [0.0] * hidden
325:         for vector in second:
326:             for k in range(hidden):
327:                 pooled[k] += vector[k]
328:         if second:
329:             inverse = 1.0 / float(len(second))
330:             for k in range(hidden):
331:                 pooled[k] *= inverse
```

`evolution_trainer/size_invariant.py:382-403` (one scalar logit per action, the
same genome on any spec):

```python
382:     def score(self, genome: Sequence[float], env: DynamicEnv, state: State,
383:               actions: Sequence[Action], subgoal_mask: Sequence[int]) -> List[float]:
384:         """One logit per legal action; the SAME genome scores any spec."""
385:         pooled, global_latent, node_latent = self.encode_graph(
386:             genome, env, state, subgoal_mask)
387:         hidden = self.hidden
388:         out: List[float] = []
389:         for action in actions:
390:             feature = self.action_features(env.spec, action, node_latent)
391:             action_pre = self._matvec_plus(genome, self.w_a, hidden,
392:                                            self.action_dim, feature, self.b_a)
393:             action_latent = [math.tanh(value) for value in action_pre]
394:             combined = [pooled[k] + global_latent[k] + action_latent[k]
395:                         for k in range(hidden)]
396:             combined_pre = self._matvec_plus(genome, self.w_c, hidden, hidden,
397:                                              combined, self.b_c)
398:             latent = [math.tanh(value) for value in combined_pre]
399:             score = genome[self.b_out]
400:             for k in range(hidden):
401:                 score += genome[self.w_out + k] * latent[k]
402:             out.append(score)
403:         return out
```

`docs/evolution/SIZE_INVARIANT.md:39-43, 62-76` (the invariant statement and the
constant 1153 with `H=12`):

```markdown
> One genome JSON (the same weights, no re-allocation, no re-shaping) runs on any
> spec built by `dynamic_env`. The genome parameter count and the input/output
> shapes are INDEPENDENT of the spec's node count, arity and action count. The
> input trait vectors are per-node (size-invariant) and every matrix is
> fixed-size.
...
Genome layout (all shapes fixed; `H = hidden = 12` by default, `NODE_DIM = 25`,
`ACTION_DIM = 4 + 8 + 1 + H = 25`, `GLOBAL_DIM = 4`):
...
With `H=12` the genome length is the constant **1153** for every spec.
```

The earlier spec-sized policy (reference only, still quoted by `harness.py`) is
`evolution_trainer/genome.py:24-34`:

```python
24: class PolicyNet:
25:     """A one-hidden-layer tanh MLP; input = state features + action features."""
26:
27:     def __init__(self, in_dim: int, hidden: int = 12) -> None:
...
32:         self.in_dim = in_dim
33:         self.hidden = hidden
34:         self.size = hidden * in_dim + hidden + hidden + 1
```

`evolution_trainer/features.py:45-79` (the target-conditioned input used by
`harness.py`; the new module reuses this encoding idea but with fixed widths):

```python
45: def state_dim(spec: Spec) -> int:
46:     """Length of the feature vector produced by :func:`state_features`."""
47:     return 3 * len(spec.objectives) + len(spec.node_types) + 3
...
50: def state_features(env: DynamicEnv, state: State,
51:                    subgoal_mask: Sequence[int]) -> Tuple[float, ...]:
...
63:     objs = raw[:n_obj]
64:     target = raw[n_obj:2 * n_obj]
...
72:     features.extend(float(v) for v in objs)
73:     features.extend(float(v) for v in target)
74:     features.extend(float(v) / node_total for v in counts)
...
78:     features.extend(float(m) for m in subgoal_mask)
79:     return tuple(features)
```

`evolution_trainer/features.py:107-108` (the old spec-derived action width,
exactly the dependence the new module removes):

```python
107:         self.dim = (len(KIND_ORDER) + len(self.vocab) + 1
108:                     + 2 * self.max_arity)
```

### 2.3 Dynamic-nodes environment

`dynamic_env/engine.py:736-755` (the state vector always exposes the elementar
1/0 objective nodes AND the target, so the policy is target-conditioned from the
first observation):

```python
736:     def state_vector(self, state: State) -> Tuple[Any, ...]:
737:         """The interface Unit B trains on: objective flags + target + structure.
738:
739:         The vector is fixed-length for a given spec and always exposes the
740:         elementar 1/0 objective nodes AND the target assignment, so the policy
741:         can be target-conditioned from the first observation.
742:         """
743:         type_names = tuple(sorted(self.spec.node_types))
744:         type_counts = tuple(
745:             sum(1 for n in state.nodes if n.type == name) for name in type_names
746:         )
747:         fingerprint = 0
748:         for ch in state.identity():
749:             fingerprint = (fingerprint * 131 + ord(ch)) % (2 ** 61 - 1)
750:         return (
751:             self.objective_vector(state)
752:             + self.target_vector()
753:             + type_counts
754:             + (len(state.nodes), state.step, fingerprint)
755:         )
```

`dynamic_env/engine.py:470-535` (the action space is DERIVED per spec; the
module's "next token" is one of these actions):

```python
470: def legal_actions(spec: Spec, state: State,
471:                   allow_pop: bool = False) -> Tuple[Action, ...]:
...
483:     for axiom in sorted(spec.build_axioms, key=lambda a: a.id):
...
489:         if axiom.arity == 0:
490:             actions.append(Action(kind="build", axiom_id=axiom.id, operands=()))
491:         elif len(eligible) >= axiom.arity:
492:             for combo in itertools.permutations(eligible, axiom.arity):
493:                 actions.append(
494:                     Action(kind="build", axiom_id=axiom.id, operands=tuple(combo))
495:                 )
...
515:     if allow_pop and topmost_work_node(spec, state) is not None:
516:         actions.append(Action(kind=POP))
517:
518:     actions.sort(key=lambda a: a.key())
```

`dynamic_env/spec.py:101-130` (node types are DATA, with the role/semantics
schema the module embeds):

```python
101: @dataclass(frozen=True)
102: class NodeType:
103:     name: str
104:     fields: Tuple[FieldSpec, ...]
105:     semantics: str
106:     role: str
107:     value_field: Optional[str]
...
113:         if semantics not in VALID_SEMANTICS:
114:             raise SpecError("node type %r: unknown semantics %r" % (name, semantics))
115:         if role not in VALID_ROLES:
116:             raise SpecError("node type %r: unknown role %r" % (name, role))
```

`dynamic_env/spec.py:54-56` (the entire schema vocabulary; the node-type space
is bounded by these, and the shipped specs use 1 or 4 types each):

```python
54: VALID_FIELD_TYPES = ("int", "str", "bool", "node", "nodes", "any")
55: VALID_SEMANTICS = ("literal", "combination")
56: VALID_ROLES = ("value", "tag", "objective", "plain")
```

Shipped specs and their node-type counts (raw `node_types` keys):

```
data/dynamic_env/spec_minimal.json          spec_minimal          node_types: flag (1)
data/dynamic_env/spec_dynamic_group.json    spec_dynamic_group    node_types: num,op,grp,flag (4)
data/dynamic_env/spec_multi_step.json       spec_multi_step       node_types: num,op,grp,flag (4)
data/dynamic_env/spec_dynamic_axiom.json    spec_dynamic_axiom    node_types: num,op,grp,flag (4)
```

`docs/evolution/DYNAMIC_NODES.md:50-58` (the one construct that covers every
expression; this is what the node encoder must remain able to represent):

```markdown
A node is `(id, type, fields)`. Its value is intrinsic only for `literal`
types (the declared `value_field`); a `combination` node has **no intrinsic
meaning**: its value is whatever the ACTIVE combine axiom keyed on
`(tag value, arity)` computes from its operand values. This is the dynamic
grouping node: `tag` is a node id and `items` is a tuple of node ids, so a
combination node expresses a COMBINATION of existing nodes and is the single
construct from which every math/logic expression is defined.
```

### 2.4 Curriculum

`evolution_trainer/curriculum.py:70-99` (exact BFS planning; independent of the
engine helper and cross-checked in tests):

```python
70: def bfs_min_steps(spec: Spec) -> Tuple[Optional[int], Optional[Tuple[str, ...]]]:
71:     """Exact minimal number of actions to the goal and one witness word.
...
77:     """
78:     env = DynamicEnv(spec)
79:     start = env.reset()
80:     if env.goal_reached(start):
81:         return 0, ()
82:     frontier: List[Any] = [start]
83:     seen = {start.identity()}
84:     depth = 0
85:     while frontier and depth < spec.max_steps:
86:         depth += 1
87:         nxt: List[Any] = []
88:         for state in frontier:
89:             for action in env.legal_actions(state):
90:                 result = env.step(state, action)
91:                 ident = result.state.identity()
92:                 if ident in seen:
93:                     continue
94:                 seen.add(ident)
95:                 if result.done:
96:                     return depth, result.state.history
97:                 nxt.append(result.state)
98:         frontier = nxt
99:     return None, None
```

`evolution_trainer/curriculum.py:102-149` (the unique solution path and the
strict gate; the "only one step is correct per state" relation):

```python
102: def solution_path(spec: Spec) -> Dict[str, Any]:
103:     """The unique BFS solution as a map ``state identity -> correct action key``.
...
112:     env = DynamicEnv(spec)
113:     minimum, word = bfs_min_steps(spec)
...
116:     state = env.reset()
117:     correct: Dict[str, str] = {}
118:     for key in word:
119:         correct[state.identity()] = key
...
127:         state = env.step(state, match).state
128:     return {"min_steps": minimum, "word": tuple(word), "correct": correct}
...
140:     def gate(env: DynamicEnv, state: Any, action: Action) -> bool:
141:         expected = table.get(state.identity())
142:         if expected is None:
...
146:             return True
147:         return action.key() == expected
```

`evolution_trainer/curriculum.py:152-165` (the operator MAX rule that also
defines the step budget for every rollout):

```python
152: def minimal_and_max(spec: Spec) -> Dict[str, Any]:
153:     """BFS min plus the OPERATOR max rule: min+2, or min*2 for the axiom spec."""
154:     minimum, word = bfs_min_steps(spec)
...
158:     if "axiom" in spec_id:
159:         budget = max(minimum + 2, minimum * 2)
160:         rule = "max(min+2, min*2)"
161:     else:
162:         budget = minimum + 2
163:         rule = "min+2"
```

`docs/evolution/CURRICULUM.md:12-24, 37-45` (the four stages and the exact
budgets; the flywheel reuses stage S4's protocol):

```markdown
1. **Start minimal.** S1 on `spec_minimal`; the strict stage is sub-second.
2. **Dynamic nodes with deterministic tests, strict steps.** S2 ...
3. **Little by little let the agent choose.** S3: strict sequence first, then
   free choice where a WRONG step is DISCARDED ...
4. **Multi-step goals.** S4: `spec_multi_step` and `spec_dynamic_axiom` with the
   BFS-exact MINIMAL step count, an operator MAX budget and terminal failure at
   MAX ...
...
| spec | BFS min | MAX rule | MAX |
| `spec_minimal` | 1 | (S1 fixed) | - |
| `spec_dynamic_group` | 2 | (S3 fixed) | - |
| `spec_multi_step` | 2 | min + 2 | 4 |
| `spec_dynamic_axiom` | 3 | max(min+2, min*2) | 6 |
```

### 2.5 Selection, held-out evaluation and masking

`evolution_trainer/size_selection.py:8-30` (the three fixes; the flywheel eval
adopts the same anti-artefact rules):

```python
8: 1. MEAN-SOLVE-RATE SELECTION. Selection and checkpointing rank a candidate by
9:    its solve rate on ``N = 30`` FRESH validation episodes ...
...
25: 3. ARGMAX-COLLAPSE COUNTER. ``masked_legal_actions`` removes provably dead
26:    actions ... and
27:    the argmax is taken over the masked scores only. Validation episodes use an
28:    EPSILON-GREEDY protocol (``epsilon = 0.1``) with fixed episode seeds, so the
29:    checkpoint metric is not a single deterministic argmax trajectory.
```

`evolution_trainer/size_selection.py:67-71, 106-113` (the exact constants and the
explicit per-episode seed):

```python
67: VAL_SEED = 20261010
68: POOL_SIZE = 64
69: SELECTION_EPISODES = 30
70: SELECTION_SEED = 20261010
71: SELECTION_EPSILON = 0.1
...
106: def episode_seed(selection_seed: int, generation: int, index: int) -> int:
...
113:     return int(selection_seed) * 1000000 + int(generation) * 1000 + int(index)
```

`docs/evolution/SELECTION_MEAN_RATE.md:25-37` (the fitness formulas; the new
module keeps these as the comparison baseline, not as its own objective):

```markdown
BEFORE  fitness = shaped_return + solved_rate_weight * solved_rate
                  # one best training episode on the fixed 7-state bundle
AFTER   fitness = shaped_return + solved_rate_weight * mean_solved_rate_30
                  + val_weight * val_solved_rate
```

`evolution_trainer/heldout_eval.py:63-74` (held-out constants; note this file has
already migrated to `SizeInvariantNet`, the others did not):

```python
63: HIDDEN = 12
...
65: STEP_COST = 0.05
66: INTERMEDIATE_REWARD = 0.30
67: GUARD_BONUS = 0.20
68: SUBGOAL_REWARD = 0.50
69: GOAL_REWARD = 1.00
...
72: HELDOUT_SEED = 12345
...
74: PERTURB_COUNT = 30
```

Held-out cases ship as DATA and are never used in training
(`docs/evolution/CURRICULUM.md:62-68`):

```markdown
* `data/dynamic_env/heldout/spec_multi_step_heldout.json` - the same shape as
  `spec_multi_step` with a DIFFERENT target objective state (8 instead of 4,
  reachable as `add(5,3)`).
* `data/dynamic_env/heldout/spec_dynamic_group_deep.json` - a deeper dynamic
  grouping with a LARGER arity (3 operands: `sum(1,2,5)=8`).
```

### 2.6 The solvability rule (the flywheel filter)

`evolution_trainer/solvability.py:1-35` (the decidable oracle and its three
verdicts; this is the rule the flywheel filter must follow):

```python
1: """Decidable solvability oracle for the automath action spaces (evaluation only).
...
8: Verdicts
9: --------
10: ``SOLVABLE``   - a witness action path was found AND re-executed to the goal.
11: ``UNSOLVABLE`` - SOUND: either the canonical non-subtree prune fires (no pop
12:                  action exists, so a stack node that is not a subtree of the
13:                  fixed target can never be consumed), or a bounded exhaustive
14:                  search closed the reachable space without a plan.
15: ``UNKNOWN``    - the node or time budget was exhausted before a plan or a
16:                  closure proof. A budget timeout is NEVER reported as
17:                  UNSOLVABLE.
...
35: Evaluation only: no training, no torch, no checkpoint write.
36: """
```

`evolution_trainer/solvability.py:454-480` (the entry point the flywheel calls):

```python
454: def classify(case, action_set: str = "current",
455:              node_budget: int = DEFAULT_NODE_BUDGET,
456:              time_budget: float = DEFAULT_TIME_BUDGET) -> Dict[str, Any]:
457:     """Classify ONE case; see the module docstring for the verdict contract."""
458:     problem = _adapt(case, action_set)
...
470:     # A provided witness (e.g. the 4344 solvable-only pool) is re-executed and
471:     # only accepted when it really reaches the goal.
472:     provided = getattr(case, "witness", None)
473:     if provided is not None and problem.kind == "dyn":
474:         final = problem.replay(tuple(provided))
475:         if final is not None and problem.is_goal(final):
476:             t0 = time.perf_counter()
477:             return _result(SOLVABLE, "provided_witness_replay",
478:                            list(provided), len(provided), 0, t0,
479:                            node_budget, time_budget)
480:     return _search(problem, node_budget, time_budget)
```

`docs/evolution/UNSOLVABLE.md:22-28, 331-359` (the three verdicts and the
graceful-handling rules the flywheel filter mirrors):

```markdown
| verdict | meaning |
| `SOLVABLE` | a concrete action path was found and **re-executed** to the goal |
| `UNSOLVABLE` | a **sound** proof that no plan exists (see section 3) |
| `UNKNOWN` | a budget was exhausted before either proof; never reported as `UNSOLVABLE` |
...
* **Skip provably-unsolvable forms.** Forms classified `UNSOLVABLE` cannot be
  solved by construction; feeding them to a planner trains against an
  impossible target and dilutes the reward. They are removed from the training
  mix by default.
* **Bounded attempt for `UNKNOWN`.** Any form the oracle could not settle is
  still trainable, but each attempt gets a hard step budget; on exhaustion the
  episode yields the shaped partial reward and ends ...
...
* **Solve rate is over solvable forms.** `solve_rate = solved / SOLVABLE`, with
  `UNKNOWN` and `UNSOLVABLE` counts always reported next to it so the
  denominator is auditable.
```

### 2.7 Prior measured evidence (quoted, not re-derived)

From `gen-fitness/COMPARISON.md` (task 4342, branch `gen-fitness @ 1b8d24e`):

```markdown
**An optimal generalizing agent per Axioms/Goal NOW EXISTS as the explicit
planner/value function over `REDUCTION_TABLE`.** The bounded deterministic search
solves **114/114 ext114** and **32/33 core33** ... with **zero training**.
```

from the same file, the head-to-head table:

```markdown
| **PLANNER** (explicit search, no training) | **32/33** | **114/114** | **0/40** |
| **GF-full** (A-target-conditioned-net, source_generation 228) | **14/33** | **12/114** | **0/40** |
```

From `planner-pattern-goals/FINAL_EVIDENCE.md` (task 4354 lineage, branch
`planner-pattern-goals @ c69edd7`, created from `gen-fitness @ 1b8d24e`):

```markdown
Headline: the explicit planner now generalizes from fixed-state goals to
pattern/`ExprGoal` goals by searching with the environment's own goal test.
`core33` goes `32/33 -> 33/33` (the historic `expr_two/prefix1` failure is
solved), `ext114` stays `114/114`, `unseen40` stays `0/40`, zero training, pure
standard library, deterministic.
```

From `selection-mean-rate/COMPARISON.md` (task 4344):

```markdown
| 4344 | `selection-mean-rate` @ `70fcb00` | port fixes 1-3 onto the size-invariant/curriculum backbone | **recorded metric now reproducible** (30/30 -> 30/30 on a fresh seed); shipped specs 30/30 **with the canonical mask**; still no unmasked/held-out generalization |
```

From `selection-mean-rate/FINAL_EVIDENCE.md`:

```markdown
3. **Honest caveat: the FIX-3 canonical-subtree mask is load-bearing.** The same
   new genomes with the mask **OFF** collapse to 0/30 (`spec_multi_step`,
   `spec_dynamic_group_deep`) and 1-2/30 (`spec_dynamic_axiom`).
```

From `unsolvable-handling/FINAL_EVIDENCE.md` (task 4349, three seeds, 30-episode
held-out harness, val64):

```markdown
| seed | val64 current | val64 pop | delta |
| 7 | 49/64, cov 0.765625, mean 0.497917 | 64/64, cov 1.0, mean 0.621875 | +15 cases |
| 13 | 46/64, cov 0.71875, mean 0.481771 | 46/64, cov 0.71875, mean 0.481771 | +0 cases (per-case identical) |
| 42 | 50/64, cov 0.78125, mean 0.498958 | 60/64, cov 0.9375, mean 0.543229 | +10 cases |
```

From the wiki `Projects/Automath/index.md` (the project-level verdict):

```markdown
**learned agents still do NOT generalize past the demo bundle — honest
NO GOOD RESULT on the learned-selection route** ...; **the EXPLICIT
PLANNER over the reduction table now MEETS the "optimal agent per Axioms/Goal"
objective for FIXED-STATE goals** (114/114 ext114, 32/33 core33, zero training,
deterministic — thread 4342).
```

Reading for this design: the reward/evolution ceiling is the representation and
the training signal, not the environment. The explicit planner already solves the
fixed-state families, so the LLM-like module is judged on the LEARNED route and
must not be allowed to hide a masked planner: the eval protocol below reports the
mask ON and OFF separately and keeps the planner as an upper-bound control.

---

## 3. Exact architecture

Package name: `llm_like/` (new). Runtime: `torch` CPU only. The engine
(`dynamic_env`) and the stdlib-only trainer are imported, never modified.

### 3.1 Interfaces (no engine changes)

* Input at step `t`: `env: DynamicEnv`, `state: State` (stdlib), the legal
  action tuple `env.legal_actions(state)`, and `subgoal_mask`.
* Per-node input is built by a new `llm_like/features.py` that reuses the
  `size_invariant.node_features` semantics (role, semantics, value, is-dynamic,
  objective current/target/mask, in/out degree, plus an id hash), but returns a
  torch tensor instead of a list.
* Context is the last `K = 8` `(state latent, chosen action latent)` pairs;
  at `t < K` the context is zero-padded and masked.
* Output: one logit per legal action (policy) and one scalar per transition
  (value). The number of actions is never a tensor width; actions are scored in a
  loop or a padded batch with a mask.

### 3.2 Layers and dimensions

Fixed hyperparameters (primary config):

| symbol | value | meaning |
| --- | ---: | --- |
| `d` | 64 | model width |
| `K` | 8 | context length (last states/actions) |
| `N_TYPES` | 10 | node-type cap (shipped specs use 1 or 4) |
| `N_KINDS` | 5 | action kinds: build, combine, set, clear, pop |
| `NODE_SCALARS` | 13 | role 4 + semantics 2 + value 1 + dynamic 1 + objective 3 + in/out degree 2 |
| `ID_HASH` | 8 | signed feature-hash width (spec-agnostic) |
| `NODE_IN` | 21 | `NODE_SCALARS + ID_HASH` |
| `ACTION_IN` | `N_KINDS + ID_HASH + 1 + d` = 78 | kind one-hot 5 + id hash 8 + arity 1 + mean referenced node latent `d` |

Layer table (all sizes exact):

```
node embedding      : Embedding(10, 64)                        10 x 64
node projection     : Linear(21 -> 64) + bias                  64 x 21 + 64
message passing (1) : tanh(W_self h + W_neigh * mean_neigh(h) + b)
                      W_self 64x64 ; W_neigh 64x64 ; b_msg 64
state pooling       : masked mean over nodes -> s_t in R^64
action projection   : Linear(78 -> 64) + bias                  64 x 78 + 64
context projection  : Linear(128 -> 64) + bias                 64 x 128 + 64
position embedding  : Embedding(8, 64)                         8 x 64
attention (1 head)  : W_q, W_k, W_v, W_o : 64x64 each (+4 biases of 64)
                      softmax(Q K^T / sqrt(64)) V, then W_o
policy head         : c = context output ; logit_j = w_out . tanh(W_c [c ; a_j] + b_c)
                      W_c 64x128, b_c 64, w_out 64
value head          : v(s,s') = w_v . tanh(W_v [s_t ; s_{t+1}] + b_v)
                      W_v 64x128, b_v 64, w_v 64
```

The attention block is ONE head, ONE layer, NO feed-forward sublayer, NO
layer-norm stack. The graph encoder is ONE message-passing round. This is the
SIMPLE-ML constraint made structural, not a claim.

### 3.3 Embedding of the at-most-10 node-type space, stack and goal conditioning

* Node type: every node's declared `type` string is resolved to a stable index
  in the spec's sorted `node_types` list (the shipped specs have 1 or 4 types;
  the cap is 10). Index `0` is reserved for "unknown/pad". `Embedding(10, 64)`
  maps it to `e_type`. This is the explicit token-like embedding of the
  node-type space, replacing the old spec-sized `state_dim`.
* Node identity and vocabulary: the 8-wide signed feature hash (the same
  `hash_vector` idea as `size_invariant.py`) keeps axiom/tag/objective/node-id
  strings distinguishable without a per-spec vocabulary table, so the module is
  size-invariant in the same sense as the 1153-gene genome.
* Stack and goal conditioning: the graph node set carries the stack/grouping
  structure; `objective_values`, `objective_targets` and `objective_mask` are per
  node (3 scalars in `NODE_SCALARS`), so the target is an input from step 0,
  exactly as `dynamic_env/engine.py:736-755` promises. The global scalars
  (node count, step, active axiom count, objective count) are fed to the context
  projection by concatenation with the pooled state when `t = 0`.

### 3.4 Context and attention (single head, no transformer stack)

For each step `tau` the context token is `u_tau = tanh(W_ctx [s_tau ; a_chosen_tau]
+ b_ctx)`. With the current state latent `s_t`, the module computes
`Q = W_q u_t` and `K_tau = W_k u_tau`, `V_tau = W_v u_tau` over the last `K`
tokens (plus the current one). The masked softmax attention output is
`alpha = softmax(Q K^T / sqrt(d))`, `c = W_o (alpha V)`. `c` is concatenated with
the current state latent for the policy and value heads. The mask is the
zero-padding mask for `t < K` and, in the data flywheel, the trajectory boundary
mask (attention never crosses an episode boundary).

### 3.5 Value model over (state, next-state) transitions

`v(s, s') = w_v . tanh(W_v [s ; s'] + b_v)`, a scalar in `R`. It is trained on
the same solved trajectories with a Monte-Carlo return-to-go target (the
normalized discounted sum of the existing reward shaping, or the normalized BFS
distance-to-goal when the witness is available). It is used (i) as the auxiliary
loss and (ii) optionally at inference to rank candidate reductions: score each
legal action by `(1 - beta) * policy_logit + beta * v(s, s')`, `beta = 0.5`.

### 3.6 EXACT parameter count

Counting rule (one pass, no weight tying, all biases included):

```
node_type_emb(10,d)              = 10*d
W_node(d,21)+b                   = d*21 + d
W_self(d,d)                      = d*d
W_neigh(d,d)                     = d*d
b_msg(d)                         = d
W_a(d,5+8+1+d)+b                 = d*(14+d) + d
W_ctx(d,2d)+b                    = d*(2*d) + d
pos_emb(8,d)                     = 8*d
W_q,W_k,W_v,W_o(4 x d,d)+4b      = 4*d*d + 4*d
policy W_c(d,2d)+b+w_out(d)      = d*(2*d) + d + d
value  W_v(d,2d)+b+w_v(d)        = d*(2*d) + d + d
```

Exact totals (computed, not estimated):

| `d` | params | float32 size | under the about-300k target |
| ---: | ---: | ---: | --- |
| 32 | 15,392 | 0.059 MiB | yes |
| **64 (primary)** | **57,408** | **0.219 MiB** | **yes** |
| 96 | 126,048 | 0.481 MiB | yes |
| 128 | 221,312 | 0.844 MiB | yes (largest config considered) |

Primary breakdown (`d = 64`, `K = 8`):

```
node_type_emb(10,d)              = 640
W_node(d,21)+b                   = 1408
W_self(d,d)                      = 4096
W_neigh(d,d)                     = 4096
b_msg(d)                         = 64
W_a(d,5+8+1+d)+b                 = 5056
W_ctx(d,2d)+b                    = 8256
pos_emb(8,d)                     = 512
W_q,W_k,W_v,W_o(4*d*d)+4b        = 16640
policy W_c(d,2d)+b+w_out(d)      = 8320
value  W_v(d,2d)+b+w_v(d)        = 8320
TOTAL                            = 57408
float32 bytes = 229632 ; MiB = 0.2190
```

The count above was produced by a scratch Node.js script outside the repo (no
script is committed); the formula is reproducible by hand from the layer table.
The primary config is **57,408 parameters**, well under the 300k target, and even
the widest config considered (`d = 128`) is **221,312**.

### 3.7 SIMPLE-ML / host constraint

* Box: 2 vCPU, 3814 MB, no swap, no GPU. Parameters (0.22 MiB float32) and Adam
  optimizer state (about 3x that) are negligible. The dominant memory is the
  replay buffer of trajectories and the environment itself; the implementation
  must bound the buffer (see 5.3).
* `torch` is already an existing training-only dependency of this repo
  (`requirements-train.txt`: `torch>=2.0`; the comment records the measured
  host `torch 2.14.1+cpu`). No new dependency is introduced. The shipped default
  install (`requirements.txt`) does NOT pull torch, so the stdlib-only test paths
  stay torch-free; `llm_like` tests are a separate, torch-gated command.
* Training loop: CPU only, no network. The environment rollouts dominate cost,
  which is why the flywheel keeps a bounded buffer and the module stays at tens
  of thousands of parameters.

---

## 4. Training objective

### 4.1 Next-action cross-entropy (self-supervised on solved trajectories)

For each kept step in a solved trajectory the label is the action that was
executed at that step (the "next token"). With logits `z_j` over the legal
actions `A(s)`:

```
L_policy = - (1 / N) * sum_t log softmax_j(z_j | s_t, context_t)[a_t]
```

with the cross-entropy computed over the legal action set only (illegal actions
never enter the softmax). Two label sources, both self-generated:

1. **Behavioural cloning of own successes**: the action actually taken in an
   episode that reached the goal.
2. **Best-first self-correction (optional, still self-generated)**: at a state
   where the rollout failed, the value head ranks the legal successors and the
   highest-value successor is treated as a pseudo-label only if it lies on a
   verified solved continuation (never a hand-authored hint).

### 4.2 Value MSE

```
L_value = (1 / N) * sum_t ( v(s_t, s_{t+1}) - G_t )^2
```

where `G_t` is the normalized return-to-go of the solved trajectory (existing
reward shaping from `EvoConfig`, section 2.1), optionally replaced by the
normalized BFS distance-to-goal `d(s_t)` when `solution_path` supplies the
witness:

```
G_t = 1 - d(s_t) / max(1, d(s_0))        (in [0, 1], 1 at the goal)
```

### 4.3 Full loss, optimizer, schedule

```
L = L_policy + lambda_v * L_value        default lambda_v = 0.5
```

* Optimizer: `torch.optim.Adam`, `lr = 1e-3`, no weight decay.
* Batch: all steps of the trajectories collected in one flywheel round (bounded
  by `max_buffer_steps`, section 5.3); mini-batches of 256 steps.
* Epochs per round: `1` (the flywheel, not long training, is the outer loop).
* Determinism: torch seeded from the run seed; `torch.use_deterministic_algorithms(True)`
  where available; the environment RNG is a single seeded `random.Random`, as in
  the evolution trainer.

---

## 5. Data flywheel (self-generated curriculum)

### 5.1 Loop

```
seed:      buffer = []                        # optional one-time BFS-witness seed
repeat R rounds:
  1. ROLL OUT the current model on a form pool (greedy + epsilon=0.1).
  2. FILTER the pool with the solvability rule (5.2):
       UNSOLVABLE -> never rolled out, never trained on
       UNKNOWN    -> bounded attempt only (spec.max_steps); partial reward kept
       SOLVABLE   -> eligible
  3. KEEP only episodes that reached the goal (solved trajectories).
  4. APPEND kept trajectories to the bounded replay buffer.
  5. EXCLUDE any form whose canonical path is already fully covered by the
     buffer (self-curriculum: new material is what the agent has NOT solved).
  6. TRAIN next-action CE + value MSE on the buffer (section 4).
  7. EVALUATE (section 6) and record the learning-curve point.
  8. STOP at the pre-registered budget or when held-out solution rate plateaus
     for P consecutive rounds (P = 3), never earlier.
```

The pool is the same DATA family the repo already ships: the four training specs
plus the two held-out specs, plus the solvable-only validation pool builder
`size_selection.validation_pool(64)` (whose cases are all SOLVABLE by witness
replay). The agent's own successes decide which forms stay in the buffer, so the
curriculum is generated by the agent, not authored.

### 5.2 Solvable-only filter (the rule, exactly)

For every candidate form, call
`evolution_trainer.solvability.classify(case, action_set="current" | "pop")` and
branch on the verdict, which is one of exactly three values
(`evolution_trainer/solvability.py:8-17`):

* `SOLVABLE` -> keep. The oracle re-executes the witness before emitting the
  verdict, so the label cannot be wrong.
* `UNSOLVABLE` -> drop before any rollout. It is a sound proof that no plan
  exists; training on it "dilutes the reward" and is prohibited by
  `docs/evolution/UNSOLVABLE.md` section 6.
* `UNKNOWN` -> keep the form for a bounded attempt only, with the hard
  `spec.max_steps` budget, and never count it as a failure:
  `solve_rate = solved / SOLVABLE` with `UNKNOWN` and `UNSOLVABLE` reported next
  to it (`docs/evolution/UNSOLVABLE.md:357-359`).

The filter is evaluation-only and stdlib; it adds no torch dependency.

### 5.3 Trajectory record and bounded buffer

One kept step record (superset of the `collect=True` trace in
`harness.py:185-190`):

```json
{
  "round": 0, "spec_id": "spec_multi_step", "episode": 3, "t": 2,
  "node_features": [[...], "..."], "actions": ["build:...", "..."],
  "legal_mask": [true, ...], "chosen": 1, "action_key": "build:...",
  "objective_vector": [0, 1], "target": [1, 1], "reward": 0.35,
  "goal": false, "return_to_go": 0.71
}
```

Bounded buffer: `max_buffer_steps = 200_000` step records (a step record is a
few hundred float32, so the buffer stays in the tens of MiB, inside the 3.7 GiB
box); FIFO eviction of the oldest round first. The raw per-form JSON evidence is
written to an out directory, never into the repo (scratch under
`/opt/automath/...`, mirror under `/opt/workspace/tmp/...`).

---

## 6. Evaluation protocol

Purpose: prove a LEARNING CURVE on unseen/solvable forms, not a single lucky
episode (the 4339 artefact) and not a masked planner (the 4344 caveat).

### 6.1 Learning curves

The x-axis is the cumulative number of SELF-GENERATED episodes (the flywheel's
own output), not wall time and not gradient steps. Two curves per run:

* `solution_rate(k)` on the held-out solvable set (6.2), and
* `time_to_solve(k)` = mean actions-to-solve and mean wall seconds per solved
  episode (`eval_mean_steps_to_solve` semantics from `harness.py:225-227`).

Each point is the mean over the same fixed episode seeds; raw per-episode
traces are stored.

### 6.2 Held-out solvable forms (never in training)

* `data/dynamic_env/heldout/spec_multi_step_heldout.json` and
  `spec_dynamic_group_deep.json` (2 cases; shipped as DATA).
* `evolution_trainer.size_selection.validation_pool(64)` (all SOLVABLE by
  witness replay, disjoint from every training canonical path).
* A FRESH held-out pool built by the same generator with a seed distinct from
  the training seed and from `VAL_SEED = 20261010` (e.g. `EVAL_SEED = 424242`,
  the repo's held-out eval seed), to avoid tuning against one fixed draw.
* The 30 perturbation variants from `heldout_eval.build_perturbations`
  (`PERTURB_COUNT = 30`).

Every held-out case is classified with the three-way oracle, and the report is
the three-way decomposition: `SOLVABLE-SOLVED`, `SOLVABLE-UNSOLVED`,
`PROVABLY-UNSOLVABLE`. The `UNSOLVABLE` bucket is never counted as failure.

### 6.3 Mask ON and mask OFF, separately

The 4344 result showed the canonical-subtree mask is load-bearing (30/30 with the
mask, 0/30 without). The LLM-like module is a learned policy, so the headline
number must be reported with the mask OFF; the mask-ON number is reported beside
it as the "planner-assisted" control, never as the learned result.

### 6.4 Seeds, baselines and controls

* Seeds `7 / 13 / 42` (the repo's three-seed convention), same as tasks
  4339/4344/4349.
* Baselines on the same protocol: (i) the 1153-gene size-invariant genome +
  canonical mask (30/30 shipped), (ii) the explicit planner
  (`core33 33/33`, `ext114 114/114`, zero training), (iii) an UNTRAINED
  `llm_like` net (the random-policy control; must not solve), (iv) the
  single-episode-artefact protocol from 4339, shown to be non-reproducible.
* Negative control: `unseen40` forms that are `PROVABLY-UNSOLVABLE` must score 0
  and are reported separately; a model that "solves" them is a red flag.

### 6.5 Raw evidence format

Per seed: `heldout_<seed>.json` with per-case verdicts and traces, a
`learning_curve.csv` (`episodes, solution_rate, mean_steps, wall_s`), a
`DONE` marker, and `SHA256SUMS`. No host address, IP, or credential in any
artifact.

---

## 7. Module layout and non-goals

Proposed files (implementation unit):

```
llm_like/__init__.py
llm_like/features.py    # node/action/context feature builders (torch tensors)
llm_like/model.py       # LLMAgentNet: graph encoder + 1 attention head + heads
llm_like/flywheel.py    # rollout -> solvability filter -> kept trajectories -> buffer
llm_like/train.py       # CE + value MSE loop, checkpoints, learning-curve CSV
llm_like/eval.py        # held-out three-way eval, per-seed JSON
llm_like/tests.py       # torch-gated unit checks (shapes, determinism, param count)
docs/evolution/LLM_LIKE.md   # this design
```

Non-goals (explicitly out of scope): changing `dynamic_env`, changing the
stdlib-only `evolution_trainer` contract, adding a new third-party dependency
beyond the existing `torch` training dependency, deploying any transformer/LLM,
and claiming a result before the eval protocol in section 6 is run.

## 8. Risks and honest limits

* The learned route has failed to generalize three times on this repo (4336,
  4339, 4344); the default expectation is an honest negative unless the learning
  curve in section 6 moves. The value head and the attention head are hypotheses,
  not known wins.
* The feature hash can collide (documented in `SIZE_INVARIANT.md` section 6).
* One message-passing round and one attention head may under-fit deep grouping;
  more rounds/heads would be a fixed, size-invariant change, but only after the
  primary config is measured.
* Held-out solvable coverage is small (2 shipped specs + pools); the fresh pool
  seed is there to reduce, not remove, overfitting to the eval draw.
* `harness.py`, `mix_train.py`, `rl_train.py`, `compare_arms.py` still bind the
  old spec-sized `PolicyNet`; the implementation unit must not assume they
  describe the current net.

## 9. Planned reproduction commands (implementation unit)

```sh
# stdlib paths stay torch-free and must stay green
/opt/automath/venv/bin/python -m dynamic_env.tests
/opt/automath/venv/bin/python -m evolution_trainer.tests
/opt/automath/venv/bin/python -m evolution_trainer.curriculum_tests

# torch-only module (new)
/opt/automath/venv/bin/python -m llm_like.tests
/opt/automath/venv/bin/python -m llm_like.train --seed 7  --rounds 40 --out out/llm_like/s7
/opt/automath/venv/bin/python -m llm_like.eval  --seed 7  --heldout --out out/llm_like/s7
# per-seed loop 7 / 13 / 42, then the learning-curve CSV and SHA256SUMS
```

## 10. Design-before-implementation statement

This file is the design. It was written and committed on branch
`llm-like-learning` before any `llm_like/` code exists in the repository. The
commit that introduces this file contains only this file (proof:
`git show --stat HEAD` names `docs/evolution/LLM_LIKE.md` and nothing else). The
parameter counts, architecture, objectives, flywheel filter and eval protocol
above are frozen as the implementation contract; any deviation is a new design
revision, not a silent change.
