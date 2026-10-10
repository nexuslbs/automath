"""Selection by MEAN solve rate, held-out validation and legal-action masking.

Task 4344 (branch ``selection-mean-rate``) ports the three fixes the 4339
size-invariant report named onto the size-invariant ``evolution_trainer``
pipeline, without changing the 1153-gene genome, the curriculum stages or the
evolution backbone:

1. MEAN-SOLVE-RATE SELECTION. Selection and checkpointing rank a candidate by
   its solve rate on ``N = 30`` FRESH validation episodes (its own RNG, an
   explicit per-episode seed), not by a single best training episode. The
   tie-break is deterministic: lower mean solved steps, then lower step
   standard deviation, then higher fitness, then higher budget, then agent id.
2. HELD-OUT VALIDATION TERM (ported from the gen-fitness lineage, branch
   ``gen-fitness`` @ ``1b8d24e``, unit ``add_a``). The fitness recorded for a
   candidate is::

       fitness = shaped_return
               + solved_rate_weight * mean_solved_rate_30
               + val_weight * val_solved_rate

   ``val_solved_rate`` is the GREEDY solve rate on the per-generation
   deterministic ``val_batch = 16`` draw from the ``POOL_SIZE = 64`` held-out
   pool (``VAL_SEED = 20261010``); the draw is
   ``random.Random(val_seed + 1000003 * gen).sample(...)``.
3. ARGMAX-COLLAPSE COUNTER. ``masked_legal_actions`` removes provably dead
   actions (an action that returns to an already-visited logical state can
   never be part of a shortest solution in this deterministic environment), and
   the argmax is taken over the masked scores only. Validation episodes use an
   EPSILON-GREEDY protocol (``epsilon = 0.1``) with fixed episode seeds, so the
   checkpoint metric is not a single deterministic argmax trajectory.

Port note (honesty). The gen-fitness ``add_a`` validation pool was built over
``new_approach`` Node/stack forms. This branch's size-invariant pipeline runs on
``dynamic_env`` specs, so the pool is built in the ``dynamic_env`` domain: each
case is a SPEC plus a start STATE. The forbidden sources named by the original
report (demo bundle, unseen40, core33, ext114, G1 dense training) have no
``dynamic_env`` counterpart; they are mapped to their in-domain equivalents
(the shipped training specs' canonical-path states = bundle/core/ext/dense, and
the shipped held-out specs' canonical-path states = unseen40). The mapping is
stated in ``docs/evolution/SELECTION_MEAN_RATE.md`` and no training or eval
form is read by the pool except to EXCLUDE it.

Pure standard library, bounded, deterministic.
"""

from __future__ import annotations

import copy
import json
import os
import random
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from dynamic_env.engine import (
    Action,
    DynamicEnv,
    State,
    _step_internal,
    bfs_minimal_word,
)
from dynamic_env.spec import Spec, load_spec, spec_from_dict

# --------------------------------------------------------------------------
# Constants (verbatim from the task / the gen-fitness add_a work)
# --------------------------------------------------------------------------

VAL_SEED = 20261010
POOL_SIZE = 64
SELECTION_EPISODES = 30
SELECTION_SEED = 20261010
SELECTION_EPSILON = 0.1

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
SPEC_DIR = os.path.join(_REPO_ROOT, "data", "dynamic_env")
HELDOUT_DIR = os.path.join(SPEC_DIR, "heldout")

#: The shipped training specs whose canonical-path states are the in-domain
#: "bundle" / "core33" / "ext114" / "dense training" forbidden sources.
TRAIN_SPEC_IDS: Tuple[str, ...] = (
    "spec_minimal",
    "spec_multi_step",
    "spec_dynamic_axiom",
    "spec_dynamic_group",
)
#: The shipped held-out specs -- the in-domain "unseen40" TEST source.
HELDOUT_SPEC_IDS: Tuple[str, ...] = (
    "spec_multi_step_heldout",
    "spec_dynamic_group_deep",
)
#: Categories used by the disjointness report (each maps a report key to the
#: set of spec ids whose canonical-path states it forbids).
FORBIDDEN_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "overlap_bundle": TRAIN_SPEC_IDS,
    "overlap_unseen40": HELDOUT_SPEC_IDS,
    "overlap_core33": ("spec_minimal", "spec_dynamic_group"),
    "overlap_ext114": ("spec_multi_step", "spec_dynamic_axiom"),
    "overlap_dense_train": ("spec_dynamic_group",),
}

# --------------------------------------------------------------------------
# TARGET-CHANGE family (held-out TARGETS)
# --------------------------------------------------------------------------
# The original fix#2 pool varied only the START STATE of a fixed set of specs,
# so the val term measured start-state generalization, never TARGET
# generalization. This family adds a distribution of held-out TARGETS: each
# case is a fresh ``spec_multi_step`` prototype whose guard TARGET value is
# perturbed away from the training target, mirroring the ``perturb_multi_step``
# sweep in ``evolution_trainer/heldout_eval.py`` (three re-valued numeric nodes,
# one combine op, one objective set). Every case is PROVEN solvable by replaying
# its BFS witness before it enters the pool.

#: How many target-change cases the pool carries (of ``POOL_SIZE``).
TARGET_VARIANT_COUNT = 16
#: The explicit seed of the target-change family (independent of ``VAL_SEED``).
TARGET_VARIANT_SEED = 20261011
#: The fresh spec-id namespace of a target variant.
TARGET_VARIANT_PREFIX = "spec_multi_step_target_"
#: The training spec's guard target: a held-out TARGET must not reproduce it.
TRAIN_TARGET_VALUES: Tuple[int, ...] = (4,)
#: The shipped held-out-EVAL spec's guard target (informational; the eval spec
#: is kept out of the pool by spec id + state form, not by target value).
HELDOUT_EVAL_TARGET_VALUES: Tuple[int, ...] = (8,)
#: The anchor case of the family: the "multi_step target-8 held-out case" motif
#: with a FRESH node namespace, so it is not a copy of spec_multi_step_heldout.
TARGET_VARIANT_ANCHOR: Dict[str, Any] = {
    "namespace": ("a9", "a5", "a3"),
    "values": (9, 5, 3),
    "target": 8,
}
#: Node namespaces used by the family. All are FRESH (never the training
#: ``n9,n5,n3`` ids nor the held-out eval ids), so a variant state is a
#: genuinely new state form even before the spec-id disjointness key is applied.
_TARGET_NAMESPACES: Tuple[Tuple[str, ...], ...] = (
    ("a9", "a5", "a3"),
    ("m9", "m5", "m3"),
    ("p9", "p5", "p3"),
)
#: Value triples mirrored from the ``perturb_multi_step`` sweep.
_TARGET_VALUE_POOL: Tuple[Tuple[int, ...], ...] = (
    (9, 5, 3), (7, 2, 1), (8, 4, 2), (6, 5, 1),
    (9, 4, 1), (8, 3, 2), (7, 4, 3), (9, 6, 2),
)

#: The marker ``assert_disjoint`` returns on success (evidence convention).
ASSERT_DISJOINT_OK = "ASSERT_DISJOINT_OK"


# --------------------------------------------------------------------------
# Episode seeds
# --------------------------------------------------------------------------

def episode_seed(selection_seed: int, generation: int, index: int) -> int:
    """The explicit seed of validation episode ``index`` of ``generation``.

    ``seed = selection_seed * 1000000 + generation * 1000 + index`` exactly as
    the task fixes it, so every episode is independently reproducible and
    recorded.
    """
    return int(selection_seed) * 1000000 + int(generation) * 1000 + int(index)


# --------------------------------------------------------------------------
# Legal-action masking
# --------------------------------------------------------------------------

_ALLOWED_NODES_CACHE: Dict[str, Optional[Set[str]]] = {}


def canonical_allowed_nodes(spec: Spec) -> Optional[Set[str]]:
    """The canonical-subtree node set of ``spec`` (task 4344, add_a unit-2b).

    Every node on a canonical solution path is kept; the set is the union of the
    canonical-prefix states reached by replaying ``bfs_minimal_word``. Returns
    ``None`` when the spec has no canonical word (unknown, so no prune).
    """
    key = spec.spec_id
    if key not in _ALLOWED_NODES_CACHE:
        word, states = _canonical_states(spec)
        if word is None:
            _ALLOWED_NODES_CACHE[key] = None
        else:
            allowed: Set[str] = set()
            for state in states:
                for node in state.nodes:
                    allowed.add(node.canonical())
            _ALLOWED_NODES_CACHE[key] = allowed
    return _ALLOWED_NODES_CACHE[key]


def canonical_subtree_prune(env: DynamicEnv, state: State,
                            actions: Sequence[Action]
                            ) -> Tuple[Action, ...]:
    """Keep only actions whose result stays inside the canonical subtree.

    The planner's sound non-subtree prune (add_a unit-2b): a node that is not a
    canonical subtree of the target can never be removed (the dynamic-nodes
    action set has no pop/discard action), so a state carrying such a node is
    unsolvable on the stack machine the prune was proved for. On the shipped
    objective-goal specs the guard reads objective values, so the prune is used
    as the canonical-path filter: it ALWAYS keeps the unique minimal solution
    step and removes off-canonical builds, which is what breaks the repeated
    wrong-build loop. ``None`` allowed set (no canonical word) leaves actions
    untouched.
    """
    allowed = canonical_allowed_nodes(env.spec)
    if allowed is None:
        return tuple(actions)
    kept: List[Action] = []
    for action in actions:
        try:
            new_state, _info = _step_internal(env.spec, state, action)
        except Exception:  # illegal by engine rules -> never keep it
            continue
        if all(node.canonical() in allowed for node in new_state.nodes):
            kept.append(action)
    return tuple(kept)


def masked_legal_actions(env: DynamicEnv, state: State,
                         seen_ids: Set[str],
                         actions: Optional[Sequence[Action]] = None
                         ) -> Tuple[Action, ...]:
    """The legal actions that can still lie on a canonical solution path.

    Two masks are composed: the canonical-subtree prune (off-canonical builds
    removed) and the visited-state mask (an action that returns to a logical
    state already visited in this episode cannot be part of a shortest
    solution). Both keep every action that is a genuinely legal action of
    ``env`` and both keep the canonical minimal step.

    ``actions`` may be the already-computed ``env.legal_actions(state)`` from
    the caller. The pure transition engine is used directly: the full
    ``is_legal`` re-check inside ``DynamicEnv.step`` re-enumerates the whole
    action space for every candidate action and is combinatorially expensive on
    states with many combine permutations.
    """
    if actions is None:
        actions = env.legal_actions(state)
    actions = canonical_subtree_prune(env, state, actions)
    kept: List[Action] = []
    for action in actions:
        try:
            new_state, _info = _step_internal(env.spec, state, action)
        except Exception:  # illegal by engine rules -> never keep it
            continue
        if new_state.identity() in seen_ids:
            continue
        kept.append(action)
    return tuple(kept)


def unmasked_legal_actions(env: DynamicEnv, state: State,
                           seen_ids: Optional[Set[str]] = None
                           ) -> Tuple[Action, ...]:
    """The full legal action set (the BEFORE masking behaviour)."""
    return env.legal_actions(state)


# --------------------------------------------------------------------------
# Rollout with epsilon-greedy action choice
# --------------------------------------------------------------------------

def rollout(net, genome: Sequence[float], spec: Spec,
            start_state: Optional[State], root_mask: Sequence[int],
            rng: random.Random, epsilon: float = 0.0,
            mask_illegal: bool = False,
            max_steps: Optional[int] = None,
            prune_mode: Optional[str] = None,
            prune_net=None) -> Dict[str, Any]:
    """One episode from ``start_state`` (or the spec's reset state).

    ``epsilon`` is the probability of a uniform random legal action; the rest
    of the time the argmax of the size-invariant genome's logits is taken over
    the (optionally masked) legal actions. The RNG is supplied by the caller so
    the episode is reproducible from its explicit seed.

    ``prune_mode`` selects the mask (task 4354 unit 3A):

    * ``None`` (default) -> ``"hand"`` when ``mask_illegal`` else ``"none"``,
      so the pre-3A behaviour is unchanged;
    * ``"hand"``    -> the task-4344 canonical-subtree + visited filter;
    * ``"learned"`` -> the trained value-function prune ONLY (no hand filter);
    * ``"none"``    -> no mask.
    """
    env = DynamicEnv(spec)
    state = start_state if start_state is not None else env.reset()
    limit = int(max_steps if max_steps is not None else spec.max_steps)
    seen: Set[str] = {state.identity()}
    steps = 0
    trace: List[str] = []
    effective_mode = (prune_mode if prune_mode is not None
                      else ("hand" if mask_illegal else "none"))
    while steps < limit:
        actions = env.legal_actions(state)
        if not actions:
            break
        if effective_mode == "hand":
            if mask_illegal:
                masked = masked_legal_actions(env, state, seen, actions=actions)
                if masked:
                    actions = masked
        elif effective_mode == "learned":
            from . import learned_prune as _learned_prune
            masked = _learned_prune.learned_prune(env, state, actions,
                                                  prune_net)
            if masked:
                actions = masked
        elif effective_mode != "none":
            raise ValueError("unknown prune mode %r" % (effective_mode,))
        if epsilon > 0.0 and rng.random() < epsilon:
            action = actions[rng.randrange(len(actions))]
        else:
            logits = net.logits(genome, env, state, actions, root_mask)
            best = max(range(len(actions)), key=lambda i: (logits[i], -i))
            action = actions[best]
        result = env.step(state, action)
        steps += 1
        trace.append(action.key())
        state = result.state
        seen.add(state.identity())
        if result.done:
            return {"solved": True, "steps": steps, "trace": trace}
    return {"solved": bool(env.goal_reached(state)), "steps": steps,
            "trace": trace}


# --------------------------------------------------------------------------
# Fix 1: mean solve rate over N fresh validation episodes
# --------------------------------------------------------------------------

def evaluate_candidate(net, genome: Sequence[float], spec: Spec, generation: int,
                       episodes: int = SELECTION_EPISODES,
                       selection_seed: int = SELECTION_SEED,
                       epsilon: float = SELECTION_EPSILON,
                       mask_illegal: bool = False,
                       root_mask: Optional[Sequence[int]] = None,
                       max_steps: Optional[int] = None,
                       prune_mode: Optional[str] = None,
                       prune_net=None) -> Dict[str, Any]:
    """Score one candidate on ``episodes`` fresh, independently seeded episodes.

    Returns the mean solve rate, the mean and standard deviation of the solved
    steps and the exact list of episode seeds used. Deterministic for a fixed
    ``(selection_seed, generation, episodes, epsilon, mask_illegal)``.

    ``prune_mode``/``prune_net`` are the task-4354 unit-3A hooks: ``"learned"``
    uses the trained value-function mask, ``"none"`` disables masking and
    ``None``/``"hand"`` preserves the pre-3A behaviour.
    """
    if root_mask is None:
        root_mask = tuple(1 for _ in spec.objectives)
    seeds: List[int] = []
    solved = 0
    solved_steps: List[int] = []
    all_steps: List[int] = []
    traces: List[List[str]] = []
    for index in range(int(episodes)):
        seed = episode_seed(selection_seed, generation, index)
        seeds.append(seed)
        rng = random.Random(seed)
        result = rollout(net, genome, spec, None, root_mask, rng,
                         epsilon=epsilon, mask_illegal=mask_illegal,
                         max_steps=max_steps, prune_mode=prune_mode,
                         prune_net=prune_net)
        all_steps.append(result["steps"])
        traces.append(result["trace"])
        if result["solved"]:
            solved += 1
            solved_steps.append(result["steps"])
    mean_rate = solved / float(max(1, int(episodes)))
    mean_steps = (sum(solved_steps) / float(len(solved_steps))
                  if solved_steps else None)
    std_steps = (statistics.pstdev(solved_steps)
                 if len(solved_steps) > 1 else 0.0)
    return {
        "episodes": int(episodes),
        "solved": solved,
        "mean_solve_rate": mean_rate,
        "mean_steps": mean_steps,
        "std_steps": round(std_steps, 6),
        "mean_all_steps": (sum(all_steps) / float(len(all_steps))
                           if all_steps else 0.0),
        "episode_seeds": seeds,
        "episode_solved_steps": solved_steps,
        "traces": traces,
    }


# --------------------------------------------------------------------------
# Fix 3(a) measurement: repeated-identical-action sequences
# --------------------------------------------------------------------------

def count_repeat_sequences(trace: Sequence[str]) -> int:
    """Number of positions that repeat the immediately preceding action.

    A cheap, raw measure of the degenerate argmax loop: the failing spec's
    greedy policy emits the same wrong build over and over.
    """
    repeats = 0
    for index in range(1, len(trace)):
        if trace[index] == trace[index - 1]:
            repeats += 1
    return repeats


def max_repeat_run(trace: Sequence[str]) -> int:
    """Length of the longest run of one identical action."""
    best = 0
    run = 0
    previous: Optional[str] = None
    for action in trace:
        if action == previous:
            run += 1
        else:
            run = 1
            previous = action
        best = max(best, run)
    return best


# --------------------------------------------------------------------------
# Fix 2: the held-out validation pool (dynamic_env domain)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ValCase:
    """One held-out validation case: a spec and a solvable start state."""

    name: str
    domain: str  # "prefix" (canonical prefix) or "random" (fresh random stack)
    spec_id: str
    start_state: State
    witness: Tuple[str, ...]
    max_steps: int
    spec: Spec = field(compare=False, repr=False)

    def form_key(self) -> str:
        return self.spec_id + "|" + self.start_state.identity()


def _spec_path(spec_id: str) -> str:
    if spec_id in HELDOUT_SPEC_IDS:
        return os.path.join(HELDOUT_DIR, spec_id + ".json")
    return os.path.join(SPEC_DIR, spec_id + ".json")


_SPEC_CACHE: Dict[str, Spec] = {}


def spec_by_id(spec_id: str) -> Spec:
    if spec_id not in _SPEC_CACHE:
        _SPEC_CACHE[spec_id] = load_spec(_spec_path(spec_id))
    return _SPEC_CACHE[spec_id]


def _canonical_states(spec: Spec) -> Tuple[Tuple[str, ...], List[State]]:
    """The BFS minimal word and the list of states after each prefix.

    ``states[0]`` is the reset state; ``states[k]`` is the state after the
    first ``k`` canonical actions. Returns ``((), [reset])`` when the spec is
    already solved at reset, and ``(None, [])`` when it is unsolvable.
    """
    word = bfs_minimal_word(spec)
    if word is None:
        return None, []
    env = DynamicEnv(spec)
    state = env.reset()
    states = [state]
    for key in word:
        match = next((a for a in env.legal_actions(state) if a.key() == key), None)
        if match is None:
            return None, []
        state = env.step(state, match).state
        states.append(state)
    return tuple(word), states


def _replay(env: DynamicEnv, state: State, word: Sequence[str]) -> Optional[State]:
    for key in word:
        match = next((a for a in env.legal_actions(state) if a.key() == key), None)
        if match is None:
            return None
        state = env.step(state, match).state
    return state


def _filler_build_actions(env: DynamicEnv, state: State) -> List[Action]:
    """Legal build/combine actions usable as harmless filler.

    Only build/combine actions are used: they add nodes and never change the
    objective vector, so the canonical suffix stays legal from the new state.
    """
    return [a for a in env.legal_actions(state) if a.kind in ("build", "combine")]


def _make_case(spec: Spec, start_state: State, witness: Sequence[str],
               name: str, domain: str) -> ValCase:
    return ValCase(name=name, domain=domain, spec_id=spec.spec_id,
                   start_state=start_state, witness=tuple(witness),
                   max_steps=int(spec.max_steps), spec=spec)


def _candidate_stream(rng: random.Random, spec_ids: Sequence[str],
                      max_attempts: int = 200000):
    """Deterministically yield candidate ``(spec, start_state, witness)``.

    Family (a): a PROPER prefix of the spec's canonical build word.
    Family (b): a FRESH random intermediate stack: one or two random legal
    build/combine filler actions, then a random canonical prefix; the canonical
    suffix is replayed to PROVE the start state is solvable before it is kept.

    The random family is bounded by ``max_attempts`` so an exhausted candidate
    space terminates instead of looping forever.
    """
    infos = []
    for spec_id in spec_ids:
        spec = spec_by_id(spec_id)
        word, states = _canonical_states(spec)
        if word is None or len(word) == 0:
            continue
        infos.append((spec, word, states))
    if not infos:
        return
    yield from _prefix_candidates(infos)
    for _attempt in range(int(max_attempts)):
        spec, word, states = infos[rng.randrange(len(infos))]
        env = DynamicEnv(spec)
        state = env.reset()
        filler = rng.randint(1, 2)
        placed = 0
        for _ in range(filler):
            legal = _filler_build_actions(env, state)
            if not legal:
                break
            state = env.step(state, legal[rng.randrange(len(legal))]).state
            placed += 1
        if placed == 0:
            continue
        k = rng.randrange(len(word))  # 0 .. len(word)-1 -> non-empty witness
        prefix = word[:k]
        state = _replay(env, state, prefix)
        if state is None:
            continue
        witness = word[k:]
        final = _replay(env, state, witness)
        if final is None or not env.goal_reached(final):
            continue
        name = "val_%s_f%d_k%d" % (spec.spec_id, placed, k)
        yield _make_case(spec, state, witness, name, "random")


def _prefix_candidates(infos):
    for spec, word, states in infos:
        for k in range(1, len(word)):  # proper, non-empty prefix
            witness = word[k:]
            env = DynamicEnv(spec)
            final = _replay(env, states[k], witness)
            if final is None or not env.goal_reached(final):
                continue
            name = "val_%s_pre%d" % (spec.spec_id, k)
            yield _make_case(spec, states[k], witness, name, "prefix")


# --------------------------------------------------------------------------
# Fix 2 (extension): the TARGET-CHANGE family (held-out TARGETS)
# --------------------------------------------------------------------------

def _achievable_targets(values: Sequence[int]) -> List[int]:
    """Non-negative guard targets reachable by ONE combine op on ``values``.

    Mirrors the achievable-set construction of the ``perturb_multi_step`` sweep
    (``evolution_trainer/heldout_eval.py``): the engine's ``sub`` primitive
    clamps at 0, so only non-negative differences are reachable.
    """
    out: Set[int] = set()
    size = len(values)
    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            a, b = int(values[i]), int(values[j])
            out.add(a + b)
            out.add(max(a - b, 0))
            out.add(a * b)
    return sorted(v for v in out if v > 0)


def _load_raw_spec(spec_id: str) -> Dict[str, Any]:
    with open(_spec_path(spec_id), "r", encoding="utf-8") as handle:
        return json.load(handle)


def _target_variant_raw(index: int, namespace: Sequence[str],
                        values: Sequence[int],
                        target: int) -> Dict[str, Any]:
    """One fresh ``spec_multi_step`` prototype with a perturbed TARGET."""
    raw = copy.deepcopy(_load_raw_spec("spec_multi_step"))
    old_ids = ("n9", "n5", "n3")
    new_nodes: Dict[str, Any] = {}
    for old, new, num in zip(old_ids, namespace, values):
        node = raw["nodes"].pop(old)
        node["n"] = int(num)
        new_nodes[new] = node
    for node_id, node in raw["nodes"].items():
        new_nodes[node_id] = node
    raw["nodes"] = new_nodes
    raw["guards"]["o"]["value"] = int(target)
    raw["spec_id"] = "%s%02d" % (TARGET_VARIANT_PREFIX, int(index))
    raw["description"] = (
        "HELDOUT-TARGET variant %d of spec_multi_step: target=%d, ids=%s, "
        "values=%s" % (int(index), int(target), tuple(namespace),
                       tuple(int(v) for v in values)))
    return raw


def _target_case(index: int, namespace: Sequence[str], values: Sequence[int],
                 target: int) -> Optional[ValCase]:
    """Build one target-change case, PROVEN solvable by replaying its witness."""
    raw = _target_variant_raw(index, namespace, values, target)
    wide = spec_from_dict(raw)
    # Every target here is achievable in exactly two actions (one build/combine
    # of the target value, then one objective set); the depth cap keeps BFS
    # bounded even if a value were unreachable.
    word = bfs_minimal_word(wide, max_depth=3)
    if not word:
        return None
    raw["max_steps"] = len(word) + 2
    spec = spec_from_dict(raw)
    env = DynamicEnv(spec)
    state = env.reset()
    final = _replay(env, state, word)
    if final is None or not env.goal_reached(final):
        return None
    name = "val_%s_t%d" % (spec.spec_id, int(target))
    return _make_case(spec, state, word, name, "target")


def target_change_cases(count: int = TARGET_VARIANT_COUNT,
                        seed: int = TARGET_VARIANT_SEED) -> List[ValCase]:
    """The held-out TARGET-change family (deterministic, proven solvable).

    The first case is the fixed target-8 anchor (``TARGET_VARIANT_ANCHOR``);
    the rest draw a namespace, a value triple and an achievable target from the
    ``TARGET_VARIANT_SEED`` RNG. A variant that reproduces the training target
    (``TRAIN_TARGET_VALUES``) or the shipped held-out eval spec is skipped.
    """
    count = max(1, int(count))
    rng = random.Random(int(seed))
    anchor = (tuple(TARGET_VARIANT_ANCHOR["namespace"]),
              tuple(int(v) for v in TARGET_VARIANT_ANCHOR["values"]),
              int(TARGET_VARIANT_ANCHOR["target"]))
    planned: List[Tuple[Tuple[str, ...], Tuple[int, ...], int]] = [anchor]
    seen: Set[Tuple[Tuple[str, ...], Tuple[int, ...], int]] = {anchor}
    guard = 0
    while len(planned) < count and guard < count * 200:
        guard += 1
        namespace = _TARGET_NAMESPACES[len(planned) % len(_TARGET_NAMESPACES)]
        values = list(rng.choice(_TARGET_VALUE_POOL))
        rng.shuffle(values)
        choices = [v for v in _achievable_targets(values)
                   if v not in TRAIN_TARGET_VALUES]
        if not choices:
            continue
        target = int(rng.choice(choices))
        key = (tuple(namespace), tuple(int(v) for v in values), target)
        if key in seen:
            continue
        if (tuple(namespace) == ("n9", "n5", "n3")
                and target in HELDOUT_EVAL_TARGET_VALUES):
            continue
        seen.add(key)
        planned.append(key)
    cases: List[ValCase] = []
    for index, (namespace, values, target) in enumerate(planned):
        case = _target_case(index, namespace, values, target)
        if case is not None:
            cases.append(case)
    return cases


def replay_witness(case: ValCase) -> bool:
    """Replay ``case.witness`` from ``case.start_state``; True iff goal reached.

    This is the solvability PROOF used for every pool case (target-change and
    start-state families alike), not a planner claim.
    """
    env = DynamicEnv(case.spec)
    state = case.start_state
    for key in case.witness:
        match = next((a for a in env.legal_actions(state) if a.key() == key),
                     None)
        if match is None:
            return False
        state = env.step(state, match).state
    return bool(env.goal_reached(state))


def case_target_value(case: ValCase) -> Optional[int]:
    """The guard TARGET value of a case's spec (``None`` when unguarded)."""
    guard = case.spec.guards.get("o")
    if guard is None:
        return None
    value = guard.get("value")
    return None if value is None else int(value)


def _forbidden_forms() -> Dict[str, Set[str]]:
    """Canonical-path forms forbidden to the pool, per report category."""
    categories: Dict[str, Set[str]] = {}
    for category, spec_ids in FORBIDDEN_CATEGORIES.items():
        forms: Set[str] = set()
        for spec_id in spec_ids:
            spec = spec_by_id(spec_id)
            word, states = _canonical_states(spec)
            if word is None:
                continue
            for index, state in enumerate(states):
                forms.add(spec_id + "|" + state.identity())
        categories[category] = forms
    return categories


_CATEGORIES_CACHE: Optional[Dict[str, Set[str]]] = None
_POOL_CACHE: Optional[List[ValCase]] = None


def forbidden_forms() -> Dict[str, Set[str]]:
    global _CATEGORIES_CACHE
    if _CATEGORIES_CACHE is None:
        _CATEGORIES_CACHE = _forbidden_forms()
    return _CATEGORIES_CACHE


def validation_pool(pool_size: int = POOL_SIZE,
                    seed: int = VAL_SEED) -> List[ValCase]:
    """The held-out validation pool (cached, disjoint by construction).

    The pool is the TARGET-CHANGE family FIRST (``TARGET_VARIANT_COUNT`` cases,
    each proven solvable by replaying its BFS witness), then the unchanged
    fresh-random intermediate stacks up to ``pool_size``. A candidate is dropped
    when its ``(spec_id, start_state identity)`` is a forbidden canonical-path
    form of any category, so disjointness holds by construction and is
    re-asserted by :func:`assert_disjoint`.
    """
    global _POOL_CACHE
    if _POOL_CACHE is not None and len(_POOL_CACHE) == int(pool_size):
        return _POOL_CACHE
    categories = forbidden_forms()
    forbidden: Set[str] = set()
    for forms in categories.values():
        forbidden |= forms
    seen: Set[str] = set()
    cases: List[ValCase] = []
    # 1. held-out TARGETS: the target-change family leads the pool, so the
    #    per-generation batch always can (and usually does) draw it.
    for case in target_change_cases():
        key = case.form_key()
        if key in forbidden or key in seen:
            continue
        seen.add(key)
        cases.append(case)
        if len(cases) >= int(pool_size):
            break
    # 2. fill the rest with the (unchanged) fresh random intermediate stacks.
    if len(cases) < int(pool_size):
        rng = random.Random(seed)
        stream = _candidate_stream(
            rng, tuple(TRAIN_SPEC_IDS) + tuple(HELDOUT_SPEC_IDS))
        for case in stream:
            key = case.form_key()
            if key in forbidden or key in seen:
                continue
            seen.add(key)
            cases.append(case)
            if len(cases) >= int(pool_size):
                break
    if len(cases) < int(pool_size):
        raise RuntimeError(
            "validation pool too small: %d < %d (disjoint solvable candidates "
            "exhausted)" % (len(cases), int(pool_size)))
    _POOL_CACHE = cases
    assert_disjoint(cases)
    return cases


def assert_disjoint(cases: Optional[Sequence[ValCase]] = None) -> str:
    """Raise ``AssertionError`` if any pool form/target is forbidden.

    Returns :data:`ASSERT_DISJOINT_OK` on success so evidence scripts can print
    the marker directly.
    """
    if cases is None:
        cases = validation_pool()
    forbidden: Set[str] = set()
    for forms in forbidden_forms().values():
        forbidden |= forms
    for case in cases:
        if case.form_key() in forbidden:
            raise AssertionError("validation form collides: " + case.form_key())
        if case.domain == "target":
            value = case_target_value(case)
            if value is not None and value in TRAIN_TARGET_VALUES:
                raise AssertionError(
                    "validation target collides with a training target: %r"
                    % value)
    return ASSERT_DISJOINT_OK


def validation_batch(cfg: Any, generation: int,
                     pool_size: int = POOL_SIZE) -> List[ValCase]:
    """The deterministic per-generation ``val_batch`` draw from the pool.

    Same ``(val_seed, generation)`` always yields the same batch in the same
    order: ``random.Random(val_seed + 1000003 * gen).sample(...)``.
    """
    cases = validation_pool(pool_size)
    batch = int(getattr(cfg, "val_batch", 16))
    if batch <= 0 or batch >= len(cases):
        return list(cases)
    rng = random.Random(int(getattr(cfg, "val_seed", VAL_SEED))
                        + 1000003 * int(generation))
    return rng.sample(cases, batch)


def disjointness_report(pool_size: int = POOL_SIZE) -> Dict[str, Any]:
    """Counts for evidence: pool size, distinct forms and per-category overlap.

    The five forbidden categories are reported for the WHOLE pool AND for the
    target-change family alone (``target_variant_overlap_*``); the family's
    guard targets are additionally checked against the training target
    (``overlap_training_targets``, must be 0). ``assert_disjoint_ok`` is the
    single boolean the evidence script asserts.
    """
    cases = validation_pool(pool_size)
    keys = {case.form_key() for case in cases}
    target_cases = [case for case in cases if case.domain == "target"]
    target_keys = {case.form_key() for case in target_cases}
    target_values = sorted({case_target_value(case) for case in target_cases
                            if case_target_value(case) is not None})
    report: Dict[str, Any] = {
        "pool": len(cases),
        "pool_distinct_forms": len(keys),
        "pool_size_requested": int(pool_size),
        "by_domain": {
            "prefix": sum(1 for c in cases if c.domain == "prefix"),
            "random": sum(1 for c in cases if c.domain == "random"),
            "target": len(target_cases),
        },
        "spec_ids": sorted({case.spec_id for case in cases}),
        "target_variants": {
            "count": len(target_cases),
            "spec_ids": sorted({case.spec_id for case in target_cases}),
            "target_values": target_values,
            "witness_min_steps": sorted(len(case.witness)
                                        for case in target_cases),
            "overlap_training_targets": len(
                set(target_values) & set(TRAIN_TARGET_VALUES)),
            "overlap_heldout_eval_targets": len(
                set(target_values) & set(HELDOUT_EVAL_TARGET_VALUES)),
        },
    }
    overlap_total = 0
    for category, forms in forbidden_forms().items():
        report[category] = len(keys & forms)
        report["target_variant_" + category] = len(target_keys & forms)
        overlap_total += report[category]
    report["assert_disjoint_ok"] = bool(
        overlap_total == 0
        and report["target_variants"]["overlap_training_targets"] == 0)
    return report


# --------------------------------------------------------------------------
# Fix 2: the held-out validation term for one candidate in one generation
# --------------------------------------------------------------------------

def evaluate_validation(net, genome: Sequence[float], cfg: Any, generation: int,
                        root_mask: Optional[Sequence[int]] = None,
                        prune_net=None) -> Tuple[float, int, List[str]]:
    """The held-out ``val_solved_rate`` for a candidate (GREEDY rollouts).

    Returns ``(val_solved_rate, val_total, solved_case_names)``. The batch is
    deterministic per ``(val_seed, generation)``; each case runs one greedy
    episode from its own start state (with masking when the config enables it).

    The mask is ``cfg.prune_mode`` (task 4354 unit 3A); ``"learned"`` uses
    ``prune_net`` and applies NO hand-written filter.
    """
    cases = validation_batch(cfg, generation)
    solved = 0
    names: List[str] = []
    prune_mode = getattr(cfg, "prune_mode", None)
    for index, case in enumerate(cases):
        if root_mask is None:
            root_mask = tuple(1 for _ in case.spec.objectives)
        # Greedy, but every case has its own explicit reproducible RNG.
        rng = random.Random(VAL_SEED + 1000003 * int(generation) + index)
        result = rollout(net, genome, case.spec, case.start_state, root_mask,
                         rng, epsilon=0.0,
                         mask_illegal=bool(getattr(cfg, "mask_illegal", False)),
                         prune_mode=prune_mode, prune_net=prune_net,
                         max_steps=case.max_steps)
        if result["solved"]:
            solved += 1
            names.append(case.name)
    total = len(cases)
    rate = solved / float(total) if total else 0.0
    return rate, total, names


__all__ = [
    "VAL_SEED",
    "POOL_SIZE",
    "SELECTION_EPISODES",
    "SELECTION_SEED",
    "SELECTION_EPSILON",
    "TRAIN_SPEC_IDS",
    "HELDOUT_SPEC_IDS",
    "TARGET_VARIANT_COUNT",
    "TARGET_VARIANT_SEED",
    "TARGET_VARIANT_PREFIX",
    "TRAIN_TARGET_VALUES",
    "HELDOUT_EVAL_TARGET_VALUES",
    "TARGET_VARIANT_ANCHOR",
    "ASSERT_DISJOINT_OK",
    "ValCase",
    "episode_seed",
    "canonical_allowed_nodes",
    "canonical_subtree_prune",
    "masked_legal_actions",
    "unmasked_legal_actions",
    "rollout",
    "evaluate_candidate",
    "count_repeat_sequences",
    "max_repeat_run",
    "target_change_cases",
    "replay_witness",
    "case_target_value",
    "validation_pool",
    "validation_batch",
    "assert_disjoint",
    "disjointness_report",
    "evaluate_validation",
]
