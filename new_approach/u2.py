"""U2 runner: bare-minimum training, generalization validation, honest verdict.

Two commands (both pure standard library, bounded, deterministic):

    # 1. BARE-MINIMUM TRAIN (fixed seed; saves the learned tables)
    python -m new_approach.u2 --mode train \
        --model /opt/automath/tmp/u2_agent.json \
        --comp  /opt/automath/tmp/u2_comp.json

    # 2. GENERALIZATION VALIDATION (loads the trained tables)
    python -m new_approach.u2 --mode validate \
        --model /opt/automath/tmp/u2_agent.json \
        --comp  /opt/automath/tmp/u2_comp.json

Unit of work (dispatch U2):

1. BARE-MINIMUM TRAIN. A minimal training set of episodes whose initial state
   is the bare minimum (the empty stack) and whose supervision is the planner's
   unique optimal action word to a fixed end state, plus one dynamic/pattern
   (``ExprGoal``) case. The agent is a deterministic tabular Q-learner
   (``TabularAgent``); a pure reward-driven control is also measured.
2. GENERALIZATION VALIDATION. New initial states (every non-trivial prefix of
   each trained goal's optimal word) that are NOT the trained initial state;
   the trained agent is placed there and must reach the same end state.
3. SIMPLE -> COMPLEX. The curriculum moves from the basic node types
   (``Zero``/``One``/``Change``) to the dynamic ``Group`` node grouping basic
   nodes (arities 1..3 and a nested grouping), bounded to small trees.
4. Structured failure feedback ``state / expected / actual / reason`` is printed
   for every failed case; every case is timed (including sub-second).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .agent import (
    CompositionalAgent,
    EpisodeResult,
    Step,
    TabularAgent,
)
from .axioms import AxiomSet
from .env import (
    ExprGoal,
    FixedStateGoal,
    MinimalEnv,
    bfs_plan,
    plan,
    simulate,
)
from .expressions import eq, nat
from .nodes import Change, Group, Node, One, Zero, size

AX = AxiomSet()
SEED = 20261009


# --------------------------------------------------------------------------
# Curriculum
# --------------------------------------------------------------------------

def curriculum() -> List[Tuple[str, str, Node]]:
    """(name, level, target) from simplest to more complex."""
    return [
        # Level 0: the bare minimum node types.
        ("zero", "basic", Zero()),
        ("one", "basic", One()),
        # Level 1: the unary Change node.
        ("two", "change", Change(One())),
        ("three", "change", Change(Change(One()))),
        # Level 2: the dynamic Group node grouping basic nodes.
        ("g2", "group", Group(One(), (Zero(),))),
        ("g3", "group", Group(One(), (One(), One()))),
        ("g4", "group", Group(One(), (One(), One(), One()))),
        ("g_add", "group", Group(nat(1), (nat(2), nat(2)))),
        # Level 3: the dynamic node grouping a Change and a Group (complex).
        ("g_grouped", "complex", Group(Change(One()), (Group(One(), (Zero(),)),))),
    ]


def held_out_targets() -> List[Tuple[str, Node]]:
    """Targets NOT in the training set, for out-of-distribution probing."""
    return [
        ("h_grouped", Group(Change(One()), (Group(One(), (Zero(), One())),))),
    ]


# --------------------------------------------------------------------------
# Case plumbing
# --------------------------------------------------------------------------

@dataclass
class Case:
    name: str
    kind: str
    key: str
    goal_name: str
    env: MinimalEnv
    expected_word: Tuple[str, ...]
    max_steps: int
    level: str = ""


def make_fixed_case(name: str, target: Node, initial: Sequence[Node],
                    kind: str, level: str = "") -> Case:
    env = MinimalEnv(FixedStateGoal(target), initial_stack=tuple(initial),
                     max_actions=size(target) + 4)
    word = tuple(plan(target, initial))
    return Case(name=name, kind=kind, key="T:" + target.canonical(),
                goal_name=env.goal.describe(), env=env, expected_word=word,
                max_steps=max(size(target) + 2, len(word) + 2), level=level)


def make_expr_case(name: str, goal: ExprGoal, kind: str,
                   level: str = "pattern") -> Case:
    env = MinimalEnv(goal, axioms=AX, initial_stack=(), max_actions=8)
    word = bfs_plan(env, max_actions=8)
    if word is None:
        raise RuntimeError("no plan for expression goal " + name)
    return Case(name=name, kind=kind, key="E:" + goal.name,
                goal_name=goal.describe(), env=env, expected_word=tuple(word),
                max_steps=len(word) + 2, level=level)


def make_expr_case_start(name: str, goal: ExprGoal,
                         initial: Sequence[Node]) -> Case:
    env = MinimalEnv(goal, axioms=AX, initial_stack=tuple(initial),
                     max_actions=8)
    word = bfs_plan(env, max_actions=8)
    if word is None:
        raise RuntimeError("no plan from new initial state " + name)
    return Case(name=name, kind="validation", key="E:" + goal.name,
                goal_name=goal.describe(), env=env, expected_word=tuple(word),
                max_steps=len(word) + 2, level="pattern")


def training_cases() -> List[Case]:
    cases: List[Case] = []
    for name, level, target in curriculum():
        cases.append(make_fixed_case(name, target, (), "train", level))
    # One dynamic/pattern (ExprGoal) case: reach a node whose value == 2.
    pattern = ExprGoal(lambda cur: eq(cur, nat(2)), AX, name="value==2")
    cases.append(make_expr_case("expr_two", pattern, "train"))
    return cases


def validation_cases(fixed_targets: List[Tuple[str, str, Node]]) -> List[Case]:
    """NEW initial states = every non-trivial prefix of each trained word."""
    cases: List[Case] = []
    for name, level, target in fixed_targets:
        word = tuple(plan(target))
        for i in range(1, len(word) + 1):
            stack, error = simulate((), word[:i])
            if error is not None:
                raise RuntimeError("prefix simulation failed: " + error)
            label = "already-goal" if i == len(word) else ("prefix%d" % i)
            cases.append(make_fixed_case(
                "%s/%s" % (name, label), target, stack, "validation", level))
    # New initial states for the ExprGoal pattern case.
    pattern = ExprGoal(lambda cur: eq(cur, nat(2)), AX, name="value==2")
    word = tuple(bfs_plan(MinimalEnv(pattern, axioms=AX, max_actions=8),
                          max_actions=8))
    for i in range(1, len(word) + 1):
        stack, error = simulate((), word[:i])
        if error is not None:
            raise RuntimeError("pattern prefix simulation failed")
        cases.append(make_expr_case_start(
            "expr_two/prefix%d" % i, pattern, stack))
    return cases


def evaluate_plan(env: MinimalEnv, word: Sequence[str],
                  goal_name: str) -> EpisodeResult:
    """Execute a fixed action word (used for the compositional agent)."""
    state = env.reset()
    result = EpisodeResult(passed=False, goal=goal_name,
                           start=state.canonical(), trace=[state.canonical()])
    for action in word:
        nxt = env.step(state, action)
        goal = env.goal_achieved(nxt)
        reward = 1.0 if goal else -0.01
        result.total_reward += reward
        result.steps.append(Step(action=action, state=nxt.canonical(),
                                 reward=reward, goal=goal))
        result.trace.append(nxt.canonical())
        state = nxt
        if goal:
            break
    result.passed = env.goal_achieved(state)
    result.final_state = state.canonical()
    if not result.passed:
        result.reason = "fixed plan did not reach the goal"
        result.feedback = env.feedback(state)
    return result


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def render_case(case: Case, result: EpisodeResult, wall: float) -> str:
    verdict = "PASS" if result.passed else "FAIL"
    actions = [s.action for s in result.steps]
    lines = [
        "[CASE] %s kind=%s level=%s wall=%.6fs" % (
            case.name, case.kind, case.level or "-", wall),
        "  goal: %s" % case.goal_name,
        "  start=%s" % result.start,
        "  actions=%s" % (actions,),
        "  trace: %s" % " -> ".join(result.trace),
        "  goal_satisfied=%s reward=%.4f wall=%.6fs %s" % (
            result.passed, result.total_reward, wall, verdict),
    ]
    if not result.passed:
        if result.feedback is not None:
            lines.append("  feedback: %s" % result.feedback.render())
        if result.reason:
            lines.append("  reason: %s" % result.reason)
    return "\n".join(lines)


def print_header(title: str) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)


# --------------------------------------------------------------------------
# Mode: train
# --------------------------------------------------------------------------

def run_train(model_path: str, comp_path: str, episodes: int) -> int:
    started = time.perf_counter()
    train = training_cases()
    print_header("U2 STEP 1: BARE-MINIMUM TRAIN")
    print("seed=%d episodes_per_problem=%d" % (SEED, episodes))
    print("training episodes: %d (each starts from the bare-minimum empty stack)"
          % len(train))
    print("training goals:")
    for c in train:
        print("  - %-12s level=%-8s key=%-24s optimal_word=%s"
              % (c.name, c.level, c.key, list(c.expected_word)))
    print("-" * 72)

    agent = TabularAgent(episodes=episodes, seed=SEED, use_demo=True)
    t0 = time.perf_counter()
    agent.train([(c.key, c.env, c.expected_word, c.max_steps) for c in train])
    train_wall = time.perf_counter() - t0

    # Determinism: a second independent run with the same seed.
    twin = TabularAgent(episodes=episodes, seed=SEED, use_demo=True)
    twin.train([(c.key, c.env, c.expected_word, c.max_steps) for c in train])
    digest = agent.digest()
    twin_digest = twin.digest()
    deterministic = digest == twin_digest

    comp = CompositionalAgent()
    for c in train:
        comp.observe_trajectory(c.env, c.expected_word)

    agent.save(model_path)
    comp.save(comp_path)
    print("train wall=%.6fs learned_states=%d digest=%s"
          % (train_wall, agent.learned_states(), digest))
    print("determinism: same seed -> same table: %s (%s vs %s)"
          % (deterministic, digest, twin_digest))
    print("compositional learned rules: %s conflicts=%d"
          % (comp.shapes(), comp.conflicts))
    print("saved model=%s comp=%s" % (model_path, comp_path))
    print("TRAIN_EXIT=%d" % (0 if deterministic else 1))
    print("train total wall=%.6fs" % (time.perf_counter() - started))
    return 0 if deterministic else 1


# --------------------------------------------------------------------------
# Mode: validate
# --------------------------------------------------------------------------

def run_validate(model_path: str, comp_path: str, episodes: int,
                 verbose_cases: bool) -> int:
    started = time.perf_counter()
    fixed = list(curriculum())
    train = training_cases()
    validation = validation_cases(fixed)
    heldout = held_out_targets()

    agent = TabularAgent.load(model_path)
    comp = CompositionalAgent.load(comp_path)
    print_header("U2 STEP 2: GENERALIZATION VALIDATION")
    print("loaded model=%s digest=%s learned_states=%d"
          % (model_path, agent.digest(), agent.learned_states()))
    print("loaded compositional rules: %s conflicts=%d"
          % (comp.shapes(), comp.conflicts))
    print("-" * 72)

    # -- train-set reachability ------------------------------------------
    print("TRAIN-SET REACHABILITY (agent from the trained initial state)")
    train_pass = 0
    train_walls: List[float] = []
    for c in train:
        t0 = time.perf_counter()
        result = agent.act_episode(c.key, c.env, c.max_steps, c.goal_name)
        wall = time.perf_counter() - t0
        train_walls.append(wall)
        train_pass += int(result.passed)
        if verbose_cases:
            print(render_case(c, result, wall))
    print("train reachability: %d/%d PASS" % (train_pass, len(train)))
    print("-" * 72)

    # -- generalization validation (new initial states) ------------------
    print("GENERALIZATION VALIDATION (NEW initial states, same trained goals)")
    val_pass = 0
    val_walls: List[float] = []
    per_level: Dict[str, List[int]] = {}
    for c in validation:
        t0 = time.perf_counter()
        result = agent.act_episode(c.key, c.env, c.max_steps, c.goal_name)
        wall = time.perf_counter() - t0
        val_walls.append(wall)
        val_pass += int(result.passed)
        stats = per_level.setdefault(c.level, [0, 0])
        stats[0] += int(result.passed)
        stats[1] += 1
        if verbose_cases:
            print(render_case(c, result, wall))
    print("validation: %d/%d PASS (new initial states)"
          % (val_pass, len(validation)))
    for level in sorted(per_level):
        ok, total = per_level[level]
        print("  level %-8s %d/%d PASS" % (level, ok, total))
    print("-" * 72)

    # -- out-of-distribution held-out targets ----------------------------
    print("HELD-OUT TARGET PROBE (targets NOT in the training set)")
    held_pass = comp_pass = 0
    for name, target in heldout:
        case = make_fixed_case(name, target, (), "heldout")
        t0 = time.perf_counter()
        flat = agent.act_episode(case.key, case.env, case.max_steps,
                                 case.goal_name)
        flat_wall = time.perf_counter() - t0
        held_pass += int(flat.passed)
        print(render_case(case, flat, flat_wall))
        env2 = MinimalEnv(FixedStateGoal(target), max_actions=size(target) + 4)
        t0 = time.perf_counter()
        try:
            word = comp.plan(target)
        except KeyError as exc:
            word = []
            print("  compositional missing rule: %s" % (exc,))
        comp_res = evaluate_plan(env2, word, env2.goal.describe())
        comp_wall = time.perf_counter() - t0
        comp_pass += int(comp_res.passed)
        comp_case = Case(name + "/COMPOSITIONAL", "heldout", "comp",
                         env2.goal.describe(), env2, tuple(word),
                         len(word) + 2)
        print("  compositional plan=%s" % (word,))
        print(render_case(comp_case, comp_res, comp_wall))
    print("held-out: flat tabular %d/%d PASS, compositional %d/%d PASS"
          % (held_pass, len(heldout), comp_pass, len(heldout)))
    print("-" * 72)

    # -- control: pure reward-driven Q-learning --------------------------
    print("CONTROL: pure reward-driven Q-learning (use_demo=False)")
    control = TabularAgent(episodes=episodes, seed=SEED, use_demo=False)
    control.train([(c.key, c.env, c.expected_word, c.max_steps) for c in train])
    ctrl_pass = 0
    ctrl_walls: List[float] = []
    for c in validation:
        t0 = time.perf_counter()
        result = control.act_episode(c.key, c.env, c.max_steps, c.goal_name)
        ctrl_walls.append(time.perf_counter() - t0)
        ctrl_pass += int(result.passed)
    print("control validation: %d/%d PASS (learned_states=%d)"
          % (ctrl_pass, len(validation), control.learned_states()))
    print("-" * 72)

    # -- verdict ---------------------------------------------------------
    total_wall = time.perf_counter() - started
    all_walls = train_walls + val_walls
    print("VERDICT")
    print("1. reduction (U1): 4 node types; <=10 met; 3/2/1 possible under"
          " explicit definitions")
    print("2. loaded agent digest=%s learned_states=%d (deterministic train)"
          % (agent.digest(), agent.learned_states()))
    print("3. generalization (new initial states): %d/%d PASS"
          % (val_pass, len(validation)))
    print("4. train reachability: %d/%d PASS; held-out flat: %d/%d;"
          " held-out compositional: %d/%d"
          % (train_pass, len(train), held_pass, len(heldout),
             comp_pass, len(heldout)))
    print("5. control (pure Q): %d/%d PASS" % (ctrl_pass, len(validation)))
    if all_walls:
        print("6. per-case wall time: min=%.6fs median=%.6fs max=%.6fs"
              % (min(all_walls), statistics.median(all_walls), max(all_walls)))
    print("7. total validate wall time=%.6fs" % total_wall)
    ok = train_pass == len(train) and val_pass == len(validation)
    print("VALIDATE_EXIT=%d" % (0 if ok else 1))
    return 0 if ok else 1


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="U2 train + validation runner")
    parser.add_argument("--mode", choices=("train", "validate", "all"),
                        default="all")
    parser.add_argument("--model", default="/opt/automath/tmp/u2_agent.json")
    parser.add_argument("--comp", default="/opt/automath/tmp/u2_comp.json")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--no-verbose-cases", dest="verbose_cases",
                        action="store_false", default=True)
    args = parser.parse_args(argv)

    if args.mode == "train":
        return run_train(args.model, args.comp, args.episodes)
    if args.mode == "validate":
        return run_validate(args.model, args.comp, args.episodes,
                            args.verbose_cases)
    rc = run_train(args.model, args.comp, args.episodes)
    rc2 = run_validate(args.model, args.comp, args.episodes, args.verbose_cases)
    return 0 if (rc == 0 and rc2 == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
