#!/usr/bin/env python3
"""Per-case runner for the automath goal environment (unit U4a).

Every registry case builds its ``GoalEnv`` through the SAME helper its test
module uses (the helper is factored into that module), so a case can no longer
silently fall back to the default ``HaveScratch(Void)`` environment.  A case
that only a scripted oracle could drive is listed with an explicit ``SKIP``
reason instead of a fake pass.

Examples:
    python scripts/run_case.py --list
    python scripts/run_case.py --selftest
    python scripts/run_case.py --case binary_int --agent simple --max-steps 20
    python scripts/run_case.py --case all --agent simple --max-steps 40 --json
    python scripts/run_case.py --case boolean_lt --agent llm \\
        --base-url http://localhost:8000 --model local-model --max-steps 5
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time

# Allow `python scripts/run_case.py` from the repo root: put the repo root on
# sys.path so `env`, `agent` and `test_suite` import.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# --------------------------------------------------------------- helpers ----
def _goal_of(env):
    from env.meta_env import MetaInfo
    from env.state import IGoal

    meta = env.full_state.meta.apply().real(MetaInfo)
    return meta.goal.apply().real(IGoal)


def _goal_symbol(goal, limit: int = 240) -> str:
    from env import symbol

    try:
        text = str(symbol.Symbol.default(goal))
    except Exception as e:  # pragma: no cover - defensive only
        text = f'<unrenderable {type(e).__name__}: {e}>'
    flat = ' '.join(text.split())
    if len(flat) > limit:
        flat = flat[:limit - 3] + '...'
    return flat


def _goal_rng_offset(state) -> int:
    """Stable per-goal offset so each goal case explores its own stream."""
    import hashlib

    from env import symbol
    from env.meta_env import MetaInfo
    from env.state import IGoal

    try:
        meta = state.meta.apply().real(MetaInfo)
        goal = meta.goal.apply().real(IGoal)
        text = str(symbol.Symbol.default(goal))
    except Exception:  # pragma: no cover - defensive only
        text = type(state).__name__
    digest = hashlib.sha256(text.encode('utf-8')).digest()
    return int.from_bytes(digest[:4], 'big')


def _allowed_counts(env) -> tuple[int, int]:
    """(declared allowed actions, basic actions the agent can emit)."""
    from env.meta_env import MetaInfo

    meta = env.full_state.meta.apply().real(MetaInfo)
    allowed = meta.allowed_actions.apply().as_tuple
    basic = meta.allowed_basic_actions.apply().as_tuple
    return len(allowed), len(basic)


def _last_action_name(state) -> str:
    from env.core import Optional

    error = state.last_action_error()
    if error is not None:
        return error[0]
    action_data_opt = state.last_action_data
    action_data = action_data_opt.value
    if action_data is None:
        return 'none'
    action_opt = action_data.action.apply().real(Optional)
    action = action_opt.value
    return type(action).__name__ if action is not None else 'none'


def _print_step(record: dict) -> None:
    print(
        f"step {record['step']}: action={record['action']} ok={record['ok']} "
        f"cost={record['cost']} wall_ms={record['wall_ms']} "
        f"err={record['err']}"
    )


# ---------------------------------------------------------------- registry ---
def _build_registry():
    """Ordered case specs. Test modules are imported lazily.

    Each ``builder(max_steps)`` returns a fresh ``GoalEnv`` constructed by the
    test module's own helper.
    """
    from env import core
    from test_suite import (
        arithmetic_test,
        boolean_test,
        control_flow_test,
        indices_test,
    )

    def arithmetic_eq(cases_fn, index: int = 0):
        def builder(max_steps: int):
            case = cases_fn()[index]
            return arithmetic_test.build_result_env(
                core.Eq(case['raw_expr'], case['correct_expr']),
                max_steps=max_steps,
            )
        return builder

    def boolean_lt(max_steps: int):
        return boolean_test.build_env(
            boolean_test.less_than_case(),
            max_steps=max_steps,
        )

    def indices(max_steps: int):
        return indices_test.build_case(max_steps=max_steps)[0]

    def control_flow(max_steps: int):
        return control_flow_test.build_case(max_steps=max_steps)[0]

    return [
        {
            'name': 'binary_int',
            'area': 'arithmetic',
            'goal_type': 'HaveResultScratch',
            'detail': 'Eq[BinaryInt]',
            'source': 'arithmetic_test.test_binary_int_basic',
            'builder': arithmetic_eq(arithmetic_test.binary_int_basic_cases),
            'skip': None,
        },
        {
            'name': 'signed_int',
            'area': 'arithmetic',
            'goal_type': 'HaveResultScratch',
            'detail': 'Eq[SignedInt]',
            'source': 'arithmetic_test.test_signed_int_basic',
            'builder': arithmetic_eq(arithmetic_test.signed_int_basic_cases),
            'skip': None,
        },
        {
            'name': 'int_to_binary',
            'area': 'arithmetic',
            'goal_type': 'HaveResultScratch',
            'detail': 'Eq[IntToBinary]',
            'source': 'arithmetic_test.test_int_to_binary',
            'builder': arithmetic_eq(arithmetic_test.int_to_binary_cases),
            'skip': None,
        },
        {
            'name': 'boolean_lt',
            'area': 'boolean',
            'goal_type': 'HaveResultScratch',
            'detail': 'LessThan(Integer,Integer)',
            'source': 'boolean_test.test_boolean',
            'builder': boolean_lt,
            'skip': None,
        },
        {
            'name': 'indices',
            'area': 'indices',
            'goal_type': 'HaveScratch',
            'detail': 'Void (index/args-group initial state)',
            'source': 'indices_test.test_indices',
            'builder': indices,
            'skip': None,
        },
        {
            'name': 'control_flow',
            'area': 'control_flow',
            'goal_type': 'HaveScratch',
            'detail': 'Void (If/Loop/function initial state)',
            'source': 'control_flow_test.test_control_flow',
            'builder': control_flow,
            'skip': None,
        },
    ]


def _find_case(registry, name: str):
    for spec in registry:
        if spec['name'] == name:
            return spec
    raise KeyError(name)


# ----------------------------------------------------------------- agents ----
class _SimplePolicy:
    """Seeded, deterministic explorer over the allowed action catalogue.

    It is deliberately not a solver: it is the "arbitrary agent" baseline.  The
    RNG is seeded with ``seed`` and offset by a stable hash of the goal, so
    different goal cases explore different action/argument sequences under the
    same seed.
    """

    def __init__(self, state, seed: int = 0):
        from agent.llm_agent import allowed_action_entries

        self._entries = allowed_action_entries(state)
        offset = _goal_rng_offset(state)
        self._rng = random.Random((seed ^ offset) & 0xFFFFFFFF)

    def select_action(self, state):
        from env.action import RawAction

        if not self._entries:
            return RawAction.with_raw_args(0, 0, 0, 0)
        _, index = self._entries[self._rng.randrange(len(self._entries))]
        arg1 = self._rng.randrange(0, 3)
        arg2 = self._rng.randrange(0, 3)
        arg3 = self._rng.randrange(0, 2)
        return RawAction.with_raw_args(index, arg1, arg2, arg3)


def _make_policy(agent_kind: str, state, seed: int, base_url, model):
    if agent_kind == 'llm':
        from agent.llm_agent import LlmAgent

        return LlmAgent(base_url=base_url, model=model)
    return _SimplePolicy(state, seed=seed)


# -------------------------------------------------------------- execution ----
def run_case(spec: dict, agent_kind: str, max_steps: int, seed: int,
             base_url: str | None, model: str | None) -> dict:
    env = spec['builder'](max_steps)
    policy = _make_policy(agent_kind, env.full_state, seed, base_url, model)
    goal_symbol = _goal_symbol(_goal_of(env))

    steps = []
    start = time.perf_counter()
    for i in range(1, max_steps + 1):
        step_start = time.perf_counter()
        try:
            action = policy.select_action(env.full_state)
            next_state, reward, terminated, truncated = env.step(action)
        except Exception as e:
            wall_ms = (time.perf_counter() - step_start) * 1000.0
            record = {
                'step': i,
                'action': 'RAISED',
                'ok': False,
                'cost': 0,
                'wall_ms': round(wall_ms, 3),
                'err': f'{type(e).__name__}: {e}'[:200],
                'reward': 0.0,
                'goal': False,
            }
            steps.append(record)
            _print_step(record)
            break

        error = next_state.last_action_error()
        try:
            cost = next_state.final_cost().as_int
        except Exception:
            cost = 0
        wall_ms = env.step_times[-1] * 1000.0 if env.step_times else 0.0
        record = {
            'step': i,
            'action': _last_action_name(next_state),
            'ok': error is None,
            'cost': cost,
            'wall_ms': round(wall_ms, 3),
            'err': error[1] if error is not None else 'none',
            'reward': round(reward, 6),
            'goal': bool(next_state.goal_achieved()),
        }
        steps.append(record)
        _print_step(record)
        if terminated or truncated:
            break

    elapsed = time.perf_counter() - start
    final_state = env.full_state
    total_reward = sum(s['reward'] for s in steps)
    summary = {
        'actions': len(steps),
        'reward': round(total_reward, 6),
        'goal': bool(final_state.goal_achieved()),
        'total_wall_s': round(elapsed, 6),
        'max_history': final_state.history_amount(),
        'dropped_history': final_state.dropped_history_count(),
    }
    print(
        f"SUMMARY actions={summary['actions']} reward={summary['reward']} "
        f"goal={summary['goal']} total_wall_s={summary['total_wall_s']} "
        f"max_history={summary['max_history']} "
        f"dropped_history={summary['dropped_history']}"
    )
    return {
        'case': spec['name'],
        'area': spec['area'],
        'source': spec['source'],
        'goal_type': spec['goal_type'],
        'goal_detail': spec['detail'],
        'goal_symbol': goal_symbol,
        'agent': agent_kind,
        'seed': seed,
        'max_steps': max_steps,
        'steps': steps,
        'summary': summary,
        'solved': summary['goal'],
    }


# ---------------------------------------------------------------- commands ---
def _print_list() -> None:
    registry = _build_registry()
    print(f"{'case':<14} {'goal':<46} {'allowed':<8} {'basic':<6} source")
    for spec in registry:
        if spec['skip']:
            print(f"{spec['name']:<14} SKIP: {spec['skip']}")
            continue
        env = spec['builder'](1)
        allowed, basic = _allowed_counts(env)
        goal = f"{spec['goal_type']}({spec['detail']})"
        print(f"{spec['name']:<14} {goal:<46} {allowed:<8} {basic:<6} "
              f"{spec['source']}")


def _print_selftest(max_steps: int) -> None:
    registry = _build_registry()
    for spec in registry:
        if spec['skip']:
            print(f"== {spec['name']} == SKIP: {spec['skip']}")
            continue
        env = spec['builder'](max_steps)
        state = env.full_state
        allowed, basic = _allowed_counts(env)
        print(f"== {spec['name']} ==")
        print(f"goal_type: {spec['goal_type']}")
        print(f"goal_detail: {spec['detail']}")
        print(f"goal_symbol: {_goal_symbol(_goal_of(env))}")
        print(f"goal_achieved_initially: {state.goal_achieved()}")
        print(f"allowed_actions: {allowed} (basic emitted: {basic})")
        print('observation:')
        print(state.render_observation(max_nodes=40))
        print()


def _run_one(spec: dict, args) -> dict:
    print(f"===== case {spec['name']} ({spec['area']}) =====")
    return run_case(
        spec=spec,
        agent_kind=args.agent,
        max_steps=args.max_steps,
        seed=args.seed,
        base_url=args.base_url,
        model=args.model,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', default='binary_int',
                        help="registry case name, or 'all'")
    parser.add_argument('--list', action='store_true',
                        help='print case name, goal type and allowed action count')
    parser.add_argument('--selftest', action='store_true',
                        help='print each case goal, initial observation and goal flag')
    parser.add_argument('--agent', choices=['simple', 'llm'], default='simple')
    parser.add_argument('--max-steps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--base-url', default=None)
    parser.add_argument('--model', default=None)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)

    if args.list:
        _print_list()
        return 0
    if args.selftest:
        _print_selftest(args.max_steps)
        return 0

    registry = _build_registry()
    names = [spec['name'] for spec in registry]
    if args.case == 'all':
        specs = registry
    else:
        try:
            specs = [_find_case(registry, args.case)]
        except KeyError:
            print(f"unknown case {args.case!r}; known: {', '.join(names)}",
                  file=sys.stderr)
            return 2

    results = []
    failed = 0
    for spec in specs:
        if spec['skip']:
            print(f"SKIP {spec['name']}: {spec['skip']}")
            continue
        try:
            results.append(_run_one(spec, args))
        except Exception as e:
            failed += 1
            print(f"CASE ERROR {spec['name']}: {type(e).__name__}: {e}",
                  file=sys.stderr)
            results.append({'case': spec['name'], 'error':
                            f'{type(e).__name__}: {e}'})

    if args.json:
        payload = results[0] if len(results) == 1 else results
        print(json.dumps(payload, indent=2))
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
