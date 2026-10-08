#!/usr/bin/env python3
"""Per-case runner for the bounded-state / legible-failure design (unit U3a).

Examples:
    python scripts/run_case.py --module test_suite.basic_test --case default \
        --agent simple --max-steps 10
    python scripts/run_case.py --agent llm --base-url http://localhost:8000 \
        --model local-model --max-steps 5 --json
"""
from __future__ import annotations
import argparse
import importlib
import json
import os
import sys
import time

# Allow `python scripts/run_case.py` from the repo root: put the repo root on
# sys.path so `env` and `agent` import.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _build_default_env(max_steps: int):
    from env import core
    from env.goal_env import GoalEnv
    from env.node_types import ESSENTIAL_ACTIONS, HaveScratch

    goal = HaveScratch.with_goal(core.Void())
    return GoalEnv(goal=goal, allowed_actions=ESSENTIAL_ACTIONS, max_steps=max_steps)


def _build_env(module_name: str, case: str, max_steps: int):
    if module_name:
        module = importlib.import_module(module_name)
        builder = getattr(module, 'build_env', None)
        if builder is not None:
            try:
                return builder(case=case, max_steps=max_steps)
            except TypeError:
                return builder()
    return _build_default_env(max_steps)


class _SimplePolicy:
    """Cycles through the allowed basic actions with zero raw args."""

    def __init__(self, state):
        from agent.llm_agent import allowed_action_entries
        self._entries = allowed_action_entries(state)
        self._cursor = 0

    def select_action(self, state):
        from env.action import RawAction
        if not self._entries:
            return RawAction.with_raw_args(0, 0, 0, 0)
        _, index = self._entries[self._cursor % len(self._entries)]
        self._cursor += 1
        return RawAction.with_raw_args(index, 0, 0, 0)


def _last_action_name(state) -> str:
    from env.core import Optional
    from env.full_state import BaseActionData
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


def run(module_name: str, case: str, agent_kind: str, max_steps: int,
        base_url: str | None, model: str | None) -> dict:
    env = _build_env(module_name, case, max_steps)
    if agent_kind == 'llm':
        from agent.llm_agent import LlmAgent
        policy = LlmAgent(base_url=base_url, model=model)
    else:
        policy = _SimplePolicy(env.full_state)

    steps = []
    start = time.perf_counter()
    for i in range(1, max_steps + 1):
        action = policy.select_action(env.full_state)
        next_state, reward, terminated, truncated = env.step(action)
        error = next_state.last_action_error()
        try:
            cost = next_state.final_cost().as_int
        except Exception:
            cost = 0
        wall_ms = env.step_times[-1] * 1000.0
        record = {
            'step': i,
            'action': _last_action_name(next_state),
            'ok': error is None,
            'cost': cost,
            'wall_ms': round(wall_ms, 3),
            'err': error[1] if error is not None else 'none',
            'reward': round(reward, 6),
            'goal': next_state.goal_achieved(),
        }
        steps.append(record)
        print(
            f"step {i}: action={record['action']} ok={record['ok']} "
            f"cost={record['cost']} wall_ms={record['wall_ms']} "
            f"err={record['err']}"
        )
        if terminated or truncated:
            break

    elapsed = time.perf_counter() - start
    final_state = env.full_state
    total_reward = sum(s['reward'] for s in steps)
    summary = {
        'actions': len(steps),
        'reward': round(total_reward, 6),
        'goal': final_state.goal_achieved(),
        'total_wall_s': round(elapsed, 6),
        'max_history': final_state.history_amount(),
        'dropped_history': final_state.dropped_history_count(),
    }
    print(
        f"SUMMARY actions={summary['actions']} reward={summary['reward']} "
        f"goal={summary['goal']} total_wall_s={summary['total_wall_s']} "
        f"max_history={summary['max_history']}"
    )
    return {'case': case, 'module': module_name, 'agent': agent_kind,
            'steps': steps, 'summary': summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module', default='', help='test_suite.<mod> to load')
    parser.add_argument('--case', default='default', help='case name')
    parser.add_argument('--agent', choices=['simple', 'llm'], default='simple')
    parser.add_argument('--max-steps', type=int, default=10)
    parser.add_argument('--base-url', default=None)
    parser.add_argument('--model', default=None)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)

    try:
        result = run(
            module_name=args.module,
            case=args.case,
            agent_kind=args.agent,
            max_steps=args.max_steps,
            base_url=args.base_url,
            model=args.model,
        )
    except Exception as e:
        print(f'RUNNER ERROR: {type(e).__name__}: {e}', file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
