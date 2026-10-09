#!/usr/bin/env python
"""Bounded training and held-out evaluation of the torch SmartAgent over the
bounded MACRO-action space (unit U5).

TRAIN split (default): the ``HaveResultScratch`` family cases that share the
built-in 4-macro result catalogue: ``binary_int``, ``int_to_binary``,
``boolean_lt``.  The training loop uses ONLY those cases: the agent's action
space is the built-in catalogue of the FIRST train case (the result family), so
no held-out case, its goal values nor its catalogue can influence training.

HELD-OUT split (default): ``signed_int`` (same family, unseen case),
``indices`` and ``control_flow`` (the ``HaveScratch`` family).  The DQN has a
FIXED output dimension equal to the training catalogue size, so evaluation
replays the TRAIN catalogue on every case; the held-out cases are never used to
build it.

Everything runs with NO network: the loop only selects over the frozen
in-process macro catalogue.  Reproducible:

    python scripts/train_macro_smart.py train \
        --train-cases binary_int,int_to_binary,boolean_lt \
        --max-episodes 60 --max-wall-s 300 --max-steps 12 --seed 1 \
        --checkpoint tmp/macro_smart.pt --jsonl tmp/train_macro_smart.jsonl

    python scripts/train_macro_smart.py eval \
        --checkpoint tmp/macro_smart.pt \
        --cases binary_int,signed_int,int_to_binary,boolean_lt,indices,control_flow \
        --max-steps 12 --out results/heldout_macro_smart.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from run_case import _build_registry  # noqa: E402

TRAIN_CASES_DEFAULT = 'binary_int,int_to_binary,boolean_lt'
HELDOUT_CASES_DEFAULT = 'signed_int,indices,control_flow'
LOSS_RE = re.compile(r'Total Loss:\s*([0-9.]+)')


# ------------------------------------------------------------------ helpers --
def _find(registry, name):
    for spec in registry:
        if spec['name'] == name:
            return spec
    raise KeyError(name)


def _train_catalogue(specs):
    """The macro catalogue shared by the TRAIN split (train-only influence)."""
    from env.macro_action import MacroActionEnv

    env = specs[0]['builder'](1)
    wrapped = MacroActionEnv(env)
    return wrapped.macro_actions


def _make_agent(action_space_size, seed):
    from agent.smart_agent import SmartAgent
    from config import agent_settings as s

    return SmartAgent(
        action_space_size=action_space_size,
        input_dim=s.INPUT_DIM,
        feature_dim=s.FEATURE_DIM,
        hidden_dim=s.HIDDEN_DIM,
        hidden_amount=s.HIDDEN_AMOUNT,
        learning_rate=s.LEARNING_RATE,
        gamma=s.GAMMA,
        epsilon_start=s.EPSILON_START,
        epsilon_end=s.EPSILON_END,
        epsilon_decay=s.EPSILON_DECAY,
        replay_buffer_capacity=s.REPLAY_BUFFER_CAPACITY,
        batch_size=s.BATCH_SIZE,
        target_update_frequency=s.TARGET_UPDATE_FREQUENCY,
        device=s.DEVICE,
        dropout_rate=s.DROPOUT_RATE,
        seed=seed,
    )


def _state_fields(env):
    state = env.full_state
    try:
        cost = state.final_cost().as_int
    except Exception:
        cost = 0
    return {
        'goal': bool(state.goal_achieved()),
        'cost': cost,
        'history_amount': state.history_amount(),
        'dropped_history': state.dropped_history_count(),
        'last_action_error': state.last_action_error(),
    }


# ------------------------------------------------------------------- train --
def cmd_train(args):
    from env.macro_action import MacroActionEnv

    registry = _build_registry()
    train_specs = [_find(registry, n) for n in args.train_cases.split(',') if n]
    catalogue = _train_catalogue(train_specs)
    action_space_size = len(catalogue)
    print(f'TRAIN cases: {[s["name"] for s in train_specs]}')
    print(f'action_space_size={action_space_size} '
          f'macros={[m.name for m in catalogue]}')
    agent = _make_agent(action_space_size, args.seed)

    start = time.perf_counter()
    episodes = []
    update_count = 0
    for episode in range(1, args.max_episodes + 1):
        if time.perf_counter() - start >= args.max_wall_s:
            print(f'WALL CAP reached after {episode - 1} episodes')
            break
        spec = train_specs[(episode - 1) % len(train_specs)]
        env = MacroActionEnv(spec['builder'](args.max_steps), catalogue=catalogue)
        prev_state = env.full_state
        macros = []
        rewards = []
        primitive_steps = 0
        losses = []
        terminated = truncated = False
        agent.reset()
        for _ in range(1, args.max_steps + 1):
            action = agent.select_action(env.full_state)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                next_state, reward, terminated, truncated = env.step(action)
                agent.train(
                    state=prev_state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    terminated=terminated,
                    truncated=truncated,
                )
            captured = buf.getvalue()
            found = LOSS_RE.findall(captured)
            if found:
                losses.append(float(found[-1]))
                update_count += 1
            if env.last_macro is not None:
                macros.append(env.last_macro.name)
            primitive_steps += len(env.last_primitive_trace)
            rewards.append(reward)
            prev_state = next_state
            if terminated or truncated:
                break
        fields = _state_fields(env)
        row = {
            'episode': episode,
            'case': spec['name'],
            'macro_steps': len(macros),
            'primitive_steps': primitive_steps,
            'reward': round(sum(rewards), 6),
            'goal': fields['goal'],
            'cost': fields['cost'],
            'epsilon': round(float(agent.epsilon), 8),
            'loss': round(losses[-1], 6) if losses else None,
            'updates': len(losses),
            'macros': macros,
        }
        episodes.append(row)
        print('EPISODE ' + json.dumps(row))
    elapsed = time.perf_counter() - start
    os.makedirs(os.path.dirname(os.path.abspath(args.checkpoint)), exist_ok=True)
    agent.save(args.checkpoint)
    summary = {
        'mode': 'train',
        'train_cases': [s['name'] for s in train_specs],
        'action_space_size': action_space_size,
        'macros': [m.name for m in catalogue],
        'episodes_run': len(episodes),
        'max_episodes': args.max_episodes,
        'max_wall_s': args.max_wall_s,
        'wall_s': round(elapsed, 6),
        'updates': update_count,
        'epsilon_final': round(float(agent.epsilon), 8),
        'checkpoint': args.checkpoint,
        'episodes': episodes,
    }
    if args.jsonl:
        with open(args.jsonl, 'w', encoding='utf-8') as handle:
            for row in episodes:
                handle.write(json.dumps(row) + '\n')
    if args.summary_json:
        with open(args.summary_json, 'w', encoding='utf-8') as handle:
            json.dump(summary, handle, indent=2)
    print('TRAIN_SUMMARY ' + json.dumps({k: v for k, v in summary.items()
                                         if k != 'episodes'}))
    return 0


# -------------------------------------------------------------------- eval --
def cmd_eval(args):
    from env.macro_action import MacroActionEnv

    registry = _build_registry()
    names = [n for n in args.cases.split(',') if n]
    specs = [_find(registry, n) for n in names]
    # The action space must equal the training one; derive it from the train
    # split only (never from the held-out case being evaluated).
    train_specs = [_find(registry, n) for n in args.train_cases.split(',') if n]
    catalogue = _train_catalogue(train_specs)
    agent = _make_agent(len(catalogue), args.seed)
    agent.load(args.checkpoint)
    agent.epsilon = 0.0  # pure exploitation of the trained checkpoint

    results = []
    for spec in specs:
        env = MacroActionEnv(spec['builder'](args.max_steps), catalogue=catalogue)
        start_fields = _state_fields(env)
        start_wall = time.perf_counter()
        steps = []
        total_primitives = 0
        for i in range(1, args.max_steps + 1):
            action = agent.select_action(env.full_state)
            action_start = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                next_state, reward, terminated, truncated = env.step(action)
            action_wall = time.perf_counter() - action_start
            macro = env.last_macro.name if env.last_macro is not None else None
            trace = env.last_primitive_trace
            total_primitives += len(trace)
            steps.append({
                'step': i,
                'macro': macro,
                'reward': round(reward, 6),
                'ok': next_state.last_action_error() is None,
                'err': next_state.last_action_error()[1]
                       if next_state.last_action_error() is not None else 'none',
                'goal': bool(next_state.goal_achieved()),
                'wall_s': round(action_wall, 6),
                'primitives': trace,
            })
            if terminated or truncated:
                break
        total_wall = time.perf_counter() - start_wall
        end_fields = _state_fields(env)
        summary = {
            'case': spec['name'],
            'goal_type': spec['goal_type'],
            'goal_detail': spec['detail'],
            'goal_symbol': _goal_symbol(env),
            'action_space_size': len(catalogue),
            'start_state': start_fields,
            'end_state': end_fields,
            'macro_steps': len(steps),
            'primitive_steps': total_primitives,
            'total_reward': round(sum(s['reward'] for s in steps), 6),
            'goal_reached': end_fields['goal'],
            'total_wall_s': round(total_wall, 6),
            'max_action_wall_s': round(max((s['wall_s'] for s in steps), default=0.0), 6),
            'sub_second': total_wall < 1.0,
            'max_history_size': end_fields['history_amount'],
            'dropped_history': end_fields['dropped_history'],
            'macro_actions_taken': [s['macro'] for s in steps],
            'steps': steps,
        }
        results.append(summary)
        print('CASE ' + json.dumps({k: v for k, v in summary.items()
                                    if k != 'steps'}))
        print(f'  start: {json.dumps(start_fields)}')
        print(f'  end:   {json.dumps(end_fields)}')

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        payload = {
            'model': args.checkpoint,
            'train_cases': [s['name'] for s in train_specs],
            'train_macros': [m.name for m in catalogue],
            'cases': results,
        }
        with open(args.out, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2)
    print('EVAL_SUMMARY ' + json.dumps([
        {k: v for k, v in r.items() if k not in ('steps', 'macro_actions_taken')}
        for r in results
    ]))
    return 0


def _goal_symbol(env):
    try:
        from run_case import _goal_of, _goal_symbol as _render
        inner = env.inner_env if hasattr(env, 'inner_env') else env
        return _render(_goal_of(inner))
    except Exception as e:  # pragma: no cover - defensive
        return f'<unavailable: {type(e).__name__}: {e}>'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)

    t = sub.add_parser('train')
    t.add_argument('--train-cases', default=TRAIN_CASES_DEFAULT)
    t.add_argument('--max-episodes', type=int, default=60)
    t.add_argument('--max-wall-s', type=float, default=300.0)
    t.add_argument('--max-steps', type=int, default=12)
    t.add_argument('--seed', type=int, default=1)
    t.add_argument('--checkpoint', default='tmp/macro_smart.pt')
    t.add_argument('--jsonl', default=None)
    t.add_argument('--summary-json', default=None)
    t.set_defaults(func=cmd_train)

    e = sub.add_parser('eval')
    e.add_argument('--cases', default=(TRAIN_CASES_DEFAULT + ',' +
                                       HELDOUT_CASES_DEFAULT))
    e.add_argument('--train-cases', default=TRAIN_CASES_DEFAULT)
    e.add_argument('--max-steps', type=int, default=12)
    e.add_argument('--seed', type=int, default=1)
    e.add_argument('--checkpoint', default='tmp/macro_smart.pt')
    e.add_argument('--out', default='results/heldout_macro_smart.json')
    e.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
