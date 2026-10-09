#!/usr/bin/env python3
"""Offline LLM macro-action prior builder (unit U4, docs/REPORT.md option C).

The self-hosted llama.cpp server (OpenAI-compatible ``/v1``) is called ONCE per
requested candidate, OUTSIDE the training loop, to propose MACRO-ACTION
candidates for a goal family.  Every returned candidate is VALIDATED against the
primitive catalogue (``env/node_types.py`` ``ESSENTIAL_ACTIONS``) and against
the concrete environment's allowed basic-action group; invalid candidates are
DROPPED with a reason.  Accepted candidates plus the per-call raw response,
latency and token counts are written to ``prior/macro_prior_<model>.json``.

The training loop uses NO LLM and NO network: it only reads the accepted JSON
through ``scripts/run_case.py --macro-prior <file>``.

Usage:
    python scripts/build_macro_prior.py --families result,scratch
    python scripts/build_macro_prior.py --base-url http://127.0.0.1:8080/v1 \\
        --model minicpm5-1b --candidates-per-family 4
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import urllib.error
import urllib.request

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Goal family -> a registry case that exercises it (used for the env check).
FAMILY_CASE = {'result': 'binary_int', 'scratch': 'indices'}

SCHEMA_TEXT = (
    'Return EXACTLY ONE JSON object and nothing else, with this shape:\n'
    '{"name": "<short macro name>", "steps": ['
    '{"action": "<primitive action name>", "args": [<int>, <int>, <int>]}, ...]}\n'
    'Rules: "steps" is a non-empty list; each step names one of the ALLOWED '
    'PRIMITIVE ACTIONS above; "args" is exactly three integers (or the literal '
    'string "from_int:<TypeName>" for one of the allowed type tokens).'
)


def _load_run_case():
    path = os.path.join(_REPO_ROOT, 'scripts', 'run_case.py')
    spec = importlib.util.spec_from_file_location('run_case', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _essential_names():
    from env.node_types import ESSENTIAL_ACTIONS

    return {action.__name__ for action in ESSENTIAL_ACTIONS}


def _group_names(env):
    """(allowed basic-action names, from-int type-token names) for a case env."""
    from env import meta_env

    meta = env.full_state.meta.apply().real(meta_env.MetaInfo)
    allowed = meta.allowed_basic_actions.apply().real(
        meta_env.GeneralTypeGroup).as_tuple
    group = meta.nested_arg((
        meta_env.MetaInfo.idx_from_int_group,
        meta_env.SubtypeOuterGroup.idx_subtypes,
    )).apply().cast(meta_env.GeneralTypeGroup)
    return (
        [node.type.__name__ for node in allowed],
        [node.type.__name__ for node in group.as_tuple],
    )


def _build_env(family):
    """Build the registry case env for a family; None when unavailable."""
    try:
        run_case = _load_run_case()
        registry = run_case._build_registry()
        case = FAMILY_CASE[family]
        spec = run_case._find_case(registry, case)
        return case, spec['builder'](10)
    except Exception as e:  # pragma: no cover - static-only fallback
        print(f'[warn] cannot build env for family {family!r}: {e!r}',
              file=sys.stderr)
        return None, None


def _prompt(family, allowed, tokens, candidate_index):
    actions = ', '.join(allowed)
    type_tokens = ', '.join(f'"from_int:{t}"' for t in tokens) or '(none)'
    return (
        f'You are proposing macro-actions for a reinforcement-learning agent '
        f'that solves symbolic math goals. Goal family: {family}.\n'
        f'ALLOWED PRIMITIVE ACTIONS: {actions}.\n'
        f'ALLOWED TYPE TOKENS: {type_tokens}.\n'
        f'A macro-action is a short deterministic sequence of those primitives '
        f'with fixed arguments; it must be a plausible solution recipe for the '
        f'{family} family. Propose candidate #{candidate_index}.\n'
        f'{SCHEMA_TEXT}'
    )


def _extract_json(text):
    start = text.find('{')
    end = text.rfind('}')
    if start < 0 or end <= start:
        return None
    return text[start:end + 1]


def _call(base_url, model, prompt, max_tokens, temperature, timeout):
    url = base_url.rstrip('/') + '/chat/completions'
    payload = json.dumps({
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': temperature,
        'max_tokens': max_tokens,
    }).encode('utf-8')
    request = urllib.request.Request(
        url, data=payload, headers={'Content-Type': 'application/json'})
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode('utf-8', 'replace')
        latency = time.time() - started
        body = json.loads(raw)
        content = body['choices'][0]['message']['content']
        usage = body.get('usage') or {}
        return {
            'ok': True,
            'content': content,
            'latency_s': round(latency, 3),
            'prompt_tokens': usage.get('prompt_tokens'),
            'completion_tokens': usage.get('completion_tokens'),
            'error': None,
        }
    except urllib.error.HTTPError as e:
        return {'ok': False, 'content': None, 'latency_s': round(
            time.time() - started, 3), 'prompt_tokens': None,
            'completion_tokens': None, 'error': f'HTTP {e.code}: {e.reason}'}
    except Exception as e:  # urllib timeout / connection / JSON
        return {'ok': False, 'content': None, 'latency_s': round(
            time.time() - started, 3), 'prompt_tokens': None,
            'completion_tokens': None, 'error': f'{type(e).__name__}: {e}'}


def _validate(raw_text, family, essential, allowed, tokens, env):
    """Return (macro_or_None, reason). Static + concrete-env validation."""
    text = _extract_json(raw_text or '')
    if text is None:
        return None, 'no JSON object in response'
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f'JSON decode error: {e}'
    if not isinstance(obj, dict):
        return None, 'top level is not an object'
    name = obj.get('name')
    if not isinstance(name, str) or not name:
        return None, 'missing non-empty "name"'
    steps = obj.get('steps')
    if not isinstance(steps, list) or not steps:
        return None, 'missing non-empty "steps"'
    clean_steps = []
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            return None, f'step #{i} is not an object'
        action = step.get('action')
        if not isinstance(action, str) or not action:
            return None, f'step #{i} has no "action"'
        if action not in essential:
            return None, f'step #{i} action {action!r} not in ESSENTIAL_ACTIONS'
        if allowed and action not in allowed:
            return None, f'step #{i} action {action!r} not allowed in this env'
        args = step.get('args', [0, 0, 0])
        if not isinstance(args, list) or len(args) != 3:
            return None, f'step #{i} "args" is not a list of 3'
        for arg in args:
            if isinstance(arg, bool) or not isinstance(arg, (int, str)):
                return None, f'step #{i} arg {arg!r} is not int or token str'
            if isinstance(arg, str) and not arg.startswith('from_int:'):
                return None, f'step #{i} bad arg token {arg!r}'
            if isinstance(arg, str) and tokens and arg[len('from_int:'):] not in tokens:
                return None, f'step #{i} type token {arg!r} not in this env'
        clean_steps.append({'action': action, 'args': list(args)})
    macro = {
        'name': name,
        'description': f'LLM proposed candidate for family {family}',
        'family': family,
        'steps': clean_steps,
    }
    if env is not None:
        try:
            from env.macro_action import MacroAction, MacroActionEnv, MacroStep

            built = MacroAction(
                name=name, description=macro['description'], family=family,
                steps=tuple(MacroStep(s['action'], tuple(s['args']))
                            for s in clean_steps))
            MacroActionEnv(env, catalogue=[built]).expand(built)
        except Exception as e:
            return None, f'env expansion failed: {type(e).__name__}: {e}'
    return macro, 'accepted'


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8080/v1')
    parser.add_argument('--model', default='minicpm5-1b')
    parser.add_argument('--families', default='result,scratch')
    parser.add_argument('--candidates-per-family', type=int, default=4)
    parser.add_argument('--max-calls', type=int, default=12)
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--timeout', type=float, default=120.0)
    parser.add_argument('--out', default=None)
    args = parser.parse_args(argv)

    out_path = args.out or os.path.join(
        _REPO_ROOT, 'prior', f'macro_prior_{args.model}.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    essential = _essential_names()
    families = [f.strip() for f in args.families.split(',') if f.strip()]

    envs = {}
    for family in families:
        case, env = _build_env(family)
        allowed, tokens = _group_names(env) if env is not None else ([], [])
        envs[family] = (case, env, allowed, tokens)
        print(f'[family {family}] case={case} '
              f'allowed={len(allowed)} types={len(tokens)}')

    calls = []
    accepted = []
    validation = []
    call_index = 0
    for family in families:
        case, env, allowed, tokens = envs[family]
        for candidate_index in range(1, args.candidates_per_family + 1):
            if call_index >= args.max_calls:
                print(f'[budget] reached --max-calls={args.max_calls}')
                break
            call_index += 1
            prompt = _prompt(family, allowed, tokens, candidate_index)
            result = _call(args.base_url, args.model, prompt,
                           args.max_tokens, args.temperature, args.timeout)
            record = {
                'call': call_index,
                'family': family,
                'case': case,
                'candidate': candidate_index,
                'ok': result['ok'],
                'latency_s': result['latency_s'],
                'prompt_tokens': result['prompt_tokens'],
                'completion_tokens': result['completion_tokens'],
                'error': result['error'],
                'raw_response': (result['content'] or '')[:4000],
                'accepted': False,
                'name': None,
                'reason': None,
            }
            if not result['ok']:
                record['reason'] = result['error']
                print(f'[call {call_index}] family={family} FAILED '
                      f'{result["error"]}')
            else:
                macro, reason = _validate(
                    result['content'], family, essential, allowed, tokens, env)
                record['reason'] = reason
                if macro is not None:
                    macro['name'] = _unique_name(macro['name'], accepted)
                    record['accepted'] = True
                    record['name'] = macro['name']
                    accepted.append(macro)
                    print(f'[call {call_index}] family={family} ACCEPT '
                          f'{macro["name"]} '
                          f'latency={result["latency_s"]}s')
                else:
                    print(f'[call {call_index}] family={family} REJECT '
                          f'latency={result["latency_s"]}s reason={reason}')
            calls.append(record)
            validation.append({
                'call': call_index, 'family': family,
                'name': record['name'], 'accepted': record['accepted'],
                'reason': record['reason'],
            })

    artifact = {
        'generated_by': 'scripts/build_macro_prior.py',
        'model': args.model,
        'base_url': args.base_url,
        'temperature': args.temperature,
        'max_tokens': args.max_tokens,
        'max_calls': args.max_calls,
        'calls_used': len(calls),
        'accepted_count': len(accepted),
        'calls': calls,
        'validation': validation,
        'macros': accepted,
    }
    with open(out_path, 'w', encoding='utf-8') as handle:
        json.dump(artifact, handle, indent=2)
        handle.write('\n')
    print(f'[write] {out_path} calls={len(calls)} accepted={len(accepted)}')
    if not accepted:
        print('[error] no candidate survived validation', file=sys.stderr)
        return 1
    return 0


def _unique_name(name, accepted):
    existing = {m['name'] for m in accepted}
    if name not in existing:
        return name
    suffix = 2
    while f'{name}_{suffix}' in existing:
        suffix += 1
    return f'{name}_{suffix}'


if __name__ == '__main__':
    raise SystemExit(main())
