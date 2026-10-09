"""Unit tests for the bounded macro-action layer (design unit U3, fix 2).

Run with pytest (the repo's test entry point style)::

    python -m pytest test_suite/macro_action_test.py -v

Every test builds the same binary_int case the runner uses and drives the
``MacroActionEnv``.  The key claims proven here:

* a macro-action expands to its exact primitive sequence;
* the expansion reaches the intended state deterministically (same history,
  same goal flag on repeated runs);
* the macro selection space is the small macro count, not the primitive count.
"""
from env import core
from env.macro_action import (
    MacroActionEnv,
    catalogue_for_state,
    catalogue_from_json,
    default_catalogue,
    goal_family,
)
from env.node_types import ESSENTIAL_ACTIONS, HaveResultScratch
from scripts.run_case import _build_registry, _find_case


def _binary_int_env(max_steps: int = 10):
    spec = _find_case(_build_registry(), 'binary_int')
    return spec['builder'](max_steps)


def _history_actions(state):
    actions = []
    for i in range(state.history_amount()):
        _, action_data_opt = state.at_history(i + 1)
        action_data = action_data_opt.value
        action = action_data.action.apply().real(core.Optional).value
        actions.append(type(action).__name__)
    return actions


def test_catalogue_is_small_and_family_scoped():
    env = _binary_int_env()
    assert goal_family(env.full_state) == 'result'
    catalogue = catalogue_for_state(env.full_state)
    assert [m.name for m in catalogue] == [
        'result_true', 'result_false', 'result_write_true', 'result_check',
    ]
    # The whole point of fix 2: the agent chooses between a handful of macros,
    # not 22-34 typed actions with three integer arguments each.
    assert len(catalogue) < len(ESSENTIAL_ACTIONS)
    assert env.action_space_size() > len(catalogue)
    # The built-in catalogue spans all families and is deterministic.
    assert default_catalogue() == default_catalogue()


def test_macro_expands_to_primitive_sequence():
    env = _binary_int_env()
    macro_env = MacroActionEnv(env)
    macro = macro_env.macro_actions[0]
    raw_actions = macro.expand(macro_env)
    assert len(raw_actions) == 3
    # Resolve each RawAction back to the primitive it names.
    names = []
    for raw in raw_actions:
        action = raw.to_action(macro_env.full_state)
        names.append(type(action).__name__)
    assert names == ['CreateScratch', 'DefineScratchFromInt', 'VerifyGoal']
    # describe() is deterministic across two independent environments.
    other = MacroActionEnv(_binary_int_env())
    assert macro.describe(macro_env) == other.macro_actions[0].describe(other)
    assert 'CreateScratch(0, 0, 0)' in macro.describe(macro_env)


def test_macro_reaches_goal_deterministically():
    first = None
    for _ in range(2):
        env = _binary_int_env()
        macro_env = MacroActionEnv(env)
        next_state, reward, terminated, truncated = macro_env.step(1)
        assert terminated is True
        assert next_state.goal_achieved() is True
        assert truncated is False
        assert _history_actions(next_state) == [
            'CreateScratch', 'DefineScratchFromInt', 'VerifyGoal',
        ]
        # One primitive step per macro step: bounded history still applies.
        assert next_state.history_amount() == 3
        snapshot = (_history_actions(next_state), reward, terminated)
        if first is None:
            first = snapshot
        else:
            assert snapshot == first
    # The three primitives carry the ordinary per-step reward and the macro
    # reward is their sum.
    env = _binary_int_env()
    macro_env = MacroActionEnv(env)
    _, macro_reward, _, _ = macro_env.step(1)
    per_primitive = [p['reward'] for p in macro_env.last_primitive_trace]
    assert len(per_primitive) == 3
    # The trace stores rewards rounded to 6 decimals, so compare at that scale.
    assert abs(round(macro_reward, 6) - sum(per_primitive)) < 1e-6


def test_json_injection_seam_round_trip():
    import json
    import tempfile

    env = _binary_int_env()
    macro_env = MacroActionEnv(env)
    payload = {
        'family': 'result',
        'macros': [
            {
                'name': 'llm_proposed_result_true',
                'description': 'injected candidate',
                'family': 'result',
                'steps': [
                    {'action': 'CreateScratch', 'args': [0, 0, 0]},
                    {'action': 'DefineScratchFromInt',
                     'args': [1, 'from_int:IntBoolean', 1]},
                    {'action': 'VerifyGoal',
                     'args': [0, 'from_int:StateScratchIndex', 1]},
                ],
            }
        ],
    }
    directory = tempfile.mkdtemp(prefix='macro-prior-')
    path = directory + '/macros.json'
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle)
    catalogue = catalogue_from_json(path)
    assert [m.name for m in catalogue] == ['llm_proposed_result_true']
    injected_env = MacroActionEnv(_binary_int_env(), catalogue=catalogue)
    assert injected_env.action_space_size() == 1
    _, _, terminated, _ = injected_env.step(1)
    assert terminated is True
    assert injected_env.full_state.goal_achieved() is True


def test_json_injection_rejects_malformed():
    import json
    import tempfile

    from env.macro_action import MacroActionError

    directory = tempfile.mkdtemp(prefix='macro-bad-')
    path = directory + '/bad.json'
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump({'macros': [{'name': 'x', 'steps': []}]}, handle)
    try:
        catalogue_from_json(path)
    except MacroActionError as e:
        assert 'steps' in str(e)
    else:  # pragma: no cover - the loader must reject this
        raise AssertionError('malformed catalogue was accepted')
