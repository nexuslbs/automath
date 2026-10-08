"""Small-LLM action proposer (design unit U3a, item e).

A BaseAgent that turns a bounded observation plus the allowed action type
names into an OpenAI-compatible chat completion request, parses the reply as
JSON ``{"action": "<TypeName>", "args": [...]}`` and returns a RawAction.

Stdlib only (urllib.request): no requests, no openai.
"""
from __future__ import annotations
import json
import os
import typing
import urllib.error
import urllib.request

from env import meta_env
from env.action import RawAction
from env.full_state import FullState


class LlmUnavailableError(RuntimeError):
    """Raised when the configured LLM endpoint cannot be reached."""


def allowed_action_entries(state: FullState) -> list[tuple[str, int]]:
    """Return (TypeName, action_index) for every allowed basic action."""
    meta = state.meta.apply().real(meta_env.MetaInfo)
    allowed = meta.allowed_basic_actions.apply().real(
        meta_env.GeneralTypeGroup
    ).as_tuple
    entries: list[tuple[str, int]] = []
    for index, type_node in enumerate(allowed, start=1):
        node_type = type_node.type
        entries.append((node_type.__name__, index))
    return entries


def build_messages(state: FullState, max_nodes: int = 40) -> list[dict[str, str]]:
    """The exact messages the agent posts to the chat endpoint."""
    observation = state.render_observation(max_nodes=max_nodes)
    entries = allowed_action_entries(state)
    action_lines = [
        f'- {name} (action_index={index}, args: [int, int, int])'
        for name, index in entries
    ]
    system = (
        'You choose the next action for a symbolic-math environment. '
        'Reply with a single JSON object and nothing else: '
        '{"action": "<ActionTypeName>", "args": [int, int, int]}. '
        'The action must be one of the allowed action type names.'
    )
    user = (
        'Observation:\n'
        f'{observation}\n\n'
        'Allowed action types:\n'
        + '\n'.join(action_lines)
        + '\n\nChoose the next action as JSON.'
    )
    return [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': user},
    ]


def build_payload(state: FullState, model: str, max_nodes: int = 40) -> dict:
    return {
        'model': model,
        'messages': build_messages(state, max_nodes=max_nodes),
        'temperature': 0.0,
    }


class LlmAgent:
    """BaseAgent-compatible policy backed by an OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
        max_nodes: int = 40,
    ):
        self.base_url = (
            base_url
            or os.environ.get('AUTOMATH_LLM_BASE_URL')
            or 'http://localhost:8000'
        ).rstrip('/')
        self.model = model or os.environ.get('AUTOMATH_LLM_MODEL') or 'local-model'
        self.timeout = timeout
        self.max_nodes = max_nodes
        self.last_prompt: list[dict[str, str]] | None = None
        self.last_error: str | None = None

    def build_payload(self, state: FullState) -> dict:
        return build_payload(state, model=self.model, max_nodes=self.max_nodes)

    def _chat(self, payload: dict) -> str:
        url = f'{self.base_url}/v1/chat/completions'
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode('utf-8')
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            raise LlmUnavailableError(
                f'no LLM reachable at {url}: {e}'
            ) from e
        try:
            data = json.loads(body)
            return data['choices'][0]['message']['content']
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise LlmUnavailableError(
                f'LLM at {url} returned an unusable response: {e}'
            ) from e

    @staticmethod
    def _extract_json(reply: str) -> dict:
        text = reply.strip()
        if text.startswith('```'):
            text = text.strip('`')
            if text.lower().startswith('json'):
                text = text[4:]
        start = text.find('{')
        end = text.rfind('}')
        if start < 0 or end < start:
            raise ValueError(f'no JSON object in reply: {reply[:200]!r}')
        return json.loads(text[start:end + 1])

    def select_action(self, state: FullState) -> RawAction:
        payload = self.build_payload(state)
        self.last_prompt = payload['messages']
        reply = self._chat(payload)
        entries = allowed_action_entries(state)
        try:
            data = self._extract_json(reply)
            name = str(data['action'])
            args = list(data.get('args') or [])
            index = next(
                (idx for entry_name, idx in entries if entry_name == name),
                None,
            )
            if index is None:
                raise ValueError(f'unknown action type: {name!r}')
            ints = [int(args[i]) if i < len(args) else 0 for i in range(3)]
            self.last_error = None
            return RawAction.with_raw_args(index, ints[0], ints[1], ints[2])
        except (ValueError, KeyError, TypeError, IndexError) as e:
            # Structured NO-OP: action_index 0 is rejected by the env as a
            # typed error row, never a Python crash.
            self.last_error = f'{type(e).__name__}: {e}'
            return RawAction.with_raw_args(0, 0, 0, 0)

    def train(self, *args, **kwargs) -> None:
        """No online learning for the LLM proposer."""

    def reset(self) -> None:
        self.last_error = None


def selftest() -> dict:
    """Print (and return) the EXACT request this agent would send."""
    from env.goal_env import GoalEnv
    from env.node_types import HaveScratch, ESSENTIAL_ACTIONS
    from env import core

    goal = HaveScratch.with_goal(core.Void())
    env = GoalEnv(goal=goal, allowed_actions=ESSENTIAL_ACTIONS, max_steps=3)
    agent = LlmAgent(base_url='http://localhost:8000', model='selftest-model')
    payload = agent.build_payload(env.full_state)
    url = f'{agent.base_url}/v1/chat/completions'
    print(f'POST {url}')
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == '__main__':
    selftest()
