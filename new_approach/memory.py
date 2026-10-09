"""Flavor C (task 4333, thread 4332): the separate MEMORY COMPARTMENT.

Thread-4332 intent (verbatim): "a /memory store keeps what was learned; the
main part mostly static (improved by fixed training) and the memory is updated
AFTER EVERY INFERENCE over a context to remember previous states".

Design
------
The evolved policy net (flavor A) is the STATIC main part: its weights are
fixed for the whole lifetime of one agent, and are changed only by the
evolutionary training loop.  ``MemoryStore`` is the SEPARATE, non-heritable
compartment: a bounded in-process map

    key   = canonical (target, stack)   (post-order canonical strings, see
            ``env.State.canonical`` and ``nodes.Node.canonical``)
    value = the best action observed for that exact state plus its observed
            reward and the last generation that touched it

The rollout calls ``MemoryStore.update`` after EVERY inference (action
selection), so a state revisited later in the same run is answered from what
was already learned.  Action selection adds a BOUNDED memory term

    alpha_mem * memory_score(target, stack, action)

to the static net score, where ``memory_score`` is the (clamped) observed
reward of the stored best action (0.0 for every other action, 0.0 on a miss).
With ``alpha_mem = 0.0`` the memory term is identically zero, so a flavor-C
config with ``alpha_mem = 0`` reproduces flavor-A behavior exactly.

Boundedness (the feasibility control)
-------------------------------------
The store is a ``collections.OrderedDict`` with a HARD entry cap.  On
overflow the least-recently-used entry is evicted (``lru=True``); with
``lru=False`` it degrades to FIFO.  ``hash_keys=True`` replaces the canonical
string key with an 8-byte blake2b digest, roughly halving key RAM.
``update_every_inference=False`` turns the after-inference hook off;
``enabled=False`` turns the whole compartment off.  ``drop_old`` implements
the optional per-generation shard + drop-old policy.

Nothing here is persisted to disk, and nothing here is unbounded.  The prior
OOM lesson on this host was an UNBOUNDED dense run (RC137); this module exists
so the memory term can never repeat it.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

#: Default hard cap on live entries (config key ``memory.cap``).
DEFAULT_CAP = 200_000
#: Default blend weight of the memory term (config key ``memory.alpha_mem``).
DEFAULT_ALPHA_MEM = 1.0


def canonical_state_key(target, stack: Sequence) -> str:
    """Canonical ``(target, stack)`` key string.

    Uses the SAME canonicalization the rest of the code base uses: each node's
    ``canonical()`` (post-order, via the ``nodes`` module) joined by ``|`` and
    ``[..]``.  The target is rendered first so the key identifies the case; a
    missing target degrades to ``"<none>"`` (only possible on the legacy
    non-target path).
    """
    tgt = target.canonical() if target is not None else "<none>"
    stk = "[" + ", ".join(n.canonical() for n in stack) + "]"
    return "%s|%s" % (tgt, stk)


def hash_state_key(key: str) -> bytes:
    """Stable 8-byte digest of a canonical key (halves key RAM)."""
    return hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()


def _clamp_unit(value: float) -> float:
    if value > 1.0:
        return 1.0
    if value < -1.0:
        return -1.0
    return float(value)


class MemoryStore:
    """Bounded in-process (target, stack) -> best-action memory.

    Parameters mirror ``config/evolution_targetC.json`` key ``memory``:
    ``enabled``, ``cap``, ``lru``, ``hash_keys``, ``alpha_mem``,
    ``update_every_inference`` and ``seed``.
    """

    def __init__(
        self,
        cap: int = DEFAULT_CAP,
        lru: bool = True,
        hash_keys: bool = False,
        alpha_mem: float = DEFAULT_ALPHA_MEM,
        enabled: bool = True,
        update_every_inference: bool = True,
        seed: int = 20261009,
    ) -> None:
        cap = int(cap)
        if cap <= 0:
            raise ValueError("memory cap must be positive, got %r" % (cap,))
        self.cap = cap
        self.lru = bool(lru)
        self.hash_keys = bool(hash_keys)
        self.alpha_mem = float(alpha_mem)
        self.enabled = bool(enabled)
        self.update_every_inference = bool(update_every_inference)
        self.seed = int(seed)
        # value row: [best_action: str, best_reward: float,
        #             last_generation: Optional[int], visits: int]
        self._d: "OrderedDict[object, List[object]]" = OrderedDict()
        # observability counters (cheap, never affect behaviour)
        self.hits = 0
        self.misses = 0
        self.updates = 0
        self.evictions = 0

    # -- key handling ---------------------------------------------------
    def key_for(self, target, stack: Sequence) -> object:
        """The stored key object for a state (str, or 8-byte digest)."""
        key = canonical_state_key(target, stack)
        return hash_state_key(key) if self.hash_keys else key

    # -- read -----------------------------------------------------------
    def lookup(self, target, stack) -> Optional[List[object]]:
        """Return the entry row for an exact hit, or ``None``."""
        if not self.enabled:
            return None
        k = self.key_for(target, stack)
        entry = self._d.get(k)
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        if self.lru:
            self._d.move_to_end(k)
        return entry

    def best_action(self, target, stack) -> Optional[str]:
        """The stored best action for an exact hit, or ``None``."""
        entry = self.lookup(target, stack)
        return None if entry is None else str(entry[0])

    def score(self, target, stack, action: str) -> float:
        """Bounded memory term for ``action`` at ``(target, stack)``.

        Exact hit and action equals the stored best action -> the observed
        reward, clamped to [-1, 1].  Everything else -> 0.0.
        """
        if not self.enabled:
            return 0.0
        entry = self.lookup(target, stack)
        if entry is None or str(entry[0]) != action:
            return 0.0
        return _clamp_unit(float(entry[1]))

    # -- write (the after-inference hook) -------------------------------
    def update(self, target, stack, action: str, reward: float,
               generation: Optional[int] = None) -> bool:
        """Record one observed (state, action, reward) after an inference.

        Keeps the best-reward action per exact state, bumps the visit count and
        evicts the oldest entry when the hard cap is exceeded.  Returns True
        when an entry was written or merged.
        """
        if not self.enabled or not self.update_every_inference:
            return False
        k = self.key_for(target, stack)
        reward = float(reward)
        entry = self._d.get(k)
        if entry is not None:
            entry[3] = int(entry[3]) + 1  # type: ignore[operator]
            if reward > float(entry[1]):
                entry[1] = reward
                entry[0] = action
            if generation is not None:
                entry[2] = int(generation)
            if self.lru:
                self._d.move_to_end(k)
        else:
            self._d[k] = [action, reward, generation, 1]
            if len(self._d) > self.cap:
                self._d.popitem(last=False)  # evict LRU / FIFO oldest
                self.evictions += 1
        self.updates += 1
        return True

    # -- optional per-generation shard + drop-old ------------------------
    def drop_old(self, keep_from_generation: int) -> int:
        """Drop entries last touched before ``keep_from_generation``.

        Only entries written with an explicit ``generation`` participate; rows
        with ``last_generation is None`` are kept.  Returns the number dropped.
        """
        drop = [k for k, row in self._d.items()
                if row[2] is not None and int(row[2]) < keep_from_generation]
        for k in drop:
            del self._d[k]
        return len(drop)

    # -- introspection --------------------------------------------------
    def __len__(self) -> int:
        return len(self._d)

    def stats(self) -> Dict[str, object]:
        return {
            "entries": len(self._d),
            "cap": self.cap,
            "lru": self.lru,
            "hash_keys": self.hash_keys,
            "alpha_mem": self.alpha_mem,
            "enabled": self.enabled,
            "update_every_inference": self.update_every_inference,
            "hits": self.hits,
            "misses": self.misses,
            "updates": self.updates,
            "evictions": self.evictions,
        }


def memory_from_config(cfg_memory: Optional[dict], seed: int) -> Optional[
        MemoryStore]:
    """Build a ``MemoryStore`` from a config block, or ``None`` when disabled.

    ``cfg_memory`` is the ``memory`` object of an ``EvoConfig``; an empty or
    missing block (or ``enabled: false``) returns ``None``, so every existing
    config reproduces flavor A exactly.
    """
    block = dict(cfg_memory or {})
    if not block.get("enabled", False):
        return None
    return MemoryStore(
        cap=int(block.get("cap", DEFAULT_CAP)),
        lru=bool(block.get("lru", True)),
        hash_keys=bool(block.get("hash_keys", False)),
        alpha_mem=float(block.get("alpha_mem", DEFAULT_ALPHA_MEM)),
        enabled=True,
        update_every_inference=bool(block.get("update_every_inference", True)),
        seed=int(block.get("seed", seed)),
    )
