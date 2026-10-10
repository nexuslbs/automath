"""The SIMPLE-ML llm-like network (design sections 3.2 - 3.6).

Architecture, exactly as the frozen layer table in ``docs/evolution/LLM_LIKE.md``
section 3.2 (all widths constants, ``d = 64`` primary)::

    node embedding      : Embedding(10, d)
    node projection     : Linear(21 -> d) + bias
    message passing (1) : tanh(W_self h + W_neigh * mean_neigh(h) + b_msg)
    state pooling       : masked mean over nodes -> s_t in R^d
    action projection   : Linear(14 + d -> d) + bias
    context projection  : Linear(2d -> d) + bias
    position embedding  : Embedding(K, d)
    attention (1 head)  : W_q, W_k, W_v, W_o : d x d each (+4 biases of d)
    policy head         : logit_j = w_out . tanh(W_c [c ; a_j] + b_c)
    value head          : v(s,s') = w_v . tanh(W_v [s ; s'] + b_v)

ONE message-passing round, ONE attention head, ONE layer, NO feed-forward
sublayer, NO layer-norm stack. The parameter count is checked at runtime and is
printed by every training/eval entry point; at ``d = 64`` it is exactly 57,408
(design section 3.6), well under the 300,000 target.

The global scalars of design section 3.3 (node count, step, active-axiom count,
objective count) are represented inside the per-node features and the context
window; the frozen layer table (section 3.6) has no separate global projection,
so adding one would change the contracted parameter count. That choice is
documented here rather than silently deviating from either.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from . import features as F

DEFAULT_WIDTH = 64
DEFAULT_CONTEXT = 8


class LLMAgentNet(nn.Module):
    """Graph encoder + exactly one attention head + policy and value heads."""

    def __init__(self, d: int = DEFAULT_WIDTH, k_context: int = DEFAULT_CONTEXT,
                 n_types: int = F.N_TYPES) -> None:
        super().__init__()
        if d <= 0:
            raise ValueError("width must be positive")
        if k_context <= 0:
            raise ValueError("context length must be positive")
        self.d = int(d)
        self.k = int(k_context)

        # graph encoder (design 3.2, 3.3)
        self.node_type_emb = nn.Embedding(int(n_types), self.d)
        self.node_proj = nn.Linear(F.NODE_IN, self.d)
        self.w_self = nn.Linear(self.d, self.d, bias=False)
        self.w_neigh = nn.Linear(self.d, self.d, bias=False)
        self.b_msg = nn.Parameter(torch.zeros(self.d))

        # action / context encoders
        self.action_proj = nn.Linear(F.ACTION_BASE + self.d, self.d)
        self.context_proj = nn.Linear(2 * self.d, self.d)
        self.pos_emb = nn.Embedding(self.k, self.d)

        # exactly ONE attention head, ONE layer
        self.w_q = nn.Linear(self.d, self.d)
        self.w_k = nn.Linear(self.d, self.d)
        self.w_v = nn.Linear(self.d, self.d)
        self.w_o = nn.Linear(self.d, self.d)

        # policy and value heads
        self.policy = nn.Linear(2 * self.d, self.d)
        self.w_out = nn.Linear(self.d, 1, bias=False)
        self.value_proj = nn.Linear(2 * self.d, self.d)
        self.w_v_head = nn.Linear(self.d, 1, bias=False)

        self.reset_parameters()

    # -- parameter accounting ---------------------------------------------

    def param_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def param_breakdown(self) -> Dict[str, int]:
        """The exact layer-by-layer count of design section 3.6."""
        d = self.d
        return {
            "node_type_emb(10,d)": int(self.node_type_emb.weight.numel()),
            "W_node(21,d)+b": int(self.node_proj.weight.numel()
                                  + self.node_proj.bias.numel()),
            "W_self(d,d)": int(self.w_self.weight.numel()),
            "W_neigh(d,d)": int(self.w_neigh.weight.numel()),
            "b_msg(d)": int(self.b_msg.numel()),
            "W_a(14+d,d)+b": int(self.action_proj.weight.numel()
                                 + self.action_proj.bias.numel()),
            "W_ctx(2d,d)+b": int(self.context_proj.weight.numel()
                                 + self.context_proj.bias.numel()),
            "pos_emb(K,d)": int(self.pos_emb.weight.numel()),
            "attention(4d*d+4d)": int(
                self.w_q.weight.numel() + self.w_q.bias.numel()
                + self.w_k.weight.numel() + self.w_k.bias.numel()
                + self.w_v.weight.numel() + self.w_v.bias.numel()
                + self.w_o.weight.numel() + self.w_o.bias.numel()),
            "policy(2d,d)+b+w_out(d)": int(
                self.policy.weight.numel() + self.policy.bias.numel()
                + self.w_out.weight.numel()),
            "value(2d,d)+b+w_v(d)": int(
                self.value_proj.weight.numel() + self.value_proj.bias.numel()
                + self.w_v_head.weight.numel()),
            "TOTAL": self.param_count(),
        }

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # -- graph encoder -----------------------------------------------------

    def encode_graph(self, node_features, type_idx, adj
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One message-passing round + masked mean pooling.

        ``node_features`` is (N, 21), ``type_idx`` is (N,) and ``adj`` is
        (N, N). Returns ``(pooled (d,), node_latent (N, d))``.
        """
        nf = torch.as_tensor(node_features, dtype=torch.float32)
        ti = torch.as_tensor(type_idx, dtype=torch.long)
        adjacency = torch.as_tensor(adj, dtype=torch.float32)
        if nf.dim() == 1:
            nf = nf.unsqueeze(0)
        if ti.dim() == 0:
            ti = ti.unsqueeze(0)
        if adjacency.dim() == 1:
            adjacency = adjacency.unsqueeze(0)

        h0 = torch.tanh(self.node_proj(nf) + self.node_type_emb(ti))
        degree = adjacency.sum(dim=1, keepdim=True).clamp(min=1.0)
        neigh_mean = (adjacency / degree) @ h0
        h1 = torch.tanh(self.w_self(h0) + self.w_neigh(neigh_mean) + self.b_msg)
        pooled = h1.mean(dim=0) if h1.shape[0] > 0 else torch.zeros(self.d)
        return pooled, h1

    # -- action encoder ----------------------------------------------------

    def action_latents(self, action_features, action_ref_local,
                       node_latent: torch.Tensor) -> torch.Tensor:
        """Project each action feature + its referenced-node mean to R^d."""
        af = torch.as_tensor(action_features, dtype=torch.float32)
        if af.dim() == 1:
            af = af.unsqueeze(0)
        count = af.shape[0]
        ref_mean = torch.zeros(count, self.d)
        for j, refs in enumerate(action_ref_local):
            if refs:
                idx = torch.as_tensor(sorted(set(int(r) for r in refs)),
                                      dtype=torch.long)
                ref_mean[j] = node_latent.index_select(0, idx).mean(dim=0)
        combined = torch.cat([af, ref_mean], dim=1)
        return torch.tanh(self.action_proj(combined))

    # -- context + one attention head --------------------------------------

    def context_tokens(self, hist_states: Sequence[torch.Tensor],
                       hist_actions: Sequence[torch.Tensor],
                       current_state: torch.Tensor
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the padded ``K`` (state, chosen action) context tokens.

        The last token is the current-state query token (with a zero action
        slot, because the action at ``t`` is the prediction target). Tokens are
        front-padded so the query is always the last row; the mask marks the
        real (unpadded) rows.
        """
        pairs = list(zip(hist_states, hist_actions))[-(self.k - 1):]
        tokens: List[torch.Tensor] = []
        for state_latent, action_latent in pairs:
            token = torch.tanh(self.context_proj(
                torch.cat([state_latent, action_latent])))
            tokens.append(token)
        zero_action = torch.zeros(self.d, dtype=current_state.dtype)
        query = torch.tanh(self.context_proj(
            torch.cat([current_state, zero_action])))
        tokens.append(query)

        pad = self.k - len(tokens)
        padded = torch.zeros(self.k, self.d, dtype=current_state.dtype)
        for i, token in enumerate(tokens):
            padded[pad + i] = token
        positions = torch.arange(self.k, dtype=torch.long)
        padded = padded + self.pos_emb(positions)
        mask = torch.zeros(self.k, dtype=torch.float32)
        mask[pad:] = 1.0
        return padded, mask

    def attend(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """One head: softmax(Q K^T / sqrt(d)) V, then W_o. Returns c in R^d."""
        query = self.w_q(tokens[-1])
        keys = self.w_k(tokens)
        values = self.w_v(tokens)
        scores = (keys @ query) / math.sqrt(float(self.d))
        scores = scores.masked_fill(mask <= 0.0, -1.0e9)
        weights = torch.softmax(scores, dim=0)
        return self.w_o((weights.unsqueeze(1) * values).sum(dim=0))

    # -- heads -------------------------------------------------------------

    def policy_logits(self, context_vec: torch.Tensor,
                      action_latents: torch.Tensor) -> torch.Tensor:
        """One logit per legal action (the action count is never a width)."""
        count = action_latents.shape[0]
        expanded = context_vec.unsqueeze(0).expand(count, self.d)
        hidden = torch.tanh(self.policy(torch.cat([expanded, action_latents],
                                                  dim=1)))
        return self.w_out(hidden).squeeze(1)

    def value(self, state_latent: torch.Tensor,
              next_state_latent: torch.Tensor) -> torch.Tensor:
        """A scalar score for the ``(s, s')`` transition."""
        pair = torch.cat([state_latent, next_state_latent]).unsqueeze(0)
        return self.w_v_head(torch.tanh(self.value_proj(pair))).squeeze()

    # -- convenience: a full step from raw feature dicts -------------------

    def forward_step(self, hist_states, hist_actions, node_features, type_idx,
                     adj, action_features, action_ref_local
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a step and return (pooled, node_latent, action_latents, c)."""
        pooled, node_latent = self.encode_graph(node_features, type_idx, adj)
        action_latent = self.action_latents(action_features, action_ref_local,
                                            node_latent)
        tokens, mask = self.context_tokens(hist_states, hist_actions, pooled)
        context_vec = self.attend(tokens, mask)
        return pooled, node_latent, action_latent, context_vec


def build_model(d: int = DEFAULT_WIDTH, k_context: int = DEFAULT_CONTEXT,
                seed: Optional[int] = None) -> LLMAgentNet:
    """Instantiate the net (optionally seeded) and assert the target bound."""
    if seed is not None:
        torch.manual_seed(int(seed))
    net = LLMAgentNet(d=d, k_context=k_context)
    if net.param_count() > 300000:
        raise AssertionError("parameter count %d exceeds the 300k target"
                             % net.param_count())
    return net
