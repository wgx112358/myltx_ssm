"""Per-modality SSM memory module for long-term context compression.

Design:
  - Each DiT block gets two SSM modules (video + audio)
  - SSM state s ∈ R^{d_state, d_model} stores compressed history
  - On token eviction: compress evicted tokens into state via gated delta rule
  - On generation: query state to retrieve long-range context
  - Gated fusion merges SSM output with local attention output

The SSM state is orthogonal to KV-cache:
  - SSM: persistent, accumulative, never recached (memory layer)
  - KV-cache: sliding, recacheable on prompt switch (control layer)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SSMConfig:
    """Configuration for SSM memory modules."""

    enabled: bool = False
    d_state: int = 64
    gate_bias: float = -2.0  # init bias for fusion gate (starts near 0)


@dataclass
class SSMState:
    """Container for per-layer, per-modality SSM states.

    States are keyed by ``(layer_idx, modality)`` where modality is
    ``"video"`` or ``"audio"``.
    """

    states: dict[tuple[int, str], torch.Tensor] = field(default_factory=dict)

    def get(self, layer_idx: int, modality: str) -> torch.Tensor | None:
        return self.states.get((layer_idx, modality))

    def set(self, layer_idx: int, modality: str, state: torch.Tensor) -> None:
        self.states[(layer_idx, modality)] = state

    def clone(self) -> "SSMState":
        return SSMState(
            states={k: v.clone() for k, v in self.states.items()}
        )

    def scale_(self, factor: float) -> "SSMState":
        """Scale all stored state tensors in-place.

        Useful for switch-aware memory attenuation when prompt context changes.
        """
        if factor < 0:
            raise ValueError(f"factor must be non-negative, got {factor}")
        if factor == 1.0 or not self.states:
            return self

        for key, tensor in self.states.items():
            self.states[key] = tensor * factor
        return self

    @staticmethod
    def empty() -> "SSMState":
        return SSMState()


class SSMMemoryModule(nn.Module):
    """Gated delta-rule SSM memory for a single modality stream.

    Parameters per module ≈ 4 * d_state * d_model + d_model
    For video (d_model=4096, d_state=64): ~1.05M params
    For audio (d_model=2048, d_state=64): ~0.53M params

    State shape: [batch, d_state, d_model]
    """

    def __init__(self, d_model: int, d_state: int = 64, gate_bias: float = -2.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # --- Compression path (eviction -> state update) ---
        self.W_key = nn.Linear(d_model, d_state, bias=False)
        self.W_beta = nn.Linear(d_model, d_state, bias=True)

        # --- Retrieval path (current tokens -> query state) ---
        self.W_query = nn.Linear(d_model, d_state, bias=False)

        # --- Fusion gate (scalar per token) ---
        self.fusion_gate = nn.Linear(d_model, 1, bias=True)

        self._init_weights(gate_bias)

    def _init_weights(self, gate_bias: float) -> None:
        nn.init.constant_(self.fusion_gate.bias, gate_bias)
        for proj in (self.W_key, self.W_query):
            nn.init.normal_(proj.weight, std=0.02)
        nn.init.normal_(self.W_beta.weight, std=0.02)
        nn.init.zeros_(self.W_beta.bias)

    def compress(
        self,
        state: torch.Tensor,
        evicted_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Compress evicted tokens into SSM state.

        Args:
            state: [B, d_state, d_model] current SSM state.
            evicted_tokens: [B, n_evicted, d_model] tokens leaving the window.

        Returns:
            Updated state: [B, d_state, d_model].
        """
        if evicted_tokens.shape[1] == 0:
            return state

        # Write keys: distribute each token across memory slots
        keys = self.W_key(evicted_tokens)  # [B, n, d_state]
        write_weights = F.softmax(keys, dim=1)  # normalise over tokens

        # Aggregate: weighted sum of token values per memory slot
        # [B, d_state, n] @ [B, n, d_model] -> [B, d_state, d_model]
        update = torch.bmm(write_weights.transpose(1, 2), evicted_tokens)

        # Content-dependent decay
        summary = evicted_tokens.mean(dim=1)  # [B, d_model]
        beta = torch.sigmoid(self.W_beta(summary))  # [B, d_state]

        return beta.unsqueeze(2) * state + update

    def query(
        self,
        state: torch.Tensor,
        current_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Retrieve long-range context from SSM state.

        Args:
            state: [B, d_state, d_model].
            current_tokens: [B, seq_len, d_model].

        Returns:
            Gated SSM output: [B, seq_len, d_model].
        """
        q = self.W_query(current_tokens)  # [B, seq_len, d_state]
        retrieved = torch.bmm(q, state)  # [B, seq_len, d_model]
        gate = torch.sigmoid(self.fusion_gate(current_tokens))  # [B, seq_len, 1]
        return gate * retrieved

    def init_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.zeros(
            batch_size, self.d_state, self.d_model, device=device, dtype=dtype
        )
