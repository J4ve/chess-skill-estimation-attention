"""
Self-contained attention module for the RatingNet extension.

Implements additive (Bahdanau-style) and scaled-dot-product self-attention
over BiLSTM outputs, with **causal-cumulative** variants that attend only
over plies 1..t at ply t.  The causal variants are wired into the rating
head so that attention receives gradient during training while preserving
the per-move output shape and introducing no lookahead.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_causal_mask(
    seq_len: int,
    lengths: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Build a (batch, seq, seq) boolean causal + padding mask.

    Entry ``[b, t, s]`` is ``True`` iff ``s <= t`` AND ``s < lengths[b]``.
    This allows each query position *t* to attend only to past-and-present
    keys within the valid (non-padded) region of the sequence.
    """
    lengths = lengths.to(device)
    # (seq, seq) lower-triangular causal mask
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    # (batch, 1, seq) padding mask — True for valid positions
    pad_mask = (
        torch.arange(seq_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    ).unsqueeze(1)
    # Combine: (batch, seq, seq)
    return causal.unsqueeze(0) & pad_mask


class BahdanauAttention(nn.Module):
    """Additive attention over a sequence of hidden states.

    Given a sequence H of shape (batch, seq, hidden_dim), the module computes
    an energy score for each query-key pair:

        e(t, s) = v^T · tanh(W_q · h_t + W_k · h_s)

    where W_q and W_k project the query and key into a shared attention space,
    and v is a learned scoring vector (Bahdanau et al. 2015).  This is the
    full additive (query-key) form: the importance of key s depends on the
    query position t, unlike a content-only variant where scores are
    query-independent.

    Args:
        hidden_dim: Dimension of each LSTM output vector.
        attention_dim: Intermediate projection size for the score function.
    """

    def __init__(self, hidden_dim: int, attention_dim: int = 64, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.key_projection = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.query_projection = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.v = nn.Parameter(torch.randn(attention_dim) / math.sqrt(attention_dim))

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Global (non-causal) attention — used for diagnostics / return_attention path.

        Uses a mean-pooled query (average over valid positions) so the output
        is a single context vector, matching the original interface.

        Args:
            hidden_states: (batch, seq, hidden_dim)
            mask: Optional (batch, seq) boolean/tensor mask. True/1 means keep.

        Returns:
            context: (batch, hidden_dim) attention-weighted context vector.
            weights: (batch, seq) per-step attention weights (sum to 1 over seq).
        """
        # Key projection: (batch, seq, attention_dim)
        keys = self.key_projection(hidden_states)
        # Global query: mean-pool hidden states as the query
        if mask is not None:
            mask_expanded = mask.bool().unsqueeze(-1).float()  # (batch, seq, 1)
            query_vec = (hidden_states * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp_min(1)
        else:
            query_vec = hidden_states.mean(dim=1)  # (batch, hidden_dim)
        # Query projection: (batch, attention_dim)
        query_proj = self.query_projection(query_vec)  # (batch, attention_dim)
        # Additive scoring: e(s) = v^T · tanh(W_q·q + W_k·h_s)
        # (batch, 1, attn_dim) + (batch, seq, attn_dim)
        combined = torch.tanh(query_proj.unsqueeze(1) + keys)
        scores = torch.matmul(combined, self.v)  # (batch, seq)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), float("-inf"))
        weights = F.softmax(scores, dim=-1)  # (batch, seq)
        context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        return context, weights

    def forward_causal(
        self,
        hidden_states: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Causal-cumulative attention: at ply *t*, attend only over plies 1..t.

        Full additive scoring: ``e(t, s) = v^T · tanh(W_q·h_t + W_k·h_s)``
        so the importance of key *s* differs per query position *t*.

        Returns a **per-ply** context tensor (not a single global vector), so
        the per-move output shape is preserved and no lookahead is introduced.

        Args:
            hidden_states: (batch, seq, hidden_dim)
            lengths: (batch,) actual sequence lengths.

        Returns:
            context: (batch, seq, hidden_dim) per-ply context vectors.
            weights: (batch, seq, seq) causal attention weights.  ``weights[b, t, :]``
                sums to 1 over the valid keys ``s <= t, s < lengths[b]``.
        """
        batch, seq, _ = hidden_states.size()
        # Key projection: (batch, seq_k, attention_dim)
        keys = self.key_projection(hidden_states)
        # Query projection: (batch, seq_q, attention_dim)
        queries = self.query_projection(hidden_states)
        # Additive scoring: e(t, s) = v^T · tanh(W_q·h_t + W_k·h_s)
        # queries: (batch, seq_q, 1, attn_dim) + keys: (batch, 1, seq_k, attn_dim)
        combined = torch.tanh(queries.unsqueeze(2) + keys.unsqueeze(1))  # (batch, seq_q, seq_k, attn_dim)
        scores = torch.matmul(combined, self.v)  # (batch, seq_q, seq_k)
        # Causal + padding mask: (batch, seq, seq)
        mask = _build_causal_mask(seq, lengths, hidden_states.device)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)  # (batch, seq_q, seq_k)
        # Handle rows that are fully masked (padded query positions):
        weights = weights.nan_to_num(0.0)
        # Per-ply context: (batch, seq, hidden_dim)
        context = torch.bmm(weights, hidden_states)
        return context, weights



class SelfAttention(nn.Module):
    """Simple scaled dot-product self-attention over LSTM outputs.

    Computes attention weights by treating each time step as both key and
    query, using a learned linear projection. Useful as an alternative to
    Bahdanau attention in ablations.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Global (non-causal) self-attention — used for diagnostics.

        Returns:
            output: (batch, seq, hidden_dim)
            step_importance: (batch, seq) mean attention received per position.
        """
        batch, seq, _ = hidden_states.size()
        q = self.q_proj(hidden_states).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool().unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(scores, dim=-1)  # (batch, heads, seq, seq)
        weights = self.dropout(weights)
        attn_output = torch.matmul(weights, v)  # (batch, heads, seq, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq, self.hidden_dim)
        output = self.out_proj(attn_output)
        # Per-step importance (batch, seq), averaged over heads and query
        # positions, matching BahdanauAttention's weights shape.
        step_importance = weights.mean(dim=(1, 2))
        return output, step_importance

    def forward_causal(
        self,
        hidden_states: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Causal-cumulative self-attention: query *t* attends to keys 1..t only.

        Args:
            hidden_states: (batch, seq, hidden_dim)
            lengths: (batch,) actual sequence lengths.

        Returns:
            output: (batch, seq, hidden_dim) attended representations.
            step_importance: (batch, seq) mean attention received per position.
        """
        batch, seq, _ = hidden_states.size()
        q = self.q_proj(hidden_states).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Causal + padding mask: (batch, seq, seq) → (batch, 1, seq, seq) for heads
        causal_mask = _build_causal_mask(seq, lengths, hidden_states.device).unsqueeze(1)
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)  # (batch, heads, seq, seq)
        weights = weights.nan_to_num(0.0)
        weights = self.dropout(weights)
        attn_output = torch.matmul(weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq, self.hidden_dim)
        output = self.out_proj(attn_output)
        step_importance = weights.mean(dim=(1, 2))
        return output, step_importance
