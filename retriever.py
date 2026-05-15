"""
models/retriever.py
Adaptive Retriever R_φ with differentiable threshold τ.

Key components (Section 3 of the paper):
  - Item / user encoders producing dense embeddings.
  - Relevance scorer: s_{u,i} = R_φ(H^t_u, i)  [Eq. 5]
  - Smooth relaxation of hard threshold selection [Eq. 6-7]
  - Soft weight injection into generator embedding space [Eq. 9]
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Item & User Encoders
# ---------------------------------------------------------------------------

class ItemEncoder(nn.Module):
    """
    Learnable item embedding table.
    In the full system this would be initialised from text features
    (e.g. frozen LLM CLS embeddings); here we use a trainable lookup.
    """

    def __init__(self, num_items: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            item_ids: (B, L) or (B,) – item indices
        Returns:
            embeddings: (..., embedding_dim)
        """
        return self.embedding(item_ids)


class UserHistoryEncoder(nn.Module):
    """
    Encode a variable-length user history into a single context vector
    via a mean-pooled projection (lightweight; can be replaced with
    Transformer / GRU for stronger modelling).
    """

    def __init__(self, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        history_embs: torch.Tensor,   # (B, L, emb_dim)
        history_lens: torch.Tensor,   # (B,)  actual lengths
    ) -> torch.Tensor:                # (B, hidden_dim)
        # Masked mean pooling
        mask = torch.arange(history_embs.size(1), device=history_embs.device)
        mask = mask.unsqueeze(0) < history_lens.unsqueeze(1)  # (B, L)
        mask = mask.unsqueeze(-1).float()                      # (B, L, 1)
        pooled = (history_embs * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.proj(pooled)                               # (B, hidden_dim)


# ---------------------------------------------------------------------------
# Retriever scoring head
# ---------------------------------------------------------------------------

class RetrieverHead(nn.Module):
    """
    Bi-linear scorer: s_{u,i} = f(u_ctx)^T W f(i)
    where f(u_ctx) comes from UserHistoryEncoder and
          f(i)     comes from ItemEncoder.
    """

    def __init__(self, user_dim: int, item_dim: int) -> None:
        super().__init__()
        self.W = nn.Linear(item_dim, user_dim, bias=False)
        nn.init.xavier_uniform_(self.W.weight)

    def forward(
        self,
        user_ctx: torch.Tensor,   # (B, user_dim)
        item_embs: torch.Tensor,  # (B, C, item_dim)  C = candidate pool size
    ) -> torch.Tensor:            # (B, C)
        # item_embs projected to user space
        item_proj = self.W(item_embs)            # (B, C, user_dim)
        scores = torch.bmm(item_proj, user_ctx.unsqueeze(-1)).squeeze(-1)  # (B, C)
        return scores


# ---------------------------------------------------------------------------
# Adaptive Retriever (full module)
# ---------------------------------------------------------------------------

class AdaptiveRetriever(nn.Module):
    """
    Full adaptive retriever R_φ with learnable threshold τ.

    Forward pass:
      1. Encode user history -> user context vector.
      2. Encode candidate items -> item embeddings.
      3. Compute relevance scores s_{u,i}  [Eq. 5]
      4. Compute soft weights w̃_i via sigmoid relaxation [Eq. 6-7]
      5. Return scores, soft weights, and the adaptive candidate set d̃.
    """

    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        hidden_dim: int,
        tau_init: float = 0.3,
        gamma: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.item_encoder = ItemEncoder(num_items, embedding_dim)
        self.user_encoder = UserHistoryEncoder(embedding_dim, hidden_dim)
        self.scorer = RetrieverHead(hidden_dim, embedding_dim)

        # Learnable threshold τ  (scalar, jointly optimised)
        self.tau = nn.Parameter(torch.tensor(tau_init))
        self.gamma = gamma
        self.eps = eps

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def compute_scores(
        self,
        histories: torch.Tensor,       # (B, L)
        history_lens: torch.Tensor,    # (B,)
        candidate_pools: torch.Tensor, # (B, C)
    ) -> torch.Tensor:                 # (B, C) relevance scores
        hist_embs = self.item_encoder(histories)            # (B, L, emb)
        user_ctx  = self.user_encoder(hist_embs, history_lens)  # (B, hid)
        cand_embs = self.item_encoder(candidate_pools)      # (B, C, emb)
        scores = self.scorer(user_ctx, cand_embs)           # (B, C)
        return scores

    def soft_weights(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Eq. 7:  w̃_i = max( σ((s_{u,i} − τ) / γ), ε )
        """
        w = torch.sigmoid((scores - self.tau) / self.gamma)
        w = torch.clamp(w, min=self.eps)
        return w                                            # (B, C)

    def adaptive_set_mask(self, soft_w: torch.Tensor) -> torch.Tensor:
        """
        Eq. 6:  item i is in d̃  iff  w̃_i > ε
        Returns a boolean mask (B, C).
        """
        return soft_w > self.eps

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        histories: torch.Tensor,        # (B, L)
        history_lens: torch.Tensor,     # (B,)
        candidate_pools: torch.Tensor,  # (B, C)
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            scores    : (B, C) raw relevance scores
            soft_w    : (B, C) soft selection weights w̃_i
            mask      : (B, C) bool – adaptive candidate set membership
        """
        scores = self.compute_scores(histories, history_lens, candidate_pools)
        soft_w = self.soft_weights(scores)
        mask   = self.adaptive_set_mask(soft_w)

        return {
            "scores": scores,
            "soft_w": soft_w,
            "mask":   mask,
        }

    # ------------------------------------------------------------------
    # Soft weight scaling for generator embeddings [Eq. 9]
    # ------------------------------------------------------------------

    def scale_generator_embeddings(
        self,
        gen_embs: torch.Tensor,   # (B, C, gen_emb_dim) – from generator's embed layer
        soft_w: torch.Tensor,     # (B, C)
    ) -> torch.Tensor:            # (B, C, gen_emb_dim)
        """
        ẽ_{d_j} = w̃_{d_j} · e_{d_j}   [Eq. 9]
        Inject soft weights into the generator's embedding space so that
        gradients flow back to φ and τ through ∂w̃/∂s and ∂w̃/∂τ.
        """
        return gen_embs * soft_w.unsqueeze(-1)
